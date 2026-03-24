from __future__ import annotations

from typing import cast

import torch
from torch import nn

from kyma.data import KymaTimeFeatures, KymaTokenizedPiece
from kyma.eval import (
    LongHorizonEvalSpec,
    evaluate_long_horizon,
    slice_long_horizon_piece,
)
from kyma.model import KymaAutoregressiveLM, KymaLMState


class CarrySensitiveLM(nn.Module):
    def __init__(self, *, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def _target_ids(
        self,
        input_ids: torch.Tensor,
        *,
        tokens_processed: int,
    ) -> torch.Tensor:
        if tokens_processed <= 1:
            return torch.zeros_like(input_ids)
        return input_ids + 1

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        time_features: torch.Tensor | None = None,
        time_feature_mask: torch.Tensor | None = None,
        state: KymaLMState | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KymaLMState]:
        del time_features, time_feature_mask

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        batch_size, seq_len = map(int, input_ids.shape)
        offset = 0 if state is None else state.tokens_processed
        logits = torch.full(
            (batch_size, seq_len, self.vocab_size),
            -1_000.0,
            dtype=torch.float32,
        )
        for position in range(seq_len):
            target_ids = self._target_ids(
                input_ids[:, position],
                tokens_processed=offset + position + 1,
            )
            logits[:, position, :].scatter_(1, target_ids.unsqueeze(-1), 0.0)
        next_state = KymaLMState(layer_states=(), tokens_processed=offset + seq_len)
        if not return_state:
            return logits
        return logits, next_state

    def step(
        self,
        input_ids: torch.Tensor,
        *,
        time_features: torch.Tensor | None = None,
        time_feature_mask: torch.Tensor | None = None,
        state: KymaLMState | None = None,
    ) -> tuple[torch.Tensor, KymaLMState]:
        del time_features, time_feature_mask

        if input_ids.ndim == 0:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim == 2 and input_ids.shape[1] == 1:
            input_ids = input_ids.squeeze(1)
        processed = (0 if state is None else state.tokens_processed) + 1
        target_ids = self._target_ids(input_ids, tokens_processed=processed)
        logits = torch.full(
            (input_ids.shape[0], self.vocab_size),
            -1_000.0,
            dtype=torch.float32,
        )
        logits.scatter_(1, target_ids.unsqueeze(-1), 0.0)
        next_state = KymaLMState(layer_states=(), tokens_processed=processed)
        return logits, next_state


def _make_piece(
    piece_id: str,
    token_ids: list[int],
    absolute_times_ms: list[float],
) -> KymaTokenizedPiece:
    values = torch.zeros((len(token_ids), 4), dtype=torch.float32)
    values[:, 0] = torch.tensor(absolute_times_ms, dtype=torch.float32)
    values[:, 1] = torch.tensor(absolute_times_ms, dtype=torch.float32)
    values[:, 2] = 0.0
    values[:, 3] = 120.0
    valid = torch.ones((len(token_ids), 4), dtype=torch.bool)
    return KymaTokenizedPiece(
        piece_id=piece_id,
        tokens=tuple(f"tok-{idx}" for idx in token_ids),
        token_ids=torch.tensor(token_ids, dtype=torch.long),
        time_features=KymaTimeFeatures(values=values, valid=valid),
        metadata={},
        source_path=None,
    )


def test_slice_long_horizon_piece_extracts_real_time_window() -> None:
    piece = _make_piece(
        "piece-a",
        [1, 2, 3, 4, 5, 6],
        [0.0, 1_000.0, 2_000.0, 3_000.0, 4_000.0, 5_000.0],
    )

    horizon_slice = slice_long_horizon_piece(
        piece,
        prompt_length_s=2,
        continuation_length_s=3,
    )

    assert horizon_slice is not None
    assert torch.equal(horizon_slice.prompt_ids, torch.tensor([1, 2, 3]))
    assert torch.equal(horizon_slice.reference_ids, torch.tensor([4, 5, 6]))


def test_evaluate_long_horizon_reports_reset_interval_ablations() -> None:
    model = cast(KymaAutoregressiveLM, CarrySensitiveLM(vocab_size=16))
    pieces = [
        _make_piece(
            "piece-a",
            [1, 2, 3, 4, 5, 6],
            [0.0, 1_000.0, 2_000.0, 3_000.0, 4_000.0, 5_000.0],
        )
    ]
    spec = LongHorizonEvalSpec(
        prompt_lengths_s=[2],
        continuation_lengths_s=[3],
        state_carry_reset_intervals=[0, 1],
        report_horizon_nll=True,
        report_structure_metrics=True,
    )

    report = evaluate_long_horizon(
        model,
        pieces,
        spec=spec,
        device="cpu",
    )

    assert len(report.buckets) == 2
    no_reset_bucket = next(
        bucket for bucket in report.buckets if bucket.state_carry_reset_interval == 0
    )
    reset_bucket = next(
        bucket for bucket in report.buckets if bucket.state_carry_reset_interval == 1
    )

    assert no_reset_bucket.examples_evaluated == 1
    assert no_reset_bucket.mean_continuation_nll == 0.0
    assert no_reset_bucket.examples[0].generated_ids == (4, 5, 6)
    assert no_reset_bucket.examples[0].reference_ids == (4, 5, 6)
    assert [point.horizon_index for point in no_reset_bucket.horizon_nll] == [1, 2, 3]
    assert all(point.mean_nll == 0.0 for point in no_reset_bucket.horizon_nll)

    assert reset_bucket.examples_evaluated == 1
    assert reset_bucket.mean_continuation_nll > 0.0
    assert reset_bucket.horizon_nll[0].mean_nll == 0.0
    assert reset_bucket.horizon_nll[1].mean_nll > 0.0
