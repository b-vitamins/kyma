from __future__ import annotations

from typing import cast

import torch
from torch import nn

from kyma.data import KymaTimeFeatures, KymaTokenizedPiece
from kyma.eval import (
    ShortContextParitySpec,
    evaluate_short_context_parity,
    slice_short_context_piece,
)
from kyma.model import KymaAutoregressiveLM, KymaLMState


class RuleBasedLM(nn.Module):
    def __init__(self, *, vocab_size: int, eos_token_id: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id

    def _next_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return (input_ids + 1).clamp(max=self.eos_token_id)

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
        logits = torch.full(
            (batch_size, seq_len, self.vocab_size),
            -1_000.0,
            dtype=torch.float32,
        )
        next_ids = self._next_ids(input_ids)
        logits.scatter_(2, next_ids.unsqueeze(-1), 0.0)
        next_state = KymaLMState(
            layer_states=(),
            tokens_processed=(0 if state is None else state.tokens_processed) + seq_len,
        )
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
        logits = torch.full(
            (input_ids.shape[0], self.vocab_size),
            -1_000.0,
            dtype=torch.float32,
        )
        next_ids = self._next_ids(input_ids)
        logits.scatter_(1, next_ids.unsqueeze(-1), 0.0)
        next_state = KymaLMState(
            layer_states=(),
            tokens_processed=(0 if state is None else state.tokens_processed) + 1,
        )
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


def test_slice_short_context_piece_extracts_prompt_and_targets() -> None:
    piece = _make_piece(
        "piece-a",
        [1, 2, 3, 4, 5],
        [0.0, 1_000.0, 2_000.0, 3_000.0, 4_000.0],
    )

    parity_slice = slice_short_context_piece(
        piece,
        prompt_duration_s=2,
        continuation_tokens=2,
    )

    assert parity_slice is not None
    assert torch.equal(parity_slice.prompt_ids, torch.tensor([1, 2, 3]))
    assert torch.equal(parity_slice.reference_ids, torch.tensor([4, 5]))
    assert torch.equal(parity_slice.eval_input_ids, torch.tensor([1, 2, 3, 4]))
    assert torch.equal(parity_slice.eval_target_ids, torch.tensor([2, 3, 4, 5]))
    assert torch.equal(
        parity_slice.eval_loss_mask,
        torch.tensor([False, False, True, True]),
    )


def test_slice_short_context_piece_skips_short_examples() -> None:
    piece = _make_piece(
        "piece-short",
        [1, 2, 3],
        [0.0, 1_000.0, 2_000.0],
    )

    parity_slice = slice_short_context_piece(
        piece,
        prompt_duration_s=2,
        continuation_tokens=2,
    )

    assert parity_slice is None


def test_evaluate_short_context_parity_reports_teacher_forced_nll() -> None:
    model = cast(KymaAutoregressiveLM, RuleBasedLM(vocab_size=8, eos_token_id=5))
    pieces = [
        _make_piece(
            "piece-a",
            [1, 2, 3, 4, 5],
            [0.0, 1_000.0, 2_000.0, 3_000.0, 4_000.0],
        ),
        _make_piece(
            "piece-b",
            [2, 3, 4, 5, 5],
            [0.0, 1_000.0, 2_000.0, 3_000.0, 4_000.0],
        ),
    ]
    spec = ShortContextParitySpec(
        prompt_durations_s=[2],
        continuation_tokens=2,
        temperature=0.0,
        min_p=0.1,
    )

    report = evaluate_short_context_parity(
        model,
        pieces,
        spec=spec,
        device="cpu",
        collect_generations=True,
    )

    assert len(report.buckets) == 1
    bucket = report.buckets[0]
    assert bucket.prompt_duration_s == 2
    assert bucket.examples_evaluated == 2
    assert bucket.total_continuation_tokens == 4
    assert bucket.mean_continuation_nll == 0.0
    assert bucket.perplexity == 1.0
    assert bucket.examples[0].generated_ids == (4, 5)
    assert bucket.examples[0].reference_ids == (4, 5)
    assert bucket.examples[1].generated_ids == (5, 5)
    assert bucket.examples[1].reference_ids == (5, 5)
