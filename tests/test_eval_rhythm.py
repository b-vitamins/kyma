from __future__ import annotations

from typing import cast

import torch
from torch import nn

from kyma.data import KymaTimeFeatures, KymaTokenizedPiece
from kyma.eval import RhythmEvalSpec, evaluate_rhythm
from kyma.model import KymaAutoregressiveLM, KymaLMState


class RuleBasedLM(nn.Module):
    def __init__(self, *, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def _next_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
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


def _make_piece() -> KymaTokenizedPiece:
    values = torch.tensor(
        [
            [0.0, 0.0, 0.0, 120.0],
            [10.0, 10.0, 0.0, 120.0],
            [10.0, 20.0, 0.5, 90.0],
            [10.0, 30.0, 0.0, 90.0],
            [10.0, 40.0, 0.5, 140.0],
        ],
        dtype=torch.float32,
    )
    valid = torch.ones_like(values, dtype=torch.bool)
    return KymaTokenizedPiece(
        piece_id="piece-a",
        tokens=(
            ("prefix", "instrument", "piano"),
            ("onset", 0),
            ("dur", 10),
            ("onset", 20),
            ("dur", 30),
        ),
        token_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
        time_features=KymaTimeFeatures(values=values, valid=valid),
        metadata={},
        source_path=None,
    )


def test_evaluate_rhythm_reports_expected_metrics() -> None:
    model = cast(KymaAutoregressiveLM, RuleBasedLM(vocab_size=8))
    spec = RhythmEvalSpec(
        report_onset_nll=True,
        report_duration_nll=True,
        report_tempo_consistency=True,
        report_beat_phase_consistency=True,
    )

    report = evaluate_rhythm(model, [_make_piece()], spec=spec, device="cpu")

    assert report.onset_nll.count == 2
    assert report.onset_nll.value == 0.0
    assert report.duration_nll.count == 2
    assert report.duration_nll.value == 0.0
    assert report.tempo_consistency.count == 2
    assert report.tempo_consistency.value == 1.0
    assert report.beat_phase_consistency.count == 2
    assert report.beat_phase_consistency.value == 1.0
    assert len(report.examples) == 1
    assert report.examples[0].piece_id == "piece-a"
