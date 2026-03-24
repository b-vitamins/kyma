from __future__ import annotations

from typing import cast

import torch
from torch import nn

from kyma.data import KymaTimeFeatures, KymaTokenizedPiece
from kyma.eval import (
    StreamingEvalSpec,
    estimate_decode_session_bytes,
    evaluate_streaming,
    slice_streaming_piece,
)
from kyma.inference import prefill_decode_session
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


def test_slice_streaming_piece_extracts_prompt_and_future_tokens() -> None:
    piece = _make_piece(
        "piece-a",
        [1, 2, 3, 4, 5, 6],
        [0.0, 1_000.0, 2_000.0, 3_000.0, 4_000.0, 5_000.0],
    )

    streaming_slice = slice_streaming_piece(
        piece,
        session_length_s=2,
        max_benchmark_tokens=3,
    )

    assert streaming_slice is not None
    assert torch.equal(streaming_slice.prompt_ids, torch.tensor([1, 2, 3]))
    assert torch.equal(streaming_slice.future_ids, torch.tensor([4, 5, 6]))


def test_estimate_decode_session_bytes_counts_logits() -> None:
    model = cast(KymaAutoregressiveLM, RuleBasedLM(vocab_size=8))
    piece = _make_piece(
        "piece-a",
        [1, 2, 3, 4],
        [0.0, 1_000.0, 2_000.0, 3_000.0],
    )
    streaming_slice = slice_streaming_piece(piece, session_length_s=1)

    assert streaming_slice is not None
    session = prefill_decode_session(
        model,
        streaming_slice.prompt_ids.unsqueeze(0),
        time_features=streaming_slice.prompt_time_features.unsqueeze(0),
        time_feature_mask=streaming_slice.prompt_time_feature_mask.unsqueeze(0),
    )
    assert estimate_decode_session_bytes(session) == 32


def test_evaluate_streaming_reports_latency_throughput_and_memory() -> None:
    model = cast(KymaAutoregressiveLM, RuleBasedLM(vocab_size=8))
    pieces = [
        _make_piece(
            "piece-a",
            [1, 2, 3, 4, 5, 6],
            [0.0, 1_000.0, 2_000.0, 3_000.0, 4_000.0, 5_000.0],
        )
    ]
    spec = StreamingEvalSpec(
        interactive_session_lengths_s=[1, 2],
        report_time_to_first_note_ms=True,
        report_decode_throughput=True,
        report_memory_growth=True,
    )

    report = evaluate_streaming(
        model,
        pieces,
        spec=spec,
        device="cpu",
        max_benchmark_tokens=3,
    )

    assert len(report.buckets) == 2
    assert [bucket.session_length_s for bucket in report.buckets] == [1, 2]
    assert all(bucket.examples_evaluated == 1 for bucket in report.buckets)
    assert all(bucket.mean_time_to_first_token_ms >= 0.0 for bucket in report.buckets)
    assert all(
        bucket.mean_decode_throughput_tokens_per_s > 0.0 for bucket in report.buckets
    )
    assert [bucket.mean_session_memory_bytes for bucket in report.buckets] == [
        32.0,
        32.0,
    ]
