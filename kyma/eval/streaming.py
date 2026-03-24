"""Streaming systems evaluation for Kyma decode sessions."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, cast

import torch

from kyma.data import TIME_FEATURE_NAMES, KymaTokenizedPiece
from kyma.eval.protocol import StreamingEvalSpec
from kyma.inference import (
    KymaDecodeSession,
    KymaSamplingConfig,
    advance_decode_session,
    prefill_decode_session,
    sample_next_token,
)
from kyma.model import KymaAutoregressiveLM

_ABSOLUTE_TIME_FEATURE_IDX = TIME_FEATURE_NAMES.index("absolute_time_ms")


@dataclass(frozen=True)
class StreamingEvalSlice:
    """Prompt and future tokens used for streaming benchmark simulation."""

    piece_id: str
    session_length_s: int
    prompt_ids: torch.Tensor
    prompt_time_features: torch.Tensor
    prompt_time_feature_mask: torch.Tensor
    future_ids: torch.Tensor
    future_time_features: torch.Tensor
    future_time_feature_mask: torch.Tensor

    @property
    def prompt_token_count(self) -> int:
        return int(self.prompt_ids.shape[0])

    @property
    def future_token_count(self) -> int:
        return int(self.future_ids.shape[0])


@dataclass(frozen=True)
class StreamingExampleResult:
    """Per-piece streaming benchmark result."""

    piece_id: str
    session_length_s: int
    prompt_token_count: int
    future_token_count: int
    time_to_first_token_ms: float
    decode_throughput_tokens_per_s: float
    session_memory_bytes: int


@dataclass(frozen=True)
class StreamingBucketResult:
    """Aggregate streaming benchmark result for one session length."""

    session_length_s: int
    examples_evaluated: int
    mean_time_to_first_token_ms: float
    mean_decode_throughput_tokens_per_s: float
    mean_session_memory_bytes: float
    examples: tuple[StreamingExampleResult, ...]


@dataclass(frozen=True)
class StreamingReport:
    """Full report for the streaming systems evaluation track."""

    spec: StreamingEvalSpec
    buckets: tuple[StreamingBucketResult, ...]


def _end_index_for_time(piece: KymaTokenizedPiece, *, end_time_s: int) -> int:
    if end_time_s <= 0:
        raise ValueError("end_time_s must be positive.")

    threshold_ms = float(end_time_s * 1000)
    absolute_ms = piece.time_features.values[:, _ABSOLUTE_TIME_FEATURE_IDX]
    valid = piece.time_features.valid[:, _ABSOLUTE_TIME_FEATURE_IDX]
    within = valid & (absolute_ms <= threshold_ms)
    if not bool(within.any()):
        return 0
    return int(torch.nonzero(within, as_tuple=False)[-1].item()) + 1


def slice_streaming_piece(
    piece: KymaTokenizedPiece,
    *,
    session_length_s: int,
    max_benchmark_tokens: int = 128,
) -> StreamingEvalSlice | None:
    """Extract one streaming benchmark slice from a tokenized piece."""

    if max_benchmark_tokens <= 0:
        raise ValueError("max_benchmark_tokens must be positive.")

    prompt_end = _end_index_for_time(piece, end_time_s=session_length_s)
    if prompt_end == 0 or prompt_end >= int(piece.token_ids.shape[0]):
        return None

    future_end = min(prompt_end + max_benchmark_tokens, int(piece.token_ids.shape[0]))
    return StreamingEvalSlice(
        piece_id=piece.piece_id,
        session_length_s=session_length_s,
        prompt_ids=piece.token_ids[:prompt_end].clone(),
        prompt_time_features=piece.time_features.values[:prompt_end].clone(),
        prompt_time_feature_mask=piece.time_features.valid[:prompt_end].clone(),
        future_ids=piece.token_ids[prompt_end:future_end].clone(),
        future_time_features=piece.time_features.values[prompt_end:future_end].clone(),
        future_time_feature_mask=piece.time_features.valid[
            prompt_end:future_end
        ].clone(),
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _estimate_tensor_tree_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        tensor = value
        return int(tensor.numel()) * int(tensor.element_size())
    if is_dataclass(value):
        return sum(
            _estimate_tensor_tree_bytes(getattr(value, field.name))
            for field in fields(value)
        )
    if isinstance(value, tuple):
        items = cast(tuple[Any, ...], value)
        return sum(_estimate_tensor_tree_bytes(item) for item in items)
    if isinstance(value, list):
        items = cast(list[Any], value)
        return sum(_estimate_tensor_tree_bytes(item) for item in items)
    if isinstance(value, dict):
        items = cast(dict[Any, Any], value)
        return sum(_estimate_tensor_tree_bytes(item) for item in items.values())
    return 0


def estimate_decode_session_bytes(session: KymaDecodeSession) -> int:
    """Estimate the tensor-backed memory footprint of a decode session."""

    return _estimate_tensor_tree_bytes(session)


def _benchmark_example(
    model: KymaAutoregressiveLM,
    streaming_slice: StreamingEvalSlice,
    *,
    device: torch.device,
    spec: StreamingEvalSpec,
) -> StreamingExampleResult:
    prompt_ids = streaming_slice.prompt_ids.unsqueeze(0).to(device=device)
    prompt_time_features = streaming_slice.prompt_time_features.unsqueeze(0).to(
        device=device
    )
    prompt_time_feature_mask = streaming_slice.prompt_time_feature_mask.unsqueeze(0).to(
        device=device
    )

    _synchronize(device)
    start = time.perf_counter()
    session = prefill_decode_session(
        model,
        prompt_ids,
        time_features=prompt_time_features,
        time_feature_mask=prompt_time_feature_mask,
    )
    if spec.report_time_to_first_note_ms:
        sample_next_token(
            session.next_logits,
            KymaSamplingConfig(max_new_tokens=1, temperature=0.0),
        )
    _synchronize(device)
    elapsed_prefill = time.perf_counter() - start

    time_to_first_token_ms = (
        elapsed_prefill * 1000.0 if spec.report_time_to_first_note_ms else 0.0
    )
    session_memory_bytes = (
        estimate_decode_session_bytes(session) if spec.report_memory_growth else 0
    )

    throughput = 0.0
    if spec.report_decode_throughput and streaming_slice.future_token_count > 1:
        benchmark_session = session
        future_ids = streaming_slice.future_ids.to(device=device)
        future_time_features = streaming_slice.future_time_features.to(device=device)
        future_time_feature_mask = streaming_slice.future_time_feature_mask.to(
            device=device
        )

        _synchronize(device)
        start = time.perf_counter()
        for idx in range(streaming_slice.future_token_count - 1):
            benchmark_session = advance_decode_session(
                model,
                benchmark_session,
                future_ids[idx : idx + 1],
                time_features=future_time_features[idx : idx + 1].unsqueeze(0),
                time_feature_mask=future_time_feature_mask[idx : idx + 1].unsqueeze(0),
            )
        _synchronize(device)
        elapsed_decode = time.perf_counter() - start
        if elapsed_decode > 0.0:
            throughput = (streaming_slice.future_token_count - 1) / elapsed_decode

    return StreamingExampleResult(
        piece_id=streaming_slice.piece_id,
        session_length_s=streaming_slice.session_length_s,
        prompt_token_count=streaming_slice.prompt_token_count,
        future_token_count=streaming_slice.future_token_count,
        time_to_first_token_ms=time_to_first_token_ms,
        decode_throughput_tokens_per_s=throughput,
        session_memory_bytes=session_memory_bytes,
    )


@torch.no_grad()
def evaluate_streaming(
    model: KymaAutoregressiveLM,
    pieces: Sequence[KymaTokenizedPiece],
    *,
    spec: StreamingEvalSpec,
    device: str | torch.device = "cpu",
    max_benchmark_tokens: int = 128,
) -> StreamingReport:
    """Benchmark decode latency, throughput, and memory over session length."""

    resolved_device = torch.device(device)
    was_training = model.training
    model.eval()
    model.to(resolved_device)

    bucket_results: list[StreamingBucketResult] = []
    for session_length_s in spec.interactive_session_lengths_s:
        examples: list[StreamingExampleResult] = []
        for piece in pieces:
            streaming_slice = slice_streaming_piece(
                piece,
                session_length_s=session_length_s,
                max_benchmark_tokens=max_benchmark_tokens,
            )
            if streaming_slice is None:
                continue
            examples.append(
                _benchmark_example(
                    model,
                    streaming_slice,
                    device=resolved_device,
                    spec=spec,
                )
            )

        count = len(examples)
        mean_ttf = (
            0.0
            if count == 0
            else sum(example.time_to_first_token_ms for example in examples) / count
        )
        mean_throughput = (
            0.0
            if count == 0
            else sum(example.decode_throughput_tokens_per_s for example in examples)
            / count
        )
        mean_memory = (
            0.0
            if count == 0
            else sum(example.session_memory_bytes for example in examples) / count
        )
        bucket_results.append(
            StreamingBucketResult(
                session_length_s=session_length_s,
                examples_evaluated=count,
                mean_time_to_first_token_ms=mean_ttf,
                mean_decode_throughput_tokens_per_s=mean_throughput,
                mean_session_memory_bytes=mean_memory,
                examples=tuple(examples),
            )
        )

    if was_training:
        model.train()

    return StreamingReport(spec=spec, buckets=tuple(bucket_results))


__all__ = [
    "StreamingBucketResult",
    "StreamingEvalSlice",
    "StreamingExampleResult",
    "StreamingReport",
    "estimate_decode_session_bytes",
    "evaluate_streaming",
    "slice_streaming_piece",
]
