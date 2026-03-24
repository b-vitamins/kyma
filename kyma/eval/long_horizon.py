"""Long-horizon evaluation for recurrent-state carry in Kyma."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from kyma.data import TIME_FEATURE_NAMES, KymaTokenizedPiece
from kyma.eval.protocol import LongHorizonEvalSpec
from kyma.inference import (
    KymaSamplingConfig,
    advance_decode_session,
    generate,
    prefill_decode_session,
)
from kyma.model import KymaAutoregressiveLM

_ABSOLUTE_TIME_FEATURE_IDX = TIME_FEATURE_NAMES.index("absolute_time_ms")


@dataclass(frozen=True)
class LongHorizonSlice:
    """Prompt and continuation tensors for long-horizon evaluation."""

    piece_id: str
    prompt_length_s: int
    continuation_length_s: int
    prompt_ids: torch.Tensor
    prompt_time_features: torch.Tensor
    prompt_time_feature_mask: torch.Tensor
    reference_ids: torch.Tensor
    reference_time_features: torch.Tensor
    reference_time_feature_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.prompt_ids.ndim != 1:
            raise ValueError("prompt_ids must have shape (prompt_len,).")
        if self.reference_ids.ndim != 1:
            raise ValueError("reference_ids must have shape (continuation_len,).")
        if self.prompt_time_features.shape[:1] != self.prompt_ids.shape:
            raise ValueError("prompt time features must align with prompt ids.")
        if self.reference_time_features.shape[:1] != self.reference_ids.shape:
            raise ValueError("reference time features must align with reference ids.")

    @property
    def continuation_token_count(self) -> int:
        return int(self.reference_ids.shape[0])


@dataclass(frozen=True)
class HorizonNLLPoint:
    """Mean negative log-likelihood at a fixed continuation horizon."""

    horizon_index: int
    mean_nll: float
    example_count: int


@dataclass(frozen=True)
class LongHorizonExampleResult:
    """Per-piece long-horizon result for one reset interval."""

    piece_id: str
    prompt_length_s: int
    continuation_length_s: int
    state_carry_reset_interval: int
    continuation_token_count: int
    mean_continuation_nll: float
    horizon_nll: tuple[float, ...]
    reference_ids: tuple[int, ...] | None = None
    generated_ids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class LongHorizonBucketResult:
    """Aggregate long-horizon results for one evaluation bucket."""

    prompt_length_s: int
    continuation_length_s: int
    state_carry_reset_interval: int
    examples_evaluated: int
    total_continuation_tokens: int
    mean_continuation_nll: float
    horizon_nll: tuple[HorizonNLLPoint, ...]
    examples: tuple[LongHorizonExampleResult, ...]


@dataclass(frozen=True)
class LongHorizonReport:
    """Full report for the long-horizon evaluation track."""

    spec: LongHorizonEvalSpec
    buckets: tuple[LongHorizonBucketResult, ...]


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


def _tensor_to_int_tuple(tensor: torch.Tensor) -> tuple[int, ...]:
    flat = tensor.detach().cpu().reshape(-1)
    return tuple(int(value.item()) for value in flat)


def slice_long_horizon_piece(
    piece: KymaTokenizedPiece,
    *,
    prompt_length_s: int,
    continuation_length_s: int,
) -> LongHorizonSlice | None:
    """Extract one prompt/continuation pair from a tokenized piece."""

    if continuation_length_s <= 0:
        raise ValueError("continuation_length_s must be positive.")

    prompt_end = _end_index_for_time(piece, end_time_s=prompt_length_s)
    continuation_end = _end_index_for_time(
        piece,
        end_time_s=prompt_length_s + continuation_length_s,
    )
    if prompt_end == 0 or continuation_end <= prompt_end:
        return None

    return LongHorizonSlice(
        piece_id=piece.piece_id,
        prompt_length_s=prompt_length_s,
        continuation_length_s=continuation_length_s,
        prompt_ids=piece.token_ids[:prompt_end].clone(),
        prompt_time_features=piece.time_features.values[:prompt_end].clone(),
        prompt_time_feature_mask=piece.time_features.valid[:prompt_end].clone(),
        reference_ids=piece.token_ids[prompt_end:continuation_end].clone(),
        reference_time_features=piece.time_features.values[
            prompt_end:continuation_end
        ].clone(),
        reference_time_feature_mask=piece.time_features.valid[
            prompt_end:continuation_end
        ].clone(),
    )


def _teacher_forced_horizon_losses(
    model: KymaAutoregressiveLM,
    horizon_slice: LongHorizonSlice,
    *,
    device: torch.device,
    state_carry_reset_interval: int,
) -> tuple[float, ...]:
    if state_carry_reset_interval < 0:
        raise ValueError("state_carry_reset_interval must be non-negative.")

    session = prefill_decode_session(
        model,
        horizon_slice.prompt_ids.unsqueeze(0).to(device=device),
        time_features=horizon_slice.prompt_time_features.unsqueeze(0).to(device=device),
        time_feature_mask=horizon_slice.prompt_time_feature_mask.unsqueeze(0).to(
            device=device
        ),
    )

    losses: list[float] = []
    tokens_since_reset = 0
    for idx in range(horizon_slice.continuation_token_count):
        target = horizon_slice.reference_ids[idx : idx + 1].to(device=device)
        token_loss = F.cross_entropy(
            session.next_logits,
            target,
            reduction="none",
        )
        losses.append(float(token_loss.item()))
        if idx == horizon_slice.continuation_token_count - 1:
            break

        input_ids = horizon_slice.reference_ids[idx : idx + 1].to(device=device)
        step_time_features = horizon_slice.reference_time_features[idx : idx + 1]
        step_time_feature_mask = horizon_slice.reference_time_feature_mask[
            idx : idx + 1
        ]
        step_time_features = step_time_features.unsqueeze(0).to(device=device)
        step_time_feature_mask = step_time_feature_mask.unsqueeze(0).to(device=device)
        tokens_since_reset += 1

        if (
            state_carry_reset_interval > 0
            and tokens_since_reset >= state_carry_reset_interval
        ):
            session = prefill_decode_session(
                model,
                input_ids,
                time_features=step_time_features,
                time_feature_mask=step_time_feature_mask,
            )
            tokens_since_reset = 0
        else:
            session = advance_decode_session(
                model,
                session,
                input_ids,
                time_features=step_time_features,
                time_feature_mask=step_time_feature_mask,
            )

    return tuple(losses)


@torch.no_grad()
def evaluate_long_horizon(
    model: KymaAutoregressiveLM,
    pieces: Sequence[KymaTokenizedPiece],
    *,
    spec: LongHorizonEvalSpec,
    device: str | torch.device = "cpu",
    collect_generations: bool | None = None,
) -> LongHorizonReport:
    """Evaluate continuation quality under explicit recurrent-state reset ablations."""

    resolved_device = torch.device(device)
    was_training = model.training
    model.eval()
    model.to(resolved_device)

    if collect_generations is None:
        collect_generations = spec.report_structure_metrics

    bucket_results: list[LongHorizonBucketResult] = []
    for prompt_length_s in spec.prompt_lengths_s:
        for continuation_length_s in spec.continuation_lengths_s:
            for reset_interval in spec.state_carry_reset_intervals:
                total_tokens = 0
                weighted_nll_sum = 0.0
                horizon_values: dict[int, list[float]] = defaultdict(list)
                examples: list[LongHorizonExampleResult] = []

                for piece in pieces:
                    horizon_slice = slice_long_horizon_piece(
                        piece,
                        prompt_length_s=prompt_length_s,
                        continuation_length_s=continuation_length_s,
                    )
                    if horizon_slice is None:
                        continue

                    token_losses = _teacher_forced_horizon_losses(
                        model,
                        horizon_slice,
                        device=resolved_device,
                        state_carry_reset_interval=reset_interval,
                    )
                    if not token_losses:
                        continue

                    continuation_token_count = len(token_losses)
                    mean_nll = sum(token_losses) / continuation_token_count
                    weighted_nll_sum += sum(token_losses)
                    total_tokens += continuation_token_count
                    if spec.report_horizon_nll:
                        for horizon_index, token_loss in enumerate(
                            token_losses,
                            start=1,
                        ):
                            horizon_values[horizon_index].append(token_loss)

                    generated_ids: tuple[int, ...] | None = None
                    reference_ids: tuple[int, ...] | None = None
                    if collect_generations:
                        continuation_count = horizon_slice.continuation_token_count
                        generation = generate(
                            model,
                            horizon_slice.prompt_ids.unsqueeze(0).to(
                                device=resolved_device
                            ),
                            sampling_config=KymaSamplingConfig(
                                max_new_tokens=continuation_count,
                                temperature=0.0,
                            ),
                            prompt_time_features=horizon_slice.prompt_time_features.unsqueeze(
                                0
                            ).to(device=resolved_device),
                            prompt_time_feature_mask=(
                                horizon_slice.prompt_time_feature_mask.unsqueeze(0).to(
                                    device=resolved_device
                                )
                            ),
                            continuation_time_features=(
                                horizon_slice.reference_time_features[:-1]
                                .unsqueeze(0)
                                .to(device=resolved_device)
                                if continuation_count > 1
                                else None
                            ),
                            continuation_time_feature_mask=(
                                horizon_slice.reference_time_feature_mask[:-1]
                                .unsqueeze(0)
                                .to(device=resolved_device)
                                if continuation_count > 1
                                else None
                            ),
                        )
                        generated_ids = _tensor_to_int_tuple(
                            generation.generated_ids[0]
                        )
                        reference_ids = _tensor_to_int_tuple(
                            horizon_slice.reference_ids
                        )

                    examples.append(
                        LongHorizonExampleResult(
                            piece_id=piece.piece_id,
                            prompt_length_s=prompt_length_s,
                            continuation_length_s=continuation_length_s,
                            state_carry_reset_interval=reset_interval,
                            continuation_token_count=continuation_token_count,
                            mean_continuation_nll=mean_nll,
                            horizon_nll=token_losses,
                            reference_ids=reference_ids,
                            generated_ids=generated_ids,
                        )
                    )

                horizon_nll = tuple(
                    HorizonNLLPoint(
                        horizon_index=horizon_index,
                        mean_nll=sum(values) / len(values),
                        example_count=len(values),
                    )
                    for horizon_index, values in sorted(horizon_values.items())
                )
                mean_continuation_nll = (
                    0.0 if total_tokens == 0 else weighted_nll_sum / total_tokens
                )
                bucket_results.append(
                    LongHorizonBucketResult(
                        prompt_length_s=prompt_length_s,
                        continuation_length_s=continuation_length_s,
                        state_carry_reset_interval=reset_interval,
                        examples_evaluated=len(examples),
                        total_continuation_tokens=total_tokens,
                        mean_continuation_nll=mean_continuation_nll,
                        horizon_nll=horizon_nll,
                        examples=tuple(examples),
                    )
                )

    if was_training:
        model.train()

    return LongHorizonReport(spec=spec, buckets=tuple(bucket_results))


__all__ = [
    "HorizonNLLPoint",
    "LongHorizonBucketResult",
    "LongHorizonExampleResult",
    "LongHorizonReport",
    "LongHorizonSlice",
    "evaluate_long_horizon",
    "slice_long_horizon_piece",
]
