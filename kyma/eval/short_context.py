"""Short-context parity evaluation for Kyma language models."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch.nn import functional as F

from kyma.data import TIME_FEATURE_NAMES, KymaTokenizedPiece
from kyma.eval.protocol import ShortContextParitySpec
from kyma.inference import KymaSamplingConfig, generate
from kyma.model import KymaAutoregressiveLM

_ABSOLUTE_TIME_FEATURE_IDX = TIME_FEATURE_NAMES.index("absolute_time_ms")


@dataclass(frozen=True)
class ShortContextParitySlice:
    """Prompt and continuation tensors extracted from one tokenized piece."""

    piece_id: str
    prompt_duration_s: int
    prompt_ids: torch.Tensor
    prompt_time_features: torch.Tensor
    prompt_time_feature_mask: torch.Tensor
    reference_ids: torch.Tensor
    continuation_time_features: torch.Tensor
    continuation_time_feature_mask: torch.Tensor
    eval_input_ids: torch.Tensor
    eval_target_ids: torch.Tensor
    eval_loss_mask: torch.Tensor
    eval_time_features: torch.Tensor
    eval_time_feature_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.prompt_ids.ndim != 1:
            raise ValueError("prompt_ids must have shape (prompt_len,).")
        if self.reference_ids.ndim != 1:
            raise ValueError("reference_ids must have shape (continuation_len,).")
        if self.eval_input_ids.ndim != 1 or self.eval_target_ids.ndim != 1:
            raise ValueError("eval input and target ids must be one-dimensional.")
        if self.eval_loss_mask.shape != self.eval_input_ids.shape:
            raise ValueError("eval_loss_mask must align with eval_input_ids.")

    @property
    def prompt_token_count(self) -> int:
        return int(self.prompt_ids.shape[0])

    @property
    def continuation_token_count(self) -> int:
        return int(self.reference_ids.shape[0])


@dataclass(frozen=True)
class ShortContextExampleResult:
    """Per-piece short-context continuation result."""

    piece_id: str
    prompt_duration_s: int
    prompt_token_count: int
    continuation_token_count: int
    continuation_nll: float
    reference_ids: tuple[int, ...]
    generated_ids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ShortContextBucketResult:
    """Aggregate results for one prompt-duration bucket."""

    prompt_duration_s: int
    examples_evaluated: int
    total_continuation_tokens: int
    mean_continuation_nll: float
    perplexity: float
    examples: tuple[ShortContextExampleResult, ...]


@dataclass(frozen=True)
class ShortContextParityReport:
    """Full report for the short-context parity evaluation track."""

    spec: ShortContextParitySpec
    buckets: tuple[ShortContextBucketResult, ...]


def _prompt_length_for_duration(
    piece: KymaTokenizedPiece,
    *,
    prompt_duration_s: int,
) -> int:
    if prompt_duration_s <= 0:
        raise ValueError("prompt_duration_s must be positive.")

    threshold_ms = float(prompt_duration_s * 1000)
    absolute_ms = piece.time_features.values[:, _ABSOLUTE_TIME_FEATURE_IDX]
    valid = piece.time_features.valid[:, _ABSOLUTE_TIME_FEATURE_IDX]
    within_prompt = valid & (absolute_ms <= threshold_ms)
    if not bool(within_prompt.any()):
        return 0
    return int(torch.nonzero(within_prompt, as_tuple=False)[-1].item()) + 1


def slice_short_context_piece(
    piece: KymaTokenizedPiece,
    *,
    prompt_duration_s: int,
    continuation_tokens: int,
) -> ShortContextParitySlice | None:
    """Extract one short-context parity slice from a tokenized piece."""

    if continuation_tokens <= 0:
        raise ValueError("continuation_tokens must be positive.")

    prompt_len = _prompt_length_for_duration(
        piece,
        prompt_duration_s=prompt_duration_s,
    )
    total_tokens = int(piece.token_ids.shape[0])
    if prompt_len == 0 or prompt_len + continuation_tokens > total_tokens:
        return None

    continuation_feature_len = max(continuation_tokens - 1, 0)
    eval_input_len = prompt_len + continuation_tokens - 1
    eval_loss_mask = torch.zeros((eval_input_len,), dtype=torch.bool)
    eval_loss_mask[prompt_len - 1 :] = True

    return ShortContextParitySlice(
        piece_id=piece.piece_id,
        prompt_duration_s=prompt_duration_s,
        prompt_ids=piece.token_ids[:prompt_len].clone(),
        prompt_time_features=piece.time_features.values[:prompt_len].clone(),
        prompt_time_feature_mask=piece.time_features.valid[:prompt_len].clone(),
        reference_ids=piece.token_ids[
            prompt_len : prompt_len + continuation_tokens
        ].clone(),
        continuation_time_features=piece.time_features.values[
            prompt_len : prompt_len + continuation_feature_len
        ].clone(),
        continuation_time_feature_mask=piece.time_features.valid[
            prompt_len : prompt_len + continuation_feature_len
        ].clone(),
        eval_input_ids=piece.token_ids[:eval_input_len].clone(),
        eval_target_ids=piece.token_ids[1 : eval_input_len + 1].clone(),
        eval_loss_mask=eval_loss_mask,
        eval_time_features=piece.time_features.values[:eval_input_len].clone(),
        eval_time_feature_mask=piece.time_features.valid[:eval_input_len].clone(),
    )


def _continuation_nll(
    model: KymaAutoregressiveLM,
    parity_slice: ShortContextParitySlice,
    *,
    device: torch.device,
) -> tuple[float, int]:
    input_ids = parity_slice.eval_input_ids.unsqueeze(0).to(device=device)
    target_ids = parity_slice.eval_target_ids.unsqueeze(0).to(device=device)
    loss_mask = parity_slice.eval_loss_mask.unsqueeze(0).to(device=device)
    time_features = parity_slice.eval_time_features.unsqueeze(0).to(device=device)
    time_feature_mask = parity_slice.eval_time_feature_mask.unsqueeze(0).to(
        device=device
    )

    logits = cast(
        torch.Tensor,
        model(
            input_ids,
            time_features=time_features,
            time_feature_mask=time_feature_mask,
        ),
    )
    token_loss = F.cross_entropy(
        logits.transpose(1, 2),
        target_ids,
        reduction="none",
    )
    masked = token_loss[loss_mask]
    valid_tokens = int(masked.numel())
    if valid_tokens == 0:
        return 0.0, 0
    return float(masked.mean().item()), valid_tokens


def _tensor_to_int_tuple(tensor: torch.Tensor) -> tuple[int, ...]:
    flat_tensor = tensor.detach().cpu().reshape(-1)
    return tuple(int(value.item()) for value in flat_tensor)


@torch.no_grad()
def evaluate_short_context_parity(
    model: KymaAutoregressiveLM,
    pieces: Sequence[KymaTokenizedPiece],
    *,
    spec: ShortContextParitySpec,
    device: str | torch.device = "cpu",
    collect_generations: bool = True,
) -> ShortContextParityReport:
    """Evaluate Aria-style short-context continuation on tokenized pieces."""

    resolved_device = torch.device(device)
    was_training = model.training
    model.eval()
    model.to(resolved_device)

    sampling_config = KymaSamplingConfig(
        max_new_tokens=spec.continuation_tokens,
        temperature=spec.temperature,
        min_p=spec.min_p,
    )
    bucket_results: list[ShortContextBucketResult] = []

    for prompt_duration_s in spec.prompt_durations_s:
        weighted_nll_sum = 0.0
        total_tokens = 0
        examples: list[ShortContextExampleResult] = []

        for piece in pieces:
            parity_slice = slice_short_context_piece(
                piece,
                prompt_duration_s=prompt_duration_s,
                continuation_tokens=spec.continuation_tokens,
            )
            if parity_slice is None:
                continue

            continuation_nll, valid_tokens = _continuation_nll(
                model,
                parity_slice,
                device=resolved_device,
            )
            weighted_nll_sum += continuation_nll * valid_tokens
            total_tokens += valid_tokens

            generated_ids: tuple[int, ...] | None = None
            if collect_generations:
                continuation_values = (
                    parity_slice.continuation_time_features.unsqueeze(0).to(
                        device=resolved_device
                    )
                    if parity_slice.continuation_time_features.numel() > 0
                    else None
                )
                continuation_mask = (
                    parity_slice.continuation_time_feature_mask.unsqueeze(0).to(
                        device=resolved_device
                    )
                    if parity_slice.continuation_time_feature_mask.numel() > 0
                    else None
                )
                generation = generate(
                    model,
                    parity_slice.prompt_ids.unsqueeze(0).to(device=resolved_device),
                    sampling_config=sampling_config,
                    prompt_time_features=parity_slice.prompt_time_features.unsqueeze(
                        0
                    ).to(device=resolved_device),
                    prompt_time_feature_mask=(
                        parity_slice.prompt_time_feature_mask.unsqueeze(0).to(
                            device=resolved_device
                        )
                    ),
                    continuation_time_features=continuation_values,
                    continuation_time_feature_mask=continuation_mask,
                )
                generated_ids = _tensor_to_int_tuple(generation.generated_ids[0])

            examples.append(
                ShortContextExampleResult(
                    piece_id=piece.piece_id,
                    prompt_duration_s=prompt_duration_s,
                    prompt_token_count=parity_slice.prompt_token_count,
                    continuation_token_count=parity_slice.continuation_token_count,
                    continuation_nll=continuation_nll,
                    reference_ids=_tensor_to_int_tuple(parity_slice.reference_ids),
                    generated_ids=generated_ids,
                )
            )

        mean_nll = 0.0 if total_tokens == 0 else weighted_nll_sum / total_tokens
        perplexity = 1.0 if total_tokens == 0 else math.exp(mean_nll)
        bucket_results.append(
            ShortContextBucketResult(
                prompt_duration_s=prompt_duration_s,
                examples_evaluated=len(examples),
                total_continuation_tokens=total_tokens,
                mean_continuation_nll=mean_nll,
                perplexity=perplexity,
                examples=tuple(examples),
            )
        )

    if was_training:
        model.train()

    return ShortContextParityReport(spec=spec, buckets=tuple(bucket_results))


__all__ = [
    "ShortContextBucketResult",
    "ShortContextExampleResult",
    "ShortContextParityReport",
    "ShortContextParitySlice",
    "evaluate_short_context_parity",
    "slice_short_context_piece",
]
