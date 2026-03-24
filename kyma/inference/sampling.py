"""Stateful autoregressive sampling for Kyma language models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from kyma.data import TIME_FEATURE_NAMES, KymaTimeFeatures
from kyma.model import KymaAutoregressiveLM, KymaLMState

TimeFeatureInput = torch.Tensor | KymaTimeFeatures


@dataclass(frozen=True)
class KymaSamplingConfig:
    """Sampling parameters for autoregressive Kyma decoding."""

    max_new_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    min_p: float | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative.")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1].")
        if self.min_p is not None and not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p must be in [0, 1] when provided.")
        if self.min_p is not None and self.top_p < 1.0:
            raise ValueError("top_p and min_p are mutually exclusive.")
        if self.eos_token_id is not None and self.eos_token_id < 0:
            raise ValueError("eos_token_id must be non-negative when provided.")
        if self.pad_token_id is not None and self.pad_token_id < 0:
            raise ValueError("pad_token_id must be non-negative when provided.")

    def stop_fill_token_id(self) -> int:
        """Token used to pad already-finished rows inside a decode batch."""

        if self.pad_token_id is not None:
            return self.pad_token_id
        if self.eos_token_id is not None:
            return self.eos_token_id
        return 0


@dataclass(frozen=True)
class KymaDecodeSession:
    """Decode session holding recurrent state plus the next-token logits."""

    next_logits: torch.Tensor
    recurrent_state: KymaLMState

    def __post_init__(self) -> None:
        if self.next_logits.ndim != 2:
            raise ValueError("next_logits must have shape (batch, vocab).")
        if self.next_logits.shape[0] <= 0:
            raise ValueError("next_logits must have a positive batch dimension.")

    def detach(self) -> KymaDecodeSession:
        return KymaDecodeSession(
            next_logits=self.next_logits.detach(),
            recurrent_state=self.recurrent_state.detach(),
        )

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> KymaDecodeSession:
        return KymaDecodeSession(
            next_logits=self.next_logits.to(device=device, dtype=dtype),
            recurrent_state=self.recurrent_state.to(device=device, dtype=dtype),
        )


@dataclass(frozen=True)
class KymaGenerationResult:
    """Rectangular generated output plus per-row completion metadata."""

    prompt_ids: torch.Tensor
    generated_ids: torch.Tensor
    sequence_ids: torch.Tensor
    generated_lengths: torch.Tensor
    finished: torch.Tensor

    def __post_init__(self) -> None:
        if self.prompt_ids.ndim != 2:
            raise ValueError("prompt_ids must have shape (batch, prompt_len).")
        if self.generated_ids.ndim != 2:
            raise ValueError("generated_ids must have shape (batch, steps).")
        if self.sequence_ids.ndim != 2:
            raise ValueError("sequence_ids must have shape (batch, total_len).")
        if self.generated_lengths.ndim != 1:
            raise ValueError("generated_lengths must have shape (batch,).")
        if self.finished.ndim != 1:
            raise ValueError("finished must have shape (batch,).")

    @property
    def steps(self) -> int:
        return int(self.generated_ids.shape[1])


def _normalize_prompt_ids(prompt_ids: torch.Tensor) -> torch.Tensor:
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if prompt_ids.ndim != 2:
        raise ValueError(
            f"prompt_ids must have shape (batch, T) or (T,); got {prompt_ids.shape}."
        )
    if prompt_ids.shape[1] <= 0:
        raise ValueError("prompt_ids must contain at least one token.")
    return prompt_ids


def _normalize_step_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
    if input_ids.ndim == 0:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.ndim == 2 and input_ids.shape[1] == 1:
        input_ids = input_ids.squeeze(1)
    if input_ids.ndim != 1:
        raise ValueError(
            "input_ids must have shape (batch,), (batch, 1), or a scalar token."
        )
    return input_ids


def _normalize_step_time_features(
    *,
    batch_size: int,
    time_features: TimeFeatureInput | None,
    time_feature_mask: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if time_features is None:
        if time_feature_mask is not None:
            raise ValueError(
                "time_feature_mask cannot be provided without time_features."
            )
        return None, None

    if isinstance(time_features, KymaTimeFeatures):
        values = time_features.values
        mask = time_features.valid if time_feature_mask is None else time_feature_mask
    else:
        values = time_features
        mask = time_feature_mask

    if values.ndim == 1:
        if batch_size != 1:
            raise ValueError("Unbatched step time features require batch_size == 1.")
        values = values.unsqueeze(0)
    if values.ndim == 2:
        if values.shape[0] != batch_size:
            raise ValueError("step time features must align with the batch size.")
        values = values.unsqueeze(1)
    if values.ndim != 3 or values.shape[1] != 1:
        raise ValueError(
            "step time features must have shape (batch, 1, feature_dim), "
            "(batch, feature_dim), or (feature_dim,)."
        )
    if values.shape[-1] != len(TIME_FEATURE_NAMES):
        raise ValueError(
            f"step time features must expose {len(TIME_FEATURE_NAMES)} features."
        )

    if mask is None:
        mask = torch.ones_like(values, dtype=torch.bool)
    else:
        if mask.ndim == 1:
            if batch_size != 1:
                raise ValueError("Unbatched step time masks require batch_size == 1.")
            mask = mask.unsqueeze(0)
        if mask.ndim == 2:
            if mask.shape[0] != batch_size:
                raise ValueError("step time masks must align with the batch size.")
            mask = mask.unsqueeze(1)
        if mask.shape != values.shape:
            raise ValueError("step time feature masks must match the feature shape.")

    return values, mask.to(dtype=torch.bool)


def _normalize_continuation_time_features(
    *,
    batch_size: int,
    required_steps: int,
    time_features: TimeFeatureInput | None,
    time_feature_mask: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if required_steps == 0:
        if time_feature_mask is not None and time_features is None:
            raise ValueError(
                "time_feature_mask cannot be provided without time_features."
            )
        return None, None
    if time_features is None:
        if time_feature_mask is not None:
            raise ValueError(
                "time_feature_mask cannot be provided without time_features."
            )
        return None, None

    if isinstance(time_features, KymaTimeFeatures):
        values = time_features.values
        mask = time_features.valid if time_feature_mask is None else time_feature_mask
    else:
        values = time_features
        mask = time_feature_mask

    if values.ndim == 2:
        if batch_size != 1:
            raise ValueError(
                "Unbatched continuation time features require batch_size == 1."
            )
        values = values.unsqueeze(0)
    if values.ndim != 3:
        raise ValueError(
            "continuation_time_features must have shape "
            "(batch, steps, feature_dim) or (steps, feature_dim)."
        )
    if values.shape[0] != batch_size:
        raise ValueError("continuation_time_features must align with the batch size.")
    if values.shape[1] < required_steps:
        raise ValueError(
            "continuation_time_features must provide at least one feature row per "
            "consumed generated token."
        )
    if values.shape[-1] != len(TIME_FEATURE_NAMES):
        raise ValueError(
            "continuation_time_features must expose "
            f"{len(TIME_FEATURE_NAMES)} features."
        )

    if mask is None:
        mask = torch.ones_like(values, dtype=torch.bool)
    else:
        if mask.ndim == 2:
            if batch_size != 1:
                raise ValueError(
                    "Unbatched continuation time masks require batch_size == 1."
                )
            mask = mask.unsqueeze(0)
        if mask.shape != values.shape:
            raise ValueError(
                "continuation_time_feature_mask must match the continuation "
                "time-feature shape."
            )

    return values, mask.to(dtype=torch.bool)


def _masked_continuation_step(
    time_features: torch.Tensor | None,
    time_feature_mask: torch.Tensor | None,
    *,
    step_index: int,
    active_mask: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if time_features is None or time_feature_mask is None:
        return None, None

    step_values = time_features[:, step_index : step_index + 1, :].clone()
    step_mask = time_feature_mask[:, step_index : step_index + 1, :].clone()
    inactive = (~active_mask).view(active_mask.shape[0], 1, 1)
    step_values = step_values.masked_fill(inactive, 0.0)
    step_mask = step_mask & (~inactive)
    return step_values, step_mask


def _sample_top_p(
    probs: torch.Tensor,
    *,
    top_p: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > top_p
    probs_sort = probs_sort.masked_fill(mask, 0.0)
    probs_sort = probs_sort / probs_sort.sum(dim=-1, keepdim=True)
    sampled = torch.multinomial(probs_sort, num_samples=1, generator=generator)
    return torch.gather(probs_idx, -1, sampled).squeeze(-1)


def _sample_min_p(
    probs: torch.Tensor,
    *,
    min_p: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    p_max, _ = torch.max(probs, dim=-1, keepdim=True)
    keep_mask = probs >= (min_p * p_max)
    masked_probs = probs.masked_fill(~keep_mask, 0.0)
    masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(masked_probs, num_samples=1, generator=generator).squeeze(
        -1
    )


@torch.no_grad()
def prefill_decode_session(
    model: KymaAutoregressiveLM,
    prompt_ids: torch.Tensor,
    *,
    time_features: TimeFeatureInput | None = None,
    time_feature_mask: torch.Tensor | None = None,
    state: KymaLMState | None = None,
) -> KymaDecodeSession:
    """Consume a prompt and return a session ready to emit the next token."""

    prompt_ids = _normalize_prompt_ids(prompt_ids)
    logits, next_state = cast(
        tuple[torch.Tensor, KymaLMState],
        model(
            prompt_ids,
            time_features=time_features,
            time_feature_mask=time_feature_mask,
            state=state,
            return_state=True,
        ),
    )
    return KymaDecodeSession(
        next_logits=logits[:, -1, :],
        recurrent_state=next_state,
    )


@torch.no_grad()
def advance_decode_session(
    model: KymaAutoregressiveLM,
    session: KymaDecodeSession,
    input_ids: torch.Tensor,
    *,
    time_features: TimeFeatureInput | None = None,
    time_feature_mask: torch.Tensor | None = None,
) -> KymaDecodeSession:
    """Advance a decode session by consuming one generated token per row."""

    step_input_ids = _normalize_step_input_ids(input_ids)
    if step_input_ids.shape[0] != session.next_logits.shape[0]:
        raise ValueError("input_ids batch size must match the decode session.")

    step_values, step_mask = _normalize_step_time_features(
        batch_size=int(step_input_ids.shape[0]),
        time_features=time_features,
        time_feature_mask=time_feature_mask,
    )
    next_logits, next_state = model.step(
        step_input_ids,
        time_features=step_values,
        time_feature_mask=step_mask,
        state=session.recurrent_state,
    )
    return KymaDecodeSession(next_logits=next_logits, recurrent_state=next_state)


def sample_next_token(
    logits: torch.Tensor,
    config: KymaSamplingConfig,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample the next token from batched logits."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, vocab).")
    if config.temperature == 0.0:
        return torch.argmax(logits, dim=-1)

    scaled_logits = logits.to(dtype=torch.float32) / config.temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    if config.min_p is not None:
        return _sample_min_p(probs, min_p=config.min_p, generator=generator)
    if config.top_p < 1.0:
        return _sample_top_p(probs, top_p=config.top_p, generator=generator)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate(
    model: KymaAutoregressiveLM,
    prompt_ids: torch.Tensor,
    *,
    sampling_config: KymaSamplingConfig,
    prompt_time_features: TimeFeatureInput | None = None,
    prompt_time_feature_mask: torch.Tensor | None = None,
    continuation_time_features: TimeFeatureInput | None = None,
    continuation_time_feature_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    state: KymaLMState | None = None,
) -> KymaGenerationResult:
    """Generate a rectangular continuation from a prompt."""

    prompt_ids = _normalize_prompt_ids(prompt_ids)
    batch_size = int(prompt_ids.shape[0])
    required_continuation_steps = max(sampling_config.max_new_tokens - 1, 0)
    continuation_values, continuation_mask = _normalize_continuation_time_features(
        batch_size=batch_size,
        required_steps=required_continuation_steps,
        time_features=continuation_time_features,
        time_feature_mask=continuation_time_feature_mask,
    )

    session = prefill_decode_session(
        model,
        prompt_ids,
        time_features=prompt_time_features,
        time_feature_mask=prompt_time_feature_mask,
        state=state,
    )
    fill_token_id = sampling_config.stop_fill_token_id()
    finished = torch.zeros(batch_size, dtype=torch.bool, device=prompt_ids.device)
    generated_lengths = torch.zeros(
        batch_size,
        dtype=torch.long,
        device=prompt_ids.device,
    )
    generated_steps: list[torch.Tensor] = []

    for step_index in range(sampling_config.max_new_tokens):
        sampled_ids = sample_next_token(
            session.next_logits,
            sampling_config,
            generator=generator,
        )
        if finished.any():
            fill_tokens = torch.full_like(sampled_ids, fill_token_id)
            sampled_ids = torch.where(finished, fill_tokens, sampled_ids)

        active_before_step = ~finished
        generated_lengths = torch.where(
            active_before_step,
            generated_lengths + 1,
            generated_lengths,
        )
        generated_steps.append(sampled_ids.unsqueeze(1))

        if sampling_config.eos_token_id is not None:
            finished = finished | (sampled_ids == sampling_config.eos_token_id)

        if step_index == sampling_config.max_new_tokens - 1 or bool(finished.all()):
            break

        active_after_step = ~finished
        step_values, step_mask = _masked_continuation_step(
            continuation_values,
            continuation_mask,
            step_index=step_index,
            active_mask=active_after_step,
        )
        step_input_ids = sampled_ids
        if finished.any():
            fill_tokens = torch.full_like(step_input_ids, fill_token_id)
            step_input_ids = torch.where(active_after_step, step_input_ids, fill_tokens)
        session = advance_decode_session(
            model,
            session,
            step_input_ids,
            time_features=step_values,
            time_feature_mask=step_mask,
        )

    generated_ids = torch.cat(generated_steps, dim=1)
    sequence_ids = torch.cat((prompt_ids, generated_ids), dim=1)
    return KymaGenerationResult(
        prompt_ids=prompt_ids,
        generated_ids=generated_ids,
        sequence_ids=sequence_ids,
        generated_lengths=generated_lengths,
        finished=finished,
    )


__all__ = [
    "KymaDecodeSession",
    "KymaGenerationResult",
    "KymaSamplingConfig",
    "advance_decode_session",
    "generate",
    "prefill_decode_session",
    "sample_next_token",
]
