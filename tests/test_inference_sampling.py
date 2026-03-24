from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
import torch
from torch import nn

from kyma.inference import (
    KymaSamplingConfig,
    advance_decode_session,
    generate,
    prefill_decode_session,
    sample_next_token,
)
from kyma.model import (
    KymaAutoregressiveLM,
    KymaEvalDifferentiators,
    KymaLMState,
    KymaLongContextConfig,
    KymaModelConfig,
    KymaTimeConditioningConfig,
)


@dataclass(frozen=True)
class FakeMixerState:
    running: torch.Tensor

    def detach(self) -> FakeMixerState:
        return FakeMixerState(running=self.running.detach())

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> FakeMixerState:
        return FakeMixerState(running=self.running.to(device=device, dtype=dtype))


class IdentityMixer(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(d_model))

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> FakeMixerState:
        resolved_dtype = torch.float32 if dtype is None else dtype
        return FakeMixerState(
            running=torch.zeros((batch_size, 1), device=device, dtype=resolved_dtype)
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        state: FakeMixerState | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, FakeMixerState]:
        y = self.proj(x)
        next_running = x.new_full((x.shape[0], 1), float(x.shape[1]))
        if state is not None:
            next_running = next_running + state.running.to(
                device=x.device,
                dtype=x.dtype,
            )
        next_state = FakeMixerState(running=next_running)
        if not return_state:
            return y
        return y, next_state


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


def _build_config() -> KymaModelConfig:
    return KymaModelConfig(
        d_model=16,
        n_layers=2,
        d_state=8,
        expand=2,
        d_head=8,
        d_conv=4,
        chunk_size=8,
        vocab_size=32,
        dropout_p=0.0,
        ffn_mult=2,
        max_segment_len=8,
        time_conditioning=KymaTimeConditioningConfig(
            learned_positional_embedding=False,
            delta_time_features=True,
            absolute_time_features=True,
            beat_phase_features=True,
            tempo_features=True,
            feature_mlp_dim=12,
        ),
        long_context=KymaLongContextConfig(),
        differentiators=KymaEvalDifferentiators(),
    )


def _fake_mixer_factory(config: KymaModelConfig) -> nn.Module:
    return IdentityMixer(config.d_model)


def _time_features(
    batch: int,
    seq_len: int,
    *,
    start_idx: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.zeros((batch, seq_len, 4), dtype=torch.float32)
    mask = torch.ones((batch, seq_len, 4), dtype=torch.bool)
    for offset in range(seq_len):
        position = start_idx + offset
        values[:, offset, 0] = float(position + 1) * 10.0
        values[:, offset, 1] = float(position) * 10.0
        values[:, offset, 2] = 0.25 * position
        values[:, offset, 3] = 120.0
    return values, mask


def test_prefill_and_advance_match_model_boundaries() -> None:
    model = KymaAutoregressiveLM(_build_config(), mixer_factory=_fake_mixer_factory)
    model.eval()

    prompt_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    prompt_values, prompt_mask = _time_features(batch=1, seq_len=3)
    step_values, step_mask = _time_features(batch=1, seq_len=1, start_idx=3)

    full_logits, full_state = cast(
        tuple[torch.Tensor, KymaLMState],
        model(
            prompt_ids,
            time_features=prompt_values,
            time_feature_mask=prompt_mask,
            return_state=True,
        ),
    )

    session = prefill_decode_session(
        model,
        prompt_ids,
        time_features=prompt_values,
        time_feature_mask=prompt_mask,
    )
    assert torch.allclose(session.next_logits, full_logits[:, -1, :], atol=1e-6)
    assert session.recurrent_state.tokens_processed == full_state.tokens_processed

    manual_step_logits, manual_step_state = model.step(
        torch.tensor([4], dtype=torch.long),
        time_features=step_values.squeeze(0),
        time_feature_mask=step_mask.squeeze(0),
        state=full_state,
    )
    advanced = advance_decode_session(
        model,
        session,
        torch.tensor([4], dtype=torch.long),
        time_features=step_values.squeeze(0),
        time_feature_mask=step_mask.squeeze(0),
    )
    assert torch.allclose(advanced.next_logits, manual_step_logits, atol=1e-6)
    assert (
        advanced.recurrent_state.tokens_processed == manual_step_state.tokens_processed
    )


def test_sampling_config_rejects_ambiguous_probability_filters() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        KymaSamplingConfig(max_new_tokens=1, top_p=0.9, min_p=0.1)


def test_sample_next_token_supports_greedy_and_probability_filters() -> None:
    logits = torch.tensor([[5.0, 1.0, 0.0]], dtype=torch.float32)

    greedy = sample_next_token(
        logits,
        KymaSamplingConfig(max_new_tokens=1, temperature=0.0),
    )
    assert torch.equal(greedy, torch.tensor([0], dtype=torch.long))

    top_p = sample_next_token(
        logits,
        KymaSamplingConfig(max_new_tokens=1, top_p=0.7),
        generator=torch.Generator().manual_seed(0),
    )
    assert torch.equal(top_p, torch.tensor([0], dtype=torch.long))

    min_p = sample_next_token(
        logits,
        KymaSamplingConfig(max_new_tokens=1, min_p=0.9),
        generator=torch.Generator().manual_seed(0),
    )
    assert torch.equal(min_p, torch.tensor([0], dtype=torch.long))


def test_generate_tracks_stop_tokens_and_lengths() -> None:
    model = cast(KymaAutoregressiveLM, RuleBasedLM(vocab_size=8, eos_token_id=5))
    prompt_ids = torch.tensor([[3], [4]], dtype=torch.long)
    continuation_values = torch.zeros((2, 2, 4), dtype=torch.float32)
    continuation_mask = torch.ones((2, 2, 4), dtype=torch.bool)

    result = generate(
        model,
        prompt_ids,
        sampling_config=KymaSamplingConfig(
            max_new_tokens=3,
            temperature=0.0,
            eos_token_id=5,
            pad_token_id=0,
        ),
        continuation_time_features=continuation_values,
        continuation_time_feature_mask=continuation_mask,
    )

    assert result.steps == 2
    assert torch.equal(
        result.generated_ids,
        torch.tensor([[4, 5], [5, 0]], dtype=torch.long),
    )
    assert torch.equal(result.generated_lengths, torch.tensor([2, 1], dtype=torch.long))
    assert torch.equal(result.finished, torch.tensor([True, True]))
    assert torch.equal(
        result.sequence_ids,
        torch.tensor([[3, 4, 5], [4, 5, 0]], dtype=torch.long),
    )
