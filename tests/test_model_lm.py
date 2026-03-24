from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn

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
                device=x.device, dtype=x.dtype
            )
        next_state = FakeMixerState(running=next_running)
        if not return_state:
            return y
        return y, next_state


def _build_config(*, learned_positional_embedding: bool = False) -> KymaModelConfig:
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
        max_segment_len=4,
        time_conditioning=KymaTimeConditioningConfig(
            learned_positional_embedding=learned_positional_embedding,
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


def _time_features(batch: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.zeros((batch, seq_len, 4), dtype=torch.float32)
    mask = torch.ones((batch, seq_len, 4), dtype=torch.bool)
    for position in range(seq_len):
        values[:, position, 0] = float(position + 1) * 10.0
        values[:, position, 1] = float(position) * 10.0
        values[:, position, 2] = 0.25 * position
        values[:, position, 3] = 120.0
    return values, mask


def test_stateful_forward_and_step_match() -> None:
    model = KymaAutoregressiveLM(_build_config(), mixer_factory=_fake_mixer_factory)
    model.eval()

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    time_values, time_mask = _time_features(batch=1, seq_len=input_ids.shape[1])

    full_logits, full_state = model(
        input_ids,
        time_features=time_values,
        time_feature_mask=time_mask,
        return_state=True,
    )

    state = model.init_state(batch_size=1)
    step_logits: list[torch.Tensor] = []
    for idx in range(input_ids.shape[1]):
        logits, state = model.step(
            input_ids[:, idx],
            time_features=time_values[:, idx : idx + 1, :],
            time_feature_mask=time_mask[:, idx : idx + 1, :],
            state=state,
        )
        step_logits.append(logits.unsqueeze(1))

    stacked_logits = torch.cat(step_logits, dim=1)
    assert torch.allclose(stacked_logits, full_logits, atol=1e-6)
    assert full_state.tokens_processed == input_ids.shape[1]
    assert state.tokens_processed == input_ids.shape[1]
    assert len(state.layer_states) == model.config.n_layers


def test_forward_rejects_time_feature_shape_mismatch() -> None:
    model = KymaAutoregressiveLM(_build_config(), mixer_factory=_fake_mixer_factory)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    bad_time_values = torch.zeros((1, 2, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match="align with input_ids"):
        model(input_ids, time_features=bad_time_values)


def test_learned_position_embeddings_enforce_segment_limit() -> None:
    model = KymaAutoregressiveLM(
        _build_config(learned_positional_embedding=True),
        mixer_factory=_fake_mixer_factory,
    )
    state = KymaLMState(
        layer_states=model.init_state(batch_size=1).layer_states, tokens_processed=4
    )
    time_values, time_mask = _time_features(batch=1, seq_len=1)

    with pytest.raises(ValueError, match="exceeds max_segment_len"):
        model.step(
            torch.tensor([1], dtype=torch.long),
            time_features=time_values,
            time_feature_mask=time_mask,
            state=state,
        )


def test_state_object_supports_detach_and_device_transfer() -> None:
    model = KymaAutoregressiveLM(_build_config(), mixer_factory=_fake_mixer_factory)
    state = model.init_state(batch_size=2, device="cpu", dtype=torch.float32)
    detached = state.detach()
    moved = detached.to(device="cpu", dtype=torch.float32)

    assert isinstance(moved, KymaLMState)
    assert len(moved.layer_states) == model.config.n_layers
    assert all(
        isinstance(layer_state, FakeMixerState) for layer_state in moved.layer_states
    )
