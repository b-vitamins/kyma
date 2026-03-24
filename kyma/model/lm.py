"""SLinOSS-backed autoregressive language model for Kyma."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol, TypeAlias, cast

import torch
from torch import nn
from torch.nn import functional as F

from kyma.data import TIME_FEATURE_NAMES, KymaTimeFeatures
from kyma.model.config import KymaModelConfig

TimeFeatureInput: TypeAlias = torch.Tensor | KymaTimeFeatures

_TIME_FEATURE_INDEX = {name: idx for idx, name in enumerate(TIME_FEATURE_NAMES)}


class StatefulMixerProtocol(Protocol):
    """Minimal runtime contract for mixer blocks used by Kyma."""

    def forward(
        self,
        x: torch.Tensor,
        *,
        state: Any | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]: ...

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Any: ...


MixerFactory: TypeAlias = Callable[[KymaModelConfig], nn.Module]


def build_slinoss_mixer(config: KymaModelConfig) -> nn.Module:
    """Build the default SLinOSS mixer for a Kyma block."""

    try:
        layers_module = import_module("slinoss.layers")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "slinoss is required for the default Kyma model. "
            "Install project dependencies before instantiating the LM."
        ) from exc

    mixer_cls = layers_module.SLinOSSMixer
    scan_backend_name = config.backends.scan_backend
    scan_backend_cls = {
        "auto": layers_module.AutoScanBackend,
        "reference": layers_module.ReferenceScanBackend,
        "cute": layers_module.CuteScanBackend,
    }[scan_backend_name]
    return mixer_cls(
        config.d_model,
        d_state=config.d_state,
        expand=config.expand,
        d_head=config.d_head,
        d_conv=config.d_conv,
        chunk_size=config.chunk_size,
        normalize_bc=True,
        backend=scan_backend_cls(),
    )


def _maybe_detach_state(state: Any) -> Any:
    detach = getattr(state, "detach", None)
    if callable(detach):
        return detach()
    return state


def _maybe_to_state(
    state: Any,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Any:
    to_fn = getattr(state, "to", None)
    if callable(to_fn):
        return to_fn(device=device, dtype=dtype)
    return state


@dataclass(frozen=True)
class KymaLMState:
    """Per-layer recurrent state for stateful Kyma decoding."""

    layer_states: tuple[Any, ...]
    tokens_processed: int = 0

    def __post_init__(self) -> None:
        if self.tokens_processed < 0:
            raise ValueError("tokens_processed must be non-negative.")

    def detach(self) -> KymaLMState:
        return KymaLMState(
            layer_states=tuple(
                _maybe_detach_state(state) for state in self.layer_states
            ),
            tokens_processed=self.tokens_processed,
        )

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> KymaLMState:
        return KymaLMState(
            layer_states=tuple(
                _maybe_to_state(state, device=device, dtype=dtype)
                for state in self.layer_states
            ),
            tokens_processed=self.tokens_processed,
        )


class KymaTimeConditioner(nn.Module):
    """Project structured musical-time features into the model width."""

    def __init__(
        self,
        *,
        config: KymaModelConfig,
    ) -> None:
        super().__init__()
        self.selected_feature_names = config.time_conditioning.selected_feature_names()
        if not self.selected_feature_names:
            raise ValueError(
                "KymaTimeConditioner requires at least one selected feature."
            )

        self.input_dim = sum(
            2 if feature_name == "beat_phase" else 1
            for feature_name in self.selected_feature_names
        )
        hidden_dim = config.time_conditioning.feature_mlp_dim
        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, config.d_model),
            nn.Dropout(config.dropout_p),
        )

    def forward(
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if values.ndim != 3 or values.shape[-1] != len(TIME_FEATURE_NAMES):
            raise ValueError(
                "time feature values must have shape "
                f"(batch, T, {len(TIME_FEATURE_NAMES)})."
            )
        if valid_mask is None:
            valid_mask = torch.ones_like(values, dtype=torch.bool)
        if valid_mask.shape != values.shape:
            raise ValueError("time feature mask must match the values tensor shape.")

        pieces: list[torch.Tensor] = []
        for feature_name in self.selected_feature_names:
            feature_idx = _TIME_FEATURE_INDEX[feature_name]
            feature = values[..., feature_idx : feature_idx + 1]
            feature_valid = valid_mask[..., feature_idx : feature_idx + 1].to(
                dtype=feature.dtype
            )

            if feature_name in {"delta_time_ms", "absolute_time_ms"}:
                transformed = torch.log1p(feature / 1000.0)
                pieces.append(transformed * feature_valid)
                continue
            if feature_name == "beat_phase":
                phase = feature * (2.0 * math.pi)
                transformed = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
                pieces.append(transformed * feature_valid.expand_as(transformed))
                continue
            if feature_name == "tempo_bpm":
                transformed = torch.log(feature.clamp_min(1.0) / 120.0)
                pieces.append(transformed * feature_valid)
                continue

            raise ValueError(f"Unsupported time feature: {feature_name}")

        return self.proj(torch.cat(pieces, dim=-1))


class KymaFeedForward(nn.Module):
    """SwiGLU feedforward block."""

    def __init__(self, *, d_model: int, mult: int, dropout_p: float) -> None:
        super().__init__()
        hidden_dim = d_model * mult
        self.w1 = nn.Linear(d_model, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, d_model)
        self.w3 = nn.Linear(d_model, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.w1(x)) * self.w3(x)
        return self.dropout(self.w2(gated))


class KymaMixerBlock(nn.Module):
    """Residual mixer block that keeps the stateful mixer boundary explicit."""

    def __init__(
        self,
        *,
        config: KymaModelConfig,
        mixer_factory: MixerFactory,
    ) -> None:
        super().__init__()
        mixer = mixer_factory(config)
        self.norm1 = nn.RMSNorm(config.d_model)
        self.mixer = mixer
        self.residual_dropout = nn.Dropout(config.dropout_p)
        self.norm2 = nn.RMSNorm(config.d_model)
        self.ffn = KymaFeedForward(
            d_model=config.d_model,
            mult=config.ffn_mult,
            dropout_p=config.dropout_p,
        )

    def _stateful_mixer(self) -> StatefulMixerProtocol:
        return cast(StatefulMixerProtocol, self.mixer)

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Any:
        return self._stateful_mixer().init_state(
            batch_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        state: Any | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        normed = self.norm1(x)
        mixer = self._stateful_mixer()

        if return_state:
            mixed, next_state = cast(
                tuple[torch.Tensor, Any],
                mixer.forward(normed, state=state, return_state=True),
            )
        else:
            mixed = cast(
                torch.Tensor,
                mixer.forward(normed, state=state, return_state=False),
            )
            next_state = None

        x = x + self.residual_dropout(mixed)
        x = x + self.ffn(self.norm2(x))
        if not return_state:
            return x
        return x, next_state


class KymaAutoregressiveLM(nn.Module):
    """Kyma language model with explicit time conditioning and recurrent state."""

    def __init__(
        self,
        config: KymaModelConfig,
        *,
        mixer_factory: MixerFactory = build_slinoss_mixer,
    ) -> None:
        super().__init__()
        if (config.expand * config.d_model) % config.d_head != 0:
            raise ValueError("expand * d_model must be divisible by d_head.")

        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embed: nn.Embedding | None
        if config.time_conditioning.learned_positional_embedding:
            self.position_embed = nn.Embedding(config.max_segment_len, config.d_model)
        else:
            self.position_embed = None

        selected_time_features = config.time_conditioning.selected_feature_names()
        self.time_conditioner: KymaTimeConditioner | None
        if selected_time_features:
            self.time_conditioner = KymaTimeConditioner(config=config)
        else:
            self.time_conditioner = None

        self.embedding_dropout = nn.Dropout(config.dropout_p)
        self.blocks = nn.ModuleList(
            [
                KymaMixerBlock(config=config, mixer_factory=mixer_factory)
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        if self.position_embed is not None:
            nn.init.normal_(self.position_embed.weight, mean=0.0, std=0.01)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.lm_head:
                nn.init.xavier_uniform_(module.weight)
                bias = cast(nn.Parameter | None, module.bias)
                if bias is not None:
                    nn.init.zeros_(bias)

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> KymaLMState:
        blocks = [cast(KymaMixerBlock, block) for block in self.blocks]
        return KymaLMState(
            layer_states=tuple(
                block.init_state(batch_size, device=device, dtype=dtype)
                for block in blocks
            ),
            tokens_processed=0,
        )

    def _prepare_time_inputs(
        self,
        *,
        batch_size: int,
        seq_len: int,
        time_features: TimeFeatureInput | None,
        time_feature_mask: torch.Tensor | None,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if time_features is None:
            if time_feature_mask is not None:
                raise ValueError(
                    "time_feature_mask cannot be provided without time_features."
                )
            return None, None

        if isinstance(time_features, KymaTimeFeatures):
            values = time_features.values
            mask = (
                time_features.valid if time_feature_mask is None else time_feature_mask
            )
        else:
            values = time_features
            mask = time_feature_mask

        if values.ndim == 2:
            if batch_size != 1:
                raise ValueError(
                    "Unbatched time features can only be used with batch_size == 1."
                )
            values = values.unsqueeze(0)
        if values.ndim != 3:
            raise ValueError(
                "time_features must have shape (batch, T, feature_dim) "
                "or (T, feature_dim)."
            )
        if values.shape[:2] != (batch_size, seq_len):
            raise ValueError(
                "time_features must align with input_ids batch and length."
            )
        if values.shape[-1] != len(TIME_FEATURE_NAMES):
            raise ValueError(
                f"time_features must expose {len(TIME_FEATURE_NAMES)} base features."
            )

        if mask is None:
            mask = torch.ones_like(values, dtype=torch.bool)
        elif mask.ndim == 2:
            if batch_size != 1:
                raise ValueError(
                    "Unbatched time feature masks require batch_size == 1."
                )
            mask = mask.unsqueeze(0)

        if mask.shape != values.shape:
            raise ValueError("time_feature_mask must match time_features shape.")

        return (
            values.to(device=device, dtype=torch.float32),
            mask.to(device=device, dtype=torch.bool),
        )

    def _position_ids(
        self,
        *,
        seq_len: int,
        start: int,
        device: torch.device,
    ) -> torch.Tensor:
        stop = start + seq_len
        if stop > self.config.max_segment_len:
            raise ValueError(
                f"Position range [{start}, {stop}) exceeds max_segment_len "
                f"{self.config.max_segment_len}."
            )
        return torch.arange(start, stop, device=device, dtype=torch.long)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        time_features: TimeFeatureInput | None = None,
        time_feature_mask: torch.Tensor | None = None,
        state: KymaLMState | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KymaLMState]:
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape (batch, T); got {tuple(input_ids.shape)}."
            )
        batch_size, seq_len = map(int, input_ids.shape)
        if state is not None and len(state.layer_states) != len(self.blocks):
            raise ValueError("state layer count must match the number of model blocks.")

        x = self.token_embed(input_ids)
        position_offset = 0 if state is None else state.tokens_processed
        if self.position_embed is not None:
            positions = self._position_ids(
                seq_len=seq_len,
                start=position_offset,
                device=input_ids.device,
            )
            x = x + self.position_embed(positions).unsqueeze(0)

        time_values, time_mask = self._prepare_time_inputs(
            batch_size=batch_size,
            seq_len=seq_len,
            time_features=time_features,
            time_feature_mask=time_feature_mask,
            device=input_ids.device,
        )
        if self.time_conditioner is not None and time_values is not None:
            x = x + self.time_conditioner(time_values, time_mask)

        x = self.embedding_dropout(x)
        next_layer_states: list[Any] = []
        for layer_idx, module in enumerate(self.blocks):
            block = cast(KymaMixerBlock, module)
            layer_state = None if state is None else state.layer_states[layer_idx]
            if return_state:
                x, next_layer_state = cast(
                    tuple[torch.Tensor, Any],
                    block(x, state=layer_state, return_state=True),
                )
                next_layer_states.append(next_layer_state)
            else:
                x = cast(torch.Tensor, block(x, state=layer_state, return_state=False))

        logits = self.lm_head(self.final_norm(x))
        if not return_state:
            return logits

        return logits, KymaLMState(
            layer_states=tuple(next_layer_states),
            tokens_processed=position_offset + seq_len,
        )

    def step(
        self,
        input_ids: torch.Tensor,
        *,
        time_features: TimeFeatureInput | None = None,
        time_feature_mask: torch.Tensor | None = None,
        state: KymaLMState | None = None,
    ) -> tuple[torch.Tensor, KymaLMState]:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(1)
        elif input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError(
                "step expects input_ids with shape (batch,) or (batch, 1)."
            )

        logits, next_state = cast(
            tuple[torch.Tensor, KymaLMState],
            self.forward(
                input_ids,
                time_features=time_features,
                time_feature_mask=time_feature_mask,
                state=state,
                return_state=True,
            ),
        )
        return logits[:, -1, :], next_state


__all__ = [
    "KymaAutoregressiveLM",
    "KymaLMState",
    "KymaMixerBlock",
    "KymaTimeConditioner",
    "MixerFactory",
    "build_slinoss_mixer",
]
