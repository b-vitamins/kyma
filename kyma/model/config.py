"""Typed model configuration for Kyma."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class KymaTimeConditioningConfig:
    """Controls the explicit musical-time signals exposed to the model."""

    learned_positional_embedding: bool = False
    delta_time_features: bool = True
    absolute_time_features: bool = True
    beat_phase_features: bool = True
    tempo_features: bool = True
    feature_mlp_dim: int = 128

    def __post_init__(self) -> None:
        if self.feature_mlp_dim <= 0:
            raise ValueError("feature_mlp_dim must be positive.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KymaTimeConditioningConfig:
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def selected_feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.delta_time_features:
            names.append("delta_time_ms")
        if self.absolute_time_features:
            names.append("absolute_time_ms")
        if self.beat_phase_features:
            names.append("beat_phase")
        if self.tempo_features:
            names.append("tempo_bpm")
        return tuple(names)


@dataclass(frozen=True)
class KymaLongContextConfig:
    """Controls state-carry and contiguous-chunk training behavior."""

    state_carry_training: bool = True
    chunk_size_tokens: int = 1024
    burn_in_tokens: int = 128
    tbptt_window_tokens: int = 1024
    max_piece_tokens: int = 32768

    def __post_init__(self) -> None:
        if self.chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive.")
        if self.burn_in_tokens < 0:
            raise ValueError("burn_in_tokens must be non-negative.")
        if self.tbptt_window_tokens <= 0:
            raise ValueError("tbptt_window_tokens must be positive.")
        if self.max_piece_tokens <= 0:
            raise ValueError("max_piece_tokens must be positive.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KymaLongContextConfig:
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KymaEvalDifferentiators:
    """The project-level claims that Kyma is obligated to evaluate."""

    long_form_stateful_generation: bool = True
    real_time_interactive_continuation: bool = True
    rhythm_aware_modeling: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KymaEvalDifferentiators:
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KymaModelConfig:
    """Top-level configuration for a Kyma language model."""

    d_model: int
    n_layers: int
    d_state: int
    expand: int
    d_head: int
    d_conv: int
    chunk_size: int
    vocab_size: int
    dropout_p: float
    ffn_mult: int
    max_segment_len: int
    time_conditioning: KymaTimeConditioningConfig
    long_context: KymaLongContextConfig
    differentiators: KymaEvalDifferentiators

    def __post_init__(self) -> None:
        for field_name in (
            "d_model",
            "n_layers",
            "d_state",
            "expand",
            "d_head",
            "d_conv",
            "chunk_size",
            "vocab_size",
            "ffn_mult",
            "max_segment_len",
        ):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive.")
        if not 0.0 <= self.dropout_p < 1.0:
            raise ValueError("dropout_p must be in the range [0, 1).")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KymaModelConfig:
        return cls(
            d_model=int(data["d_model"]),
            n_layers=int(data["n_layers"]),
            d_state=int(data["d_state"]),
            expand=int(data["expand"]),
            d_head=int(data["d_head"]),
            d_conv=int(data["d_conv"]),
            chunk_size=int(data["chunk_size"]),
            vocab_size=int(data["vocab_size"]),
            dropout_p=float(data["dropout_p"]),
            ffn_mult=int(data.get("ffn_mult", 4)),
            max_segment_len=int(data["max_segment_len"]),
            time_conditioning=KymaTimeConditioningConfig.from_dict(
                data["time_conditioning"]
            ),
            long_context=KymaLongContextConfig.from_dict(data["long_context"]),
            differentiators=KymaEvalDifferentiators.from_dict(data["differentiators"]),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["time_conditioning"] = self.time_conditioning.to_dict()
        data["long_context"] = self.long_context.to_dict()
        data["differentiators"] = self.differentiators.to_dict()
        return data
