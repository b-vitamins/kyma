"""Typed model configuration for Kyma."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class KymaTimeConditioningConfig:
    """Controls the explicit musical-time signals exposed to the model."""

    learned_positional_embedding: bool = False
    delta_time_features: bool = True
    beat_phase_features: bool = True
    tempo_features: bool = True
    feature_mlp_dim: int = 128

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KymaTimeConditioningConfig:
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KymaLongContextConfig:
    """Controls state-carry and contiguous-chunk training behavior."""

    state_carry_training: bool = True
    chunk_size_tokens: int = 1024
    burn_in_tokens: int = 128
    tbptt_window_tokens: int = 1024
    max_piece_tokens: int = 32768

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
    max_segment_len: int
    time_conditioning: KymaTimeConditioningConfig
    long_context: KymaLongContextConfig
    differentiators: KymaEvalDifferentiators

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
