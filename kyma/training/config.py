"""Typed configuration for Kyma pretraining."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class KymaOptimizerConfig:
    """Optimizer settings for language-model pretraining."""

    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.lr <= 0.0:
            raise ValueError("lr must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if not 0.0 < self.beta1 < 1.0:
            raise ValueError("beta1 must be in (0, 1).")
        if not 0.0 < self.beta2 < 1.0:
            raise ValueError("beta2 must be in (0, 1).")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KymaOptimizerConfig:
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KymaScheduleConfig:
    """Learning-rate schedule settings."""

    warmup_steps: int = 0
    min_lr_scale: float = 0.1

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if not 0.0 < self.min_lr_scale <= 1.0:
            raise ValueError("min_lr_scale must be in (0, 1].")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KymaScheduleConfig:
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KymaPretrainConfig:
    """Top-level configuration for the Kyma pretraining loop."""

    batch_size: int
    max_steps: int
    grad_accum_steps: int = 1
    precision: str = "fp32"
    grad_clip_norm: float | None = 1.0
    log_every_steps: int = 10
    checkpoint_every_steps: int | None = None
    device: str = "cpu"
    optimizer: KymaOptimizerConfig = field(default_factory=KymaOptimizerConfig)
    schedule: KymaScheduleConfig = field(default_factory=KymaScheduleConfig)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if self.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be positive.")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be one of: fp32, fp16, bf16.")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive when provided.")
        if self.log_every_steps <= 0:
            raise ValueError("log_every_steps must be positive.")
        if self.checkpoint_every_steps is not None and self.checkpoint_every_steps <= 0:
            raise ValueError("checkpoint_every_steps must be positive when provided.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KymaPretrainConfig:
        return cls(
            batch_size=int(data["batch_size"]),
            max_steps=int(data["max_steps"]),
            grad_accum_steps=int(data.get("grad_accum_steps", 1)),
            precision=str(data.get("precision", "fp32")),
            grad_clip_norm=(
                None
                if data.get("grad_clip_norm") is None
                else float(data["grad_clip_norm"])
            ),
            log_every_steps=int(data.get("log_every_steps", 10)),
            checkpoint_every_steps=(
                None
                if data.get("checkpoint_every_steps") is None
                else int(data["checkpoint_every_steps"])
            ),
            device=str(data.get("device", "cpu")),
            optimizer=KymaOptimizerConfig.from_dict(data.get("optimizer", {})),
            schedule=KymaScheduleConfig.from_dict(data.get("schedule", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "max_steps": self.max_steps,
            "grad_accum_steps": self.grad_accum_steps,
            "precision": self.precision,
            "grad_clip_norm": self.grad_clip_norm,
            "log_every_steps": self.log_every_steps,
            "checkpoint_every_steps": self.checkpoint_every_steps,
            "device": self.device,
            "optimizer": self.optimizer.to_dict(),
            "schedule": self.schedule.to_dict(),
        }


__all__ = [
    "KymaOptimizerConfig",
    "KymaPretrainConfig",
    "KymaScheduleConfig",
]
