"""Checkpoint format for Kyma pretraining."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from kyma.model import KymaModelConfig
from kyma.training.config import KymaPretrainConfig

KYMA_CHECKPOINT_VERSION = 1


class GradScalerStateful(Protocol):
    """Minimal checkpointable gradient-scaler surface."""

    def is_enabled(self) -> bool: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class KymaTrainState:
    """Scalar training progress tracked across checkpoints."""

    global_step: int = 0
    optimizer_steps: int = 0
    tokens_processed: int = 0

    def __post_init__(self) -> None:
        if self.global_step < 0:
            raise ValueError("global_step must be non-negative.")
        if self.optimizer_steps < 0:
            raise ValueError("optimizer_steps must be non-negative.")
        if self.tokens_processed < 0:
            raise ValueError("tokens_processed must be non-negative.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KymaTrainState:
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KymaCheckpointBundle:
    """Structured view of a serialized Kyma pretraining checkpoint."""

    format_version: int
    model_config: KymaModelConfig
    pretrain_config: KymaPretrainConfig
    train_state: KymaTrainState
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any] | None
    scheduler_state: dict[str, Any] | None
    scaler_state: dict[str, Any] | None
    extra: dict[str, Any]


def save_pretrain_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    model_config: KymaModelConfig,
    pretrain_config: KymaPretrainConfig,
    train_state: KymaTrainState,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: GradScalerStateful | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Serialize model, optimizer, and training metadata to a checkpoint file."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": KYMA_CHECKPOINT_VERSION,
        "model_config": model_config.to_dict(),
        "pretrain_config": pretrain_config.to_dict(),
        "train_state": train_state.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "scaler_state": (
            None if scaler is None or not scaler.is_enabled() else scaler.state_dict()
        ),
        "extra": {} if extra is None else dict(extra),
    }
    torch.save(payload, checkpoint_path)


def load_pretrain_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: GradScalerStateful | None = None,
    map_location: str | torch.device | None = None,
) -> KymaCheckpointBundle:
    """Load a Kyma pretraining checkpoint and optionally restore stateful objects."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    format_version = int(payload["format_version"])
    if format_version != KYMA_CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {format_version}; "
            f"expected {KYMA_CHECKPOINT_VERSION}."
        )

    if model is not None:
        model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload["optimizer_state"] is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload["scheduler_state"] is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    scaler_state = payload.get("scaler_state")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    return KymaCheckpointBundle(
        format_version=format_version,
        model_config=KymaModelConfig.from_dict(payload["model_config"]),
        pretrain_config=KymaPretrainConfig.from_dict(payload["pretrain_config"]),
        train_state=KymaTrainState.from_dict(payload["train_state"]),
        model_state=payload["model_state"],
        optimizer_state=payload["optimizer_state"],
        scheduler_state=payload["scheduler_state"],
        scaler_state=scaler_state,
        extra=dict(payload.get("extra", {})),
    )


__all__ = [
    "GradScalerStateful",
    "KYMA_CHECKPOINT_VERSION",
    "KymaCheckpointBundle",
    "KymaTrainState",
    "load_pretrain_checkpoint",
    "save_pretrain_checkpoint",
]
