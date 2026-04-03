"""Shared utilities for training loops."""

from __future__ import annotations

from dataclasses import dataclass, field

import accelerate
import torch

from kyma.config.schemas import ProjectPaths


@dataclass
class LossTracker:
    """Track trailing and running mean loss values."""

    trailingwindow: int
    values: list[float] = field(default_factory=list)

    def update(self, value: float) -> tuple[float, float]:
        self.values.append(value)
        trailing = sum(self.values[-self.trailingwindow :]) / len(
            self.values[-self.trailingwindow :]
        )
        average = sum(self.values) / len(self.values)
        return trailing, average


def gatheredloss(accelerator: accelerate.Accelerator, loss: torch.Tensor) -> float:
    return float(accelerator.gather(loss).mean(dim=0).item())


def lrstring(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> str:
    if scheduler is not None:
        try:
            return f"{scheduler.get_last_lr()[0]:.2e}"
        except (AttributeError, IndexError, TypeError):
            return f"{optimizer.param_groups[-1]['lr']:.2e}"
    return f"{optimizer.param_groups[-1]['lr']:.2e}"


def savecheckpoint(
    accelerator: accelerate.Accelerator,
    projectpaths: ProjectPaths,
    *,
    epoch: int,
    step: int,
) -> None:
    if not accelerator.is_main_process:
        return
    accelerator.save_state(str(projectpaths.checkpoints / f"epoch{epoch}_step{step}"))
