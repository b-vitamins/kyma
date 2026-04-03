"""Optimizer and scheduler helpers."""

from __future__ import annotations

import torch
from torch import nn


def buildadamw(
    model: nn.Module,
    *,
    lr: float,
    weightdecay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-5,
) -> torch.optim.Optimizer:
    usefused = any(parameter.is_cuda for parameter in model.parameters())
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weightdecay,
        betas=betas,
        eps=eps,
        fused=usefused,
    )


def buildlinearscheduler(
    optimizer: torch.optim.Optimizer,
    *,
    totalsteps: int,
    warmupsteps: int,
    endratio: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    if totalsteps <= 0:
        raise ValueError("totalsteps must be positive.")
    warmupsteps = max(0, min(warmupsteps, totalsteps - 1))
    if warmupsteps == 0:
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=endratio,
            total_iters=totalsteps,
        )

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.000001,
        end_factor=1.0,
        total_iters=warmupsteps,
    )
    decay = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=endratio,
        total_iters=totalsteps - warmupsteps,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, decay],
        milestones=[warmupsteps],
    )
