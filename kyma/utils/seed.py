"""Random seeding helpers."""

from __future__ import annotations

import random

import torch


def setseed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
