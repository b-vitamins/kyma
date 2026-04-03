"""Prompt prefill helpers."""

from __future__ import annotations

import torch

from kyma.model.languagemodel import KymaLM
from kyma.model.state import KymaState


@torch.inference_mode()
def prefill(
    model: KymaLM,
    idxs: torch.Tensor,
    *,
    state: KymaState | None = None,
) -> tuple[torch.Tensor, KymaState]:
    return model.prefill(idxs, state=state)
