"""One-token decode helpers."""

from __future__ import annotations

import torch

from kyma.model.languagemodel import KymaLM
from kyma.model.state import KymaState


@torch.inference_mode()
def decodeone(
    model: KymaLM,
    idxs: torch.Tensor,
    *,
    state: KymaState,
) -> tuple[torch.Tensor, KymaState]:
    return model.decodeone(idxs, state)
