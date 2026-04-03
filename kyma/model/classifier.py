"""Classifier head models."""

from __future__ import annotations

import torch
from torch import nn

from kyma.model.backbone import KymaBackbone
from kyma.model.config import ModelConfig


class KymaClassifier(nn.Module):
    """Sequence classifier over per-token hidden states."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.class_size is None:
            raise ValueError("ModelConfig.class_size must be set before construction.")
        self.backbone = KymaBackbone(config)
        self.classhead = nn.Linear(config.d_model, config.class_size, bias=False)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        return self.classhead(self.backbone(src))
