"""Head and feed-forward modules used by Kyma models."""

from __future__ import annotations

import torch
from slinoss.ops.decode_linear import decode_linear
from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block with decode-one support."""

    def __init__(self, dmodel: int, *, mult: int = 4) -> None:
        super().__init__()
        hidden = dmodel * mult
        self.gate = nn.Linear(dmodel, hidden, bias=False)
        self.value = nn.Linear(dmodel, hidden, bias=False)
        self.down = nn.Linear(hidden, dmodel, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.value(x))

    def decodeone(self, x: torch.Tensor) -> torch.Tensor:
        gate = decode_linear(x, self.gate)
        value = decode_linear(x, self.value)
        return decode_linear(F.silu(gate) * value, self.down)
