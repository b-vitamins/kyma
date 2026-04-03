"""Named recurrent state objects for Kyma models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from slinoss.layers.state import SLinOSSMixerState


@dataclass
class KymaState:
    """Persistent recurrent state for a full Kyma backbone."""

    layers: list[SLinOSSMixerState]
    position: int = 0

    def clone(self) -> KymaState:
        return KymaState(
            layers=[layer.clone() for layer in self.layers],
            position=int(self.position),
        )

    def detach(self) -> KymaState:
        return KymaState(
            layers=[layer.detach() for layer in self.layers],
            position=int(self.position),
        )

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> KymaState:
        return KymaState(
            layers=[layer.to(device=device, dtype=dtype) for layer in self.layers],
            position=int(self.position),
        )
