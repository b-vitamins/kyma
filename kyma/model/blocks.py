"""Kyma residual blocks."""

from __future__ import annotations

import torch
from slinoss.layers import SLinOSSMixer
from slinoss.layers.state import SLinOSSMixerState
from torch import nn
from torch.nn import functional as F

from kyma.model.config import ModelConfig
from kyma.model.heads import SwiGLU
from kyma.model.rope import applyroperaw


class KymaBlock(nn.Module):
    """A pre-norm residual block with a SLinOSS mixer and SwiGLU MLP."""

    def __init__(self, config: ModelConfig, *, residdropout: float) -> None:
        super().__init__()
        self.nheads = config.n_heads
        self.residdropout = float(residdropout)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.mixer = SLinOSSMixer(
            config.d_model,
            d_state=config.d_state,
            expand=config.expand,
            d_head=config.d_head,
            d_conv=config.d_conv,
            chunk_size=config.chunk_size,
            dt_min=1e-3,
            dt_init_floor=1e-3,
            r_min=0.2,
            normalize_bc=True,
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ff = SwiGLU(config.d_model, mult=config.ff_mult)

    def initstate(
        self,
        batchsize: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> SLinOSSMixerState:
        return self.mixer.init_decode_state(batchsize, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        state: SLinOSSMixerState | None = None,
        returnstate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, SLinOSSMixerState]:
        mixed = self.mixer(
            applyroperaw(self.norm1(x), freqs_cis, nheads=self.nheads),
            state=state,
            return_state=returnstate,
        )
        nextstate: SLinOSSMixerState | None = None
        if returnstate:
            mixed, nextstate = mixed
        x = x + F.dropout(mixed, p=self.residdropout, training=self.training)
        ff = self.ff(self.norm2(x))
        x = x + F.dropout(ff, p=self.residdropout, training=self.training)
        if not returnstate:
            return x
        if nextstate is None:
            raise RuntimeError("Expected a next state from the mixer.")
        return x, nextstate

    def decodeoneinplace(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        state: SLinOSSMixerState,
    ) -> torch.Tensor:
        normed = self.norm1(x).unsqueeze(1)
        normed = applyroperaw(normed, freqs_cis, nheads=self.nheads).squeeze(1)
        x = x + self.mixer._step_inplace(normed, state)
        return x + self.ff.decodeone(self.norm2(x))
