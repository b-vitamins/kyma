"""Kyma language models."""

from __future__ import annotations

import torch
from slinoss.ops.decode_linear import decode_linear
from torch import nn

from kyma.model.backbone import KymaBackbone
from kyma.model.config import ModelConfig
from kyma.model.state import KymaState


class KymaLM(nn.Module):
    """Autoregressive language model with optional embedding conditioning."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.vocab_size is None:
            raise ValueError("ModelConfig.vocab_size must be set before construction.")
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.backbone = KymaBackbone(config)
        self.lmhead = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lmhead.weight = self.backbone.tokenembed.weight
        self.embeddingadapter = (
            nn.Linear(config.emb_size, config.d_model, bias=False)
            if config.emb_size is not None
            else None
        )

    def _adaptembedding(self, emb: torch.Tensor) -> torch.Tensor:
        if self.embeddingadapter is None:
            raise ValueError("This model does not support embedding conditioning.")
        if emb.ndim != 2:
            raise ValueError(
                f"Expected emb shape (batch, d_emb), got {tuple(emb.shape)}."
            )
        return self.embeddingadapter(emb)

    def initstate(
        self,
        batchsize: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> KymaState:
        return self.backbone.initstate(batchsize, device=device, dtype=dtype)

    def forward(
        self, src: torch.Tensor, emb: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden = self.backbone(
            src,
            prepended=(self._adaptembedding(emb) if emb is not None else None),
        )
        logits = self.lmhead(hidden)
        if emb is None:
            return logits
        return logits[:, 1:, :]

    def prefill(
        self,
        idxs: torch.Tensor,
        *,
        state: KymaState | None = None,
    ) -> tuple[torch.Tensor, KymaState]:
        hidden, nextstate = self.backbone(idxs, state=state, returnstate=True)
        return self.lmhead(hidden), nextstate

    @torch.no_grad()
    def decodeone(
        self,
        idxs: torch.Tensor,
        state: KymaState,
    ) -> tuple[torch.Tensor, KymaState]:
        if idxs.ndim == 2:
            if idxs.shape[1] != 1:
                raise ValueError("decodeone expects (batch,) or (batch, 1) token ids.")
            idxs = idxs[:, 0]
        elif idxs.ndim != 1:
            raise ValueError(
                f"decodeone expects (batch,) or (batch, 1); got {tuple(idxs.shape)}."
            )
        hidden = self.backbone.tokenembed(idxs)
        hidden, nextstate = self.backbone.step(hidden, state)
        return decode_linear(hidden, self.lmhead), nextstate

    @torch.no_grad()
    def primecondition(
        self,
        emb: torch.Tensor,
        state: KymaState,
    ) -> KymaState:
        hidden = self._adaptembedding(emb)
        _, nextstate = self.backbone.step(hidden, state)
        return nextstate
