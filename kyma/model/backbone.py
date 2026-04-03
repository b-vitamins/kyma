"""Backbone model shared by Kyma language, classifier, and embedding heads."""

from __future__ import annotations

from typing import cast

import torch
import torch.utils.checkpoint
from torch import nn

from kyma.model.blocks import KymaBlock
from kyma.model.config import ModelConfig
from kyma.model.state import KymaState


class KymaBackbone(nn.Module):
    """SLinOSS-backed sequence model without task-specific heads."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.vocab_size is None:
            raise ValueError("ModelConfig.vocab_size must be set before construction.")

        self.config = config
        self.tokenembed = nn.Embedding(config.vocab_size, config.d_model)
        self.posembed = nn.Embedding(config.max_seq_len, config.d_model)
        self.inputdropout = nn.Dropout(config.drop_p)
        self.blocks = nn.ModuleList([KymaBlock(config) for _ in range(config.n_layers)])
        self.outnorm = nn.LayerNorm(config.d_model)
        self.resetparameters()

    def resetparameters(self) -> None:
        nn.init.normal_(self.tokenembed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.posembed.weight, mean=0.0, std=0.01)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def initstate(
        self,
        batchsize: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> KymaState:
        if batchsize <= 0:
            raise ValueError(f"batchsize must be positive, got {batchsize}.")
        if device is None:
            device = self.tokenembed.weight.device
        if dtype is None:
            dtype = self.tokenembed.weight.dtype
        return KymaState(
            layers=[
                block.initstate(batchsize, device=device, dtype=dtype)
                for block in cast(list[KymaBlock], list(self.blocks))
            ],
            position=0,
        )

    def _positionids(
        self, length: int, *, start: int, device: torch.device
    ) -> torch.Tensor:
        end = start + length
        if end > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {end} exceeds max_seq_len {self.config.max_seq_len}."
            )
        return torch.arange(start, end, device=device, dtype=torch.long)

    def forwardembeddings(
        self,
        hidden: torch.Tensor,
        *,
        state: KymaState | None = None,
        returnstate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KymaState]:
        if hidden.ndim != 3 or hidden.shape[-1] != self.config.d_model:
            raise ValueError(
                f"Expected hidden shape (batch, T, {self.config.d_model}), "
                f"got {tuple(hidden.shape)}."
            )
        batchsize, timesteps, _ = map(int, hidden.shape)
        start = 0 if state is None else int(state.position)
        posids = self._positionids(timesteps, start=start, device=hidden.device)
        hidden = hidden + self.posembed(posids).unsqueeze(0)
        hidden = self.inputdropout(hidden)

        if (
            self.config.grad_checkpoint
            and self.training
            and state is None
            and not returnstate
        ):
            for block in cast(list[KymaBlock], list(self.blocks)):
                hidden = cast(
                    torch.Tensor,
                    torch.utils.checkpoint.checkpoint(
                        lambda x, block=block: cast(torch.Tensor, block(x)),  # noqa: B023
                        hidden,
                        use_reentrant=False,
                    ),
                )
            return self.outnorm(hidden)

        nextlayers: list = []
        for index, block in enumerate(cast(list[KymaBlock], list(self.blocks))):
            layerstate = None if state is None else state.layers[index]
            if returnstate:
                output = block(
                    hidden,
                    state=layerstate,
                    returnstate=True,
                )
                hidden, nextstate = output
                nextlayers.append(nextstate)
            else:
                hidden = block(hidden, state=layerstate, returnstate=False)

        hidden = self.outnorm(hidden)
        if not returnstate:
            return hidden

        return hidden, KymaState(layers=nextlayers, position=start + timesteps)

    def forward(
        self,
        src: torch.Tensor,
        *,
        prepended: torch.Tensor | None = None,
        state: KymaState | None = None,
        returnstate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KymaState]:
        if src.ndim != 2:
            raise ValueError(f"Expected src shape (batch, T), got {tuple(src.shape)}.")
        hidden = self.tokenembed(src)
        if prepended is not None:
            if prepended.ndim != 2 or prepended.shape[0] != hidden.shape[0]:
                raise ValueError("prepended must have shape (batch, d_model).")
            hidden = torch.cat([prepended.unsqueeze(1), hidden[:, :-1, :]], dim=1)
        return self.forwardembeddings(hidden, state=state, returnstate=returnstate)

    def step(
        self,
        hidden: torch.Tensor,
        state: KymaState,
    ) -> tuple[torch.Tensor, KymaState]:
        if hidden.ndim != 2 or hidden.shape[-1] != self.config.d_model:
            raise ValueError(
                f"Expected hidden shape (batch, {self.config.d_model}), "
                f"got {tuple(hidden.shape)}."
            )

        position = state.position
        if position >= self.config.max_seq_len:
            raise ValueError(
                "Decode position "
                f"{position} exceeds max_seq_len {self.config.max_seq_len}."
            )

        pos = self.posembed(
            torch.tensor([position], device=hidden.device, dtype=torch.long)
        )[0]
        hidden = hidden + pos
        for block, layerstate in zip(
            cast(list[KymaBlock], list(self.blocks)),
            state.layers,
            strict=True,
        ):
            hidden = block.decodeoneinplace(hidden, layerstate)
        state.position += 1
        return self.outnorm(hidden), state
