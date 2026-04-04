"""Model configuration for Kyma's SLinOSS-backed models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ModelConfig:
    d_model: int
    n_heads: int
    n_layers: int
    ff_mult: int
    drop_p: float
    max_seq_len: int
    grad_checkpoint: bool
    resid_dropout: float = 0.0
    vocab_size: int | None = None
    class_size: int | None = None
    emb_size: int | None = None
    d_state: int = 64
    expand: int = 2
    d_conv: int = 4
    chunk_size: int = 128

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}."
            )
        if self.d_head % 2 != 0:
            raise ValueError(
                f"d_head={self.d_head} must be even so RoPE can rotate head pairs."
            )

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def setvocabsize(self, vocabsize: int) -> None:
        self.vocab_size = vocabsize
