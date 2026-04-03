"""Shared configuration schemas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PackedShard:
    """Metadata describing one reusable packed-shard file."""

    name: str
    sequence_count: int
    loss_token_count: int


@dataclass(frozen=True, slots=True)
class PackedDatasetManifest:
    """Manifest stored alongside reusable packed pretraining shards."""

    format_version: int
    tokenizer_name: str
    tokenizer_config: dict[str, Any]
    max_seq_len: int
    shard_token_capacity: int
    separate_sequences: bool
    embedding_size: int | None
    sequence_count: int
    loss_token_count: int
    shards: list[PackedShard]


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Filesystem layout for a training run."""

    root: Path
    checkpoints: Path
    logs: Path
    metrics: Path
