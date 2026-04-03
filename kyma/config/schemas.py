"""Shared configuration schemas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetHeader:
    """Header stored on the first line of Kyma dataset shard files."""

    tokenizer_name: str
    tokenizer_config: dict[str, Any]
    max_seq_len: int


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Filesystem layout for a training run."""

    root: Path
    checkpoints: Path
    logs: Path
    metrics: Path
