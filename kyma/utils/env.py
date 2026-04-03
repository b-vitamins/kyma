"""Repo-local environment loading helpers."""

from __future__ import annotations

import os
import shlex
from functools import lru_cache
from pathlib import Path

REPOROOT = Path(__file__).resolve().parents[2]
ENVPATH = REPOROOT / ".env"


def parseenv(path: str | Path) -> dict[str, str]:
    """Parse a shell-style env file into a flat key/value mapping."""

    envpath = Path(path)
    if not envpath.is_file():
        return {}

    values: dict[str, str] = {}
    for rawline in envpath.read_text(encoding="utf-8").splitlines():
        line = rawline.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        parsed = shlex.split(value, posix=True)
        values[key.strip()] = parsed[0] if parsed else ""
    return values


@lru_cache(maxsize=1)
def loadrepowandbenv() -> dict[str, str]:
    """Load W&B-related keys from the repo-local `.env` into `os.environ`."""

    loaded = {
        key: value
        for key, value in parseenv(ENVPATH).items()
        if key.startswith("WANDB_") or key.startswith("KYMA_WANDB")
    }
    for key, value in loaded.items():
        os.environ.setdefault(key, value)
    return loaded
