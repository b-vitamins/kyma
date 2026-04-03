"""Configuration loading helpers."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from kyma.model.config import ModelConfig

REPOROOT = Path(__file__).resolve().parents[2]
CONFIGROOT = REPOROOT / "config"


@lru_cache(maxsize=1)
def loadconfig() -> dict[str, Any]:
    with (CONFIGROOT / "config.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def loadmodelconfig(name: str) -> dict[str, Any]:
    path = CONFIGROOT / "models" / f"{name}.json"
    if not path.is_file():
        path = CONFIGROOT / "models" / f"{name.replace('_', '')}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find model preset {name!r} in {CONFIGROOT / 'models'}."
        )
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def loadmodelschema(name: str) -> ModelConfig:
    return ModelConfig(**loadmodelconfig(name))
