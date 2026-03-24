"""Load packaged Kyma configuration files."""

from __future__ import annotations

import json
import os
from importlib.abc import Traversable
from importlib.resources import files
from typing import Any

_CONFIG_ROOT = files("kyma").joinpath("configs")


def _load_config(relative_path: str) -> dict[str, Any]:
    config_path = _CONFIG_ROOT.joinpath(relative_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {relative_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_model_config(name: str) -> dict[str, Any]:
    """Load a packaged model configuration by name."""

    return _load_config(f"model/{name}.json")


def load_eval_config(name: str) -> dict[str, Any]:
    """Load a packaged evaluation protocol by name."""

    return _load_config(f"eval/{name}.json")


def load_training_config(name: str) -> dict[str, Any]:
    """Load a packaged training configuration by name."""

    return _load_config(f"training/{name}.json")


def _config_name(path: Traversable) -> str:
    filename = os.path.basename(str(path))
    return filename[: -len(".json")]


def list_model_configs() -> list[str]:
    """Return packaged model configuration names."""

    model_dir = _CONFIG_ROOT.joinpath("model")
    return sorted(
        _config_name(path)
        for path in model_dir.iterdir()
        if str(path).endswith(".json")
    )


def list_eval_configs() -> list[str]:
    """Return packaged evaluation protocol names."""

    eval_dir = _CONFIG_ROOT.joinpath("eval")
    return sorted(
        _config_name(path) for path in eval_dir.iterdir() if str(path).endswith(".json")
    )


def list_training_configs() -> list[str]:
    """Return packaged training configuration names."""

    training_dir = _CONFIG_ROOT.joinpath("training")
    return sorted(
        _config_name(path)
        for path in training_dir.iterdir()
        if str(path).endswith(".json")
    )
