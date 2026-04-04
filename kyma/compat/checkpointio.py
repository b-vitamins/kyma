"""Checkpoint loading and normalization helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn


def _strip_orig_mod_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.replace("_orig_mod.", ""): value for key, value in state.items()}


def loadstate(
    checkpointpath: str | Path,
    *,
    device: str | torch.device = "cpu",
    striporigmod: bool = True,
) -> dict[str, torch.Tensor]:
    path = Path(checkpointpath)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device=str(device))
    else:
        try:
            state = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(path, map_location=device)

    if not isinstance(state, dict):
        raise TypeError(f"Expected a state dict at {path}, got {type(state)!r}.")

    return _strip_orig_mod_prefix(state) if striporigmod else state


def savestate(state: dict[str, torch.Tensor], savepath: str | Path) -> None:
    path = Path(savepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def acceleratecheckpointmodelpath(checkpointdir: str | Path, *, index: int = 0) -> Path:
    from accelerate.utils.constants import MODEL_NAME, SAFE_MODEL_NAME

    root = Path(checkpointdir)
    suffix = f"_{index}" if index > 0 else ""
    candidates = [
        root / f"{SAFE_MODEL_NAME}{suffix}.safetensors",
        root / f"{MODEL_NAME}{suffix}.bin",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find Accelerate model weights under {root} for index {index}."
    )


def loadacceleratemodelstate(
    checkpointdir: str | Path,
    *,
    index: int = 0,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    return loadstate(
        acceleratecheckpointmodelpath(checkpointdir, index=index),
        device=device,
    )


def convertaccelerate(
    modelfactory: Callable[[], nn.Module],
    checkpointdir: str | Path,
    savepath: str | Path,
) -> None:
    import accelerate

    accelerator = accelerate.Accelerator()
    model = accelerator.prepare(modelfactory())
    accelerator.load_state(str(checkpointdir))
    savestate(_strip_orig_mod_prefix(model.state_dict()), savepath)
