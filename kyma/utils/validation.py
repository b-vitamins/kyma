"""Small validation helpers used across Kyma."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

Exc = TypeVar("Exc", bound=Exception)


def ensure(condition: bool, message: str, exc: type[Exc] = ValueError) -> None:
    if not condition:
        raise exc(message)


def ensurefile(path: str | Path, *, label: str = "file") -> Path:
    candidate = Path(path)
    ensure(candidate.is_file(), f"{label} not found: {candidate}", FileNotFoundError)
    return candidate


def ensuredir(path: str | Path, *, label: str = "directory") -> Path:
    candidate = Path(path)
    ensure(candidate.is_dir(), f"{label} not found: {candidate}", FileNotFoundError)
    return candidate
