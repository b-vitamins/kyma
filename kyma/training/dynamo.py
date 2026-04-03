"""Torch compile helpers for Kyma training entrypoints."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from accelerate.utils import DynamoBackend, TorchDynamoPlugin

COMPILE_BACKENDS = ("no", "eager", "aot_eager", "inductor")
COMPILE_MODES = ("default", "reduce-overhead", "max-autotune")


@dataclass(frozen=True, slots=True)
class CompileConfig:
    """Serializable compile settings for training runs."""

    backend: str = "no"
    mode: str | None = None
    fullgraph: bool = False
    dynamic: bool = False
    regional: bool = False

    def __post_init__(self) -> None:
        normalized = self.backend.strip().lower()
        if normalized not in COMPILE_BACKENDS:
            raise ValueError(
                f"Unsupported compile backend {self.backend!r}; "
                f"expected one of {COMPILE_BACKENDS}."
            )
        if self.mode is not None and self.mode not in COMPILE_MODES:
            raise ValueError(
                f"Unsupported compile mode {self.mode!r}; "
                f"expected one of {COMPILE_MODES}."
            )
        object.__setattr__(self, "backend", normalized)

    @property
    def enabled(self) -> bool:
        return self.backend != "no"

    def createplugin(self) -> TorchDynamoPlugin | None:
        if not self.enabled:
            return None
        kwargs: dict[str, Any] = {
            "backend": DynamoBackend(self.backend.upper()),
            "fullgraph": self.fullgraph,
            "dynamic": self.dynamic,
            "use_regional_compilation": self.regional,
        }
        if self.mode is not None:
            kwargs["mode"] = self.mode
        return TorchDynamoPlugin(
            **kwargs,
        )

    def asdict(self) -> dict[str, Any]:
        return {
            "compile_backend": self.backend,
            "compile_mode": self.mode,
            "compile_fullgraph": self.fullgraph,
            "compile_dynamic": self.dynamic,
            "compile_regional": self.regional,
        }


def addcompileargs(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach shared compile arguments to a CLI parser."""

    parser.add_argument(
        "--compile_backend",
        choices=COMPILE_BACKENDS,
        default="no",
    )
    parser.add_argument(
        "--compile_mode",
        choices=COMPILE_MODES,
        default=None,
    )
    parser.add_argument("--compile_fullgraph", action="store_true")
    parser.add_argument("--compile_dynamic", action="store_true")
    parser.add_argument("--compile_regional", action="store_true")
    return parser
