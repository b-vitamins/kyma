"""Tokenizer adapters that keep Aria comparability explicit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class KymaTokenizerConfig:
    """Tokenizer contract for Kyma.

    The initial implementation intentionally stays Aria-compatible for baseline
    comparability, while exposing the additional time-aware features that Kyma
    will evaluate separately.
    """

    family: str = "aria-abs"
    add_time_features: bool = True
    add_beat_phase_features: bool = True
    add_tempo_features: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_abs_tokenizer(*, config_path: str | None = None) -> Any:
    """Create an Aria-compatible absolute tokenizer.

    The return type stays generic because `ariautils` does not currently ship
    precise type information.
    """

    try:
        tokenizer_module = import_module("ariautils.tokenizer")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ariautils is required for Kyma tokenization. "
            "Install project dependencies before creating tokenizers."
        ) from exc
    tokenizer_cls = tokenizer_module.AbsTokenizer

    if config_path is not None:
        return tokenizer_cls(config_path=config_path)
    return tokenizer_cls()
