"""Tokenizer loading helpers."""

from __future__ import annotations

from ariautils.tokenizer import AbsTokenizer, RelTokenizer, Tokenizer


def gettokenizer(name: str) -> Tokenizer:
    if name == "abs":
        return AbsTokenizer()
    if name == "rel":
        return RelTokenizer()
    raise ValueError(f"Unsupported tokenizer {name!r}.")
