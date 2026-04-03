"""Sequence transform helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

Item = TypeVar("Item")


def composetransforms(
    transforms: Callable[[Sequence[Item]], Sequence[Item]]
    | list[Callable[[Sequence[Item]], Sequence[Item]]]
    | None,
) -> Callable[[Sequence[Item]], Sequence[Item]] | None:
    if transforms is None:
        return None
    if callable(transforms):
        return transforms
    if not transforms:
        return None

    def composed(items: Sequence[Item]) -> Sequence[Item]:
        output = items
        for transform in transforms:
            output = transform(output)
        return output

    return composed
