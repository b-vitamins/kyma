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


def _normalizetransformedsegment(
    original: Sequence[Item],
    transformed: Sequence[Item],
    *,
    paditem: Item,
) -> list[Item]:
    output = list(transformed[: len(original)])
    if len(output) < len(original):
        output.extend([paditem] * (len(original) - len(output)))
    return output


def applytocompletewindows(
    items: Sequence[Item],
    transform: Callable[[list[Item]], Sequence[Item]],
    *,
    bostok: Item,
    eostok: Item,
    padtok: Item,
) -> list[Item]:
    """Apply a transform only to complete BOS/EOS-bounded sequences."""

    output: list[Item] = []
    active_start: int | None = None
    flush_start = 0

    for index, item in enumerate(items):
        if active_start is None:
            if item == bostok:
                output.extend(items[flush_start:index])
                active_start = index
        else:
            if item == bostok:
                output.extend(items[flush_start:index])
                flush_start = index
                active_start = index
            elif item == eostok:
                segment = list(items[active_start : index + 1])
                output.extend(
                    _normalizetransformedsegment(
                        segment,
                        transform(segment),
                        paditem=padtok,
                    )
                )
                flush_start = index + 1
                active_start = None

    output.extend(items[flush_start:])
    return _normalizetransformedsegment(items, output, paditem=padtok)


def buildpackedaugmentations(tokenizer) -> list[Callable[[Sequence[Item]], list[Item]]]:
    """Wrap Aria augmentations so packed windows mutate only complete sequences."""

    bostok = getattr(tokenizer, "bos_tok", None)
    eostok = getattr(tokenizer, "eos_tok", None)
    padtok = getattr(tokenizer, "pad_tok", None)
    if bostok is None or eostok is None or padtok is None:
        raise ValueError("Packed-window augmentation requires bos/eos/pad tokens.")

    wrapped = []
    for transform in tokenizer.export_data_aug():

        def safeapply(
            items: Sequence[Item],
            *,
            transform=transform,
            bostok=bostok,
            eostok=eostok,
            padtok=padtok,
        ) -> list[Item]:
            return applytocompletewindows(
                items,
                transform,
                bostok=bostok,
                eostok=eostok,
                padtok=padtok,
            )

        wrapped.append(safeapply)
    return wrapped
