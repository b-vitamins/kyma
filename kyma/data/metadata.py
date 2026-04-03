"""Metadata validation helpers for dataset construction."""

from __future__ import annotations

from typing import Any

from ariautils.midi import MidiDict

from kyma.config.loaders import loadconfig
from kyma.utils.validation import ensure


def validatemanualmetadata(metadata: dict[str, str]) -> None:
    valid = loadconfig()["data"]["metadata"]["manual"]
    for key, value in metadata.items():
        ensure(key in valid, f"Invalid metadata key: {key}")
        ensure(value in valid[key], f"Invalid metadata value for {key}: {value}")


def applymanualmetadata(mididict: MidiDict, metadata: dict[str, Any]) -> MidiDict:
    for key, value in metadata.items():
        if mididict.metadata.get(key) is None:
            mididict.metadata[key] = value
    return mididict
