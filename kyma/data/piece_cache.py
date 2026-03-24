"""Serialization helpers for tokenized Kyma piece caches."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import torch

from kyma.data.pieces import (
    JsonValue,
    KymaTimeFeatures,
    KymaToken,
    KymaTokenizedPiece,
)


def _token_to_json(token: KymaToken) -> JsonValue:
    if isinstance(token, str):
        return token
    return list(token)


def _token_from_json(value: JsonValue) -> KymaToken:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return tuple(value)
    raise TypeError(f"Unsupported serialized token type: {type(value).__name__}")


def piece_to_record(piece: KymaTokenizedPiece) -> dict[str, JsonValue]:
    """Serialize a tokenized piece into a JSONL-friendly record."""

    return {
        "piece_id": piece.piece_id,
        "tokens": [_token_to_json(token) for token in piece.tokens],
        "token_ids": [int(value.item()) for value in piece.token_ids],
        "time_features": {
            "values": [
                [float(value.item()) for value in row]
                for row in piece.time_features.values
            ],
            "valid": [
                [bool(value.item()) for value in row]
                for row in piece.time_features.valid
            ],
            "names": list(piece.time_features.names),
        },
        "metadata": piece.metadata,
        "source_path": piece.source_path,
    }


def piece_from_record(record: dict[str, JsonValue]) -> KymaTokenizedPiece:
    """Reconstruct a tokenized piece from a serialized cache record."""

    time_features = record["time_features"]
    if not isinstance(time_features, dict):
        raise TypeError("time_features must be a dictionary.")

    values = torch.tensor(time_features["values"], dtype=torch.float32)
    valid = torch.tensor(time_features["valid"], dtype=torch.bool)
    names_raw = time_features.get("names")
    if not isinstance(names_raw, list):
        raise TypeError("time_features.names must be a list.")

    tokens_raw = record["tokens"]
    if not isinstance(tokens_raw, list):
        raise TypeError("tokens must be a list.")
    token_ids_raw = record["token_ids"]
    if not isinstance(token_ids_raw, list):
        raise TypeError("token_ids must be a list.")
    metadata = record["metadata"]
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary.")

    source_path = record.get("source_path")
    if source_path is not None and not isinstance(source_path, str):
        raise TypeError("source_path must be a string when present.")

    return KymaTokenizedPiece(
        piece_id=str(record["piece_id"]),
        tokens=tuple(_token_from_json(token) for token in tokens_raw),
        token_ids=torch.tensor(token_ids_raw, dtype=torch.long),
        time_features=KymaTimeFeatures(
            values=values,
            valid=valid,
            names=tuple(str(name) for name in names_raw),
        ),
        metadata=metadata,
        source_path=source_path,
    )


def save_piece_cache(
    pieces: Iterable[KymaTokenizedPiece],
    path: str | Path,
) -> int:
    """Write a tokenized piece cache to JSONL and return the number of pieces."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for piece in pieces:
            json.dump(piece_to_record(piece), handle)
            handle.write("\n")
            count += 1
    return count


def load_piece_cache(
    path: str | Path,
    *,
    limit: int | None = None,
) -> list[KymaTokenizedPiece]:
    """Load tokenized pieces from a JSONL cache file."""

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided.")

    pieces: list[KymaTokenizedPiece] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if not line.strip():
                continue
            loaded_record = json.loads(line)
            if not isinstance(loaded_record, dict):
                raise TypeError("Serialized piece records must be JSON objects.")
            record = cast(dict[str, JsonValue], loaded_record)
            pieces.append(piece_from_record(record))
    return pieces


__all__ = [
    "load_piece_cache",
    "piece_from_record",
    "piece_to_record",
    "save_piece_cache",
]
