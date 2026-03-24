"""Piece-level tokenization and time-feature adapters for Kyma."""

from __future__ import annotations

import json
from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import torch

KymaToken: TypeAlias = tuple[Any, ...] | str
JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

TIME_FEATURE_NAMES: tuple[str, ...] = (
    "delta_time_ms",
    "absolute_time_ms",
    "beat_phase",
    "tempo_bpm",
)


class TokenizerLike(Protocol):
    """Minimal tokenizer contract required by the Kyma data adapters."""

    abs_time_step_ms: int

    def tokenize(self, midi_dict: Any, **kwargs: Any) -> list[KymaToken]: ...

    def encode(self, tokens: list[KymaToken]) -> list[int]: ...


@dataclass(frozen=True)
class TempoPoint:
    """Tempo change point represented in real time."""

    time_ms: int
    bpm: float

    def __post_init__(self) -> None:
        if self.time_ms < 0:
            raise ValueError("TempoPoint.time_ms must be non-negative.")
        if self.bpm <= 0.0:
            raise ValueError("TempoPoint.bpm must be positive.")


@dataclass(frozen=True)
class TempoMap:
    """Piecewise-constant tempo map used for token-level musical-time features."""

    points: tuple[TempoPoint, ...]

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("TempoMap requires at least one tempo point.")
        prev_time_ms = -1
        for point in self.points:
            if point.time_ms < prev_time_ms:
                raise ValueError("TempoMap points must be sorted by time.")
            prev_time_ms = point.time_ms

    @classmethod
    def from_midi_dict(cls, midi_dict: Any) -> TempoMap:
        """Build a tempo map from a MidiDict-like object."""

        try:
            tempo_msgs = midi_dict.tempo_msgs
            tick_to_ms = midi_dict.tick_to_ms
        except AttributeError as exc:
            raise TypeError(
                "midi_dict must expose tempo_msgs and tick_to_ms for tempo-aware "
                "feature extraction."
            ) from exc

        points: list[TempoPoint] = []
        for msg in tempo_msgs:
            tick = int(msg["tick"])
            tempo_us_per_beat = int(msg["data"])
            time_ms = int(tick_to_ms(tick))
            bpm = 60_000_000.0 / float(tempo_us_per_beat)
            if points and points[-1].time_ms == time_ms:
                points[-1] = TempoPoint(time_ms=time_ms, bpm=bpm)
            else:
                points.append(TempoPoint(time_ms=time_ms, bpm=bpm))

        if not points:
            points.append(TempoPoint(time_ms=0, bpm=120.0))
        elif points[0].time_ms != 0:
            points.insert(0, TempoPoint(time_ms=0, bpm=120.0))

        return cls(tuple(points))

    @cached_property
    def _times_ms(self) -> tuple[int, ...]:
        return tuple(point.time_ms for point in self.points)

    @cached_property
    def _cumulative_beats(self) -> tuple[float, ...]:
        cumulative: list[float] = [0.0]
        beats = 0.0
        for prev_point, next_point in zip(self.points, self.points[1:], strict=False):
            duration_ms = next_point.time_ms - prev_point.time_ms
            beats += (duration_ms * prev_point.bpm) / 60_000.0
            cumulative.append(beats)
        return tuple(cumulative)

    def bpm_at(self, time_ms: int) -> float:
        """Return the BPM in effect at the given absolute time."""

        if time_ms < 0:
            raise ValueError("time_ms must be non-negative.")
        idx = bisect_right(self._times_ms, time_ms) - 1
        return self.points[max(idx, 0)].bpm

    def beats_at(self, time_ms: int) -> float:
        """Return the accumulated quarter-note beat count at the given time."""

        if time_ms < 0:
            raise ValueError("time_ms must be non-negative.")
        idx = bisect_right(self._times_ms, time_ms) - 1
        point_idx = max(idx, 0)
        point = self.points[point_idx]
        beats = self._cumulative_beats[point_idx]
        beats += ((time_ms - point.time_ms) * point.bpm) / 60_000.0
        return beats

    def beat_phase_at(self, time_ms: int) -> float:
        """Return the fractional beat phase in the range [0, 1)."""

        return self.beats_at(time_ms) % 1.0


@dataclass(frozen=True)
class KymaTimeFeatures:
    """Dense per-token timing features and their validity mask."""

    values: torch.Tensor
    valid: torch.Tensor
    names: tuple[str, ...] = TIME_FEATURE_NAMES

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("values must have shape (sequence_length, feature_dim).")
        if self.valid.shape != self.values.shape:
            raise ValueError("valid mask must match the shape of values.")
        if self.values.shape[1] != len(self.names):
            raise ValueError("Feature name count must match feature dimension.")

    @property
    def sequence_length(self) -> int:
        return int(self.values.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.values.shape[1])


@dataclass(frozen=True)
class KymaTokenizedPiece:
    """Encoded piece representation for Kyma pretraining and evaluation."""

    piece_id: str
    tokens: tuple[KymaToken, ...]
    token_ids: torch.Tensor
    time_features: KymaTimeFeatures
    metadata: dict[str, JsonValue]
    source_path: str | None = None

    def __post_init__(self) -> None:
        if not self.piece_id:
            raise ValueError("piece_id must be non-empty.")
        if self.token_ids.ndim != 1:
            raise ValueError("token_ids must have shape (sequence_length,).")
        if int(self.token_ids.shape[0]) != len(self.tokens):
            raise ValueError("token_ids length must match the token sequence length.")
        if self.time_features.sequence_length != len(self.tokens):
            raise ValueError(
                "time_features length must match the token sequence length."
            )


def _load_mididict_class() -> type[Any]:
    try:
        midi_module = import_module("ariautils.midi")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ariautils is required for reading Aria-compatible MIDI datasets."
        ) from exc
    return midi_module.MidiDict


def iter_mididict_jsonl(path: str | Path) -> Iterator[Any]:
    """Yield MidiDict objects from an Aria-style JSONL dataset file."""

    mididict_cls = _load_mididict_class()
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield mididict_cls.from_msg_dict(json.loads(line))


def _is_note_like(token: KymaToken) -> bool:
    return (
        isinstance(token, tuple)
        and len(token) > 0
        and token[0] not in {"prefix", "onset", "dur"}
    )


def _is_duration_token(token: KymaToken) -> bool:
    return isinstance(token, tuple) and len(token) > 0 and token[0] == "dur"


def extract_time_features(
    tokens: list[KymaToken] | tuple[KymaToken, ...],
    *,
    tempo_map: TempoMap | None = None,
    abs_time_step_ms: int = 5000,
) -> KymaTimeFeatures:
    """Extract dense timing features from an Aria-compatible token sequence."""

    if abs_time_step_ms <= 0:
        raise ValueError("abs_time_step_ms must be positive.")

    values = torch.zeros((len(tokens), len(TIME_FEATURE_NAMES)), dtype=torch.float32)
    valid = torch.zeros((len(tokens), len(TIME_FEATURE_NAMES)), dtype=torch.bool)

    current_segment_ms = 0
    last_onset_ms = 0
    onset_seen = False

    onset_feature_indices: list[tuple[int, int]] = []

    for idx, token in enumerate(tokens):
        values[idx, 1] = float(current_segment_ms)
        valid[idx, 0] = True
        valid[idx, 1] = True

        if tempo_map is not None:
            values[idx, 2] = float(tempo_map.beat_phase_at(current_segment_ms))
            values[idx, 3] = float(tempo_map.bpm_at(current_segment_ms))
            valid[idx, 2] = True
            valid[idx, 3] = True

        if token == "<T>":
            current_segment_ms += abs_time_step_ms
            continue

        if not (isinstance(token, tuple) and len(token) == 2 and token[0] == "onset"):
            continue

        absolute_time_ms = current_segment_ms + int(token[1])
        delta_time_ms = (
            absolute_time_ms - last_onset_ms if onset_seen else absolute_time_ms
        )
        last_onset_ms = absolute_time_ms
        onset_seen = True

        values[idx, 0] = float(delta_time_ms)
        values[idx, 1] = float(absolute_time_ms)
        valid[idx, 0] = True
        valid[idx, 1] = True

        if tempo_map is not None:
            values[idx, 2] = float(tempo_map.beat_phase_at(absolute_time_ms))
            values[idx, 3] = float(tempo_map.bpm_at(absolute_time_ms))
            valid[idx, 2] = True
            valid[idx, 3] = True

        if idx > 0 and _is_note_like(tokens[idx - 1]):
            onset_feature_indices.append((idx, idx - 1))
        if idx + 1 < len(tokens) and _is_duration_token(tokens[idx + 1]):
            onset_feature_indices.append((idx, idx + 1))

    for source_idx, target_idx in onset_feature_indices:
        values[target_idx] = values[source_idx]
        valid[target_idx] = valid[source_idx]

    return KymaTimeFeatures(values=values, valid=valid)


def make_tokenized_piece(
    *,
    piece_id: str,
    tokens: list[KymaToken] | tuple[KymaToken, ...],
    tokenizer: TokenizerLike,
    metadata: dict[str, JsonValue] | None = None,
    source_path: str | None = None,
    time_features: KymaTimeFeatures | None = None,
) -> KymaTokenizedPiece:
    """Encode a token sequence into the canonical Kyma piece container."""

    token_tuple = tuple(tokens)
    encoded = torch.tensor(tokenizer.encode(list(token_tuple)), dtype=torch.long)
    if time_features is None:
        time_features = extract_time_features(
            token_tuple,
            abs_time_step_ms=int(getattr(tokenizer, "abs_time_step_ms", 5000)),
        )

    return KymaTokenizedPiece(
        piece_id=piece_id,
        tokens=token_tuple,
        token_ids=encoded,
        time_features=time_features,
        metadata={} if metadata is None else dict(metadata),
        source_path=source_path,
    )


def tokenize_midi_record(
    midi_dict: Any,
    *,
    tokenizer: TokenizerLike,
    piece_id: str | None = None,
    metadata: dict[str, JsonValue] | None = None,
    source_path: str | None = None,
    tokenize_kwargs: dict[str, Any] | None = None,
) -> KymaTokenizedPiece:
    """Tokenize a MidiDict-like record into the canonical Kyma piece container."""

    kwargs = {} if tokenize_kwargs is None else dict(tokenize_kwargs)
    midi_metadata = getattr(midi_dict, "metadata", {})
    merged_metadata = {**midi_metadata, **({} if metadata is None else metadata)}
    metadata_source_path = merged_metadata.get("abs_load_path")
    resolved_source_path = (
        source_path if source_path is not None else metadata_source_path
    )
    if resolved_source_path is not None and not isinstance(resolved_source_path, str):
        raise TypeError("source_path must be a string when present.")

    resolved_piece_id = piece_id if piece_id is not None else resolved_source_path
    if resolved_piece_id is None:
        raise ValueError(
            "piece_id must be provided when the MIDI metadata has no abs_load_path."
        )

    tokens = tokenizer.tokenize(midi_dict, **kwargs)
    tempo_map: TempoMap | None
    try:
        tempo_map = TempoMap.from_midi_dict(midi_dict)
    except TypeError:
        tempo_map = None
    time_features = extract_time_features(
        tokens,
        tempo_map=tempo_map,
        abs_time_step_ms=int(getattr(tokenizer, "abs_time_step_ms", 5000)),
    )
    return make_tokenized_piece(
        piece_id=resolved_piece_id,
        tokens=tokens,
        tokenizer=tokenizer,
        metadata=merged_metadata,
        source_path=resolved_source_path,
        time_features=time_features,
    )


__all__ = [
    "JsonValue",
    "KymaTimeFeatures",
    "KymaToken",
    "KymaTokenizedPiece",
    "TIME_FEATURE_NAMES",
    "TempoMap",
    "TempoPoint",
    "TokenizerLike",
    "extract_time_features",
    "iter_mididict_jsonl",
    "make_tokenized_piece",
    "tokenize_midi_record",
]
