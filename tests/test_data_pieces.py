from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import pytest
import torch

from kyma.data import (
    KymaToken,
    TempoMap,
    TempoPoint,
    extract_time_features,
    make_tokenized_piece,
    tokenize_midi_record,
)


@dataclass
class FakeTokenizer:
    abs_time_step_ms: int = 5000
    token_map: dict[KymaToken, int] = field(
        default_factory=lambda: cast(dict[KymaToken, int], {})
    )

    def tokenize(self, midi_dict: Any, **kwargs: Any) -> list[KymaToken]:
        return list(midi_dict.tokens)

    def encode(self, tokens: list[KymaToken]) -> list[int]:
        encoded: list[int] = []
        for token in tokens:
            if token not in self.token_map:
                self.token_map[token] = len(self.token_map)
            encoded.append(self.token_map[token])
        return encoded


@dataclass
class FakeMidiRecord:
    tokens: list[KymaToken]
    metadata: dict[str, object]
    tempo_msgs: list[dict[str, int]]

    def tick_to_ms(self, tick: int) -> int:
        return tick * 2


def test_tempo_map_reports_piecewise_bpm_and_beats() -> None:
    tempo_map = TempoMap(
        points=(
            TempoPoint(time_ms=0, bpm=120.0),
            TempoPoint(time_ms=1000, bpm=60.0),
        )
    )

    assert abs(tempo_map.bpm_at(250) - 120.0) < 1e-6
    assert abs(tempo_map.bpm_at(1500) - 60.0) < 1e-6
    assert abs(tempo_map.beats_at(500) - 1.0) < 1e-6
    assert abs(tempo_map.beats_at(1500) - 2.5) < 1e-6
    assert abs(tempo_map.beat_phase_at(1500) - 0.5) < 1e-6


def test_extract_time_features_propagates_onset_to_note_triplets() -> None:
    tokens: list[KymaToken] = [
        "<S>",
        ("piano", 60, 80),
        ("onset", 120),
        ("dur", 240),
        "<T>",
        ("drum", 38),
        ("onset", 40),
        "<E>",
    ]

    features = extract_time_features(tokens, abs_time_step_ms=5000)

    note_triplet = features.values[1:4]
    assert torch.equal(note_triplet[0], note_triplet[1])
    assert torch.equal(note_triplet[1], note_triplet[2])
    assert abs(features.values[1, 0].item() - 120.0) < 1e-6
    assert abs(features.values[1, 1].item() - 120.0) < 1e-6
    assert abs(features.values[5, 0].item() - 4920.0) < 1e-6
    assert abs(features.values[5, 1].item() - 5040.0) < 1e-6
    assert not bool(features.valid[1, 2].item())
    assert not bool(features.valid[1, 3].item())


def test_tokenize_midi_record_builds_piece_and_time_features() -> None:
    tokenizer = FakeTokenizer()
    midi = FakeMidiRecord(
        tokens=[
            "<S>",
            ("piano", 64, 72),
            ("onset", 120),
            ("dur", 360),
            "<E>",
        ],
        metadata={"abs_load_path": "/tmp/example.mid", "split": "train"},
        tempo_msgs=[{"tick": 0, "data": 500000}],
    )

    piece = tokenize_midi_record(midi, tokenizer=tokenizer)

    assert piece.piece_id == "/tmp/example.mid"
    assert piece.source_path == "/tmp/example.mid"
    assert piece.metadata["split"] == "train"
    assert piece.time_features.valid[:, 2].all()
    assert piece.time_features.valid[:, 3].all()
    assert abs(piece.time_features.values[1, 3].item() - 120.0) < 1e-6
    assert [int(token_id) for token_id in piece.token_ids] == [0, 1, 2, 3, 4]


def test_make_tokenized_piece_requires_matching_time_feature_length() -> None:
    tokenizer = FakeTokenizer()
    with pytest.raises(ValueError):
        make_tokenized_piece(
            piece_id="piece",
            tokens=["<S>", "<E>"],
            tokenizer=tokenizer,
            time_features=extract_time_features(["<S>"]),
        )
