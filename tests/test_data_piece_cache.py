from __future__ import annotations

from pathlib import Path

import torch

from kyma.data import (
    KymaTimeFeatures,
    KymaTokenizedPiece,
    load_piece_cache,
    piece_from_record,
    piece_to_record,
    save_piece_cache,
)


def _make_piece(piece_id: str = "piece-a") -> KymaTokenizedPiece:
    return KymaTokenizedPiece(
        piece_id=piece_id,
        tokens=("<S>", ("onset", 0), ("dur", 120)),
        token_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        time_features=KymaTimeFeatures(
            values=torch.tensor(
                [
                    [0.0, 0.0, 0.0, 120.0],
                    [0.0, 0.0, 0.0, 120.0],
                    [0.0, 0.0, 0.0, 120.0],
                ],
                dtype=torch.float32,
            ),
            valid=torch.ones((3, 4), dtype=torch.bool),
        ),
        metadata={"split": "train"},
        source_path="/tmp/example.mid",
    )


def test_piece_record_roundtrip_preserves_tokens_and_features() -> None:
    piece = _make_piece()

    record = piece_to_record(piece)
    rebuilt = piece_from_record(record)

    assert rebuilt.piece_id == piece.piece_id
    assert rebuilt.tokens == piece.tokens
    assert torch.equal(rebuilt.token_ids, piece.token_ids)
    assert torch.equal(rebuilt.time_features.values, piece.time_features.values)
    assert torch.equal(rebuilt.time_features.valid, piece.time_features.valid)
    assert rebuilt.metadata == piece.metadata
    assert rebuilt.source_path == piece.source_path


def test_piece_cache_roundtrip_loads_multiple_pieces(tmp_path: Path) -> None:
    cache_path = tmp_path / "pieces.jsonl"
    written = save_piece_cache(
        [_make_piece("piece-a"), _make_piece("piece-b")],
        cache_path,
    )

    loaded = load_piece_cache(cache_path)

    assert written == 2
    assert [piece.piece_id for piece in loaded] == ["piece-a", "piece-b"]
