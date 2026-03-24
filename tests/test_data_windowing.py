from __future__ import annotations

import torch

from kyma.data import (
    KymaStateCarryDataset,
    KymaTimeFeatures,
    KymaTokenizedPiece,
    KymaWindowSpec,
    build_training_windows,
    collate_training_windows,
)


def _make_piece(piece_id: str, length: int) -> KymaTokenizedPiece:
    token_ids = torch.arange(length, dtype=torch.long)
    values = torch.arange(length * 4, dtype=torch.float32).view(length, 4)
    valid = torch.ones((length, 4), dtype=torch.bool)
    return KymaTokenizedPiece(
        piece_id=piece_id,
        tokens=tuple(f"tok-{idx}" for idx in range(length)),
        token_ids=token_ids,
        time_features=KymaTimeFeatures(values=values, valid=valid),
        metadata={},
        source_path=None,
    )


def test_build_training_windows_marks_boundaries_and_burn_in() -> None:
    piece = _make_piece("piece-a", length=6)
    window_spec = KymaWindowSpec(
        chunk_size_tokens=4,
        burn_in_tokens=2,
        tbptt_window_tokens=4,
        max_piece_tokens=16,
    )

    windows = build_training_windows(piece, window_spec=window_spec, pad_token_id=99)

    assert len(windows) == 2
    first, second = windows

    assert first.is_piece_start is True
    assert first.carry_from_previous is False
    assert first.detach_state_after is True
    assert [int(value) for value in first.input_ids] == [0, 1, 2, 3]
    assert [int(value) for value in first.target_ids] == [1, 2, 3, 4]
    assert [bool(value) for value in first.loss_mask] == [False, False, True, True]

    assert second.is_piece_end is True
    assert second.carry_from_previous is True
    assert [int(value) for value in second.input_ids] == [4, 5, 99, 99]
    assert [int(value) for value in second.target_ids] == [5, 99, 99, 99]
    assert [bool(value) for value in second.loss_mask] == [True, False, False, False]
    assert second.time_feature_mask[2:].sum().item() == 0


def test_build_training_windows_respects_tbptt_and_piece_truncation() -> None:
    piece = _make_piece("piece-b", length=9)
    window_spec = KymaWindowSpec(
        chunk_size_tokens=3,
        burn_in_tokens=0,
        tbptt_window_tokens=6,
        max_piece_tokens=7,
    )

    windows = build_training_windows(piece, window_spec=window_spec, pad_token_id=99)

    assert len(windows) == 3
    assert [window.detach_state_after for window in windows] == [False, True, True]
    assert [int(value) for value in windows[-1].input_ids] == [6, 99, 99]
    assert [int(value) for value in windows[-1].target_ids] == [99, 99, 99]
    assert [bool(value) for value in windows[-1].loss_mask] == [False, False, False]


def test_dataset_from_pieces_and_collate_preserve_metadata() -> None:
    piece_a = _make_piece("piece-a", length=5)
    piece_b = _make_piece("piece-b", length=4)
    window_spec = KymaWindowSpec(
        chunk_size_tokens=3,
        burn_in_tokens=1,
        tbptt_window_tokens=3,
        max_piece_tokens=16,
    )

    dataset = KymaStateCarryDataset.from_pieces(
        [piece_a, piece_b],
        window_spec=window_spec,
        pad_token_id=99,
    )
    batch = collate_training_windows([dataset[0], dataset[2]])

    assert len(dataset) == 4
    assert batch["piece_ids"] == ["piece-a", "piece-b"]
    assert batch["input_ids"].shape == (2, 3)
    assert batch["time_features"].shape == (2, 3, 4)
    assert batch["is_piece_start"].tolist() == [True, True]
