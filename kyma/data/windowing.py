"""Contiguous windowing utilities for state-carry Kyma pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from kyma.data.pieces import KymaTokenizedPiece
from kyma.model.config import KymaLongContextConfig


@dataclass(frozen=True)
class KymaWindowSpec:
    """Windowing parameters for contiguous piece training."""

    chunk_size_tokens: int
    burn_in_tokens: int
    tbptt_window_tokens: int
    max_piece_tokens: int

    def __post_init__(self) -> None:
        if self.chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive.")
        if self.burn_in_tokens < 0:
            raise ValueError("burn_in_tokens must be non-negative.")
        if self.tbptt_window_tokens <= 0:
            raise ValueError("tbptt_window_tokens must be positive.")
        if self.max_piece_tokens <= 0:
            raise ValueError("max_piece_tokens must be positive.")

    @classmethod
    def from_long_context_config(cls, config: KymaLongContextConfig) -> KymaWindowSpec:
        return cls(
            chunk_size_tokens=config.chunk_size_tokens,
            burn_in_tokens=config.burn_in_tokens,
            tbptt_window_tokens=config.tbptt_window_tokens,
            max_piece_tokens=config.max_piece_tokens,
        )


@dataclass(frozen=True)
class KymaTrainingWindow:
    """Single contiguous training chunk from a tokenized piece."""

    piece_id: str
    window_index: int
    start_token_idx: int
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    time_features: torch.Tensor
    time_feature_mask: torch.Tensor
    loss_mask: torch.Tensor
    carry_from_previous: bool
    detach_state_after: bool
    is_piece_start: bool
    is_piece_end: bool

    def __post_init__(self) -> None:
        expected_shape = self.input_ids.shape
        if self.target_ids.shape != expected_shape:
            raise ValueError("target_ids must match input_ids shape.")
        if self.loss_mask.shape != expected_shape:
            raise ValueError("loss_mask must match input_ids shape.")
        if self.time_features.shape[:1] != expected_shape:
            raise ValueError("time_features must align with input_ids length.")
        if self.time_feature_mask.shape != self.time_features.shape:
            raise ValueError("time_feature_mask must match time_features shape.")


def build_training_windows(
    piece: KymaTokenizedPiece,
    *,
    window_spec: KymaWindowSpec,
    pad_token_id: int,
) -> list[KymaTrainingWindow]:
    """Slice a piece into contiguous training windows for state-carry LM training."""

    if pad_token_id < 0:
        raise ValueError("pad_token_id must be non-negative.")

    max_tokens = min(len(piece.tokens), window_spec.max_piece_tokens)
    if max_tokens == 0:
        return []

    token_ids = piece.token_ids[:max_tokens]
    time_features = piece.time_features.values[:max_tokens]
    time_feature_mask = piece.time_features.valid[:max_tokens]

    windows: list[KymaTrainingWindow] = []
    chunk = window_spec.chunk_size_tokens
    for window_index, start_idx in enumerate(range(0, max_tokens, chunk)):
        end_idx = min(start_idx + chunk, max_tokens)
        actual_len = end_idx - start_idx

        input_ids = torch.full((chunk,), pad_token_id, dtype=token_ids.dtype)
        target_ids = torch.full((chunk,), pad_token_id, dtype=token_ids.dtype)
        feature_values = torch.zeros(
            (chunk, time_features.shape[-1]), dtype=time_features.dtype
        )
        feature_mask = torch.zeros(
            (chunk, time_feature_mask.shape[-1]), dtype=time_feature_mask.dtype
        )
        loss_mask = torch.zeros((chunk,), dtype=torch.bool)

        input_ids[:actual_len] = token_ids[start_idx:end_idx]
        feature_values[:actual_len] = time_features[start_idx:end_idx]
        feature_mask[:actual_len] = time_feature_mask[start_idx:end_idx]

        for offset in range(actual_len):
            target_pos = start_idx + offset + 1
            if target_pos >= max_tokens:
                break
            target_ids[offset] = token_ids[target_pos]
            loss_mask[offset] = True

        is_piece_start = start_idx == 0
        if is_piece_start and window_spec.burn_in_tokens > 0:
            burn_in = min(window_spec.burn_in_tokens, actual_len)
            loss_mask[:burn_in] = False

        tokens_consumed = end_idx
        is_piece_end = end_idx >= max_tokens
        detach_state_after = (
            is_piece_end or tokens_consumed % window_spec.tbptt_window_tokens == 0
        )
        windows.append(
            KymaTrainingWindow(
                piece_id=piece.piece_id,
                window_index=window_index,
                start_token_idx=start_idx,
                input_ids=input_ids,
                target_ids=target_ids,
                time_features=feature_values,
                time_feature_mask=feature_mask,
                loss_mask=loss_mask,
                carry_from_previous=not is_piece_start,
                detach_state_after=detach_state_after,
                is_piece_start=is_piece_start,
                is_piece_end=is_piece_end,
            )
        )

    return windows


def collate_training_windows(
    windows: list[KymaTrainingWindow],
) -> dict[str, Any]:
    """Collate typed windows into a batch dictionary for a DataLoader."""

    if not windows:
        raise ValueError("collate_training_windows requires at least one window.")

    return {
        "piece_ids": [window.piece_id for window in windows],
        "window_indices": torch.tensor(
            [window.window_index for window in windows],
            dtype=torch.long,
        ),
        "start_token_indices": torch.tensor(
            [window.start_token_idx for window in windows],
            dtype=torch.long,
        ),
        "input_ids": torch.stack([window.input_ids for window in windows], dim=0),
        "target_ids": torch.stack([window.target_ids for window in windows], dim=0),
        "time_features": torch.stack(
            [window.time_features for window in windows],
            dim=0,
        ),
        "time_feature_mask": torch.stack(
            [window.time_feature_mask for window in windows],
            dim=0,
        ),
        "loss_mask": torch.stack([window.loss_mask for window in windows], dim=0),
        "carry_from_previous": torch.tensor(
            [window.carry_from_previous for window in windows],
            dtype=torch.bool,
        ),
        "detach_state_after": torch.tensor(
            [window.detach_state_after for window in windows],
            dtype=torch.bool,
        ),
        "is_piece_start": torch.tensor(
            [window.is_piece_start for window in windows],
            dtype=torch.bool,
        ),
        "is_piece_end": torch.tensor(
            [window.is_piece_end for window in windows],
            dtype=torch.bool,
        ),
    }


class KymaStateCarryDataset(torch.utils.data.Dataset[KymaTrainingWindow]):
    """Flat dataset of contiguous windows built from tokenized pieces."""

    def __init__(self, windows: list[KymaTrainingWindow]) -> None:
        self.windows = windows

    @classmethod
    def from_pieces(
        cls,
        pieces: list[KymaTokenizedPiece],
        *,
        window_spec: KymaWindowSpec,
        pad_token_id: int,
    ) -> KymaStateCarryDataset:
        windows: list[KymaTrainingWindow] = []
        for piece in pieces:
            windows.extend(
                build_training_windows(
                    piece,
                    window_spec=window_spec,
                    pad_token_id=pad_token_id,
                )
            )
        return cls(windows)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> KymaTrainingWindow:
        return self.windows[index]


__all__ = [
    "KymaStateCarryDataset",
    "KymaTrainingWindow",
    "KymaWindowSpec",
    "build_training_windows",
    "collate_training_windows",
]
