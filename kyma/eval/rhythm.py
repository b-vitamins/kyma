"""Rhythm-aware evaluation for symbolic timing behavior in Kyma."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from kyma.data import TIME_FEATURE_NAMES, KymaToken, KymaTokenizedPiece
from kyma.eval.protocol import RhythmEvalSpec
from kyma.model import KymaAutoregressiveLM

_BEAT_PHASE_FEATURE_IDX = TIME_FEATURE_NAMES.index("beat_phase")
_TEMPO_FEATURE_IDX = TIME_FEATURE_NAMES.index("tempo_bpm")


@dataclass(frozen=True)
class RhythmMetric:
    """Weighted scalar metric plus the token count supporting it."""

    value: float
    count: int


@dataclass(frozen=True)
class RhythmExampleResult:
    """Per-piece rhythm-aware evaluation result."""

    piece_id: str
    onset_nll: float
    duration_nll: float
    tempo_consistency: float
    beat_phase_consistency: float


@dataclass(frozen=True)
class RhythmReport:
    """Full report for the rhythm-aware evaluation track."""

    spec: RhythmEvalSpec
    onset_nll: RhythmMetric
    duration_nll: RhythmMetric
    tempo_consistency: RhythmMetric
    beat_phase_consistency: RhythmMetric
    examples: tuple[RhythmExampleResult, ...]


def _is_onset_token(token: KymaToken) -> bool:
    return isinstance(token, tuple) and len(token) == 2 and token[0] == "onset"


def _is_duration_token(token: KymaToken) -> bool:
    return isinstance(token, tuple) and len(token) > 0 and token[0] == "dur"


def _mean_or_zero(values: torch.Tensor) -> float:
    return 0.0 if values.numel() == 0 else float(values.mean().item())


@torch.no_grad()
def evaluate_rhythm(
    model: KymaAutoregressiveLM,
    pieces: Sequence[KymaTokenizedPiece],
    *,
    spec: RhythmEvalSpec,
    device: str | torch.device = "cpu",
) -> RhythmReport:
    """Evaluate rhythm-sensitive token prediction on tokenized pieces."""

    resolved_device = torch.device(device)
    was_training = model.training
    model.eval()
    model.to(resolved_device)

    onset_loss_sum = 0.0
    onset_count = 0
    duration_loss_sum = 0.0
    duration_count = 0
    tempo_correct = 0
    tempo_count = 0
    beat_correct = 0
    beat_count = 0
    examples: list[RhythmExampleResult] = []

    for piece in pieces:
        if len(piece.tokens) < 2:
            continue

        input_ids = piece.token_ids[:-1].unsqueeze(0).to(device=resolved_device)
        target_ids = piece.token_ids[1:].to(device=resolved_device)
        input_time_features = (
            piece.time_features.values[:-1].unsqueeze(0).to(device=resolved_device)
        )
        input_time_feature_mask = (
            piece.time_features.valid[:-1].unsqueeze(0).to(device=resolved_device)
        )
        logits = model(
            input_ids,
            time_features=input_time_features,
            time_feature_mask=input_time_feature_mask,
        )
        logits = logits.squeeze(0)
        token_loss = F.cross_entropy(logits, target_ids, reduction="none")
        predicted_ids = torch.argmax(logits, dim=-1)

        target_tokens = piece.tokens[1:]
        onset_mask = torch.tensor(
            [_is_onset_token(token) for token in target_tokens],
            dtype=torch.bool,
            device=resolved_device,
        )
        duration_mask = torch.tensor(
            [_is_duration_token(token) for token in target_tokens],
            dtype=torch.bool,
            device=resolved_device,
        )

        target_time_values = piece.time_features.values[1:].to(device=resolved_device)
        target_time_valid = piece.time_features.valid[1:].to(device=resolved_device)
        prev_time_values = piece.time_features.values[:-1].to(device=resolved_device)
        prev_time_valid = piece.time_features.valid[:-1].to(device=resolved_device)

        tempo_change_mask = (
            target_time_valid[:, _TEMPO_FEATURE_IDX]
            & prev_time_valid[:, _TEMPO_FEATURE_IDX]
            & (
                torch.abs(
                    target_time_values[:, _TEMPO_FEATURE_IDX]
                    - prev_time_values[:, _TEMPO_FEATURE_IDX]
                )
                > 1e-6
            )
        )
        beat_phase_mask = target_time_valid[:, _BEAT_PHASE_FEATURE_IDX] & onset_mask

        onset_nll = (
            _mean_or_zero(token_loss[onset_mask]) if spec.report_onset_nll else 0.0
        )
        duration_nll = (
            _mean_or_zero(token_loss[duration_mask])
            if spec.report_duration_nll
            else 0.0
        )
        tempo_accuracy = (
            _mean_or_zero(
                (predicted_ids[tempo_change_mask] == target_ids[tempo_change_mask]).to(
                    dtype=torch.float32
                )
            )
            if spec.report_tempo_consistency
            else 0.0
        )
        beat_phase_accuracy = (
            _mean_or_zero(
                (predicted_ids[beat_phase_mask] == target_ids[beat_phase_mask]).to(
                    dtype=torch.float32
                )
            )
            if spec.report_beat_phase_consistency
            else 0.0
        )

        if spec.report_onset_nll:
            onset_loss_sum += float(token_loss[onset_mask].sum().item())
            onset_count += int(onset_mask.sum().item())
        if spec.report_duration_nll:
            duration_loss_sum += float(token_loss[duration_mask].sum().item())
            duration_count += int(duration_mask.sum().item())
        if spec.report_tempo_consistency:
            tempo_correct += int(
                (predicted_ids[tempo_change_mask] == target_ids[tempo_change_mask])
                .sum()
                .item()
            )
            tempo_count += int(tempo_change_mask.sum().item())
        if spec.report_beat_phase_consistency:
            beat_correct += int(
                (predicted_ids[beat_phase_mask] == target_ids[beat_phase_mask])
                .sum()
                .item()
            )
            beat_count += int(beat_phase_mask.sum().item())

        examples.append(
            RhythmExampleResult(
                piece_id=piece.piece_id,
                onset_nll=onset_nll,
                duration_nll=duration_nll,
                tempo_consistency=tempo_accuracy,
                beat_phase_consistency=beat_phase_accuracy,
            )
        )

    if was_training:
        model.train()

    return RhythmReport(
        spec=spec,
        onset_nll=RhythmMetric(
            value=0.0 if onset_count == 0 else onset_loss_sum / onset_count,
            count=onset_count,
        ),
        duration_nll=RhythmMetric(
            value=0.0 if duration_count == 0 else duration_loss_sum / duration_count,
            count=duration_count,
        ),
        tempo_consistency=RhythmMetric(
            value=0.0 if tempo_count == 0 else tempo_correct / tempo_count,
            count=tempo_count,
        ),
        beat_phase_consistency=RhythmMetric(
            value=0.0 if beat_count == 0 else beat_correct / beat_count,
            count=beat_count,
        ),
        examples=tuple(examples),
    )


__all__ = [
    "RhythmExampleResult",
    "RhythmMetric",
    "RhythmReport",
    "evaluate_rhythm",
]
