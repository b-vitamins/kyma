"""Typed evaluation protocol definitions.

Kyma keeps Aria-style short-context evaluation as a baseline, but the protocol
must additionally measure the three project differentiators directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ShortContextParitySpec:
    prompt_durations_s: list[int]
    continuation_tokens: int
    temperature: float
    min_p: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShortContextParitySpec:
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LongHorizonEvalSpec:
    prompt_lengths_s: list[int]
    continuation_lengths_s: list[int]
    state_carry_reset_intervals: list[int]
    report_horizon_nll: bool
    report_structure_metrics: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LongHorizonEvalSpec:
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StreamingEvalSpec:
    interactive_session_lengths_s: list[int]
    report_time_to_first_note_ms: bool
    report_decode_throughput: bool
    report_memory_growth: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StreamingEvalSpec:
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RhythmEvalSpec:
    report_onset_nll: bool
    report_duration_nll: bool
    report_tempo_consistency: bool
    report_beat_phase_consistency: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RhythmEvalSpec:
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KymaEvalProtocol:
    short_context_parity: ShortContextParitySpec
    long_horizon: LongHorizonEvalSpec
    streaming: StreamingEvalSpec
    rhythm: RhythmEvalSpec

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KymaEvalProtocol:
        return cls(
            short_context_parity=ShortContextParitySpec.from_dict(
                data["short_context_parity"]
            ),
            long_horizon=LongHorizonEvalSpec.from_dict(data["long_horizon"]),
            streaming=StreamingEvalSpec.from_dict(data["streaming"]),
            rhythm=RhythmEvalSpec.from_dict(data["rhythm"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "short_context_parity": self.short_context_parity.to_dict(),
            "long_horizon": self.long_horizon.to_dict(),
            "streaming": self.streaming.to_dict(),
            "rhythm": self.rhythm.to_dict(),
        }
