"""RTX 3060 pilot orchestration helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from kyma.data import (
    build_aria_midi_piece_cache,
    get_abs_tokenizer,
    load_piece_cache,
)
from kyma.data.pieces import KymaTokenizedPiece
from kyma.data.windowing import KymaStateCarryDataset, KymaWindowSpec
from kyma.logging import get_logger
from kyma.model import KymaAutoregressiveLM, KymaModelConfig
from kyma.training import (
    KymaPretrainConfig,
    evaluate_language_model,
    train_language_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "rtx3060"
DEFAULT_3060_MODEL_CONFIG_PATH = EXPERIMENT_ROOT / "model.json"
DEFAULT_3060_TRAINING_CONFIG_PATH = EXPERIMENT_ROOT / "training.json"
DEFAULT_3060_CACHE_PATH = (
    REPO_ROOT
    / "artifacts"
    / "data"
    / "rtx3060-pilot"
    / "aria-midi-pruned-r0-16000.jsonl"
)
DEFAULT_3060_OUTPUT_DIR = REPO_ROOT / "artifacts" / "runs" / "rtx3060-pilot"
LOGGER = get_logger("kyma.pilot.rtx3060")


@dataclass(frozen=True)
class PilotRunSummary:
    """Serializable description of a resolved pilot run."""

    cache_path: str
    model_config_path: str
    training_config_path: str
    output_dir: str
    max_pieces: int | None
    pad_id: int
    train_piece_count: int
    val_piece_count: int
    train_window_count: int
    val_window_count: int
    train_loss_tokens: int
    val_loss_tokens: int
    effective_tokens_per_optimizer_step: int
    model_parameter_count: int
    device: str
    precision: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedPilotRun:
    """Resolved pilot inputs ready for training."""

    summary: PilotRunSummary
    model_config: KymaModelConfig
    pretrain_config: KymaPretrainConfig
    train_dataset: KymaStateCarryDataset
    val_dataset: KymaStateCarryDataset


def _load_json_dict(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _split_pieces(
    pieces: list[KymaTokenizedPiece],
    *,
    val_ratio: float,
) -> tuple[list[KymaTokenizedPiece], list[KymaTokenizedPiece]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in the range (0, 1).")
    if len(pieces) < 2:
        raise ValueError("At least two pieces are required for a train/val split.")

    train_pieces: list[KymaTokenizedPiece] = []
    val_pieces: list[KymaTokenizedPiece] = []
    threshold = int(val_ratio * 10_000)

    for piece in pieces:
        digest = hashlib.sha256(piece.piece_id.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 10_000
        target = val_pieces if bucket < threshold else train_pieces
        target.append(piece)

    if not val_pieces:
        val_pieces.append(train_pieces.pop())
    if not train_pieces:
        train_pieces.append(val_pieces.pop())

    return train_pieces, val_pieces


def _count_loss_tokens(dataset: KymaStateCarryDataset) -> int:
    return sum(int(window.loss_mask.sum().item()) for window in dataset.windows)


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_3060_pilot_cache(
    *,
    subset: str = "pruned",
    root: str | Path = REPO_ROOT / "artifacts" / "data" / "aria-midi",
    extracted_root: str | Path | None = None,
    output_path: str | Path = DEFAULT_3060_CACHE_PATH,
    tokenizer_config_path: str | None = None,
    max_pieces: int = 16_000,
    random_seed: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build the bounded pilot cache used for the 3060 run."""

    return build_aria_midi_piece_cache(
        subset=subset,
        root=root,
        extracted_root=extracted_root,
        output_path=output_path,
        tokenizer_config_path=tokenizer_config_path,
        limit=max_pieces,
        shuffle=True,
        random_seed=random_seed,
        overwrite=overwrite,
    )


def prepare_3060_pilot_run(
    *,
    cache_path: str | Path = DEFAULT_3060_CACHE_PATH,
    model_config_path: str | Path = DEFAULT_3060_MODEL_CONFIG_PATH,
    training_config_path: str | Path = DEFAULT_3060_TRAINING_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_3060_OUTPUT_DIR,
    tokenizer_config_path: str | None = None,
    max_pieces: int | None = None,
    val_ratio: float = 0.02,
) -> PreparedPilotRun:
    """Resolve configs, piece cache, and datasets for the 3060 pilot."""

    model_config = KymaModelConfig.from_dict(_load_json_dict(model_config_path))
    pretrain_config = KymaPretrainConfig.from_dict(
        _load_json_dict(training_config_path)
    )
    tokenizer = get_abs_tokenizer(config_path=tokenizer_config_path)
    pad_id = int(tokenizer.pad_id)

    pieces = load_piece_cache(cache_path, limit=max_pieces)
    train_pieces, val_pieces = _split_pieces(pieces, val_ratio=val_ratio)
    window_spec = KymaWindowSpec.from_long_context_config(model_config.long_context)
    train_dataset = KymaStateCarryDataset.from_pieces(
        train_pieces,
        window_spec=window_spec,
        pad_token_id=pad_id,
    )
    val_dataset = KymaStateCarryDataset.from_pieces(
        val_pieces,
        window_spec=window_spec,
        pad_token_id=pad_id,
    )

    model = KymaAutoregressiveLM(model_config)
    summary = PilotRunSummary(
        cache_path=str(Path(cache_path)),
        model_config_path=str(Path(model_config_path)),
        training_config_path=str(Path(training_config_path)),
        output_dir=str(Path(output_dir)),
        max_pieces=max_pieces,
        pad_id=pad_id,
        train_piece_count=len(train_pieces),
        val_piece_count=len(val_pieces),
        train_window_count=len(train_dataset),
        val_window_count=len(val_dataset),
        train_loss_tokens=_count_loss_tokens(train_dataset),
        val_loss_tokens=_count_loss_tokens(val_dataset),
        effective_tokens_per_optimizer_step=(
            pretrain_config.batch_size
            * pretrain_config.grad_accum_steps
            * window_spec.chunk_size_tokens
        ),
        model_parameter_count=_count_parameters(model),
        device=pretrain_config.device,
        precision=pretrain_config.precision,
    )
    return PreparedPilotRun(
        summary=summary,
        model_config=model_config,
        pretrain_config=pretrain_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )


def write_3060_pilot_summary(
    summary: PilotRunSummary,
    *,
    output_dir: str | Path,
) -> Path:
    """Persist a pilot summary manifest under the run directory."""

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = resolved_output_dir / "pilot-summary.json"
    output_path.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def train_3060_pilot(prepared: PreparedPilotRun) -> dict[str, Any]:
    """Execute the RTX 3060 pilot training run."""

    device = torch.device(prepared.pretrain_config.device)
    if device.type != "cuda":
        raise ValueError("The RTX 3060 pilot must target CUDA.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the current environment.")

    output_dir = Path(prepared.summary.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = write_3060_pilot_summary(
        prepared.summary,
        output_dir=output_dir,
    )
    LOGGER.info("Wrote pilot summary to %s", summary_path)

    model = KymaAutoregressiveLM(prepared.model_config)
    initial_metrics = evaluate_language_model(
        model,
        prepared.val_dataset,
        batch_size=prepared.pretrain_config.batch_size,
        device=device,
        precision=prepared.pretrain_config.precision,
    )
    train_state = train_language_model(
        model,
        prepared.train_dataset,
        model_config=prepared.model_config,
        pretrain_config=prepared.pretrain_config,
        checkpoint_dir=output_dir / "checkpoints",
    )
    final_metrics = evaluate_language_model(
        model,
        prepared.val_dataset,
        batch_size=prepared.pretrain_config.batch_size,
        device=device,
        precision=prepared.pretrain_config.precision,
    )
    report = {
        "summary": prepared.summary.to_dict(),
        "train_state": {
            "global_step": train_state.global_step,
            "optimizer_steps": train_state.optimizer_steps,
            "tokens_processed": train_state.tokens_processed,
        },
        "initial_val_loss": initial_metrics.loss,
        "final_val_loss": final_metrics.loss,
        "initial_val_tokens": initial_metrics.valid_tokens,
        "final_val_tokens": final_metrics.valid_tokens,
    }
    (output_dir / "train-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "DEFAULT_3060_CACHE_PATH",
    "DEFAULT_3060_MODEL_CONFIG_PATH",
    "DEFAULT_3060_OUTPUT_DIR",
    "DEFAULT_3060_TRAINING_CONFIG_PATH",
    "PilotRunSummary",
    "PreparedPilotRun",
    "build_3060_pilot_cache",
    "prepare_3060_pilot_run",
    "train_3060_pilot",
    "write_3060_pilot_summary",
]
