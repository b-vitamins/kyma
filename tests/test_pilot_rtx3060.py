from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from kyma.data import KymaTimeFeatures, KymaTokenizedPiece, save_piece_cache
from kyma.pilot.rtx3060 import (
    PilotRunSummary,
    prepare_3060_pilot_run,
    write_3060_pilot_summary,
)


def _make_piece(piece_id: str, token_id: int) -> KymaTokenizedPiece:
    return KymaTokenizedPiece(
        piece_id=piece_id,
        tokens=("<S>", "<E>"),
        token_ids=torch.tensor([token_id, token_id + 1], dtype=torch.long),
        time_features=KymaTimeFeatures(
            values=torch.zeros((2, 4), dtype=torch.float32),
            valid=torch.ones((2, 4), dtype=torch.bool),
        ),
        metadata={},
        source_path=f"{piece_id}.mid",
    )


def _write_config(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_prepare_3060_pilot_run_builds_train_and_val_datasets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_path = tmp_path / "pieces.jsonl"
    save_piece_cache(
        [
            _make_piece("a", 1),
            _make_piece("b", 3),
            _make_piece("c", 5),
            _make_piece("d", 7),
        ],
        cache_path,
    )

    model_config_path = tmp_path / "model.json"
    _write_config(
        model_config_path,
        {
            "d_model": 16,
            "n_layers": 2,
            "d_state": 8,
            "expand": 2,
            "d_head": 8,
            "d_conv": 4,
            "chunk_size": 16,
            "vocab_size": 32,
            "dropout_p": 0.0,
            "ffn_mult": 2,
            "max_segment_len": 128,
            "time_conditioning": {
                "learned_positional_embedding": False,
                "delta_time_features": True,
                "absolute_time_features": True,
                "beat_phase_features": True,
                "tempo_features": True,
                "feature_mlp_dim": 16,
            },
            "long_context": {
                "state_carry_training": True,
                "chunk_size_tokens": 2,
                "burn_in_tokens": 0,
                "tbptt_window_tokens": 2,
                "max_piece_tokens": 8,
            },
            "differentiators": {
                "long_form_stateful_generation": True,
                "real_time_interactive_continuation": True,
                "rhythm_aware_modeling": True,
            },
        },
    )

    training_config_path = tmp_path / "training.json"
    _write_config(
        training_config_path,
        {
            "batch_size": 1,
            "max_steps": 2,
            "grad_accum_steps": 2,
            "precision": "fp32",
            "grad_clip_norm": 1.0,
            "log_every_steps": 1,
            "checkpoint_every_steps": 1,
            "device": "cpu",
            "optimizer": {
                "lr": 0.001,
                "weight_decay": 0.0,
                "beta1": 0.9,
                "beta2": 0.95,
                "eps": 1e-08,
            },
            "schedule": {
                "warmup_steps": 0,
                "min_lr_scale": 0.1,
            },
        },
    )

    class FakeTokenizer:
        pad_id = 2

    def fake_get_abs_tokenizer(*, config_path: str | None = None) -> FakeTokenizer:
        del config_path
        return FakeTokenizer()

    monkeypatch.setattr(
        "kyma.pilot.rtx3060.get_abs_tokenizer",
        fake_get_abs_tokenizer,
    )

    prepared = prepare_3060_pilot_run(
        cache_path=cache_path,
        model_config_path=model_config_path,
        training_config_path=training_config_path,
        output_dir=tmp_path / "run",
    )

    assert prepared.summary.pad_id == 2
    assert prepared.summary.train_piece_count > 0
    assert prepared.summary.val_piece_count > 0
    assert prepared.summary.train_window_count > 0
    assert prepared.summary.val_window_count > 0
    assert prepared.summary.effective_tokens_per_optimizer_step == 4


def test_write_3060_pilot_summary_persists_json(tmp_path: Path) -> None:
    output_path = write_3060_pilot_summary(
        summary=PilotRunSummary(
            cache_path="cache.jsonl",
            model_config_path="model.json",
            training_config_path="training.json",
            output_dir=str(tmp_path / "run"),
            max_pieces=16,
            pad_id=2,
            train_piece_count=8,
            val_piece_count=2,
            train_window_count=32,
            val_window_count=8,
            train_loss_tokens=1024,
            val_loss_tokens=256,
            effective_tokens_per_optimizer_step=2048,
            model_parameter_count=1234,
            device="cuda",
            precision="fp16",
        ),
        output_dir=tmp_path / "run",
    )

    assert output_path.is_file()
    assert json.loads(output_path.read_text(encoding="utf-8"))["pad_id"] == 2
