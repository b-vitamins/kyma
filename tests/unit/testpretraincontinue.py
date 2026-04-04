from __future__ import annotations

import json
from pathlib import Path

import torch

from kyma.compat.checkpointio import loadacceleratemodelstate
from kyma.config.schemas import ProjectPaths
from kyma.training.pretrain import (
    ContinuationState,
    loadcontinuationstate,
    savecontinuationstate,
)


def test_loadacceleratemodelstate_reads_accelerate_model_file(tmp_path: Path) -> None:
    checkpointdir = tmp_path / "step10"
    checkpointdir.mkdir()
    model = torch.nn.Linear(4, 3)
    torch.save(model.state_dict(), checkpointdir / "pytorch_model.bin")

    loaded = loadacceleratemodelstate(checkpointdir)

    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded[key], value)


def test_loadcontinuationstate_reads_resume_metadata(tmp_path: Path) -> None:
    checkpointdir = tmp_path / "step42"
    checkpointdir.mkdir()
    payload = {
        "step": 42,
        "tokens_seen": 123456,
        "pass_index": 1,
        "batches_processed_in_pass": 17,
    }
    (checkpointdir / "pretrain_state.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    state = loadcontinuationstate(checkpointdir)

    assert state == ContinuationState(
        checkpoint_dir=checkpointdir.resolve(),
        source_step=42,
        source_tokens_seen=123456,
    )


def test_savecontinuationstate_writes_ancestry_metadata(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    projectpaths = ProjectPaths(
        root=root,
        checkpoints=root / "checkpoints",
        logs=root / "logs.txt",
        metrics=root / "metrics",
    )
    state = ContinuationState(
        checkpoint_dir=(tmp_path / "source" / "step100").resolve(),
        source_step=100,
        source_tokens_seen=999999,
    )

    savecontinuationstate(projectpaths, state)

    payload = json.loads((root / "continuation.json").read_text(encoding="utf-8"))
    assert payload == {
        "checkpoint_dir": str(state.checkpoint_dir),
        "source_step": 100,
        "source_tokens_seen": 999999,
    }
