from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_cli_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "kyma.cli", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "kyma <command>" in result.stdout


def test_pretrain_continue_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "kyma.training.pretrain", "continue", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--cp_dir" in result.stdout
    assert "--max_steps" in result.stdout
    assert "--mixed_precision" in result.stdout
