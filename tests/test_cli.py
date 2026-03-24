from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "kyma.cli", *args],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def test_list_configs_command_prints_packaged_name() -> None:
    result = _run_cli("list-configs", "model")
    assert "kyma-small" in result.stdout


def test_print_config_command_emits_json() -> None:
    result = _run_cli("print-config", "eval", "default")
    payload = json.loads(result.stdout)
    assert "streaming" in payload
    assert payload["rhythm"]["report_tempo_consistency"] is True


def test_training_config_commands_work() -> None:
    list_result = _run_cli("list-configs", "training")
    assert "kyma-small-pretrain" in list_result.stdout

    print_result = _run_cli("print-config", "training", "kyma-small-pretrain")
    payload = json.loads(print_result.stdout)
    assert payload["batch_size"] == 8
    assert payload["schedule"]["warmup_steps"] == 2000


def test_download_aria_midi_dry_run_emits_plan() -> None:
    result = _run_cli("download-aria-midi", "--dry-run")
    payload = json.loads(result.stdout)
    assert payload["subset"] == "pruned"
    assert payload["archive_path"].endswith("aria-midi-v1-pruned-ext.tar.gz")
