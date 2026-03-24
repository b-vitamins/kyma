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
