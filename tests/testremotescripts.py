from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    ROOT / "scripts" / "guix-run",
    ROOT / "scripts" / "remote-list",
    ROOT / "scripts" / "remote-print-config",
    ROOT / "scripts" / "remote-smoke",
    ROOT / "scripts" / "remote-shell",
    ROOT / "scripts" / "remote-rsync",
]


def _write_remote_env(tmp_path: Path) -> Path:
    env_file = tmp_path / ".env"
    env_file.write_text(
        textwrap.dedent(
            """
            KD_REMOTE_MACHINES=gpu-box,archive-box
            KD_REMOTE_DEFAULT_MACHINE=gpu-box

            KD_REMOTE_GPU_BOX_HOST=gpu.example.edu
            KD_REMOTE_GPU_BOX_USER=tester
            KD_REMOTE_GPU_BOX_PORT=2222
            KD_REMOTE_GPU_BOX_WORKDIR=/srv/project
            KD_REMOTE_GPU_BOX_AUTH=key
            KD_REMOTE_GPU_BOX_SSH_KEY=/tmp/test-key
            KD_REMOTE_GPU_BOX_PASSWORD=fallback-secret

            KD_REMOTE_ARCHIVE_BOX_HOST=archive.example.edu
            KD_REMOTE_ARCHIVE_BOX_USER=runner
            KD_REMOTE_ARCHIVE_BOX_WORKDIR=/srv/archive/project
            KD_REMOTE_ARCHIVE_BOX_AUTH=password
            KD_REMOTE_ARCHIVE_BOX_PASSWORD=super-secret
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return env_file


def _run_script(
    script_name: str, *args: str, env_file: Path
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["KD_REMOTE_ENV_FILE"] = str(env_file)
    return subprocess.run(
        [str(ROOT / "scripts" / script_name), *args],
        check=True,
        capture_output=True,
        cwd=ROOT,
        env=env,
        text=True,
    )


def test_remote_scripts_have_valid_shell_syntax() -> None:
    for script in SCRIPTS:
        subprocess.run(["sh", "-n", str(script)], check=True, cwd=ROOT)


def test_remote_list_reads_configured_machine_names(tmp_path: Path) -> None:
    env_file = _write_remote_env(tmp_path)
    result = _run_script("remote-list", env_file=env_file)
    assert result.stdout.strip().splitlines() == ["gpu-box", "archive-box"]


def test_remote_print_config_resolves_selected_machine(tmp_path: Path) -> None:
    env_file = _write_remote_env(tmp_path)
    result = _run_script(
        "remote-print-config", "--machine", "archive-box", env_file=env_file
    )
    assert "machine=archive-box" in result.stdout
    assert "host=archive.example.edu" in result.stdout
    assert "user=runner" in result.stdout
    assert "workdir=/srv/archive/project" in result.stdout
    assert "auth=password" in result.stdout
    assert "has_password=true" in result.stdout
    assert "super-secret" not in result.stdout


def test_remote_shell_dry_run_uses_resolved_machine_settings(tmp_path: Path) -> None:
    env_file = _write_remote_env(tmp_path)
    result = _run_script(
        "remote-shell",
        "--machine",
        "gpu-box",
        "--dry-run",
        "--",
        "python",
        "-V",
        env_file=env_file,
    )
    assert "SSHPASS=<redacted>" in result.stdout
    assert "sshpass" in result.stdout
    assert "ssh" in result.stdout
    assert "/tmp/test-key" in result.stdout
    assert "tester@gpu.example.edu" in result.stdout
    assert "ConnectTimeout=30" in result.stdout
    assert "ServerAliveInterval=15" in result.stdout
    assert "ServerAliveCountMax=3" in result.stdout
    assert "PreferredAuthentications=publickey,password" in result.stdout
    assert "cd /srv/project && python -V" in result.stdout
    assert "fallback-secret" not in result.stdout


def test_remote_rsync_dry_run_redacts_password_auth(tmp_path: Path) -> None:
    env_file = _write_remote_env(tmp_path)
    result = _run_script(
        "remote-rsync",
        "--machine",
        "archive-box",
        "--dry-run",
        env_file=env_file,
    )
    assert "SSHPASS=<redacted>" in result.stdout
    assert "sshpass" in result.stdout
    assert "rsync" in result.stdout
    assert "ConnectTimeout=30" in result.stdout
    assert "ServerAliveInterval=15" in result.stdout
    assert "ServerAliveCountMax=3" in result.stdout
    assert "runner@archive.example.edu:/srv/archive/project/" in result.stdout
    assert "super-secret" not in result.stdout


def test_remote_rsync_dry_run_excludes_local_artifacts_by_default(
    tmp_path: Path,
) -> None:
    env_file = _write_remote_env(tmp_path)
    result = _run_script(
        "remote-rsync",
        "--machine",
        "gpu-box",
        "--dry-run",
        env_file=env_file,
    )
    assert "--exclude artifacts/" in result.stdout
    assert "--exclude logs/" in result.stdout


def test_remote_smoke_dry_run_uses_machine_workdir(tmp_path: Path) -> None:
    env_file = _write_remote_env(tmp_path)
    result = _run_script(
        "remote-smoke", "--machine", "gpu-box", "--dry-run", env_file=env_file
    )
    assert "tester@gpu.example.edu" in result.stdout
    assert "cd /srv/project && { set -eu;" in result.stdout
    assert "nvidia-smi" in result.stdout
