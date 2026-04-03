from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_required_repository_files_exist() -> None:
    expected = [
        ROOT / "AGENTS.md",
        ROOT / "CHANGELOG.md",
        ROOT / "README.md",
        ROOT / "TODO.md",
        ROOT / "demo" / "demotorch.py",
        ROOT / "kyma" / "cli.py",
        ROOT / "pyproject.toml",
        ROOT / "pyrightconfig.json",
        ROOT / ".pre-commit-config.yaml",
        ROOT / ".env.example",
        ROOT / "scripts" / "bootstrap-venv.sh",
        ROOT / "scripts" / "pre-commit.sh",
        ROOT / "scripts" / "remotectl.py",
        ROOT / "scripts" / "README.md",
    ]

    for path in expected:
        assert path.is_file(), f"Missing required repository file: {path}"


def test_gitignore_covers_local_env_and_operator_files() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore
    assert ".remote-known-hosts" in gitignore
    assert "remote-downloads/" in gitignore
    assert ".venv/" in gitignore


def test_pre_commit_prefers_local_venv_tools() -> None:
    script = (ROOT / "scripts" / "pre-commit.sh").read_text(encoding="utf-8")
    assert ".venv" in script
    assert "bootstrap-venv.sh" in script


def test_bootstrap_installs_local_quality_tools() -> None:
    script = (ROOT / "scripts" / "bootstrap-venv.sh").read_text(encoding="utf-8")
    assert ".venv" in script
    assert "pip install --upgrade -e '.[all]'" in script


def test_project_tracks_milestones_and_change_history() -> None:
    assert (ROOT / "CHANGELOG.md").is_file()
    assert (ROOT / "TODO.md").is_file()
