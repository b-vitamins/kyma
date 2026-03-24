from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_repository_contract_files_exist() -> None:
    expected = [
        ROOT / "AGENTS.md",
        ROOT / "TODO.md",
        ROOT / "CHANGELOG.md",
        ROOT / "pyproject.toml",
        ROOT / "pyrightconfig.json",
        ROOT / ".pre-commit-config.yaml",
        ROOT / "scripts" / "bootstrap-venv.sh",
        ROOT / "scripts" / "pre-commit.sh",
    ]

    for path in expected:
        assert path.is_file(), f"Missing required repo contract file: {path}"


def test_todo_contains_milestones() -> None:
    todo = (ROOT / "TODO.md").read_text(encoding="utf-8")
    assert "M01" in todo
    assert "M12" in todo
    assert "M15" in todo


def test_gitignore_covers_local_dataset_and_artifact_caches() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "artifacts/" in gitignore
    assert "/data/" in gitignore or "data/" in gitignore
    assert "*.tar.gz" in gitignore
    assert "*.part" in gitignore


def test_pre_commit_prefers_local_venv_tools() -> None:
    script = (ROOT / "scripts" / "pre-commit.sh").read_text(encoding="utf-8")
    assert ".venv" in script
    assert "bootstrap-venv.sh" in script
