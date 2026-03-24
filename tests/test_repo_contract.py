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
        ROOT / "scripts" / "pre-commit.sh",
    ]

    for path in expected:
        assert path.is_file(), f"Missing required repo contract file: {path}"


def test_todo_contains_milestones() -> None:
    todo = (ROOT / "TODO.md").read_text(encoding="utf-8")
    assert "M01" in todo
    assert "M12" in todo
