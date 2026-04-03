from __future__ import annotations

import sys
import types
from pathlib import Path

from kyma.config.schemas import ProjectPaths
from kyma.utils.env import parseenv
from kyma.utils.wandb import _shouldenable, createwandbrun, haswandbnetrc


def test_parseenv_supports_shell_style_assignments(tmp_path: Path) -> None:
    envpath = tmp_path / ".env"
    envpath.write_text(
        "\n".join(
            [
                "# comment",
                "export WANDB_PROJECT=kyma",
                "WANDB_ENTITY='incado1010-iisc'",
            ]
        ),
        encoding="utf-8",
    )
    assert parseenv(envpath) == {
        "WANDB_PROJECT": "kyma",
        "WANDB_ENTITY": "incado1010-iisc",
    }


def test_haswandbnetrc_detects_api_entry(tmp_path: Path) -> None:
    netrcpath = tmp_path / ".netrc"
    netrcpath.write_text(
        "machine api.wandb.ai login user password example-key\n",
        encoding="utf-8",
    )
    assert haswandbnetrc(netrcpath)


def test_createwandbrun_noops_when_disabled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KYMA_WANDB", "0")
    projectroot = tmp_path / "run"
    projectroot.mkdir()
    projectpaths = ProjectPaths(
        root=projectroot,
        checkpoints=projectroot / "checkpoints",
        logs=projectroot / "logs.txt",
        metrics=projectroot / "metrics",
    )
    wandbrun = createwandbrun(
        projectpaths=projectpaths,
        jobtype="pretrain",
        name="test-run",
        group="kyma-s",
        tags=["pretrain"],
        runconfig={"epochs": 1},
    )
    assert not wandbrun.enabled


def test_shouldenable_requires_project_and_auth(monkeypatch) -> None:
    monkeypatch.setattr("kyma.utils.wandb.loadrepowandbenv", lambda: {})
    monkeypatch.delenv("KYMA_WANDB", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_PROJECT", "kyma")
    monkeypatch.setattr("kyma.utils.wandb.haswandbnetrc", lambda _path=None: False)

    assert not _shouldenable()


def test_shouldenable_accepts_project_with_netrc(monkeypatch) -> None:
    monkeypatch.setattr("kyma.utils.wandb.loadrepowandbenv", lambda: {})
    monkeypatch.delenv("KYMA_WANDB", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_PROJECT", "kyma")
    monkeypatch.setattr("kyma.utils.wandb.haswandbnetrc", lambda _path=None: True)

    assert _shouldenable()


def test_createwandbrun_noops_when_init_fails(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KYMA_WANDB", "1")

    def failinit(**_kwargs):
        raise RuntimeError

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        types.SimpleNamespace(init=failinit),
    )
    projectroot = tmp_path / "run"
    projectroot.mkdir()
    projectpaths = ProjectPaths(
        root=projectroot,
        checkpoints=projectroot / "checkpoints",
        logs=projectroot / "logs.txt",
        metrics=projectroot / "metrics",
    )

    wandbrun = createwandbrun(
        projectpaths=projectpaths,
        jobtype="pretrain",
        name="test-run",
        group="kyma-s",
        tags=["pretrain"],
        runconfig={"epochs": 1},
    )

    assert not wandbrun.enabled
