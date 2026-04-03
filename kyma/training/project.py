"""Project-directory helpers for training jobs."""

from __future__ import annotations

from pathlib import Path

from kyma.config.schemas import ProjectPaths
from kyma.utils.logging import configurelogger


def createprojectpaths(projectdir: str | None) -> ProjectPaths:
    if projectdir is None:
        experiments = Path("experiments")
        experiments.mkdir(exist_ok=True)
        index = 0
        while (experiments / str(index)).exists():
            index += 1
        root = experiments / str(index)
        root.mkdir()
    else:
        root = Path(projectdir)
        if root.is_file():
            raise FileExistsError(f"Provided project path is a file: {root}")
        if root.exists():
            if any(root.iterdir()):
                raise ValueError(f"Provided project directory is not empty: {root}")
        else:
            root.mkdir(parents=True)

    checkpoints = root / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    return ProjectPaths(
        root=root.resolve(),
        checkpoints=checkpoints.resolve(),
        logs=(root / "logs.txt").resolve(),
        metrics=(root / "metrics").resolve(),
    )


def createprojectlogger(projectpaths: ProjectPaths, *, name: str):
    return configurelogger(name, projectdir=projectpaths.root)
