"""Logging helpers."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configurelogger(
    name: str,
    *,
    projectdir: str | Path | None = None,
    consolelevel: int = logging.INFO,
    filelevel: int = logging.DEBUG,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s: [%(levelname)s] %(message)s",
    )

    console = logging.StreamHandler()
    console.setLevel(consolelevel)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if projectdir is not None:
        projectpath = Path(projectdir)
        projectpath.mkdir(parents=True, exist_ok=True)
        logfile = RotatingFileHandler(
            projectpath / "logs.txt",
            backupCount=5,
            maxBytes=1024**3,
        )
        logfile.setLevel(filelevel)
        logfile.setFormatter(formatter)
        logger.addHandler(logfile)

    return logger
