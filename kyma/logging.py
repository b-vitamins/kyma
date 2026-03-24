"""Shared logging helpers."""

from __future__ import annotations

import logging


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a process-local logger with a stable formatter."""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.propagate = False
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s: [%(levelname)s] %(message)s",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
