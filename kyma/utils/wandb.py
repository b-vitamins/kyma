"""Optional W&B observability helpers."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from kyma.config.schemas import ProjectPaths
from kyma.utils.env import loadrepowandbenv

LOGGER = logging.getLogger(__name__)


def _parsebool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _tagsfromenv() -> list[str]:
    raw = os.environ.get("WANDB_TAGS", "")
    return [part.strip() for part in raw.split(",") if part.strip()]


def _modefromenv() -> Literal["online", "offline", "disabled", "shared"] | None:
    raw = os.environ.get("WANDB_MODE")
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in {"online", "offline", "disabled", "shared"}:
        return cast(Literal["online", "offline", "disabled", "shared"], normalized)
    return None


def _shouldenable() -> bool:
    loadrepowandbenv()
    explicit = _parsebool(os.environ.get("KYMA_WANDB"))
    mode = _modefromenv()
    if explicit is False:
        return False
    if mode == "disabled":
        return False
    if explicit is True:
        return True
    if mode == "offline":
        return True
    if not os.environ.get("WANDB_PROJECT"):
        return False
    return bool(os.environ.get("WANDB_API_KEY")) or haswandbnetrc()


@dataclass(slots=True)
class WandbRun:
    """Thin optional wrapper around a W&B run object."""

    run: Any | None
    stepinterval: int = 10

    @property
    def enabled(self) -> bool:
        return self.run is not None

    def log(
        self,
        metrics: dict[str, Any],
        *,
        step: int | None = None,
        force: bool = False,
    ) -> None:
        if self.run is None:
            return
        if (
            not force
            and step is not None
            and step > 1
            and self.stepinterval > 1
            and step % self.stepinterval != 0
        ):
            return
        self.run.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def createwandbrun(
    *,
    projectpaths: ProjectPaths,
    jobtype: str,
    runconfig: dict[str, Any],
    name: str,
    group: str | None = None,
    tags: list[str] | None = None,
) -> WandbRun:
    """Create a W&B run if repo or process env requests observability."""

    if not _shouldenable():
        return WandbRun(run=None)

    try:
        import wandb
    except ImportError:
        LOGGER.warning(
            "W&B was requested but the package is not installed; continuing without it."
        )
        return WandbRun(run=None)

    stepinterval = int(os.environ.get("KYMA_WANDB_STEP_INTERVAL", "10"))
    mergedtags = [*_tagsfromenv(), *(tags or [])]
    try:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "kyma"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=os.environ.get("WANDB_NAME") or name,
            group=os.environ.get("WANDB_RUN_GROUP") or group,
            notes=os.environ.get("WANDB_NOTES") or None,
            tags=mergedtags or None,
            job_type=jobtype,
            dir=str(projectpaths.root),
            mode=_modefromenv(),
            config=runconfig,
            save_code=False,
        )
    except Exception:
        LOGGER.warning("W&B init failed; continuing without external observability.")
        return WandbRun(run=None)
    return WandbRun(run=run, stepinterval=max(1, stepinterval))


def defaultwandbname(projectpaths: ProjectPaths, *, prefix: str) -> str:
    return f"{prefix}-{projectpaths.root.name}"


def haswandbnetrc(path: str | Path | None = None) -> bool:
    netrcpath = Path(path) if path is not None else Path.home() / ".netrc"
    if not netrcpath.is_file():
        return False
    try:
        import netrc

        hosts = netrc.netrc(str(netrcpath)).hosts
    except Exception:
        return False
    return "api.wandb.ai" in hosts
