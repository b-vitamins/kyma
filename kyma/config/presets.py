"""Named preset identifiers used across the repo."""

from __future__ import annotations

from kyma.compat.ariacontracts import CATEGORY_TAGS

MODEL_PRESETS = {
    "small": "kyma-s",
    "medium": "kyma-m",
    "large": "kyma-l",
    "lm": "kyma-l",
    "embedding": "medium-emb",
}

CATEGORY_MODEL_PRESETS = {name: f"medium-{name}" for name in CATEGORY_TAGS}
