"""Named preset identifiers used across the repo."""

from __future__ import annotations

from kyma.compat.ariacontracts import CATEGORY_TAGS

MODEL_PRESETS = {
    "base": "kyma-base",
    "lm": "kyma-base",
    "embedding": "kyma-base-emb",
}

CATEGORY_MODEL_PRESETS = {name: f"kyma-base-{name}" for name in CATEGORY_TAGS}
