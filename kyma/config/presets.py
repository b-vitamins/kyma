"""Named preset identifiers used across the repo."""

from __future__ import annotations

from kyma.compat.ariacontracts import CATEGORY_TAGS

MODEL_PRESETS = {
    "lm": "medium",
    "embedding": "medium-emb",
}

CATEGORY_MODEL_PRESETS = {name: f"medium-{name}" for name in CATEGORY_TAGS}
