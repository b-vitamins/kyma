"""Aria-facing constants that Kyma preserves across workflows."""

from __future__ import annotations

CATEGORY_TAGS: dict[str, dict[str, int]] = {
    "genre": {
        "classical": 0,
        "jazz": 1,
    },
    "music_period": {
        "baroque": 0,
        "classical": 1,
        "romantic": 2,
        "impressionist": 3,
    },
    "composer": {
        "beethoven": 0,
        "debussy": 1,
        "brahms": 2,
        "rachmaninoff": 3,
        "schumann": 4,
        "mozart": 5,
        "liszt": 6,
        "bach": 7,
        "chopin": 8,
        "schubert": 9,
    },
    "form": {
        "nocturne": 0,
        "sonata": 1,
        "improvisation": 2,
        "etude": 3,
        "fugue": 4,
        "waltz": 5,
    },
    "pianist": {
        "hisaishi": 0,
        "hancock": 1,
        "bethel": 2,
        "einaudi": 3,
        "clayderman": 4,
        "ryuichi": 5,
        "yiruma": 6,
        "hillsong": 7,
    },
    "emotion": {
        "happy": 0,
        "sad": 1,
        "calm": 2,
        "tense": 3,
    },
}

DEFAULT_MODEL_PRESET = "medium"
DEFAULT_EMBEDDING_MODEL_PRESET = "medium-emb"
SUPPORTED_BACKENDS = ("torch_cuda",)
