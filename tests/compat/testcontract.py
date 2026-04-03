from __future__ import annotations

from kyma.compat.ariacontracts import CATEGORY_TAGS
from kyma.config.loaders import loadconfig, loadmodelconfig, loadmodelschema


def test_aria_category_contract_is_preserved() -> None:
    assert CATEGORY_TAGS["genre"] == {"classical": 0, "jazz": 1}
    assert CATEGORY_TAGS["emotion"]["tense"] == 3


def test_model_presets_stay_loadable() -> None:
    small = loadmodelschema("kyma-s")
    medium = loadmodelschema("kyma-m")
    config = loadmodelconfig("kyma-l")
    schema = loadmodelschema("kyma-l")
    assert small.d_model == 512
    assert medium.d_model == 832
    assert config["d_model"] == 1536
    assert schema.d_head == 64
    assert schema.chunk_size == 128


def test_data_config_is_available() -> None:
    config = loadconfig()
    assert "data" in config
    assert "pre_processing" in config["data"]
