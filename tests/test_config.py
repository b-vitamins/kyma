from __future__ import annotations

from kyma.config import (
    list_eval_configs,
    list_model_configs,
    load_eval_config,
    load_model_config,
)
from kyma.eval import KymaEvalProtocol
from kyma.model import KymaModelConfig


def test_packaged_model_config_loads_and_roundtrips() -> None:
    raw = load_model_config("kyma-small")
    config = KymaModelConfig.from_dict(raw)

    assert config.d_model == 512
    assert config.ffn_mult == 4
    assert config.long_context.state_carry_training is True
    assert config.time_conditioning.absolute_time_features is True
    assert config.time_conditioning.tempo_features is True
    assert config.to_dict() == raw


def test_packaged_eval_protocol_loads_and_roundtrips() -> None:
    raw = load_eval_config("default")
    protocol = KymaEvalProtocol.from_dict(raw)

    assert 180 in protocol.long_horizon.prompt_lengths_s
    assert protocol.streaming.report_memory_growth is True
    assert protocol.rhythm.report_onset_nll is True
    assert protocol.to_dict() == raw


def test_config_lists_include_packaged_defaults() -> None:
    assert "kyma-small" in list_model_configs()
    assert "default" in list_eval_configs()
