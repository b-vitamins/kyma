from __future__ import annotations

from argparse import ArgumentParser

import pytest

from kyma.training.dynamo import CompileConfig, addcompileargs


def test_compileconfig_disables_plugin_for_no_backend() -> None:
    config = CompileConfig()
    assert not config.enabled
    assert config.createplugin() is None


def test_compileconfig_builds_inductor_plugin() -> None:
    config = CompileConfig(
        backend="inductor",
        mode="default",
        fullgraph=True,
        dynamic=True,
        regional=True,
    )

    plugin = config.createplugin()

    assert config.enabled
    assert plugin is not None
    assert plugin.backend.value.lower() == "inductor"
    assert plugin.mode == "default"
    assert plugin.fullgraph is True
    assert plugin.dynamic is True
    assert plugin.use_regional_compilation is True


def test_compileconfig_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported compile backend"):
        CompileConfig(backend="bogus")


def test_addcompileargs_accepts_shared_flags() -> None:
    parser = addcompileargs(ArgumentParser())

    args = parser.parse_args(
        [
            "--compile_backend",
            "eager",
            "--compile_mode",
            "reduce-overhead",
            "--compile_fullgraph",
            "--compile_dynamic",
            "--compile_regional",
        ]
    )

    assert args.compile_backend == "eager"
    assert args.compile_mode == "reduce-overhead"
    assert args.compile_fullgraph is True
    assert args.compile_dynamic is True
    assert args.compile_regional is True
