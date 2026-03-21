from __future__ import annotations

import importlib

import pytest


@pytest.mark.unit
def test_plancheck_import_exposes_grouping_config_lazily() -> None:
    module = importlib.import_module("plancheck")

    grouping_config = module.GroupingConfig
    config = grouping_config()

    assert grouping_config.__name__ == "PipelineConfig"
    assert config.vocr_backend == "surya"


@pytest.mark.unit
def test_from_import_grouping_config() -> None:
    from plancheck import GroupingConfig

    config = GroupingConfig()

    assert config.enable_tocr is True


@pytest.mark.unit
def test_import_surya_backend_module() -> None:
    module = importlib.import_module("plancheck.vocr.backends.surya")

    assert module.SuryaOCRBackend.__name__ == "SuryaOCRBackend"
