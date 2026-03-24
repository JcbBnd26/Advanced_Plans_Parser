"""Tests for Phase 4 config fields and the FEATURE_VERSION constant."""

from __future__ import annotations

from pathlib import Path

import pytest

from plancheck.config import GroupingConfig
from plancheck.corrections.classifier import FEATURE_VERSION


class TestPhase4ConfigFields:
    def test_defaults(self):
        cfg = GroupingConfig()
        assert cfg.ml_drift_enabled is False
        assert Path(cfg.ml_drift_stats_path) == Path("data/drift_stats.json")
        assert cfg.ml_drift_threshold == 0.3
        assert cfg.ml_retrain_threshold == 50
        assert cfg.ml_retrain_on_startup is False
        assert cfg.ml_feature_cache_enabled is True

    def test_drift_threshold_valid(self):
        cfg = GroupingConfig(ml_drift_threshold=0.5)
        assert cfg.ml_drift_threshold == 0.5

    def test_drift_threshold_out_of_range(self):
        with pytest.raises(Exception):
            GroupingConfig(ml_drift_threshold=1.5)

    def test_drift_threshold_negative(self):
        with pytest.raises(Exception):
            GroupingConfig(ml_drift_threshold=-0.1)

    def test_retrain_threshold_valid(self):
        cfg = GroupingConfig(ml_retrain_threshold=100)
        assert cfg.ml_retrain_threshold == 100

    def test_retrain_threshold_zero_raises(self):
        with pytest.raises(Exception):
            GroupingConfig(ml_retrain_threshold=0)


class TestFeatureVersion:
    def test_feature_version_is_positive_int(self):
        assert isinstance(FEATURE_VERSION, int)
        assert FEATURE_VERSION >= 1
