"""Tests for plancheck.vocr.engine — PaddleOCR singleton caching."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from plancheck.config import GroupingConfig
from plancheck.vocr.engine import _engine_key, _ocr_cache


class TestEngineKey:
    """_engine_key derives a hashable cache key from config."""

    def test_default_config(self):
        cfg = GroupingConfig()
        key = _engine_key(cfg)
        assert key == ("mobile", False, False, False)

    def test_server_tier(self):
        cfg = GroupingConfig(vocr_model_tier="server")
        key = _engine_key(cfg)
        assert key[0] == "server"

    def test_orientation_flags(self):
        cfg = GroupingConfig(
            vocr_use_orientation_classify=True,
            vocr_use_doc_unwarping=True,
            vocr_use_textline_orientation=True,
        )
        key = _engine_key(cfg)
        assert key == ("mobile", True, True, True)

    def test_none_config(self):
        key = _engine_key(None)
        assert key == ("mobile", False, False, False)

    def test_different_configs_different_keys(self):
        k1 = _engine_key(GroupingConfig(vocr_model_tier="mobile"))
        k2 = _engine_key(GroupingConfig(vocr_model_tier="server"))
        assert k1 != k2

    def test_same_config_same_key(self):
        cfg1 = GroupingConfig()
        cfg2 = GroupingConfig()
        assert _engine_key(cfg1) == _engine_key(cfg2)

    def test_irrelevant_fields_dont_affect_key(self):
        """Fields not related to VOCR engine (e.g. grouping) shouldn't change key."""
        cfg1 = GroupingConfig(block_gap_mult=0.9)
        cfg2 = GroupingConfig(block_gap_mult=2.0)
        assert _engine_key(cfg1) == _engine_key(cfg2)
