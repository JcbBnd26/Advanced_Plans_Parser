"""Tests for plancheck.vocr.backends — OCR backend caching and factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from plancheck.config import GroupingConfig
from plancheck.vocr.backends.base import (_backend_cache, _backend_key,
                                          clear_backend_cache, get_ocr_backend)


class TestBackendKey:
    """_backend_key derives a hashable cache key from config."""

    def test_default_config(self):
        cfg = GroupingConfig()
        key = _backend_key(cfg)
        assert key == ("surya", "cpu")

    def test_gpu_device(self):
        cfg = GroupingConfig(vocr_device="gpu")
        key = _backend_key(cfg)
        assert key == ("surya", "gpu")

    def test_none_config(self):
        key = _backend_key(None)
        assert key == ("surya", "cpu")

    def test_different_devices_different_keys(self):
        k1 = _backend_key(GroupingConfig(vocr_device="cpu"))
        k2 = _backend_key(GroupingConfig(vocr_device="gpu"))
        assert k1 != k2

    def test_same_config_same_key(self):
        cfg1 = GroupingConfig()
        cfg2 = GroupingConfig()
        assert _backend_key(cfg1) == _backend_key(cfg2)

    def test_irrelevant_fields_dont_affect_key(self):
        """Fields not related to VOCR engine (e.g. grouping) shouldn't change key."""
        cfg1 = GroupingConfig(block_gap_mult=0.9)
        cfg2 = GroupingConfig(block_gap_mult=2.0)
        assert _backend_key(cfg1) == _backend_key(cfg2)


class TestBackendCache:
    """Test backend caching and eviction."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_backend_cache()

    def teardown_method(self):
        """Clear cache after each test."""
        clear_backend_cache()

    def test_clear_backend_cache_returns_count(self):
        # Empty cache returns 0
        assert clear_backend_cache() == 0

    @patch("plancheck.vocr.backends.base.SuryaOCRBackend", autospec=True)
    def test_get_ocr_backend_caches_instance(self, mock_surya):
        """Same config should return cached instance."""
        mock_instance = MagicMock()
        mock_surya.return_value = mock_instance

        # Import patched version
        with patch("plancheck.vocr.backends.surya.SuryaOCRBackend", mock_surya):
            cfg = GroupingConfig()
            backend1 = get_ocr_backend(cfg)
            backend2 = get_ocr_backend(cfg)

            # Should be cached - only one instantiation
            assert mock_surya.call_count == 1
            assert backend1 is backend2


class TestBackendFactory:
    """Test backend selection logic."""

    def setup_method(self):
        clear_backend_cache()

    def teardown_method(self):
        clear_backend_cache()

    def test_unknown_backend_raises(self):
        """Invalid backend name should raise ValueError."""
        cfg = GroupingConfig()
        # Manually override to test error handling
        cfg.__dict__["vocr_backend"] = "unknown_backend"

        with pytest.raises(ValueError, match="Unknown OCR backend"):
            get_ocr_backend(cfg)
