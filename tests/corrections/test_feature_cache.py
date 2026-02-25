"""Tests for feature cache in CorrectionStore (Phase 4.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from plancheck.corrections.store import CorrectionStore


class TestFeatureCache:
    def test_cache_and_retrieve(self, tmp_store: CorrectionStore):
        vector = [1.0, 2.0, 3.0, 4.0, 5.0]
        tmp_store.cache_features("det_001", vector, feature_version=5)
        result = tmp_store.get_cached_features("det_001", feature_version=5)
        assert result == vector

    def test_cache_miss_wrong_version(self, tmp_store: CorrectionStore):
        tmp_store.cache_features("det_002", [1.0, 2.0], feature_version=4)
        result = tmp_store.get_cached_features("det_002", feature_version=5)
        assert result is None

    def test_cache_miss_nonexistent(self, tmp_store: CorrectionStore):
        result = tmp_store.get_cached_features("det_999", feature_version=1)
        assert result is None

    def test_cache_overwrite(self, tmp_store: CorrectionStore):
        tmp_store.cache_features("det_003", [1.0], feature_version=5)
        tmp_store.cache_features("det_003", [2.0], feature_version=5)
        result = tmp_store.get_cached_features("det_003", feature_version=5)
        assert result == [2.0]

    def test_invalidate_all(self, tmp_store: CorrectionStore):
        tmp_store.cache_features("det_a", [1.0], feature_version=5)
        tmp_store.cache_features("det_b", [2.0], feature_version=5)
        n = tmp_store.invalidate_cache()
        assert n == 2
        assert tmp_store.get_cached_features("det_a", feature_version=5) is None

    def test_invalidate_by_version(self, tmp_store: CorrectionStore):
        tmp_store.cache_features("det_a", [1.0], feature_version=4)
        tmp_store.cache_features("det_b", [2.0], feature_version=5)
        n = tmp_store.invalidate_cache(feature_version=5)
        # Only version 4 entries should be deleted
        assert n == 1
        assert tmp_store.get_cached_features("det_b", feature_version=5) == [2.0]
        assert tmp_store.get_cached_features("det_a", feature_version=4) is None

    def test_cache_stats_empty(self, tmp_store: CorrectionStore):
        stats = tmp_store.cache_stats()
        assert stats["total_entries"] == 0
        assert stats["distinct_detections"] == 0
        assert stats["distinct_versions"] == 0

    def test_cache_stats_populated(self, tmp_store: CorrectionStore):
        tmp_store.cache_features("det_a", [1.0], feature_version=4)
        tmp_store.cache_features("det_b", [2.0], feature_version=5)
        tmp_store.cache_features("det_c", [3.0], feature_version=5)
        stats = tmp_store.cache_stats()
        assert stats["total_entries"] == 3
        assert stats["distinct_detections"] == 3
        assert stats["distinct_versions"] == 2
