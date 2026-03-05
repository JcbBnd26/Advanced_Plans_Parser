"""Tests for plancheck.vocr.method_stats — Level 1 adaptive confidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plancheck.vocr.method_stats import (
    MethodStats,
    get_adaptive_confidence,
    load_method_stats,
    update_method_stats,
)

# Use class constant for version
_STATS_VERSION = MethodStats.VERSION

# ── load_method_stats ───────────────────────────────────────────────────


class TestLoadMethodStats:
    """Tests for load_method_stats."""

    def test_returns_empty_when_file_missing(self, tmp_path: Path):
        stats = load_method_stats(tmp_path / "nonexistent.json")
        assert stats["version"] == _STATS_VERSION
        assert stats["total_runs"] == 0
        assert stats["methods"] == {}
        assert stats["symbols"] == {}

    def test_loads_valid_json(self, tmp_path: Path):
        p = tmp_path / "stats.json"
        data = {
            "version": _STATS_VERSION,
            "total_runs": 3,
            "methods": {"foo": {"flagged": 10, "hits": 8, "misses": 2}},
            "symbols": {
                "%": {"predicted": 5, "correct": 4, "wrong_symbol": 0, "miss": 1}
            },
            "updated_at": "2025-01-01T00:00:00",
        }
        p.write_text(json.dumps(data))
        result = load_method_stats(p)
        assert result["total_runs"] == 3
        assert result["methods"]["foo"]["hits"] == 8
        assert result["symbols"]["%"]["correct"] == 4

    def test_resets_on_version_mismatch(self, tmp_path: Path):
        p = tmp_path / "stats.json"
        p.write_text(json.dumps({"version": 999, "total_runs": 10}))
        result = load_method_stats(p)
        assert result["total_runs"] == 0

    def test_resets_on_corrupt_json(self, tmp_path: Path):
        p = tmp_path / "stats.json"
        p.write_text("{bad json")
        result = load_method_stats(p)
        assert result["total_runs"] == 0

    def test_fills_missing_keys(self, tmp_path: Path):
        p = tmp_path / "stats.json"
        p.write_text(json.dumps({"version": _STATS_VERSION}))
        result = load_method_stats(p)
        assert result["total_runs"] == 0
        assert result["methods"] == {}
        assert result["symbols"] == {}


# ── update_method_stats ─────────────────────────────────────────────────


class TestUpdateMethodStats:
    """Tests for update_method_stats."""

    def _make_candidate_stats(self, by_method=None, pvf=None):
        return {
            "total_candidates": 5,
            "total_hits": 3,
            "total_misses": 2,
            "total_pending": 0,
            "hit_rate": 0.6,
            "by_method": by_method or {},
            "area_stats": {},
            "predicted_vs_found": pvf or {},
        }

    def test_creates_file_from_scratch(self, tmp_path: Path):
        p = tmp_path / "sub" / "stats.json"
        cs = self._make_candidate_stats(
            by_method={"foo": {"flagged": 3, "hits": 2, "misses": 1}},
        )
        result = update_method_stats(p, cs)
        assert p.exists()
        assert result["total_runs"] == 1
        assert result["methods"]["foo"]["flagged"] == 3

    def test_accumulates_across_runs(self, tmp_path: Path):
        p = tmp_path / "stats.json"
        cs1 = self._make_candidate_stats(
            by_method={"a": {"flagged": 10, "hits": 8, "misses": 2}},
        )
        cs2 = self._make_candidate_stats(
            by_method={"a": {"flagged": 5, "hits": 3, "misses": 2}},
        )
        update_method_stats(p, cs1)
        result = update_method_stats(p, cs2)
        assert result["total_runs"] == 2
        assert result["methods"]["a"]["flagged"] == 15
        assert result["methods"]["a"]["hits"] == 11

    def test_accumulates_symbols(self, tmp_path: Path):
        p = tmp_path / "stats.json"
        cs = self._make_candidate_stats(
            pvf={"%": {"predicted": 4, "correct": 3, "wrong_symbol": 0, "miss": 1}},
        )
        update_method_stats(p, cs)
        result = update_method_stats(p, cs)
        assert result["symbols"]["%"]["predicted"] == 8
        assert result["symbols"]["%"]["correct"] == 6

    def test_accepts_preloaded_stats(self, tmp_path: Path):
        p = tmp_path / "stats.json"
        pre = load_method_stats(p)
        cs = self._make_candidate_stats(
            by_method={"x": {"flagged": 1, "hits": 1, "misses": 0}},
        )
        result = update_method_stats(p, cs, stats=pre)
        assert result["total_runs"] == 1

    def test_persists_to_disk(self, tmp_path: Path):
        p = tmp_path / "stats.json"
        cs = self._make_candidate_stats(
            by_method={"m": {"flagged": 2, "hits": 1, "misses": 1}},
        )
        update_method_stats(p, cs)
        reloaded = json.loads(p.read_text())
        assert reloaded["version"] == _STATS_VERSION
        assert reloaded["total_runs"] == 1
        assert reloaded["methods"]["m"]["hits"] == 1

    def test_updated_at_set(self, tmp_path: Path):
        p = tmp_path / "stats.json"
        cs = self._make_candidate_stats()
        result = update_method_stats(p, cs)
        assert result["updated_at"]  # non-empty


# ── get_adaptive_confidence ─────────────────────────────────────────────


class TestGetAdaptiveConfidence:
    """Tests for get_adaptive_confidence."""

    def test_returns_base_when_stats_none(self):
        assert get_adaptive_confidence("foo", None, 0.7) == 0.7

    def test_returns_base_when_method_not_in_stats(self):
        stats = {"methods": {}}
        assert get_adaptive_confidence("foo", stats, 0.7) == 0.7

    def test_returns_base_when_below_min_runs(self):
        stats = {"methods": {"foo": {"flagged": 3, "hits": 3, "misses": 0}}}
        assert get_adaptive_confidence("foo", stats, 0.7, min_runs=5) == 0.7

    def test_adaptive_high_hit_rate(self):
        stats = {"methods": {"m": {"flagged": 100, "hits": 90, "misses": 10}}}
        # Laplace: (90+1)/(100+2) ≈ 0.8922
        result = get_adaptive_confidence("m", stats, 0.5)
        assert 0.88 < result < 0.91

    def test_adaptive_low_hit_rate(self):
        stats = {"methods": {"m": {"flagged": 100, "hits": 10, "misses": 90}}}
        # Laplace: (10+1)/(100+2) ≈ 0.1078
        result = get_adaptive_confidence("m", stats, 0.5)
        assert 0.10 < result < 0.12

    def test_floor_clamp(self):
        stats = {"methods": {"m": {"flagged": 50, "hits": 0, "misses": 50}}}
        # Laplace: 1/52 ≈ 0.019 → clamped to floor=0.1
        result = get_adaptive_confidence("m", stats, 0.5, floor=0.1)
        assert result == 0.1

    def test_ceiling_clamp(self):
        stats = {"methods": {"m": {"flagged": 50, "hits": 50, "misses": 0}}}
        # Laplace: 51/52 ≈ 0.9808 → clamped to ceiling=0.95
        result = get_adaptive_confidence("m", stats, 0.5, ceiling=0.95)
        assert result == 0.95

    def test_exact_at_min_runs_threshold(self):
        stats = {"methods": {"m": {"flagged": 5, "hits": 3, "misses": 2}}}
        # Exactly at min_runs=5 → should use adaptive
        result = get_adaptive_confidence("m", stats, 0.5, min_runs=5)
        assert result != 0.5  # Not base confidence

    def test_custom_floor_ceiling(self):
        stats = {"methods": {"m": {"flagged": 100, "hits": 95, "misses": 5}}}
        result = get_adaptive_confidence("m", stats, 0.5, floor=0.2, ceiling=0.8)
        assert result == 0.8  # Clamped to custom ceiling


# ── Integration: detect_vocr_candidates with method_stats ───────────────


class TestAdaptiveConfidenceIntegration:
    """Verify that detect_vocr_candidates applies adaptive confidence."""

    def test_confidence_overridden_when_stats_provided(self):
        """Method with high historical hit rate should boost confidence."""
        from plancheck.config import GroupingConfig
        from plancheck.models import GlyphBox
        from plancheck.vocr.candidates import detect_vocr_candidates

        # Create a token that triggers placeholder_token detection
        tok = GlyphBox(
            x0=10,
            y0=10,
            x1=30,
            y1=20,
            text="\ufffd",
            page=0,
            origin="text",
        )
        cfg = GroupingConfig()

        # Without stats — uses hard-coded confidence
        cands_no_stats = detect_vocr_candidates(
            tokens=[tok],
            page_chars=[],
            page_lines=[],
            page_curves=[],
            page_rects=[],
            page_width=612,
            page_height=792,
            page_num=0,
            cfg=cfg,
            method_stats=None,
        )

        # With stats — placeholder_token has high hit rate
        stats = {
            "methods": {
                "placeholder_token": {"flagged": 100, "hits": 95, "misses": 5},
            }
        }
        cands_with_stats = detect_vocr_candidates(
            tokens=[tok],
            page_chars=[],
            page_lines=[],
            page_curves=[],
            page_rects=[],
            page_width=612,
            page_height=792,
            page_num=0,
            cfg=cfg,
            method_stats=stats,
        )

        # Both should produce candidates
        assert len(cands_no_stats) >= 1
        assert len(cands_with_stats) >= 1

        # Confidences should differ when stats override
        c_no = next(
            c for c in cands_no_stats if "placeholder_token" in c.trigger_methods
        )
        c_with = next(
            c for c in cands_with_stats if "placeholder_token" in c.trigger_methods
        )
        # The adaptive confidence should be different from the hard-coded one
        # (Laplace: (95+1)/(100+2) ≈ 0.9412, clamped to 0.9412)
        assert c_with.confidence != c_no.confidence

    def test_no_stats_preserves_original_confidence(self):
        """Without stats, candidates keep their original confidence."""
        from plancheck.config import GroupingConfig
        from plancheck.models import GlyphBox
        from plancheck.vocr.candidates import detect_vocr_candidates

        tok = GlyphBox(
            x0=10,
            y0=10,
            x1=30,
            y1=20,
            text="\ufffd",
            page=0,
            origin="text",
        )
        cfg = GroupingConfig()

        cands1 = detect_vocr_candidates(
            tokens=[tok],
            page_chars=[],
            page_lines=[],
            page_curves=[],
            page_rects=[],
            page_width=612,
            page_height=792,
            page_num=0,
            cfg=cfg,
            method_stats=None,
        )
        cands2 = detect_vocr_candidates(
            tokens=[tok],
            page_chars=[],
            page_lines=[],
            page_curves=[],
            page_rects=[],
            page_width=612,
            page_height=792,
            page_num=0,
            cfg=cfg,
        )

        # Both should have same confidence (no stats → no change)
        if cands1 and cands2:
            assert cands1[0].confidence == cands2[0].confidence
