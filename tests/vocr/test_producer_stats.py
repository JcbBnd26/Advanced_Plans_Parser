"""Tests for plancheck.vocr.producer_stats — Level 3 per-producer adaptation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plancheck.vocr.producer_stats import (
    _normalise_producer,
    get_producer_confidence,
    load_producer_stats,
    update_producer_stats,
)


class TestNormaliseProducer:
    def test_strips_version(self):
        assert _normalise_producer("AutoCAD 2023") == "autocad"

    def test_lowercases(self):
        assert _normalise_producer("Bluebeam Revu") == "bluebeam revu"

    def test_empty(self):
        assert _normalise_producer("") == ""

    def test_version_with_dots(self):
        assert _normalise_producer("Adobe PDF Library 15.0") == "adobe pdf library"


class TestLoadProducerStats:
    def test_missing_file(self, tmp_path: Path):
        stats = load_producer_stats(tmp_path / "none.json")
        assert stats["producers"] == {}

    def test_loads_valid(self, tmp_path: Path):
        p = tmp_path / "ps.json"
        data = {
            "version": 1,
            "producers": {
                "autocad": {
                    "total_runs": 5,
                    "methods": {"a": {"flagged": 10, "hits": 8, "misses": 2}},
                }
            },
            "updated_at": "",
        }
        p.write_text(json.dumps(data))
        stats = load_producer_stats(p)
        assert stats["producers"]["autocad"]["total_runs"] == 5


class TestUpdateProducerStats:
    def _make_cand_stats(self, by_method=None):
        return {
            "by_method": by_method or {},
            "predicted_vs_found": {},
        }

    def test_creates_file(self, tmp_path: Path):
        p = tmp_path / "ps.json"
        cs = self._make_cand_stats(
            by_method={"a": {"flagged": 5, "hits": 4, "misses": 1}}
        )
        result = update_producer_stats(p, "AutoCAD 2023", cs)
        assert p.exists()
        assert result["producers"]["autocad"]["total_runs"] == 1

    def test_accumulates(self, tmp_path: Path):
        p = tmp_path / "ps.json"
        cs = self._make_cand_stats(
            by_method={"x": {"flagged": 3, "hits": 2, "misses": 1}}
        )
        update_producer_stats(p, "AutoCAD 2023", cs)
        result = update_producer_stats(p, "AutoCAD 2024", cs)
        # Both versions normalise to "autocad"
        assert result["producers"]["autocad"]["total_runs"] == 2
        assert result["producers"]["autocad"]["methods"]["x"]["flagged"] == 6

    def test_different_producers(self, tmp_path: Path):
        p = tmp_path / "ps.json"
        cs = self._make_cand_stats(
            by_method={"m": {"flagged": 1, "hits": 1, "misses": 0}}
        )
        update_producer_stats(p, "AutoCAD", cs)
        result = update_producer_stats(p, "Bluebeam Revu", cs)
        assert "autocad" in result["producers"]
        assert "bluebeam revu" in result["producers"]

    def test_empty_producer_skipped(self, tmp_path: Path):
        p = tmp_path / "ps.json"
        cs = self._make_cand_stats()
        result = update_producer_stats(p, "", cs)
        assert result["producers"] == {}


class TestGetProducerConfidence:
    def test_none_stats(self):
        assert get_producer_confidence("m", "AutoCAD", None, 0.7) == 0.7

    def test_empty_producer(self):
        stats = {
            "producers": {
                "autocad": {
                    "methods": {"m": {"flagged": 100, "hits": 90, "misses": 10}}
                }
            }
        }
        assert get_producer_confidence("m", "", stats, 0.7) == 0.7

    def test_unknown_producer(self):
        stats = {"producers": {}}
        assert get_producer_confidence("m", "AutoCAD", stats, 0.7) == 0.7

    def test_insufficient_data(self):
        stats = {
            "producers": {
                "autocad": {"methods": {"m": {"flagged": 2, "hits": 2, "misses": 0}}}
            }
        }
        assert get_producer_confidence("m", "AutoCAD", stats, 0.7, min_runs=3) == 0.7

    def test_sufficient_data(self):
        stats = {
            "producers": {
                "autocad": {
                    "methods": {"m": {"flagged": 100, "hits": 90, "misses": 10}}
                }
            }
        }
        result = get_producer_confidence("m", "AutoCAD 2023", stats, 0.5)
        # Laplace: (90+1)/(100+2) ≈ 0.8922
        assert 0.88 < result < 0.91

    def test_floor_clamp(self):
        stats = {
            "producers": {
                "autocad": {"methods": {"m": {"flagged": 50, "hits": 0, "misses": 50}}}
            }
        }
        result = get_producer_confidence("m", "AutoCAD", stats, 0.5, floor=0.1)
        assert result == 0.1


class TestProducerIntegration:
    """Test Level 3 integration with detect_vocr_candidates."""

    def test_producer_overrides_confidence(self):
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

        # Without producer stats
        cands_base = detect_vocr_candidates(
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

        # With producer stats showing high hit rate for this method
        prod_stats = {
            "producers": {
                "autocad": {
                    "methods": {
                        "placeholder_token": {"flagged": 100, "hits": 95, "misses": 5},
                    }
                }
            }
        }
        cands_prod = detect_vocr_candidates(
            tokens=[tok],
            page_chars=[],
            page_lines=[],
            page_curves=[],
            page_rects=[],
            page_width=612,
            page_height=792,
            page_num=0,
            cfg=cfg,
            producer_stats=prod_stats,
            producer_id="AutoCAD 2023",
        )

        assert len(cands_base) >= 1
        assert len(cands_prod) >= 1

        # Confidence should differ
        c_base = next(c for c in cands_base if "placeholder_token" in c.trigger_methods)
        c_prod = next(c for c in cands_prod if "placeholder_token" in c.trigger_methods)
        assert c_prod.confidence != c_base.confidence
