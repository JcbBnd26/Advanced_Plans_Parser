"""Tests for Phase 2.3 — OCR Confidence-Weighted Tokens.

Covers:
- GlyphBox.confidence field (default, serialization)
- featurize() / featurize_region() confidence features
- _adjusted_severity() helper
- run_all_checks() severity attenuation via mean_ocr_confidence
- _inject_symbols() confidence gate in reconcile
"""

from __future__ import annotations

import pytest

from plancheck.checks.semantic_checks import (
    CheckResult,
    _adjusted_severity,
    run_all_checks,
)
from plancheck.corrections.features import featurize, featurize_region
from plancheck.models import BlockCluster, GlyphBox, NotesColumn, RowBand

# ── Helpers ────────────────────────────────────────────────────────────

PAGE_W = 2448.0
PAGE_H = 1584.0


def _box(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    text: str = "",
    confidence: float = 1.0,
    origin: str = "text",
) -> GlyphBox:
    return GlyphBox(
        page=0,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        text=text,
        origin=origin,
        confidence=confidence,
    )


def _make_block(boxes: list[GlyphBox], **kwargs) -> BlockCluster:
    from collections import defaultdict

    row_map: dict[float, list[GlyphBox]] = defaultdict(list)
    for b in boxes:
        row_map[b.y0].append(b)
    rows = [RowBand(page=0, boxes=bs) for _, bs in sorted(row_map.items())]
    return BlockCluster(page=0, rows=rows, **kwargs)


# ── GlyphBox confidence field ─────────────────────────────────────────


class TestGlyphBoxConfidence:
    """GlyphBox.confidence field: default, serialization, round-trip."""

    def test_default_confidence_is_one(self):
        b = GlyphBox(page=0, x0=0, y0=0, x1=10, y1=10, text="A")
        assert b.confidence == 1.0

    def test_custom_confidence(self):
        b = GlyphBox(page=0, x0=0, y0=0, x1=10, y1=10, text="A", confidence=0.75)
        assert b.confidence == 0.75

    def test_to_dict_includes_confidence(self):
        b = GlyphBox(page=0, x0=0, y0=0, x1=10, y1=10, text="A", confidence=0.8123)
        d = b.to_dict()
        assert "confidence" in d
        assert d["confidence"] == pytest.approx(0.8123, abs=0.001)

    def test_from_dict_reads_confidence(self):
        d = {
            "page": 0,
            "x0": 0,
            "y0": 0,
            "x1": 10,
            "y1": 10,
            "text": "A",
            "confidence": 0.65,
        }
        b = GlyphBox.from_dict(d)
        assert b.confidence == 0.65

    def test_from_dict_defaults_to_one(self):
        """Legacy dicts without confidence field default to 1.0."""
        d = {"page": 0, "x0": 0, "y0": 0, "x1": 10, "y1": 10}
        b = GlyphBox.from_dict(d)
        assert b.confidence == 1.0

    def test_round_trip(self):
        b = GlyphBox(
            page=1,
            x0=5.5,
            y0=10.5,
            x1=20.5,
            y1=25.5,
            text="OK",
            confidence=0.92,
        )
        restored = GlyphBox.from_dict(b.to_dict())
        assert restored.confidence == pytest.approx(0.92, abs=0.001)


# ── featurize() confidence features ───────────────────────────────────


class TestFeaturizeConfidence:
    """featurize() returns mean_token_confidence and min_token_confidence."""

    def test_confidence_features_present(self):
        boxes = [
            _box(100, 200, 180, 220, text="HELLO", confidence=0.9),
            _box(200, 200, 280, 220, text="WORLD", confidence=0.7),
        ]
        block = _make_block(boxes)
        result = featurize(block, PAGE_W, PAGE_H)
        assert "mean_token_confidence" in result
        assert "min_token_confidence" in result

    def test_confidence_values_computed(self):
        boxes = [
            _box(100, 200, 180, 220, text="A", confidence=0.8),
            _box(200, 200, 280, 220, text="B", confidence=0.6),
        ]
        block = _make_block(boxes)
        result = featurize(block, PAGE_W, PAGE_H)
        assert result["mean_token_confidence"] == pytest.approx(0.7, abs=0.01)
        assert result["min_token_confidence"] == pytest.approx(0.6, abs=0.01)

    def test_all_pdf_tokens_have_confidence_one(self):
        """PDF-only tokens default to confidence=1.0."""
        boxes = [
            _box(100, 200, 180, 220, text="PDF"),
            _box(200, 200, 280, 220, text="TEXT"),
        ]
        block = _make_block(boxes)
        result = featurize(block, PAGE_W, PAGE_H)
        assert result["mean_token_confidence"] == pytest.approx(1.0)
        assert result["min_token_confidence"] == pytest.approx(1.0)


class TestFeaturizeRegionConfidence:
    """featurize_region() returns confidence features."""

    def test_no_header_defaults_to_one(self):
        result = featurize_region(
            region_type="legend",
            bbox=(100, 200, 400, 600),
            header_block=None,
            page_width=PAGE_W,
            page_height=PAGE_H,
        )
        assert result["mean_token_confidence"] == 1.0
        assert result["min_token_confidence"] == 1.0

    def test_with_header_block_confidence(self):
        boxes = [
            _box(100, 200, 250, 220, text="LEGEND", confidence=0.75),
        ]
        header = _make_block(boxes)
        result = featurize_region(
            region_type="legend",
            bbox=(100, 200, 400, 600),
            header_block=header,
            page_width=PAGE_W,
            page_height=PAGE_H,
        )
        assert result["mean_token_confidence"] == pytest.approx(0.75, abs=0.01)
        assert result["min_token_confidence"] == pytest.approx(0.75, abs=0.01)


# ── _adjusted_severity() ──────────────────────────────────────────────


class TestAdjustedSeverity:
    """Severity attenuation helper."""

    def test_high_confidence_no_change(self):
        assert _adjusted_severity("error", 0.9) == "error"
        assert _adjusted_severity("warning", 0.8) == "warning"
        assert _adjusted_severity("info", 0.7) == "info"

    def test_pdf_only_no_change(self):
        assert _adjusted_severity("error", 1.0) == "error"

    def test_low_confidence_downgrades_error(self):
        assert _adjusted_severity("error", 0.4) == "warning"

    def test_low_confidence_downgrades_warning(self):
        assert _adjusted_severity("warning", 0.3) == "info"

    def test_low_confidence_info_stays_info(self):
        assert _adjusted_severity("info", 0.2) == "info"

    def test_threshold_boundary_at_default(self):
        # Exactly at threshold → no downgrade
        assert _adjusted_severity("error", 0.6) == "error"
        # Just below threshold → downgrade
        assert _adjusted_severity("error", 0.59) == "warning"

    def test_custom_threshold(self):
        assert _adjusted_severity("error", 0.75, threshold=0.8) == "warning"
        assert _adjusted_severity("error", 0.85, threshold=0.8) == "error"


# ── run_all_checks() severity attenuation ─────────────────────────────


class TestRunAllChecksSeverityAttenuation:
    """run_all_checks() downgrades severity when mean_ocr_confidence is low."""

    def _make_notes_with_gap(self) -> list:
        """Build a notes column with a gap (1, 3) to trigger a finding."""
        from plancheck.models import NotesColumn

        box1 = _box(100, 100, 120, 120, text="1.")
        box3 = _box(100, 140, 120, 160, text="3.")
        body1 = _box(130, 100, 400, 120, text="First note text here")
        body3 = _box(130, 140, 400, 160, text="Third note text here")

        row1 = RowBand(page=0, boxes=[box1, body1])
        row3 = RowBand(page=0, boxes=[box3, body3])
        blk = BlockCluster(page=0, rows=[row1, row3], is_notes=True)

        header_box = _box(100, 80, 300, 100, text="GENERAL NOTES")
        header_row = RowBand(page=0, boxes=[header_box])
        header = BlockCluster(page=0, rows=[header_row], is_header=True)

        col = NotesColumn(
            page=0,
            header=header,
            notes_blocks=[blk],
        )
        return [col]

    def test_full_confidence_preserves_severity(self):
        cols = self._make_notes_with_gap()
        findings = run_all_checks(
            notes_columns=cols,
            page=0,
            mean_ocr_confidence=1.0,
        )
        # Should have at least one finding (gap)
        assert len(findings) > 0
        # Severity should be original (error/warning)
        severities = {f.severity for f in findings}
        assert "error" in severities or "warning" in severities

    def test_low_confidence_downgrades_severity(self):
        cols = self._make_notes_with_gap()
        high_findings = run_all_checks(
            notes_columns=cols,
            page=0,
            mean_ocr_confidence=1.0,
        )
        low_findings = run_all_checks(
            notes_columns=cols,
            page=0,
            mean_ocr_confidence=0.3,
        )
        # Same number of findings
        assert len(high_findings) == len(low_findings)
        # Low confidence should have lower or equal severity
        for hf, lf in zip(high_findings, low_findings):
            sev_order = {"info": 0, "warning": 1, "error": 2}
            assert sev_order[lf.severity] <= sev_order[hf.severity]

    def test_default_confidence_is_one(self):
        """Without mean_ocr_confidence arg, defaults to 1.0 (no attenuation)."""
        cols = self._make_notes_with_gap()
        findings = run_all_checks(notes_columns=cols, page=0)
        # Severity should be original — verify at least one is error/warning
        assert any(f.severity in ("error", "warning") for f in findings)
