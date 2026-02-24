"""Tests for plancheck.corrections.features."""

from __future__ import annotations

import pytest

from plancheck.corrections.features import featurize, featurize_region
from plancheck.models import BlockCluster, GlyphBox, RowBand

# ── helpers ────────────────────────────────────────────────────────────


def _box(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    text: str = "",
    fontname: str = "Arial",
    font_size: float = 9.0,
    page: int = 0,
) -> GlyphBox:
    return GlyphBox(
        page=page,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        text=text,
        fontname=fontname,
        font_size=font_size,
    )


def _make_block(boxes: list[GlyphBox], **kwargs) -> BlockCluster:
    """Build a row-based BlockCluster from a flat list of boxes."""
    from collections import defaultdict

    row_map: dict[float, list[GlyphBox]] = defaultdict(list)
    for b in boxes:
        row_map[b.y0].append(b)
    rows = [RowBand(page=0, boxes=bs) for _, bs in sorted(row_map.items())]
    return BlockCluster(page=0, rows=rows, **kwargs)


PAGE_W = 2448.0
PAGE_H = 1584.0

# Required keys that featurize() must return
REQUIRED_KEYS = {
    "font_size_pt",
    "font_size_max_pt",
    "font_size_min_pt",
    "is_all_caps",
    "is_bold",
    "token_count",
    "row_count",
    "x_frac",
    "y_frac",
    "x_center_frac",
    "y_center_frac",
    "width_frac",
    "height_frac",
    "aspect_ratio",
    "contains_digit",
    "starts_with_digit",
    "has_colon",
    "has_period_after_num",
    "text_length",
    "avg_chars_per_token",
    "zone",
    "neighbor_count",
}


class TestFeaturizeRequiredKeys:
    """test_featurize_returns_required_keys"""

    def test_all_keys_present(self) -> None:
        boxes = [
            _box(100, 200, 180, 220, text="HELLO"),
            _box(200, 200, 280, 220, text="WORLD"),
        ]
        block = _make_block(boxes)
        result = featurize(block, PAGE_W, PAGE_H)
        assert REQUIRED_KEYS.issubset(
            result.keys()
        ), f"Missing keys: {REQUIRED_KEYS - result.keys()}"


class TestFeaturizeAllCaps:
    """test_featurize_all_caps_detection"""

    def test_uppercase_text_detected(self) -> None:
        boxes = [
            _box(100, 100, 200, 120, text="GENERAL"),
            _box(210, 100, 300, 120, text="NOTES"),
        ]
        block = _make_block(boxes)
        result = featurize(block, PAGE_W, PAGE_H)
        assert result["is_all_caps"] == 1

    def test_mixed_case_not_all_caps(self) -> None:
        boxes = [
            _box(100, 100, 200, 120, text="General"),
            _box(210, 100, 300, 120, text="Notes"),
        ]
        block = _make_block(boxes)
        result = featurize(block, PAGE_W, PAGE_H)
        assert result["is_all_caps"] == 0


class TestFeaturizeStartsWithDigit:
    """test_featurize_starts_with_digit"""

    def test_numbered_note(self) -> None:
        boxes = [
            _box(100, 100, 120, 120, text="1."),
            _box(130, 100, 400, 120, text="ALL CONCRETE SHALL BE 4000 PSI"),
        ]
        block = _make_block(boxes)
        result = featurize(block, PAGE_W, PAGE_H)
        assert result["starts_with_digit"] == 1
        assert result["has_period_after_num"] == 1

    def test_text_start(self) -> None:
        boxes = [_box(100, 100, 300, 120, text="NOTES")]
        block = _make_block(boxes)
        result = featurize(block, PAGE_W, PAGE_H)
        assert result["starts_with_digit"] == 0
        assert result["has_period_after_num"] == 0


class TestFeaturizePositionFractions:
    """test_featurize_position_fractions"""

    def test_center_of_page(self) -> None:
        # Place a block approximately at the center of the page
        cx, cy = PAGE_W / 2, PAGE_H / 2
        w, h = 100, 20
        boxes = [_box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, text="CENTER")]
        block = _make_block(boxes)
        result = featurize(block, PAGE_W, PAGE_H)

        assert abs(result["x_center_frac"] - 0.5) < 0.01
        assert abs(result["y_center_frac"] - 0.5) < 0.01

    def test_top_left_corner(self) -> None:
        boxes = [_box(0, 0, 50, 20, text="ORIGIN")]
        block = _make_block(boxes)
        result = featurize(block, PAGE_W, PAGE_H)
        assert result["x_frac"] == pytest.approx(0.0, abs=0.001)
        assert result["y_frac"] == pytest.approx(0.0, abs=0.001)


class TestFeaturizeRegion:
    """Test featurize_region produces valid output."""

    def test_returns_required_keys(self) -> None:
        result = featurize_region(
            region_type="legend",
            bbox=(100, 200, 400, 600),
            header_block=None,
            page_width=PAGE_W,
            page_height=PAGE_H,
        )
        assert REQUIRED_KEYS.issubset(result.keys())

    def test_with_header_block(self) -> None:
        boxes = [
            _box(
                100, 200, 250, 220, text="LEGEND", fontname="Arial-Bold", font_size=12.0
            )
        ]
        header = _make_block(boxes)
        result = featurize_region(
            region_type="legend",
            bbox=(100, 200, 400, 600),
            header_block=header,
            page_width=PAGE_W,
            page_height=PAGE_H,
        )
        assert result["font_size_pt"] == 12.0
        assert result["is_all_caps"] == 1
        assert result["is_bold"] == 1
