"""Unit tests for vector symbol recovery (tocr/vector_symbols.py)."""

from __future__ import annotations

import math

import pytest

from plancheck.config import GroupingConfig
from plancheck.models.tokens import GlyphBox
from plancheck.tocr.vector_symbols import (
    _classify_as_degree,
    _classify_as_slash,
    _classify_minus_lines,
    _estimate_char_width,
    _find_digit_neighbours,
    _is_small_circle,
    _line_angle_deg,
    _line_bbox,
    recover_vector_symbols,
)
from tests.conftest import make_box


# ── Helpers ────────────────────────────────────────────────────────────


def _make_line(
    x0: float, top: float, x1: float, bottom: float,
) -> dict:
    """Build a pdfplumber-style line dict."""
    return {"x0": x0, "top": top, "x1": x1, "bottom": bottom}


def _make_curve(pts: list[tuple[float, float]]) -> dict:
    """Build a pdfplumber-style curve dict."""
    return {"pts": pts}


def _make_digit_box(
    x0: float, y0: float, x1: float, y1: float,
    text: str = "3", font_size: float = 10.0,
) -> GlyphBox:
    """Create a GlyphBox containing a digit with font_size set."""
    return GlyphBox(
        page=0, x0=x0, y0=y0, x1=x1, y1=y1,
        text=text, origin="text", font_size=font_size,
    )


def _default_cfg() -> GroupingConfig:
    return GroupingConfig()


# ── Tests: _line_bbox ──────────────────────────────────────────────────


class TestLineBbox:
    def test_normalises_coordinates(self) -> None:
        ln = _make_line(10, 5, 2, 15)
        assert _line_bbox(ln) == (2, 5, 10, 15)

    def test_already_normalised(self) -> None:
        ln = _make_line(0, 0, 10, 10)
        assert _line_bbox(ln) == (0, 0, 10, 10)


# ── Tests: _line_angle_deg ─────────────────────────────────────────────


class TestLineAngle:
    def test_horizontal_line(self) -> None:
        ln = _make_line(0, 10, 20, 10)
        assert _line_angle_deg(ln) == pytest.approx(0.0)

    def test_vertical_line(self) -> None:
        ln = _make_line(10, 0, 10, 20)
        assert _line_angle_deg(ln) == pytest.approx(90.0)

    def test_diagonal_45(self) -> None:
        ln = _make_line(0, 0, 10, 10)
        assert _line_angle_deg(ln) == pytest.approx(45.0)

    def test_typical_slash_angle(self) -> None:
        # Rise 12, run 6 → ~63.4°
        ln = _make_line(0, 0, 6, 12)
        assert 60 < _line_angle_deg(ln) < 67

    def test_zero_length(self) -> None:
        ln = _make_line(5, 5, 5, 5)
        assert _line_angle_deg(ln) == 0.0


# ── Tests: _estimate_char_width ────────────────────────────────────────


class TestEstimateCharWidth:
    def test_with_digit_tokens(self) -> None:
        tokens = [
            _make_digit_box(0, 0, 12, 10, "12"),  # 12/2 = 6.0 per char
            _make_digit_box(20, 0, 38, 10, "345"),  # 18/3 = 6.0 per char
        ]
        assert _estimate_char_width(tokens) == pytest.approx(6.0)

    def test_no_digit_tokens_returns_default(self) -> None:
        tokens = [make_box(0, 0, 30, 10, "ABC")]
        assert _estimate_char_width(tokens) == 6.0

    def test_empty_returns_default(self) -> None:
        assert _estimate_char_width([]) == 6.0


# ── Tests: _is_small_circle ───────────────────────────────────────────


class TestIsSmallCircle:
    def test_circle_like_curve(self) -> None:
        # Rough circle: 4 pts forming a small square-ish bbox.
        pts = [(0, 0), (3, 0), (3, 3), (0, 3)]
        assert _is_small_circle(_make_curve(pts), max_size=5.0, max_aspect=1.4)

    def test_too_few_points(self) -> None:
        pts = [(0, 0), (3, 0), (3, 3)]
        assert not _is_small_circle(_make_curve(pts), max_size=5.0, max_aspect=1.4)

    def test_too_large(self) -> None:
        pts = [(0, 0), (20, 0), (20, 20), (0, 20)]
        assert not _is_small_circle(_make_curve(pts), max_size=5.0, max_aspect=1.4)

    def test_too_elongated(self) -> None:
        pts = [(0, 0), (10, 0), (10, 2), (0, 2)]
        assert not _is_small_circle(_make_curve(pts), max_size=15.0, max_aspect=1.4)


# ── Tests: _find_digit_neighbours ─────────────────────────────────────


class TestFindDigitNeighbours:
    def test_finds_left_and_right(self) -> None:
        left_tok = _make_digit_box(0, 50, 10, 60, "2")
        right_tok = _make_digit_box(25, 50, 35, 60, "34")
        bbox = (12, 50, 22, 60)  # In the gap
        l, r = _find_digit_neighbours(bbox, [left_tok, right_tok], max_gap=15)
        assert l is left_tok
        assert r is right_tok

    def test_no_left_neighbour(self) -> None:
        right_tok = _make_digit_box(25, 50, 35, 60, "34")
        bbox = (12, 50, 22, 60)
        l, r = _find_digit_neighbours(bbox, [right_tok], max_gap=15)
        assert l is None
        assert r is right_tok

    def test_too_far_away(self) -> None:
        left_tok = _make_digit_box(0, 50, 5, 60, "2")
        bbox = (50, 50, 55, 60)  # 45 pts away
        l, r = _find_digit_neighbours(bbox, [left_tok], max_gap=10)
        assert l is None

    def test_ignores_non_digit_tokens(self) -> None:
        alpha = make_box(0, 50, 10, 60, "ABC")
        bbox = (12, 50, 22, 60)
        l, r = _find_digit_neighbours(bbox, [alpha], max_gap=15)
        assert l is None and r is None

    def test_ignores_different_vertical_band(self) -> None:
        tok = _make_digit_box(0, 0, 10, 10, "2")
        bbox = (12, 100, 22, 110)  # Different y-band
        l, r = _find_digit_neighbours(bbox, [tok], max_gap=15)
        assert l is None and r is None


# ── Tests: _classify_as_slash ──────────────────────────────────────────


class TestClassifyAsSlash:
    def test_valid_slash(self) -> None:
        # Diagonal line at ~63° between two digit tokens.
        left = _make_digit_box(0, 50, 10, 60, "2")
        right = _make_digit_box(20, 50, 30, 60, "34")
        slash_line = _make_line(12, 60, 18, 50)  # ~63° diagonal
        cfg = _default_cfg()

        result = _classify_as_slash(slash_line, [left, right], 6.0, 0, cfg)
        assert result is not None
        assert result.text == "/"
        assert result.origin == "vector_symbol"
        assert result.confidence == 0.95

    def test_wrong_angle_rejected(self) -> None:
        # Near-horizontal line — not a slash.
        left = _make_digit_box(0, 50, 10, 60, "2")
        right = _make_digit_box(20, 50, 30, 60, "34")
        horiz_line = _make_line(12, 55, 18, 56)  # ~9.5°
        cfg = _default_cfg()

        result = _classify_as_slash(horiz_line, [left, right], 6.0, 0, cfg)
        assert result is None

    def test_too_large_rejected(self) -> None:
        left = _make_digit_box(0, 50, 10, 60, "2")
        right = _make_digit_box(60, 50, 70, 60, "34")
        big_line = _make_line(15, 90, 55, 50)  # ~45°, but huge
        cfg = _default_cfg()

        result = _classify_as_slash(big_line, [left, right], 6.0, 0, cfg)
        assert result is None

    def test_no_digit_neighbours_rejected(self) -> None:
        slash_line = _make_line(12, 60, 18, 50)
        alpha = make_box(0, 50, 10, 60, "ABC")
        cfg = _default_cfg()

        result = _classify_as_slash(slash_line, [alpha], 6.0, 0, cfg)
        assert result is None

    def test_already_in_text_rejected(self) -> None:
        left = _make_digit_box(0, 50, 10, 60, "2")
        right = _make_digit_box(20, 50, 30, 60, "34")
        # An existing "/" token covers the same position.
        existing_slash = GlyphBox(
            page=0, x0=12, y0=50, x1=18, y1=60, text="/", origin="text",
        )
        slash_line = _make_line(12, 60, 18, 50)
        cfg = _default_cfg()

        result = _classify_as_slash(
            slash_line, [left, right, existing_slash], 6.0, 0, cfg,
        )
        assert result is None

    def test_too_short_rejected(self) -> None:
        left = _make_digit_box(0, 50, 10, 60, "2")
        right = _make_digit_box(14, 50, 24, 60, "34")
        tiny_line = _make_line(11, 55, 12, 56)  # ~1.4 pts long
        cfg = _default_cfg()

        result = _classify_as_slash(tiny_line, [left, right], 6.0, 0, cfg)
        assert result is None


# ── Tests: _classify_as_degree ─────────────────────────────────────────


class TestClassifyAsDegree:
    def test_valid_degree(self) -> None:
        # Small circle at superscript height, digit to the left.
        digit = _make_digit_box(0, 50, 10, 60, "90")
        circle = _make_curve([(12, 48), (15, 48), (15, 51), (12, 51)])
        cfg = _default_cfg()

        result = _classify_as_degree(circle, [digit], 6.0, 10.0, 0, cfg)
        assert result is not None
        assert result.text == "°"
        assert result.origin == "vector_symbol"

    def test_too_large_rejected(self) -> None:
        digit = _make_digit_box(0, 50, 10, 60, "90")
        big_circle = _make_curve([(12, 45), (22, 45), (22, 55), (12, 55)])
        cfg = _default_cfg()

        result = _classify_as_degree(big_circle, [digit], 6.0, 10.0, 0, cfg)
        assert result is None

    def test_not_superscript_rejected(self) -> None:
        # Circle at same height as digit mid → not a degree.
        digit = _make_digit_box(0, 50, 10, 60, "90")
        # Centre at y=55, digit midpoint = 55 → cy >= digit_mid → rejected.
        circle = _make_curve([(12, 53), (15, 53), (15, 57), (12, 57)])
        cfg = _default_cfg()

        result = _classify_as_degree(circle, [digit], 6.0, 10.0, 0, cfg)
        assert result is None

    def test_no_digit_neighbour_rejected(self) -> None:
        alpha = make_box(0, 50, 10, 60, "ABC")
        circle = _make_curve([(12, 48), (15, 48), (15, 51), (12, 51)])
        cfg = _default_cfg()

        result = _classify_as_degree(circle, [alpha], 6.0, 10.0, 0, cfg)
        assert result is None


# ── Tests: _classify_minus_lines ──────────────────────────────────────


class TestClassifyMinusLines:
    def test_valid_minus(self) -> None:
        left = _make_digit_box(0, 50, 10, 60, "5")
        right = _make_digit_box(25, 50, 35, 60, "3")
        horiz_line = _make_line(12, 55, 22, 55)  # 0° horizontal
        cfg = _default_cfg()

        results = _classify_minus_lines([horiz_line], [left, right], 6.0, 0, cfg)
        assert len(results) == 1
        assert results[0].text == "-"

    def test_diagonal_rejected(self) -> None:
        left = _make_digit_box(0, 50, 10, 60, "5")
        diag_line = _make_line(12, 65, 22, 50)  # ~56°
        cfg = _default_cfg()

        results = _classify_minus_lines([diag_line], [left], 6.0, 0, cfg)
        assert len(results) == 0


# ── Tests: recover_vector_symbols (integration) ───────────────────────


class TestRecoverVectorSymbols:
    def test_disabled_returns_original(self) -> None:
        cfg = GroupingConfig(tocr_vector_symbols_enabled=False)
        tokens = [_make_digit_box(0, 50, 10, 60, "42")]
        result, diag = recover_vector_symbols(tokens, [], [], 0, cfg)
        assert result is tokens  # Same object, untouched
        assert diag["vector_symbols_found"] == 0

    def test_empty_graphics_returns_original(self) -> None:
        cfg = _default_cfg()
        tokens = [_make_digit_box(0, 50, 10, 60, "42")]
        result, diag = recover_vector_symbols(tokens, [], [], 0, cfg)
        assert result is tokens
        assert diag["vector_symbols_found"] == 0

    def test_slash_injected(self) -> None:
        cfg = _default_cfg()
        left = _make_digit_box(0, 50, 10, 60, "2")
        right = _make_digit_box(20, 50, 30, 60, "34")
        slash_line = _make_line(12, 60, 18, 50)  # ~63°

        result, diag = recover_vector_symbols(
            [left, right], [slash_line], [], 0, cfg,
        )
        assert diag["vector_symbols_found"] >= 1
        assert diag["by_type"].get("/", 0) >= 1
        slash_tokens = [t for t in result if t.text == "/"]
        assert len(slash_tokens) >= 1
        assert slash_tokens[0].origin == "vector_symbol"

    def test_diagnostic_only_mode(self) -> None:
        cfg = GroupingConfig(tocr_vector_symbols_inject=False)
        left = _make_digit_box(0, 50, 10, 60, "2")
        right = _make_digit_box(20, 50, 30, 60, "34")
        slash_line = _make_line(12, 60, 18, 50)

        result, diag = recover_vector_symbols(
            [left, right], [slash_line], [], 0, cfg,
        )
        # Symbols found in diagnostics but NOT injected.
        assert diag["vector_symbols_found"] >= 1
        slash_tokens = [t for t in result if t.text == "/"]
        assert len(slash_tokens) == 0

    def test_mixed_symbols(self) -> None:
        cfg = _default_cfg()
        # Digit tokens.
        d1 = _make_digit_box(0, 50, 10, 60, "2")
        d2 = _make_digit_box(20, 50, 30, 60, "34")
        d3 = _make_digit_box(50, 50, 60, 60, "90")

        # Slash between d1 and d2.
        slash_line = _make_line(12, 60, 18, 50)

        # Small circle for degree after d3 (superscript).
        degree_circle = _make_curve([(62, 48), (65, 48), (65, 51), (62, 51)])

        result, diag = recover_vector_symbols(
            [d1, d2, d3], [slash_line], [degree_circle], 0, cfg,
        )
        assert diag["vector_symbols_found"] >= 1
        texts = [t.text for t in result]
        assert "/" in texts

    def test_output_sorted_by_position(self) -> None:
        cfg = _default_cfg()
        d1 = _make_digit_box(0, 50, 10, 60, "2")
        d2 = _make_digit_box(20, 50, 30, 60, "34")
        slash_line = _make_line(12, 60, 18, 50)

        result, _ = recover_vector_symbols(
            [d1, d2], [slash_line], [], 0, cfg,
        )
        # Verify sorted by (y0, x0).
        for i in range(len(result) - 1):
            assert (result[i].y0, result[i].x0) <= (result[i + 1].y0, result[i + 1].x0)

    def test_injected_token_has_font_size(self) -> None:
        cfg = _default_cfg()
        left = _make_digit_box(0, 50, 10, 60, "2", font_size=12.0)
        right = _make_digit_box(20, 50, 30, 60, "34", font_size=12.0)
        slash_line = _make_line(12, 60, 18, 50)

        result, _ = recover_vector_symbols(
            [left, right], [slash_line], [], 0, cfg,
        )
        slash_tokens = [t for t in result if t.text == "/"]
        assert len(slash_tokens) >= 1
        assert slash_tokens[0].font_size > 0
