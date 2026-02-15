"""Unit tests for plancheck.legends module (legend-specific only).

Covers: _is_legend_header, _detect_legend_columns, _pair_symbols_with_text,
        detect_legend_regions
"""

from __future__ import annotations

import logging

import pytest
from conftest import make_block, make_box, make_graphic

from plancheck.legends import (
    _detect_legend_columns,
    _is_legend_header,
    _pair_symbols_with_text,
    detect_legend_regions,
)
from plancheck.models import BlockCluster


# ====================================================================
# Classifier Tests
# ====================================================================


class TestIsLegendHeader:
    """Tests for _is_legend_header."""

    def test_exact_legend(self):
        blk = make_block([(10, 100, 80, 112, "LEGEND")], is_header=True)
        assert _is_legend_header(blk) is True

    def test_legend_colon(self):
        blk = make_block([(10, 100, 80, 112, "LEGEND:")], is_header=True)
        assert _is_legend_header(blk) is True

    def test_legend_embedded(self):
        blk = make_block(
            [(10, 100, 80, 112, "PLAN"), (90, 100, 170, 112, "LEGEND")],
            is_header=True,
        )
        assert _is_legend_header(blk) is True

    def test_not_header_flag(self):
        """Block must have is_header=True."""
        blk = make_block([(10, 100, 80, 112, "LEGEND")], is_header=False)
        assert _is_legend_header(blk) is False

    def test_no_legend_word(self):
        blk = make_block([(10, 100, 80, 112, "NOTES:")], is_header=True)
        assert _is_legend_header(blk) is False

    def test_empty_rows(self):
        blk = BlockCluster(page=0, rows=[], is_header=True)
        assert _is_legend_header(blk) is False


# ====================================================================
# Geometry / Column Detection
# ====================================================================


class TestDetectLegendColumns:
    """Tests for _detect_legend_columns."""

    def test_single_column(self):
        syms = [
            make_graphic("rect", 10, 10, 30, 30),
            make_graphic("rect", 12, 50, 32, 70),
            make_graphic("rect", 11, 90, 31, 110),
        ]
        cols = _detect_legend_columns(syms, x_tolerance=30.0)
        assert len(cols) == 1
        assert len(cols[0]) == 3

    def test_two_columns(self):
        syms = [
            make_graphic("rect", 10, 10, 30, 30),
            make_graphic("rect", 10, 50, 30, 70),
            make_graphic("rect", 200, 10, 220, 30),
            make_graphic("rect", 200, 50, 220, 70),
        ]
        cols = _detect_legend_columns(syms, x_tolerance=30.0)
        assert len(cols) == 2
        assert len(cols[0]) == 2
        assert len(cols[1]) == 2

    def test_empty(self):
        assert _detect_legend_columns([], x_tolerance=30.0) == []


# ====================================================================
# Pairing
# ====================================================================


class TestPairSymbolsWithText:
    """Tests for _pair_symbols_with_text."""

    def test_pairs_symbol_with_text_to_right(self):
        sym = make_graphic("rect", 10, 100, 30, 120)
        text_blk = make_block([(50, 100, 200, 120, "FIRE HYDRANT")])
        entries = _pair_symbols_with_text(
            [sym],
            [text_blk],
            page=0,
            y_tolerance=20.0,
            x_gap_max=300.0,
        )
        assert len(entries) == 1
        assert entries[0].description == "FIRE HYDRANT"
        assert entries[0].symbol is sym

    def test_no_text_to_right(self):
        sym = make_graphic("rect", 500, 100, 520, 120)
        text_blk = make_block([(10, 100, 200, 120, "FAR LEFT")])
        entries = _pair_symbols_with_text(
            [sym],
            [text_blk],
            page=0,
            y_tolerance=20.0,
            x_gap_max=300.0,
        )
        assert len(entries) == 1
        assert entries[0].description == ""  # no match

    def test_too_far_y(self):
        sym = make_graphic("rect", 10, 100, 30, 120)
        text_blk = make_block([(50, 300, 200, 320, "TOO FAR")])
        entries = _pair_symbols_with_text(
            [sym],
            [text_blk],
            page=0,
            y_tolerance=20.0,
            x_gap_max=300.0,
        )
        assert len(entries) == 1
        assert entries[0].description == ""


# ====================================================================
# Detector Integration Tests
# ====================================================================


class TestDetectLegendRegions:
    """Integration tests for detect_legend_regions."""

    def test_boxed_legend(self):
        header = make_block([(100, 100, 200, 112, "LEGEND")], is_header=True)
        # Enclosing rect
        rect = make_graphic("rect", 80, 80, 500, 400)
        # Symbol inside region
        sym = make_graphic("rect", 100, 150, 120, 170)
        # Text block to the right of symbol
        text_blk = make_block([(140, 150, 400, 170, "FIRE HYDRANT")])

        regions = detect_legend_regions(
            blocks=[header, text_blk],
            graphics=[rect, sym],
            page_width=612,
            page_height=792,
        )
        assert len(regions) == 1
        assert regions[0].is_boxed is True
        assert regions[0].page == 0

    def test_no_legend_headers(self):
        blk = make_block([(10, 10, 80, 22, "NOTES:")], is_header=True)
        regions = detect_legend_regions(
            blocks=[blk],
            graphics=[],
            page_width=612,
            page_height=792,
        )
        assert regions == []

    def test_exclusion_zone_skips_header(self):
        header = make_block([(100, 100, 200, 112, "LEGEND")], is_header=True)
        regions = detect_legend_regions(
            blocks=[header],
            graphics=[],
            page_width=612,
            page_height=792,
            exclusion_zones=[(90, 90, 210, 120)],  # covers header
        )
        assert regions == []

    def test_emits_debug_logs(self, caplog):
        """Logger output should appear via caplog, not file I/O."""
        header = make_block([(100, 100, 200, 112, "LEGEND")], is_header=True)
        with caplog.at_level(logging.DEBUG, logger="plancheck.legends"):
            detect_legend_regions(
                blocks=[header],
                graphics=[],
                page_width=612,
                page_height=792,
            )
        assert any("detect_legend_regions" in r.message for r in caplog.records)
