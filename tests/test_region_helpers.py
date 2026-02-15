"""Unit tests for plancheck._region_helpers module.

Covers: _bboxes_overlap, _find_enclosing_rect, _find_symbols_in_region,
        _find_text_blocks_in_region, _extract_text_rows_from_blocks,
        _merge_same_line_rows, filter_graphics_outside_regions
"""

from __future__ import annotations

import pytest
from conftest import make_block, make_box, make_graphic

from plancheck._region_helpers import (
    _bboxes_overlap,
    _extract_text_rows_from_blocks,
    _find_enclosing_rect,
    _find_symbols_in_region,
    _find_text_blocks_in_region,
    _merge_same_line_rows,
    filter_graphics_outside_regions,
)
from plancheck.models import BlockCluster


class TestBboxesOverlap:
    """Tests for _bboxes_overlap."""

    def test_overlapping(self):
        assert _bboxes_overlap((0, 0, 100, 100), (50, 50, 150, 150)) is True

    def test_identical(self):
        assert _bboxes_overlap((10, 10, 50, 50), (10, 10, 50, 50)) is True

    def test_separated_x(self):
        assert _bboxes_overlap((0, 0, 50, 50), (60, 0, 110, 50)) is False

    def test_separated_y(self):
        assert _bboxes_overlap((0, 0, 50, 50), (0, 60, 50, 110)) is False

    def test_touching_edge(self):
        """Touching at an edge (x1a == x0b) => not separated."""
        assert _bboxes_overlap((0, 0, 50, 50), (50, 0, 100, 50)) is True


class TestFindEnclosingRect:
    """Tests for _find_enclosing_rect."""

    def test_finds_enclosing(self):
        header_bbox = (100, 100, 200, 120)
        rect = make_graphic("rect", 80, 80, 400, 300)
        result = _find_enclosing_rect(header_bbox, [rect], tolerance=20.0)
        assert result is rect

    def test_too_small_ignored(self):
        """Rects smaller than 100 wide or 50 tall are skipped."""
        header_bbox = (10, 10, 50, 30)
        rect = make_graphic("rect", 5, 5, 60, 40)  # 55 x 35
        result = _find_enclosing_rect(header_bbox, [rect])
        assert result is None

    def test_non_rect_ignored(self):
        header_bbox = (100, 100, 200, 120)
        line = make_graphic("line", 80, 80, 400, 300)
        result = _find_enclosing_rect(header_bbox, [line])
        assert result is None

    def test_no_graphics(self):
        result = _find_enclosing_rect((0, 0, 100, 100), [])
        assert result is None


class TestFindSymbolsInRegion:
    """Tests for _find_symbols_in_region."""

    def test_finds_small_symbols(self):
        region = (0, 0, 200, 200)
        sym = make_graphic("rect", 10, 10, 30, 30)  # 20x20, inside
        big = make_graphic("rect", 10, 10, 100, 100)  # 90x90, too large
        outside = make_graphic("rect", 210, 10, 230, 30)  # outside region
        result = _find_symbols_in_region(
            region, [sym, big, outside], max_symbol_size=50
        )
        assert result == [sym]

    def test_empty_graphics(self):
        assert _find_symbols_in_region((0, 0, 100, 100), []) == []


class TestFindTextBlocksInRegion:
    """Tests for _find_text_blocks_in_region."""

    def test_finds_overlapping(self):
        region = (0, 0, 300, 300)
        inside = make_block([(10, 10, 100, 30, "INSIDE")])
        outside = make_block([(400, 400, 500, 420, "OUTSIDE")])
        result = _find_text_blocks_in_region(region, [inside, outside])
        assert result == [inside]

    def test_excludes_header(self):
        region = (0, 0, 300, 300)
        header = make_block([(10, 10, 100, 30, "HEADER")], is_header=True)
        body = make_block([(10, 50, 100, 70, "BODY")])
        result = _find_text_blocks_in_region(
            region, [header, body], exclude_header=header
        )
        assert result == [body]

    def test_excludes_tables(self):
        region = (0, 0, 300, 300)
        tbl = make_block([(10, 10, 100, 30, "TABLE")], is_table=True)
        result = _find_text_blocks_in_region(region, [tbl])
        assert result == []

    def test_excludes_is_header_blocks(self):
        """Blocks with is_header=True (but not the explicit exclude_header) are still excluded."""
        region = (0, 0, 300, 300)
        other_header = make_block([(10, 50, 100, 70, "OTHER HEADER")], is_header=True)
        result = _find_text_blocks_in_region(region, [other_header])
        assert result == []


class TestExtractTextRowsFromBlocks:
    """Tests for _extract_text_rows_from_blocks."""

    def test_basic_extraction(self):
        blk = make_block(
            [
                (10, 10, 80, 22, "HELLO"),
                (10, 30, 80, 42, "WORLD"),
            ],
        )
        rows = _extract_text_rows_from_blocks([blk])
        assert len(rows) == 2
        assert rows[0][0] == "HELLO"
        assert rows[1][0] == "WORLD"
        # Each row has (text, bbox, parent_block)
        assert rows[0][2] is blk


class TestFilterGraphicsOutsideRegions:
    """Tests for filter_graphics_outside_regions."""

    def test_filters_inside(self):
        g1 = make_graphic("rect", 10, 10, 30, 30)  # inside exclusion
        g2 = make_graphic("rect", 200, 200, 220, 220)  # outside exclusion
        result = filter_graphics_outside_regions([g1, g2], [(0, 0, 100, 100)])
        assert result == [g2]

    def test_no_exclusions(self):
        g = make_graphic("rect", 10, 10, 30, 30)
        result = filter_graphics_outside_regions([g], [])
        assert result == [g]


class TestMergeSameLineRows:
    """Tests for _merge_same_line_rows."""

    def test_no_merge_single_row(self):
        header = make_block([(10, 100, 200, 112, "HEADER ONLY")])
        text, bbox = _merge_same_line_rows(header, [])
        assert text is None
        assert bbox is None

    def test_merges_adjacent_block(self):
        header = make_block(
            [
                (10, 100, 100, 112, "ODOT STANDARD"),
                (10, 118, 200, 130, "THE FOLLOWING SHALL"),
            ],
        )
        # Adjacent block on the same line as row 1 of header
        adjacent = make_block([(210, 118, 400, 130, "BE USED ON THIS PROJECT:")])
        text, bbox = _merge_same_line_rows(header, [adjacent])
        assert text is not None
        assert "THE FOLLOWING SHALL" in text
        assert "BE USED ON THIS PROJECT:" in text

    def test_does_not_merge_distant_block(self):
        header = make_block(
            [
                (10, 100, 100, 112, "HEADER"),
                (10, 118, 200, 130, "SUB HEADER"),
            ],
        )
        # Block far away in y
        far = make_block([(210, 300, 400, 312, "FAR AWAY")])
        text, bbox = _merge_same_line_rows(header, [far])
        # Should still return row1 text, but not include 'FAR AWAY'
        assert text is not None
        assert "FAR AWAY" not in text
