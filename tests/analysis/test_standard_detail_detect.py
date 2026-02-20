"""Unit tests for plancheck._standard_detail_detect module.

Covers: _is_standard_detail_header, _has_inline_entries,
        _parse_standard_detail_entries, _parse_standard_detail_entries_inline,
        _parse_standard_detail_entries_two_column, detect_standard_detail_regions
"""

from __future__ import annotations

import logging

import pytest
from conftest import make_block, make_box, make_graphic

from plancheck.analysis.standard_details import (
    _has_inline_entries,
    _is_standard_detail_header,
    _parse_standard_detail_entries,
    _parse_standard_detail_entries_from_inline_blocks,
    _parse_standard_detail_entries_inline,
    _parse_standard_detail_entries_two_column,
    detect_standard_detail_regions,
)
from plancheck.models import BlockCluster


class TestIsStandardDetailHeader:
    """Tests for _is_standard_detail_header."""

    def test_odot_standard_details(self):
        blk = make_block(
            [(10, 100, 200, 112, "ODOT"), (210, 100, 400, 112, "STANDARD DETAILS:")],
        )
        assert _is_standard_detail_header(blk) is True

    def test_standard_details_only(self):
        blk = make_block([(10, 100, 200, 112, "STANDARD DETAILS:")])
        assert _is_standard_detail_header(blk) is True

    def test_odot_standards(self):
        blk = make_block(
            [(10, 100, 80, 112, "ODOT"), (90, 100, 200, 112, "STANDARDS")],
        )
        assert _is_standard_detail_header(blk) is True

    def test_not_standard_detail(self):
        blk = make_block([(10, 100, 80, 112, "NOTES:")])
        assert _is_standard_detail_header(blk) is False

    def test_empty_rows(self):
        blk = BlockCluster(page=0, rows=[])
        assert _is_standard_detail_header(blk) is False


class TestHasInlineEntries:
    """Tests for _has_inline_entries."""

    def test_sheet_number_row(self):
        """Row 1 starts with a sheet pattern like 'BMPR-0' -> True."""
        blk = make_block(
            [
                (10, 100, 200, 112, "ODOT STANDARD DETAILS:"),
                (10, 120, 80, 132, "BMPR-0"),
                (90, 120, 250, 132, "PREFAB CULVERT"),
            ],
        )
        assert _has_inline_entries(blk) is True

    def test_numeric_sheet_number(self):
        """Row 1 starts with '621-1' -> True."""
        blk = make_block(
            [
                (10, 100, 200, 112, "STANDARD DETAILS:"),
                (10, 120, 60, 132, "621-1"),
                (70, 120, 200, 132, "SOME DETAIL"),
            ],
        )
        assert _has_inline_entries(blk) is True

    def test_subtitle_row(self):
        """Row 1 looks like a subtitle -> False."""
        blk = make_block(
            [
                (10, 100, 200, 112, "ODOT STANDARD DETAILS:"),
                (10, 120, 300, 132, "THE FOLLOWING SHALL BE USED"),
            ],
        )
        assert _has_inline_entries(blk) is False

    def test_single_row(self):
        """Only one row -> False."""
        blk = make_block([(10, 100, 200, 112, "ODOT STANDARD DETAILS:")])
        assert _has_inline_entries(blk) is False


class TestParseStandardDetailEntries:
    """Tests for _parse_standard_detail_entries (boxed region parser)."""

    def test_sheet_num_description(self):
        blk = make_block(
            [
                (10, 10, 50, 22, "SS-1"),
                (60, 10, 250, 22, "STANDARD SHOULDER SECTION"),
            ],
        )
        entries = _parse_standard_detail_entries([blk], page=0)
        assert len(entries) >= 1
        assert entries[0].sheet_number == "SS-1"
        assert "STANDARD SHOULDER SECTION" in entries[0].description

    def test_no_match(self):
        blk = make_block([(10, 10, 200, 22, "some random text")])
        entries = _parse_standard_detail_entries([blk], page=0)
        assert entries == []

    def test_empty(self):
        entries = _parse_standard_detail_entries([], page=0)
        assert entries == []


class TestParseStandardDetailEntriesInline:
    """Tests for _parse_standard_detail_entries_inline."""

    def test_inline_entries(self):
        blk = make_block(
            [
                (10, 100, 200, 112, "ODOT STANDARD DETAILS:"),
                (10, 120, 80, 132, "PCES-5-1"),
                (90, 120, 300, 132, "- PREFAB CULVERT END"),
                (10, 140, 80, 152, "SS-1"),
                (90, 140, 300, 152, "- STANDARD SHOULDER"),
            ],
        )
        entries = _parse_standard_detail_entries_inline(blk, page=0)
        assert len(entries) >= 2
        sheet_nums = [e.sheet_number for e in entries]
        assert "PCES-5-1" in sheet_nums
        assert "SS-1" in sheet_nums

    def test_no_entries(self):
        blk = make_block([(10, 100, 200, 112, "ODOT STANDARD DETAILS:")])
        entries = _parse_standard_detail_entries_inline(blk, page=0)
        assert entries == []


class TestParseStandardDetailEntriesTwoColumn:
    """Tests for _parse_standard_detail_entries_two_column."""

    def test_matched_rows(self):
        sheet_blk = make_block(
            [
                (10, 10, 60, 22, "SS-1"),
                (10, 30, 60, 42, "621-1"),
            ],
        )
        desc_blk = make_block(
            [
                (200, 10, 400, 22, "STANDARD SHOULDER"),
                (200, 30, 400, 42, "BARRIER DETAIL"),
            ],
        )
        entries = _parse_standard_detail_entries_two_column(
            [sheet_blk],
            [desc_blk],
            page=0,
        )
        assert len(entries) == 2
        assert entries[0].sheet_number == "SS-1"
        assert entries[0].description == "STANDARD SHOULDER"

    def test_no_matching_desc(self):
        sheet_blk = make_block([(10, 10, 60, 22, "SS-1")])
        entries = _parse_standard_detail_entries_two_column([sheet_blk], [], page=0)
        assert len(entries) == 1
        assert entries[0].sheet_number == "SS-1"
        assert entries[0].description == ""

    def test_empty(self):
        entries = _parse_standard_detail_entries_two_column([], [], page=0)
        assert entries == []


class TestDetectStandardDetailRegions:
    """Integration tests for detect_standard_detail_regions."""

    def test_boxed_standard_detail(self):
        header = make_block(
            [(100, 100, 300, 112, "ODOT STANDARD DETAILS:")],
            is_header=True,
        )
        rect = make_graphic("rect", 80, 80, 500, 400)
        entry_blk = make_block(
            [
                (100, 130, 150, 142, "SS-1"),
                (160, 130, 400, 142, "STANDARD SHOULDER"),
            ],
        )
        regions = detect_standard_detail_regions(
            blocks=[header, entry_blk],
            graphics=[rect],
            page_width=612,
            page_height=792,
        )
        assert len(regions) == 1

    def test_no_headers(self):
        blk = make_block([(10, 10, 80, 22, "NOTES:")], is_header=True)
        regions = detect_standard_detail_regions(
            blocks=[blk],
            graphics=[],
            page_width=612,
            page_height=792,
        )
        assert regions == []

    def test_emits_debug_logs(self, caplog):
        header = make_block(
            [(100, 100, 300, 112, "ODOT STANDARD DETAILS:")],
            is_header=True,
        )
        with caplog.at_level(logging.DEBUG, logger="plancheck.standard_details"):
            detect_standard_detail_regions(
                blocks=[header],
                graphics=[],
                page_width=612,
                page_height=792,
            )
        assert any(
            "detect_standard_detail_regions" in r.message for r in caplog.records
        )


class TestParseFromInlineBlocks:
    """Tests for _parse_standard_detail_entries_from_inline_blocks."""

    def test_basic_inline_entry(self):
        """A block whose row starts with SHEET-NUM followed by description."""
        blk = make_block(
            [
                (10, 100, 60, 112, "BMPR-0"),
                (70, 100, 400, 112, "BEST MANAGEMENT PRACTICE"),
            ],
        )
        entries = _parse_standard_detail_entries_from_inline_blocks([blk], page=1)
        assert len(entries) == 1
        assert entries[0].sheet_number == "BMPR-0"
        assert entries[0].description == "BEST MANAGEMENT PRACTICE"
        assert entries[0].page == 1

    def test_multirow_inline_block(self):
        """Multiple rows inside a single block each produce an entry."""
        blk = make_block(
            [
                (10, 100, 60, 112, "CET6D-4-2"),
                (70, 100, 300, 112, "CULVERT END TREATMENT"),
                (10, 120, 60, 132, "PCES-5-1"),
                (70, 120, 350, 132, "PREFABRICATED CULVERT END"),
            ],
        )
        entries = _parse_standard_detail_entries_from_inline_blocks([blk], page=0)
        assert len(entries) == 2
        assert entries[0].sheet_number == "CET6D-4-2"
        assert entries[1].sheet_number == "PCES-5-1"

    def test_no_match_returns_empty(self):
        """Rows that don't match the sheet-pattern are skipped."""
        blk = make_block(
            [(10, 100, 200, 112, "JUST SOME TEXT WITHOUT SHEET NUM")],
        )
        entries = _parse_standard_detail_entries_from_inline_blocks([blk], page=0)
        assert entries == []
