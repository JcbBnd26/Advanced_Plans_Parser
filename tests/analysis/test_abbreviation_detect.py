"""Unit tests for plancheck._abbreviation_detect module.

Covers: _is_abbreviation_header, _parse_abbreviation_entries,
        _parse_abbreviation_entries_two_column, detect_abbreviation_regions
"""

from __future__ import annotations

import logging

import pytest
from conftest import make_block, make_box, make_graphic

from plancheck.analysis.abbreviations import (
    _is_abbreviation_header,
    _parse_abbreviation_entries,
    _parse_abbreviation_entries_two_column,
    detect_abbreviation_regions,
)
from plancheck.models import BlockCluster


class TestIsAbbreviationHeader:
    """Tests for _is_abbreviation_header."""

    def test_abbreviations(self):
        blk = make_block([(10, 100, 150, 112, "ABBREVIATIONS")])
        assert _is_abbreviation_header(blk) is True

    def test_abbreviation_singular(self):
        blk = make_block([(10, 100, 150, 112, "ABBREVIATION")])
        assert _is_abbreviation_header(blk) is True

    def test_abbreviations_colon(self):
        blk = make_block([(10, 100, 150, 112, "ABBREVIATIONS:")])
        assert _is_abbreviation_header(blk) is True

    def test_not_abbreviation(self):
        blk = make_block([(10, 100, 80, 112, "LEGEND")])
        assert _is_abbreviation_header(blk) is False

    def test_empty_rows(self):
        blk = BlockCluster(page=0, rows=[])
        assert _is_abbreviation_header(blk) is False


class TestParseAbbreviationEntries:
    """Tests for _parse_abbreviation_entries (boxed / single-block)."""

    def test_equals_sign_format(self):
        """'AI = AREA INLET' should parse code='AI', meaning='AREA INLET'."""
        blk = make_block(
            [
                (10, 10, 60, 22, "AI"),
                (65, 10, 75, 22, "="),
                (80, 10, 200, 22, "AREA INLET"),
            ],
        )
        entries = _parse_abbreviation_entries([blk], page=0)
        assert len(entries) >= 1
        assert entries[0].code == "AI"
        assert entries[0].meaning == "AREA INLET"

    def test_space_separated_format(self):
        """'BOC BACK OF CURB' should parse via regex fallback."""
        blk = make_block(
            [
                (10, 10, 40, 22, "BOC"),
                (50, 10, 200, 22, "BACK OF CURB"),
            ],
        )
        entries = _parse_abbreviation_entries([blk], page=0)
        assert len(entries) >= 1
        assert entries[0].code == "BOC"
        assert "BACK OF CURB" in entries[0].meaning

    def test_two_column_layout(self):
        """Two distinct x-groups with left being short codes."""
        code_blk = make_block(
            [
                (10, 10, 40, 22, "AI"),
                (10, 30, 40, 42, "BOC"),
            ],
        )
        meaning_blk = make_block(
            [
                (200, 10, 350, 22, "AREA INLET"),
                (200, 30, 350, 42, "BACK OF CURB"),
            ],
        )
        entries = _parse_abbreviation_entries([code_blk, meaning_blk], page=0)
        assert len(entries) == 2

    def test_empty_blocks(self):
        entries = _parse_abbreviation_entries([], page=0)
        assert entries == []


class TestParseAbbreviationEntriesTwoColumn:
    """Tests for _parse_abbreviation_entries_two_column."""

    def test_matched_rows(self):
        code_blk = make_block(
            [
                (10, 10, 40, 22, "AI"),
                (10, 30, 40, 42, "BOC"),
            ],
        )
        meaning_blk = make_block(
            [
                (200, 10, 350, 22, "AREA INLET"),
                (200, 30, 350, 42, "BACK OF CURB"),
            ],
        )
        entries = _parse_abbreviation_entries_two_column(
            [code_blk],
            [meaning_blk],
            page=0,
        )
        assert len(entries) == 2
        assert entries[0].code == "AI"
        assert entries[0].meaning == "AREA INLET"
        assert entries[1].code == "BOC"
        assert entries[1].meaning == "BACK OF CURB"

    def test_unmatched_rows(self):
        """Rows at very different y-positions should not match."""
        code_blk = make_block([(10, 10, 40, 22, "AI")])
        meaning_blk = make_block([(200, 100, 350, 112, "FAR AWAY")])
        entries = _parse_abbreviation_entries_two_column(
            [code_blk],
            [meaning_blk],
            page=0,
        )
        assert len(entries) == 0

    def test_empty_inputs(self):
        entries = _parse_abbreviation_entries_two_column([], [], page=0)
        assert entries == []


class TestDetectAbbreviationRegions:
    """Integration tests for detect_abbreviation_regions."""

    def test_boxed_abbreviation(self):
        header = make_block([(100, 100, 250, 112, "ABBREVIATIONS")])
        rect = make_graphic("rect", 80, 80, 400, 300)
        code_blk = make_block(
            [
                (100, 130, 140, 142, "AI"),
                (145, 130, 155, 142, "="),
                (160, 130, 300, 142, "AREA INLET"),
            ],
        )
        regions = detect_abbreviation_regions(
            blocks=[header, code_blk],
            graphics=[rect],
            page_width=612,
            page_height=792,
        )
        assert len(regions) == 1
        assert regions[0].is_boxed is True

    def test_no_abbreviation_headers(self):
        blk = make_block([(10, 10, 80, 22, "NOTES:")], is_header=True)
        regions = detect_abbreviation_regions(
            blocks=[blk],
            graphics=[],
            page_width=612,
            page_height=792,
        )
        assert regions == []

    def test_emits_debug_logs(self, caplog):
        header = make_block([(100, 100, 250, 112, "ABBREVIATIONS")])
        with caplog.at_level(logging.DEBUG, logger="plancheck.abbreviations"):
            detect_abbreviation_regions(
                blocks=[header],
                graphics=[],
                page_width=612,
                page_height=792,
            )
        assert any("detect_abbreviation_regions" in r.message for r in caplog.records)
