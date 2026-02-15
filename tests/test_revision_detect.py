"""Unit tests for plancheck._revision_detect module.

Covers: _is_revision_header, _has_revision_column_headers, _is_column_header_row,
        _parse_revision_row, _parse_revision_entries, detect_revision_regions
"""

from __future__ import annotations

import logging

import pytest
from conftest import make_block, make_box, make_graphic

from plancheck._revision_detect import (
    _has_revision_column_headers,
    _is_column_header_row,
    _is_revision_header,
    _parse_revision_entries,
    _parse_revision_row,
    detect_revision_regions,
)
from plancheck.models import BlockCluster


class TestIsRevisionHeader:
    """Tests for _is_revision_header."""

    def test_revisions(self):
        blk = make_block([(10, 100, 120, 112, "REVISIONS")])
        assert _is_revision_header(blk) is True

    def test_revision_singular(self):
        blk = make_block([(10, 100, 120, 112, "REVISION")])
        assert _is_revision_header(blk) is True

    def test_revisions_colon(self):
        blk = make_block([(10, 100, 120, 112, "REVISIONS:")])
        assert _is_revision_header(blk) is True

    def test_lowercase_ignored(self):
        """Regression: header text is upper-cased internally."""
        blk = make_block([(10, 100, 120, 112, "Revisions")])
        assert _is_revision_header(blk) is True

    def test_not_revision(self):
        blk = make_block([(10, 100, 120, 112, "LEGEND")])
        assert _is_revision_header(blk) is False

    def test_extra_text_rejected(self):
        """Extra words after 'REVISIONS' should not match the strict regex."""
        blk = make_block([(10, 100, 200, 112, "REVISIONS HISTORY")])
        assert _is_revision_header(blk) is False

    def test_empty_rows(self):
        blk = BlockCluster(page=0, rows=[])
        assert _is_revision_header(blk) is False


class TestHasRevisionColumnHeaders:
    """Tests for _has_revision_column_headers."""

    def test_no_desc_date(self):
        blk = make_block(
            [
                (10, 100, 40, 112, "NO."),
                (50, 100, 180, 112, "DESCRIPTION"),
                (190, 100, 240, 112, "DATE"),
            ],
        )
        assert _has_revision_column_headers(blk) is True

    def test_desc_only(self):
        blk = make_block([(10, 100, 180, 112, "DESCRIPTION")])
        assert _has_revision_column_headers(blk) is True

    def test_no_and_date(self):
        blk = make_block(
            [(10, 100, 40, 112, "NO."), (50, 100, 100, 112, "DATE")],
        )
        assert _has_revision_column_headers(blk) is True

    def test_no_match(self):
        blk = make_block([(10, 100, 80, 112, "LEGEND")])
        assert _has_revision_column_headers(blk) is False


class TestIsColumnHeaderRow:
    """Tests for _is_column_header_row."""

    def test_no_description_date(self):
        assert _is_column_header_row("NO. DESCRIPTION DATE") is True

    def test_no_dot_date(self):
        assert _is_column_header_row("NO. DATE") is True

    def test_plain_text(self):
        assert _is_column_header_row("1 INITIAL SUBMISSION 01/15/2025") is False


class TestParseRevisionRow:
    """Tests for _parse_revision_row."""

    def test_basic_row(self):
        boxes = [
            make_box(10, 40, 30, 52, "1"),
            make_box(50, 40, 180, 52, "INITIAL"),
            make_box(190, 40, 260, 52, "01/15/2025"),
        ]
        entry = _parse_revision_row(boxes, page=0, row_bbox=(10, 40, 260, 52))
        assert entry is not None
        assert entry.number == "1"
        assert entry.date == "01/15/2025"
        assert "INITIAL" in entry.description

    def test_no_date(self):
        boxes = [
            make_box(10, 40, 30, 52, "2"),
            make_box(50, 40, 180, 52, "REVISED GRADING"),
        ]
        entry = _parse_revision_row(boxes, page=0, row_bbox=(10, 40, 180, 52))
        assert entry is not None
        assert entry.number == "2"
        assert entry.date == ""

    def test_empty_boxes(self):
        entry = _parse_revision_row([], page=0, row_bbox=(0, 0, 0, 0))
        assert entry is None


class TestParseRevisionEntries:
    """Tests for _parse_revision_entries."""

    def test_with_column_headers_and_data(self):
        header_blk = make_block(
            [
                (10, 10, 40, 22, "NO."),
                (50, 10, 180, 22, "DESCRIPTION"),
                (190, 10, 240, 22, "DATE"),
            ],
        )
        data_blk = make_block(
            [
                (10, 40, 30, 52, "1"),
                (50, 40, 180, 52, "INITIAL SUBMISSION"),
                (190, 40, 260, 52, "01/15/2025"),
            ],
        )
        region_bbox = (0, 0, 300, 100)
        entries = _parse_revision_entries(
            [header_blk, data_blk], page=0, region_bbox=region_bbox
        )
        assert len(entries) >= 1
        assert entries[0].number == "1"

    def test_empty_blocks(self):
        entries = _parse_revision_entries([], page=0, region_bbox=(0, 0, 300, 100))
        assert entries == []


class TestDetectRevisionRegions:
    """Integration tests for detect_revision_regions."""

    def test_boxed_revision(self):
        header = make_block([(100, 100, 200, 112, "REVISIONS")])
        rect = make_graphic("rect", 80, 80, 400, 300)
        data_blk = make_block(
            [
                (100, 130, 120, 142, "1"),
                (140, 130, 300, 142, "INITIAL"),
                (310, 130, 380, 142, "01/15/2025"),
            ],
        )
        regions = detect_revision_regions(
            blocks=[header, data_blk],
            graphics=[rect],
            page_width=612,
            page_height=792,
        )
        assert len(regions) == 1
        assert regions[0].is_boxed is True

    def test_no_revision_headers(self):
        blk = make_block([(10, 10, 80, 22, "NOTES:")])
        regions = detect_revision_regions(
            blocks=[blk],
            graphics=[],
            page_width=612,
            page_height=792,
        )
        assert regions == []

    def test_emits_debug_logs(self, caplog):
        header = make_block([(100, 100, 200, 112, "REVISIONS")])
        with caplog.at_level(logging.DEBUG, logger="plancheck.legends"):
            detect_revision_regions(
                blocks=[header],
                graphics=[],
                page_width=612,
                page_height=792,
            )
        assert any("detect_revision_regions" in r.message for r in caplog.records)
