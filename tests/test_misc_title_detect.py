"""Unit tests for plancheck._misc_title_detect module.

Covers: _is_misc_title_text, detect_misc_title_regions
"""

from __future__ import annotations

import logging

import pytest
from conftest import make_block, make_box, make_graphic

from plancheck._misc_title_detect import _is_misc_title_text, detect_misc_title_regions


class TestIsMiscTitleText:
    """Tests for _is_misc_title_text."""

    def test_department_of_transportation(self):
        assert _is_misc_title_text("OKLAHOMA DEPARTMENT OF TRANSPORTATION") is True

    def test_state_of(self):
        assert _is_misc_title_text("STATE OF OKLAHOMA") is True

    def test_county_of(self):
        assert _is_misc_title_text("COUNTY OF BRYAN") is True

    def test_city_of(self):
        assert _is_misc_title_text("CITY OF TULSA") is True

    def test_public_works(self):
        assert _is_misc_title_text("PUBLIC WORKS DEPARTMENT") is True

    def test_not_misc(self):
        assert _is_misc_title_text("GENERAL NOTES:") is False

    def test_too_long(self):
        long_text = "A" * 101
        assert _is_misc_title_text(long_text) is False


class TestDetectMiscTitleRegions:
    """Integration tests for detect_misc_title_regions."""

    def test_finds_odot_title(self):
        blk = make_block(
            [(100, 100, 450, 112, "OKLAHOMA DEPARTMENT OF TRANSPORTATION")],
        )
        regions = detect_misc_title_regions(
            blocks=[blk],
            graphics=[],
            page_width=612,
            page_height=792,
        )
        assert len(regions) == 1
        assert "OKLAHOMA" in regions[0].text

    def test_excludes_non_matching(self):
        blk = make_block([(10, 10, 150, 22, "GENERAL NOTES:")])
        regions = detect_misc_title_regions(
            blocks=[blk],
            graphics=[],
            page_width=612,
            page_height=792,
        )
        assert regions == []

    def test_skips_large_blocks(self):
        """Blocks with >2 rows should be skipped."""
        blk = make_block(
            [
                (10, 10, 350, 22, "OKLAHOMA DEPARTMENT OF TRANSPORTATION"),
                (10, 30, 350, 42, "SECOND LINE"),
                (10, 50, 350, 62, "THIRD LINE"),
            ],
        )
        regions = detect_misc_title_regions(
            blocks=[blk],
            graphics=[],
            page_width=612,
            page_height=792,
        )
        assert regions == []

    def test_respects_exclusion_zones(self):
        blk = make_block(
            [(100, 100, 450, 112, "OKLAHOMA DEPARTMENT OF TRANSPORTATION")],
        )
        regions = detect_misc_title_regions(
            blocks=[blk],
            graphics=[],
            page_width=612,
            page_height=792,
            exclusion_zones=[(90, 90, 460, 120)],  # covers block
        )
        assert regions == []

    def test_emits_debug_logs(self, caplog):
        blk = make_block(
            [(100, 100, 450, 112, "OKLAHOMA DEPARTMENT OF TRANSPORTATION")],
        )
        with caplog.at_level(logging.DEBUG, logger="plancheck.legends"):
            detect_misc_title_regions(
                blocks=[blk],
                graphics=[],
                page_width=612,
                page_height=792,
            )
        assert any("detect_misc_title_regions" in r.message for r in caplog.records)
