"""Tests for plancheck.zoning — page zone detection and block classification."""

import pytest
from conftest import make_box

from plancheck.analysis.zoning import (
    PageZone,
    ZoneTag,
    classify_blocks,
    detect_zones,
    zone_summary,
)
from plancheck.config import GroupingConfig
from plancheck.models import BlockCluster, NotesColumn, RowBand


class TestPageZone:
    def test_bbox(self):
        z = PageZone(tag=ZoneTag.page, x0=0, y0=0, x1=100, y1=200)
        assert z.bbox() == (0, 0, 100, 200)

    def test_area(self):
        z = PageZone(tag=ZoneTag.drawing, x0=0, y0=0, x1=100, y1=50)
        assert z.area() == 5000.0

    def test_contains_point(self):
        z = PageZone(tag=ZoneTag.notes, x0=500, y0=0, x1=700, y1=600)
        assert z.contains_point(600, 300) is True
        assert z.contains_point(100, 300) is False

    def test_overlap_fraction(self):
        z = PageZone(tag=ZoneTag.drawing, x0=0, y0=0, x1=100, y1=100)
        # Fully inside
        assert z.overlap_fraction((10, 10, 50, 50)) == 1.0
        # Half inside
        assert abs(z.overlap_fraction((50, 0, 150, 100)) - 0.5) < 1e-9
        # No overlap
        assert z.overlap_fraction((200, 200, 300, 300)) == 0.0

    def test_overlap_area(self):
        z = PageZone(tag=ZoneTag.drawing, x0=0, y0=0, x1=100, y1=100)
        assert z.overlap_area((50, 50, 150, 150)) == 2500.0


class TestDetectZones:
    def test_empty_page(self):
        zones = detect_zones(2448.0, 1584.0, [])
        assert len(zones) >= 1
        assert zones[0].tag == ZoneTag.page

    def test_page_zone_always_present(self):
        blocks = [
            BlockCluster(
                page=0,
                rows=[RowBand(page=0, boxes=[make_box(100, 100, 200, 112, "TEXT")])],
            )
        ]
        zones = detect_zones(2448.0, 1584.0, blocks)
        tags = {z.tag for z in zones}
        assert ZoneTag.page in tags

    def test_title_block_detected(self):
        """Blocks in the bottom 15% should produce a title_block zone."""
        # Page height 1000, title block at y=900 (bottom 10%)
        blocks = [
            BlockCluster(
                page=0,
                rows=[
                    RowBand(
                        page=0, boxes=[make_box(50, 900, 800, 920, "PROJECT TITLE")]
                    )
                ],
            ),
            BlockCluster(
                page=0,
                rows=[
                    RowBand(
                        page=0, boxes=[make_box(50, 930, 800, 950, "SHEET 1 OF 10")]
                    )
                ],
            ),
        ]
        zones = detect_zones(1000.0, 1000.0, blocks)
        tags = {z.tag for z in zones}
        assert ZoneTag.title_block in tags

    def test_notes_zone_detected(self):
        """Notes columns should produce a notes zone."""
        header_blk = BlockCluster(
            page=0,
            rows=[RowBand(page=0, boxes=[make_box(700, 50, 900, 65, "NOTES:")])],
            is_header=True,
        )
        notes_blk = BlockCluster(
            page=0,
            rows=[RowBand(page=0, boxes=[make_box(700, 70, 900, 82, "1. Do work")])],
            is_notes=True,
        )
        col = NotesColumn(page=0, header=header_blk, notes_blocks=[notes_blk])
        zones = detect_zones(1000.0, 1000.0, [], notes_columns=[col])
        tags = {z.tag for z in zones}
        assert ZoneTag.notes in tags

    def test_legend_zone_from_bbox(self):
        zones = detect_zones(
            1000.0,
            1000.0,
            [],
            legend_bboxes=[(500, 200, 900, 500)],
        )
        tags = {z.tag for z in zones}
        assert ZoneTag.legend in tags

    def test_abbreviation_zone_from_bbox(self):
        zones = detect_zones(
            1000.0,
            1000.0,
            [],
            abbreviation_bboxes=[(100, 300, 400, 600)],
        )
        tags = {z.tag for z in zones}
        assert ZoneTag.abbreviations in tags


class TestClassifyBlocks:
    def test_block_in_title(self):
        blocks = [
            BlockCluster(
                page=0,
                rows=[RowBand(page=0, boxes=[make_box(50, 900, 800, 920, "TITLE")])],
            ),
        ]
        zones = [
            PageZone(tag=ZoneTag.page, x0=0, y0=0, x1=1000, y1=1000),
            PageZone(tag=ZoneTag.title_block, x0=0, y0=850, x1=1000, y1=1000),
            PageZone(tag=ZoneTag.drawing, x0=0, y0=0, x1=1000, y1=850),
        ]
        result = classify_blocks(blocks, zones)
        assert result[0] == ZoneTag.title_block

    def test_block_in_drawing(self):
        blocks = [
            BlockCluster(
                page=0,
                rows=[RowBand(page=0, boxes=[make_box(100, 100, 200, 120, "DETAIL")])],
            ),
        ]
        zones = [
            PageZone(tag=ZoneTag.page, x0=0, y0=0, x1=1000, y1=1000),
            PageZone(tag=ZoneTag.title_block, x0=0, y0=850, x1=1000, y1=1000),
            PageZone(tag=ZoneTag.drawing, x0=0, y0=0, x1=1000, y1=850),
        ]
        result = classify_blocks(blocks, zones)
        assert result[0] == ZoneTag.drawing

    def test_unknown_when_no_overlap(self):
        blocks = [
            BlockCluster(
                page=0,
                rows=[RowBand(page=0, boxes=[make_box(100, 100, 200, 120, "X")])],
            ),
        ]
        # Only page zone — all other zones empty
        zones = [PageZone(tag=ZoneTag.page, x0=0, y0=0, x1=1000, y1=1000)]
        result = classify_blocks(blocks, zones)
        assert result[0] == ZoneTag.unknown


class TestZoneSummary:
    def test_summary_structure(self):
        zones = [
            PageZone(tag=ZoneTag.page, x0=0, y0=0, x1=100, y1=200, confidence=1.0),
        ]
        s = zone_summary(zones)
        assert "zones" in s
        assert len(s["zones"]) == 1
        assert s["zones"][0]["tag"] == "page"
        assert s["zones"][0]["confidence"] == 1.0
