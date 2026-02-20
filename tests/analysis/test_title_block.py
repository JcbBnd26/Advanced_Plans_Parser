"""Tests for the title-block parser module."""

from __future__ import annotations

import pytest

from plancheck.analysis.title_block import (
    TitleBlockField,
    TitleBlockInfo,
    _group_into_rows,
    extract_title_blocks,
    parse_title_block,
)
from plancheck.models import BlockCluster, GlyphBox, RowBand

# ── Helpers ────────────────────────────────────────────────────────────


def _make_block(texts: list[str], y_start: float = 0, page: int = 0) -> BlockCluster:
    """Build a BlockCluster from a list of row-text strings."""
    rows = []
    y = y_start
    for t in texts:
        box = GlyphBox(page=page, x0=10, y0=y, x1=200, y1=y + 12, text=t)
        rows.append(RowBand(page=page, boxes=[box]))
        y += 14
    return BlockCluster(page=page, rows=rows)


def _tb_box(page=0, box_type="title_block"):
    """Minimal mock structural box for title-block tests."""

    class _Box:
        def __init__(self):
            self.box_type_value = box_type
            self.x0, self.y0, self.x1, self.y1 = 400, 0, 612, 792

        @property
        def box_type(self):
            from plancheck.analysis.structural_boxes import BoxType

            return BoxType(self.box_type_value)

        def bbox(self):
            return (self.x0, self.y0, self.x1, self.y1)

    return _Box()


# ── TitleBlockInfo tests ───────────────────────────────────────────────


class TestTitleBlockInfo:
    def test_empty(self):
        info = TitleBlockInfo(page=0)
        assert info.project_name == ""
        assert info.sheet_number == ""
        assert info.date == ""
        assert info.scale == ""
        assert info.engineer == ""
        assert info.drawing_number == ""
        assert info.sheet_title == ""

    def test_get_default(self):
        info = TitleBlockInfo(page=0)
        assert info.get("missing_field", "fallback") == "fallback"

    def test_get_existing(self):
        info = TitleBlockInfo(
            page=0,
            fields=[TitleBlockField(label="project_name", value="Test Project")],
        )
        assert info.get("project_name") == "Test Project"
        assert info.project_name == "Test Project"

    def test_to_dict(self):
        info = TitleBlockInfo(
            page=1,
            bbox=(100, 200, 300, 400),
            confidence=0.75,
            raw_text="HELLO",
            fields=[
                TitleBlockField(
                    label="sheet_number",
                    value="C-1",
                    bbox=(110, 210, 190, 225),
                    confidence=0.85,
                )
            ],
        )
        d = info.to_dict()
        assert d["page"] == 1
        assert d["bbox"] == [100, 200, 300, 400]
        assert d["confidence"] == 0.75
        assert len(d["fields"]) == 1
        assert d["fields"][0]["label"] == "sheet_number"
        assert d["fields"][0]["value"] == "C-1"


# ── parse_title_block tests ───────────────────────────────────────────


class TestParseTitleBlock:
    def test_empty_blocks(self):
        info = parse_title_block(contained_blocks=[], page=0, box_bbox=(0, 0, 612, 792))
        assert info.fields == []
        assert info.raw_text == ""

    def test_project_name_extraction(self):
        blk = _make_block(["PROJECT NAME: Highway 66 Widening"])
        info = parse_title_block(
            contained_blocks=[blk], page=0, box_bbox=(0, 0, 612, 792)
        )
        assert any(f.label == "project_name" for f in info.fields)
        pn = info.project_name
        assert "Highway 66 Widening" in pn

    def test_sheet_number_extraction(self):
        blk = _make_block(["SHEET NO. C-1"])
        info = parse_title_block(
            contained_blocks=[blk], page=0, box_bbox=(0, 0, 612, 792)
        )
        assert info.sheet_number.strip() == "C-1"

    def test_date_extraction(self):
        blk = _make_block(["DATE: 01/15/2024"])
        info = parse_title_block(
            contained_blocks=[blk], page=0, box_bbox=(0, 0, 612, 792)
        )
        assert "01/15/2024" in info.date

    def test_scale_extraction_label(self):
        blk = _make_block(["SCALE: 1\" = 40'"])
        info = parse_title_block(
            contained_blocks=[blk], page=0, box_bbox=(0, 0, 612, 792)
        )
        assert info.scale != ""

    def test_scale_extraction_pattern(self):
        """Detect scale by pattern when no label present."""
        blk = _make_block(["1\" = 40'"])
        info = parse_title_block(
            contained_blocks=[blk], page=0, box_bbox=(0, 0, 612, 792)
        )
        assert info.scale != ""

    def test_nts_scale(self):
        blk = _make_block(["NTS"])
        info = parse_title_block(
            contained_blocks=[blk], page=0, box_bbox=(0, 0, 612, 792)
        )
        assert info.scale.upper() == "NTS"

    def test_engineer_extraction(self):
        blk = _make_block(["ENGINEER: John Smith"])
        info = parse_title_block(
            contained_blocks=[blk], page=0, box_bbox=(0, 0, 612, 792)
        )
        assert "John Smith" in info.engineer

    def test_drawing_number(self):
        blk = _make_block(["DWG NO. 12345"])
        info = parse_title_block(
            contained_blocks=[blk], page=0, box_bbox=(0, 0, 612, 792)
        )
        assert "12345" in info.drawing_number

    def test_multiple_fields(self):
        blk = _make_block(
            [
                "PROJECT NAME: Highway 66",
                "SHEET NO. C-3",
                "DATE: 03/20/2024",
                "SCALE: 1\" = 20'",
                "ENGINEER: Jane Doe",
            ]
        )
        info = parse_title_block(
            contained_blocks=[blk], page=0, box_bbox=(0, 0, 612, 792)
        )
        assert info.project_name != ""
        assert info.sheet_number != ""
        assert info.date != ""
        assert info.scale != ""
        assert info.engineer != ""
        # With 4 of 4 important fields found, confidence should be 1.0
        assert info.confidence == 1.0

    def test_value_on_next_line(self):
        """When label and value are on separate lines."""
        blk = _make_block(["PROJECT NAME:", "Highway 66 Reconstruction"])
        info = parse_title_block(
            contained_blocks=[blk], page=0, box_bbox=(0, 0, 612, 792)
        )
        assert "Highway 66 Reconstruction" in info.project_name

    def test_confidence_calculation(self):
        # Only date found → 1/4
        blk = _make_block(["DATE: 01/01/2024"])
        info = parse_title_block(
            contained_blocks=[blk], page=0, box_bbox=(0, 0, 612, 792)
        )
        assert info.confidence == 0.25

    def test_date_pattern_detection(self):
        """Date found by pattern (no label) in pass 2."""
        blk = _make_block(["03/15/2024"])
        info = parse_title_block(
            contained_blocks=[blk], page=0, box_bbox=(0, 0, 612, 792)
        )
        assert info.date != ""

    def test_no_duplicate_fields(self):
        blk = _make_block(
            [
                "PROJECT NAME: Alpha",
                "PROJECT TITLE: Beta",
            ]
        )
        info = parse_title_block(
            contained_blocks=[blk], page=0, box_bbox=(0, 0, 612, 792)
        )
        pn_fields = [f for f in info.fields if f.label == "project_name"]
        assert len(pn_fields) == 1

    def test_checker_and_designer(self):
        blk = _make_block(
            [
                "CHECKED BY: A. Jones",
                "DRAWN BY: B. Smith",
            ]
        )
        info = parse_title_block(
            contained_blocks=[blk], page=0, box_bbox=(0, 0, 612, 792)
        )
        assert info.get("checker") != ""
        assert info.get("designer") != ""

    def test_revision_date_vs_date(self):
        blk = _make_block(
            [
                "DATE: 01/01/2024",
                "REVISION DATE: 06/15/2024",
            ]
        )
        info = parse_title_block(
            contained_blocks=[blk], page=0, box_bbox=(0, 0, 612, 792)
        )
        assert info.date == "01/01/2024"
        assert info.get("revision_date") == "06/15/2024"


# ── extract_title_blocks tests ────────────────────────────────────────


class TestExtractTitleBlocks:
    def test_no_title_blocks(self):
        sb = _tb_box(box_type="unknown")
        result = extract_title_blocks(structural_boxes=[sb], blocks=[], page=0)
        assert result == []

    def test_with_title_block(self):
        sb = _tb_box()
        blk = _make_block(["PROJECT NAME: Test"], y_start=100)
        # Place block centre inside the title block bbox
        for row in blk.rows:
            for b in row.boxes:
                b.x0 = 450
                b.x1 = 600
                b.y0 = 100
                b.y1 = 112
        result = extract_title_blocks(structural_boxes=[sb], blocks=[blk], page=0)
        assert len(result) == 1
        assert result[0].project_name != ""

    def test_block_outside_title_block_ignored(self):
        sb = _tb_box()
        blk = _make_block(["PROJECT NAME: Test"], y_start=100)
        # Place block outside
        for row in blk.rows:
            for b in row.boxes:
                b.x0 = 10
                b.x1 = 100
        result = extract_title_blocks(structural_boxes=[sb], blocks=[blk], page=0)
        assert len(result) == 1
        assert result[0].project_name == ""  # No contained blocks


# ── _group_into_rows tests ────────────────────────────────────────────


class TestGroupIntoRows:
    def test_empty(self):
        assert _group_into_rows([]) == []

    def test_single_row(self):
        boxes = [
            GlyphBox(page=0, x0=10, y0=100, x1=50, y1=112, text="A"),
            GlyphBox(page=0, x0=60, y0=101, x1=100, y1=113, text="B"),
        ]
        rows = _group_into_rows(boxes)
        assert len(rows) == 1
        assert len(rows[0]) == 2

    def test_two_rows(self):
        boxes = [
            GlyphBox(page=0, x0=10, y0=100, x1=50, y1=112, text="A"),
            GlyphBox(page=0, x0=10, y0=130, x1=50, y1=142, text="B"),
        ]
        rows = _group_into_rows(boxes)
        assert len(rows) == 2

    def test_sorted_left_to_right(self):
        boxes = [
            GlyphBox(page=0, x0=60, y0=100, x1=100, y1=112, text="B"),
            GlyphBox(page=0, x0=10, y0=101, x1=50, y1=113, text="A"),
        ]
        rows = _group_into_rows(boxes)
        assert rows[0][0].text == "A"
        assert rows[0][1].text == "B"
