"""Tests for plancheck.models — data model invariants."""

from conftest import make_block, make_box

from plancheck.models import (
    AbbreviationEntry,
    AbbreviationRegion,
    BlockCluster,
    GlyphBox,
    GraphicElement,
    LegendEntry,
    LegendRegion,
    Line,
    MiscTitleRegion,
    NotesColumn,
    RevisionEntry,
    RevisionRegion,
    RowBand,
    Span,
    StandardDetailEntry,
    StandardDetailRegion,
    SuspectRegion,
)


class TestGlyphBox:
    def test_bbox(self):
        b = make_box(10, 20, 30, 40, "text")
        assert b.bbox() == (10, 20, 30, 40)

    def test_width_height_area(self):
        b = make_box(0, 0, 10, 5)
        assert b.width() == 10.0
        assert b.height() == 5.0
        assert b.area() == 50.0

    def test_zero_area(self):
        b = make_box(5, 5, 5, 10)
        assert b.width() == 0.0
        assert b.area() == 0.0

    def test_negative_dims_clamps(self):
        b = make_box(10, 10, 5, 5)
        # area uses max(0, ...) so negative is clamped
        assert b.area() == 0.0

    def test_to_dict_round_trip(self):
        b = GlyphBox(
            page=1,
            x0=10.1234,
            y0=20.5678,
            x1=30.9,
            y1=40.1,
            text="HELLO",
            origin="text",
            fontname="Arial",
            font_size=12.5,
        )
        d = b.to_dict()
        restored = GlyphBox.from_dict(d)
        assert restored.text == "HELLO"
        assert restored.page == 1
        assert restored.fontname == "Arial"
        assert restored.font_size == 12.5
        assert restored.origin == "text"
        # Coordinates are rounded to 3dp
        assert abs(restored.x0 - 10.123) < 0.001

    def test_from_dict_defaults(self):
        d = {"page": 0, "x0": 0, "y0": 0, "x1": 10, "y1": 10}
        b = GlyphBox.from_dict(d)
        assert b.text == ""
        assert b.origin == "text"
        assert b.fontname == ""
        assert b.font_size == 0.0


class TestLine:
    def test_bbox_from_tokens(self):
        tokens = [make_box(10, 100, 50, 112, "A"), make_box(60, 100, 120, 112, "B")]
        line = Line(line_id=0, page=0, token_indices=[0, 1])
        assert line.bbox(tokens) == (10, 100, 120, 112)

    def test_text_sorted_by_x(self):
        tokens = [
            make_box(60, 100, 120, 112, "WORLD"),
            make_box(10, 100, 50, 112, "HELLO"),
        ]
        line = Line(line_id=0, page=0, token_indices=[0, 1])
        # text() sorts by x0 → "HELLO WORLD"
        assert line.text(tokens) == "HELLO WORLD"

    def test_empty_line(self):
        line = Line(line_id=0, page=0, token_indices=[])
        assert line.bbox([]) == (0, 0, 0, 0)
        assert line.text([]) == ""

    def test_to_dict_round_trip(self):
        line = Line(line_id=5, page=1, token_indices=[0, 1, 2], baseline_y=106.5)
        line.spans = [
            Span(token_indices=[0, 1], col_id=0),
            Span(token_indices=[2], col_id=1),
        ]
        d = line.to_dict()
        restored = Line.from_dict(d)
        assert restored.line_id == 5
        assert restored.page == 1
        assert restored.token_indices == [0, 1, 2]
        assert restored.baseline_y == 106.5
        assert len(restored.spans) == 2
        assert restored.spans[0].col_id == 0


class TestSpan:
    def test_bbox_and_text(self):
        tokens = [make_box(10, 100, 50, 112, "A"), make_box(60, 100, 120, 112, "B")]
        span = Span(token_indices=[0, 1])
        assert span.bbox(tokens) == (10, 100, 120, 112)
        assert span.text(tokens) == "A B"

    def test_empty_span(self):
        span = Span(token_indices=[])
        assert span.bbox([]) == (0, 0, 0, 0)
        assert span.text([]) == ""

    def test_to_dict_round_trip(self):
        span = Span(token_indices=[3, 4], col_id=2)
        d = span.to_dict()
        restored = Span.from_dict(d)
        assert restored.token_indices == [3, 4]
        assert restored.col_id == 2


class TestBlockCluster:
    def test_bbox_from_rows(self):
        row = RowBand(page=0, boxes=[make_box(10, 100, 50, 112, "A")])
        block = BlockCluster(page=0, rows=[row])
        assert block.bbox() == (10, 100, 50, 112)

    def test_bbox_from_lines(self):
        tokens = [make_box(10, 100, 50, 112, "A"), make_box(60, 100, 120, 112, "B")]
        line = Line(line_id=0, page=0, token_indices=[0, 1])
        block = BlockCluster(page=0, lines=[line], _tokens=tokens)
        assert block.bbox() == (10, 100, 120, 112)

    def test_get_all_boxes_from_lines(self):
        tokens = [make_box(60, 100, 120, 112, "B"), make_box(10, 100, 50, 112, "A")]
        line = Line(line_id=0, page=0, token_indices=[0, 1])
        block = BlockCluster(page=0, lines=[line], _tokens=tokens)
        boxes = block.get_all_boxes()
        # Sorted by (y0, x0)
        assert boxes[0].text == "A"
        assert boxes[1].text == "B"

    def test_populate_rows_from_lines(self):
        tokens = [make_box(10, 100, 50, 112, "A"), make_box(60, 100, 120, 112, "B")]
        line = Line(line_id=0, page=0, token_indices=[0, 1])
        block = BlockCluster(page=0, lines=[line], _tokens=tokens)
        block.populate_rows_from_lines()
        assert len(block.rows) == 1
        assert len(block.rows[0].boxes) == 2
        assert block.rows[0].boxes[0].text == "A"

    def test_to_dict_round_trip(self):
        tokens = [make_box(10, 100, 50, 112, "A"), make_box(60, 100, 120, 112, "B")]
        line = Line(line_id=0, page=0, token_indices=[0, 1], baseline_y=106.0)
        line.spans = [Span(token_indices=[0, 1])]
        block = BlockCluster(
            page=0, lines=[line], _tokens=tokens, is_header=True, label="header"
        )
        block.populate_rows_from_lines()

        d = block.to_dict()
        restored = BlockCluster.from_dict(d, tokens)
        assert restored.page == 0
        assert restored.is_header is True
        assert restored.label == "header"
        assert len(restored.lines) == 1
        assert len(restored.rows) > 0
        assert restored.bbox() == block.bbox()

    def test_empty_block_bbox(self):
        block = BlockCluster(page=0)
        assert block.bbox() == (0, 0, 0, 0)


class TestNotesColumn:
    def _make_column(self):
        tokens = [
            make_box(10, 100, 80, 112, "GENERAL"),
            make_box(90, 100, 160, 112, "NOTES:"),
            make_box(10, 130, 25, 142, "1."),
            make_box(30, 130, 200, 142, "SEE PLANS"),
        ]
        line0 = Line(line_id=0, page=0, token_indices=[0, 1], baseline_y=106.0)
        line0.spans = [Span(token_indices=[0, 1])]
        line1 = Line(line_id=1, page=0, token_indices=[2, 3], baseline_y=136.0)
        line1.spans = [Span(token_indices=[2, 3])]

        header = BlockCluster(page=0, lines=[line0], _tokens=tokens, is_header=True)
        header.populate_rows_from_lines()
        notes = BlockCluster(page=0, lines=[line1], _tokens=tokens, is_notes=True)
        notes.populate_rows_from_lines()

        blocks = [header, notes]
        col = NotesColumn(page=0, header=header, notes_blocks=[notes])
        return tokens, blocks, col

    def test_header_text(self):
        _, _, col = self._make_column()
        assert "GENERAL" in col.header_text()
        assert "NOTES" in col.header_text()

    def test_base_header_text_no_contd(self):
        _, _, col = self._make_column()
        base = col.base_header_text()
        assert "GENERAL" in base

    def test_is_continuation_false(self):
        _, _, col = self._make_column()
        assert col.is_continuation() is False

    def test_is_continuation_true(self):
        tokens = [make_box(10, 100, 200, 112, "GENERAL NOTES (CONT'D):")]
        line = Line(line_id=0, page=0, token_indices=[0], baseline_y=106.0)
        line.spans = [Span(token_indices=[0])]
        header = BlockCluster(page=0, lines=[line], _tokens=tokens, is_header=True)
        header.populate_rows_from_lines()
        col = NotesColumn(page=0, header=header)
        assert col.is_continuation() is True

    def test_bbox(self):
        _, _, col = self._make_column()
        x0, y0, x1, y1 = col.bbox()
        assert x0 <= 10
        assert y0 <= 100

    def test_all_blocks(self):
        _, _, col = self._make_column()
        assert len(col.all_blocks()) == 2

    def test_to_dict_round_trip(self):
        _, blocks, col = self._make_column()
        d = col.to_dict(blocks)
        restored = NotesColumn.from_dict(d, blocks)
        assert restored.header_text() == col.header_text()
        assert len(restored.notes_blocks) == len(col.notes_blocks)

    def test_empty_column(self):
        col = NotesColumn(page=0)
        assert col.header_text() == ""
        assert col.bbox() == (0, 0, 0, 0)
        assert col.all_blocks() == []
        assert col.is_continuation() is False


class TestGraphicElement:
    def test_bbox(self):
        g = GraphicElement(page=0, element_type="rect", x0=10, y0=20, x1=100, y1=200)
        assert g.bbox() == (10, 20, 100, 200)

    def test_area(self):
        g = GraphicElement(page=0, element_type="rect", x0=0, y0=0, x1=10, y1=5)
        assert g.area() == 50.0

    def test_is_small_symbol(self):
        small = GraphicElement(page=0, element_type="line", x0=0, y0=0, x1=20, y1=15)
        assert small.is_small_symbol() is True
        big = GraphicElement(page=0, element_type="rect", x0=0, y0=0, x1=100, y1=100)
        assert big.is_small_symbol() is False


class TestLegendEntry:
    def test_bbox_both(self):
        e = LegendEntry(
            page=0, symbol_bbox=(10, 20, 30, 40), description_bbox=(50, 20, 200, 40)
        )
        x0, y0, x1, y1 = e.bbox()
        assert x0 == 10
        assert x1 == 200

    def test_bbox_empty(self):
        e = LegendEntry(page=0)
        assert e.bbox() == (0, 0, 0, 0)


class TestAbbreviationEntry:
    def test_bbox_with_bboxes(self):
        e = AbbreviationEntry(
            page=0,
            code="AI",
            meaning="AREA INLET",
            code_bbox=(10, 100, 30, 112),
            meaning_bbox=(50, 100, 200, 112),
        )
        x0, y0, x1, y1 = e.bbox()
        assert x0 == 10
        assert x1 == 200

    def test_bbox_empty(self):
        e = AbbreviationEntry(page=0)
        assert e.bbox() == (0, 0, 0, 0)


class TestRevisionEntry:
    def test_bbox_with_row(self):
        e = RevisionEntry(
            page=0, number="1", description="Initial", row_bbox=(10, 100, 500, 112)
        )
        assert e.bbox() == (10, 100, 500, 112)

    def test_bbox_no_row(self):
        e = RevisionEntry(page=0)
        assert e.bbox() == (0, 0, 0, 0)


class TestStandardDetailEntry:
    def test_bbox_combined(self):
        e = StandardDetailEntry(
            page=0,
            sheet_number="SS-1",
            sheet_bbox=(10, 100, 50, 112),
            description_bbox=(60, 100, 300, 112),
        )
        x0, y0, x1, y1 = e.bbox()
        assert x0 == 10
        assert x1 == 300

    def test_bbox_empty(self):
        e = StandardDetailEntry(page=0)
        assert e.bbox() == (0, 0, 0, 0)


class TestRegionHeaderText:
    """All *Region classes should return header text from header block."""

    def _header_block(self, text):
        row = RowBand(page=0, boxes=[make_box(10, 100, 200, 112, text)])
        return BlockCluster(page=0, rows=[row])

    def test_legend_region_header_text(self):
        r = LegendRegion(page=0, header=self._header_block("LEGEND"))
        assert r.header_text() == "LEGEND"

    def test_abbreviation_region_header_text(self):
        r = AbbreviationRegion(page=0, header=self._header_block("ABBREVIATIONS"))
        assert r.header_text() == "ABBREVIATIONS"

    def test_revision_region_header_text(self):
        r = RevisionRegion(page=0, header=self._header_block("REVISIONS"))
        assert r.header_text() == "REVISIONS"

    def test_standard_detail_header_text(self):
        r = StandardDetailRegion(page=0, header=self._header_block("STANDARD DETAILS"))
        assert r.header_text() == "STANDARD DETAILS"

    def test_no_header(self):
        assert LegendRegion(page=0).header_text() == ""
        assert AbbreviationRegion(page=0).header_text() == ""
        assert RevisionRegion(page=0).header_text() == ""
        assert StandardDetailRegion(page=0).header_text() == ""


class TestRegionBbox:
    """All *Region classes should return sensible bboxes."""

    def _header_block(self, x0, y0, x1, y1, text="HDR"):
        row = RowBand(page=0, boxes=[make_box(x0, y0, x1, y1, text)])
        return BlockCluster(page=0, rows=[row])

    def test_legend_region_bbox_from_box(self):
        r = LegendRegion(page=0, box_bbox=(10, 20, 300, 400))
        assert r.bbox() == (10, 20, 300, 400)

    def test_legend_region_bbox_from_entries(self):
        r = LegendRegion(
            page=0,
            header=self._header_block(10, 100, 200, 112),
            entries=[LegendEntry(page=0, description_bbox=(10, 120, 200, 132))],
        )
        x0, y0, x1, y1 = r.bbox()
        assert x0 == 10
        assert y0 == 100
        assert y1 == 132

    def test_empty_region_bbox(self):
        assert LegendRegion(page=0).bbox() == (0, 0, 0, 0)
        assert AbbreviationRegion(page=0).bbox() == (0, 0, 0, 0)
        assert RevisionRegion(page=0).bbox() == (0, 0, 0, 0)
        assert StandardDetailRegion(page=0).bbox() == (0, 0, 0, 0)


class TestMiscTitleRegion:
    def test_bbox_from_box(self):
        r = MiscTitleRegion(page=0, text="TITLE", box_bbox=(10, 20, 300, 40))
        assert r.bbox() == (10, 20, 300, 40)

    def test_bbox_from_block(self):
        row = RowBand(page=0, boxes=[make_box(10, 20, 300, 40, "TITLE")])
        block = BlockCluster(page=0, rows=[row])
        r = MiscTitleRegion(page=0, text="TITLE", text_block=block)
        assert r.bbox() == (10, 20, 300, 40)

    def test_bbox_empty(self):
        assert MiscTitleRegion(page=0).bbox() == (0, 0, 0, 0)


class TestSuspectRegion:
    def test_bbox(self):
        sr = SuspectRegion(
            page=0,
            x0=10,
            y0=20,
            x1=100,
            y1=40,
            word_text="TEST",
            context="ctx",
            reason="test",
        )
        assert sr.bbox() == (10, 20, 100, 40)

    def test_to_dict(self):
        sr = SuspectRegion(
            page=1,
            x0=10.5,
            y0=20.5,
            x1=100.5,
            y1=40.5,
            word_text="FUSED",
            context="header ctx",
            reason="compound",
        )
        d = sr.to_dict()
        assert d["page"] == 1
        assert d["word_text"] == "FUSED"
        assert d["reason"] == "compound"
        assert len(d["bbox"]) == 4
