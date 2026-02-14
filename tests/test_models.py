"""Tests for plancheck.models — data model invariants."""

from conftest import make_box

from plancheck.models import BlockCluster, GlyphBox, Line, RowBand, Span


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


class TestSpan:
    def test_bbox_and_text(self):
        tokens = [make_box(10, 100, 50, 112, "A"), make_box(60, 100, 120, 112, "B")]
        span = Span(token_indices=[0, 1])
        assert span.bbox(tokens) == (10, 100, 120, 112)
        assert span.text(tokens) == "A B"


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
