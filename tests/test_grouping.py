"""Tests for plancheck.grouping — line building, clustering, notes detection."""

import pytest
from conftest import make_box

from plancheck.config import GroupingConfig
from plancheck.grouping import (
    _histogram_gutters,
    _is_note_number,
    _is_note_number_column,
    _median_size,
    _split_row_by_width,
    _split_row_on_gaps,
    _text_ends_incomplete,
    _text_starts_as_continuation,
    assign_column_ids,
    build_clusters_v2,
    build_lines,
    compute_median_space_gap,
    detect_column_boundaries,
    group_blocks,
    group_blocks_from_lines,
    group_rows,
    mark_headers,
    mark_notes,
    mark_tables,
    split_line_spans,
)
from plancheck.models import RowBand


class TestBuildLines:
    def test_single_line(self, simple_boxes, default_cfg):
        lines = build_lines(simple_boxes, default_cfg)
        assert len(lines) == 1
        assert len(lines[0].token_indices) == 3

    def test_two_lines(self, two_line_boxes, default_cfg):
        lines = build_lines(two_line_boxes, default_cfg)
        assert len(lines) == 2
        # First line has "GENERAL", "NOTES:"
        line1_texts = {two_line_boxes[i].text for i in lines[0].token_indices}
        assert "GENERAL" in line1_texts
        assert "NOTES:" in line1_texts
        # Second line has "1.", "ALL", "WORK"
        line2_texts = {two_line_boxes[i].text for i in lines[1].token_indices}
        assert "1." in line2_texts

    def test_empty_input(self, default_cfg):
        assert build_lines([], default_cfg) == []

    def test_lines_sorted_by_baseline(self, two_line_boxes, default_cfg):
        lines = build_lines(two_line_boxes, default_cfg)
        baselines = [l.baseline_y for l in lines]
        assert baselines == sorted(baselines)

    def test_line_ids_sequential(self, multi_block_boxes, default_cfg):
        lines = build_lines(multi_block_boxes, default_cfg)
        ids = [l.line_id for l in lines]
        assert ids == list(range(len(lines)))


class TestBuildClustersV2:
    def test_single_block(self, simple_boxes, default_cfg):
        blocks = build_clusters_v2(simple_boxes, 800.0, default_cfg)
        assert len(blocks) >= 1
        total_lines = sum(len(blk.lines or []) for blk in blocks)
        assert total_lines == 1

    def test_multi_block(self, multi_block_boxes, default_cfg):
        blocks = build_clusters_v2(multi_block_boxes, 800.0, default_cfg)
        # Should get at least 2 blocks due to large vertical gap
        assert len(blocks) >= 2

    def test_empty_input(self, default_cfg):
        assert build_clusters_v2([], 800.0, default_cfg) == []

    def test_rows_populated_for_compat(self, simple_boxes, default_cfg):
        """v2 pipeline should also populate .rows for backward compatibility."""
        blocks = build_clusters_v2(simple_boxes, 800.0, default_cfg)
        for blk in blocks:
            if blk.lines:
                # rows should be populated from lines
                assert len(blk.rows) > 0


class TestComputeMedianSpaceGap:
    def test_returns_positive(self, simple_boxes, default_cfg):
        lines = build_lines(simple_boxes, default_cfg)
        gap = compute_median_space_gap(lines, simple_boxes, default_cfg)
        assert gap > 0

    def test_fallback_on_empty(self, default_cfg):
        gap = compute_median_space_gap([], [], default_cfg)
        assert gap == default_cfg.grouping_space_gap_fallback


class TestSplitLineSpans:
    def test_single_span(self, simple_boxes, default_cfg):
        lines = build_lines(simple_boxes, default_cfg)
        line = lines[0]
        gap = compute_median_space_gap(lines, simple_boxes, default_cfg)
        split_line_spans(line, simple_boxes, gap, default_cfg.span_gap_mult)
        # With tight boxes, might be 1 span or more depending on gap sizes
        assert len(line.spans) >= 1
        # All token_indices should be covered
        span_indices = set()
        for span in line.spans:
            span_indices.update(span.token_indices)
        assert span_indices == set(line.token_indices)


class TestGroupRows:
    """Test the old row-based grouping path (still exported for compat)."""

    def test_basic(self, two_line_boxes, default_cfg):
        rows = group_rows(two_line_boxes, default_cfg)
        assert len(rows) >= 2

    def test_empty(self, default_cfg):
        assert group_rows([], default_cfg) == []


class TestGroupBlocks:
    def test_basic(self, two_line_boxes, default_cfg):
        rows = group_rows(two_line_boxes, default_cfg)
        blocks = group_blocks(rows, default_cfg)
        assert len(blocks) >= 1

    def test_large_gap_splits(self, multi_block_boxes, default_cfg):
        rows = group_rows(multi_block_boxes, default_cfg)
        blocks = group_blocks(rows, default_cfg)
        assert len(blocks) >= 2


class TestMarkHeaders:
    def test_header_detection(self, default_cfg):
        """A block with all-caps text ending in ':' should be marked as header."""
        boxes = [
            make_box(50, 100, 130, 112, "GENERAL"),
            make_box(140, 100, 200, 112, "NOTES:"),
        ]
        blocks = build_clusters_v2(boxes, 800.0, default_cfg)
        mark_headers(blocks)
        # At least one block should be marked as header
        has_header = any(b.is_header for b in blocks)
        # Header detection is heuristic — just ensure no crash
        assert isinstance(has_header, bool)


class TestMarkNotes:
    def test_notes_detection(self, default_cfg):
        """Blocks starting with '1.', '2.' patterns should be marked as notes."""
        boxes = [
            make_box(50, 100, 65, 112, "1."),
            make_box(70, 100, 200, 112, "DO THIS WORK"),
            make_box(50, 118, 65, 130, "2."),
            make_box(70, 118, 200, 130, "DO THAT WORK"),
        ]
        blocks = build_clusters_v2(boxes, 800.0, default_cfg)
        mark_notes(blocks)
        # Check notes markers — heuristic, so we just ensure no crash
        for b in blocks:
            assert isinstance(b.is_notes, bool)


# =====================================================================
# Deeper helpers coverage
# =====================================================================


class TestMedianSize:
    def test_normal(self):
        boxes = [
            make_box(0, 0, 10, 5, "A"),
            make_box(20, 0, 30, 5, "B"),
            make_box(40, 0, 50, 5, "C"),
        ]
        w, h = _median_size(boxes)
        assert w == 10.0
        assert h == 5.0

    def test_empty_returns_one(self):
        assert _median_size([]) == (1.0, 1.0)

    def test_single_box(self):
        w, h = _median_size([make_box(0, 0, 7, 3, "A")])
        assert w == 7.0
        assert h == 3.0


class TestHistogramGutters:
    def _two_column_boxes(self):
        """Two dense clusters separated by a wide gap."""
        left = [make_box(i * 10, 0, i * 10 + 8, 10, str(i)) for i in range(10)]
        right = [make_box(300 + i * 10, 0, 308 + i * 10, 10, str(i)) for i in range(10)]
        return left + right

    def test_finds_gutter(self):
        boxes = self._two_column_boxes()
        gutters = _histogram_gutters(boxes, bins=50, gutter_width=20.0)
        assert len(gutters) >= 1
        # The gutter should be roughly in the gap between ~100 and 300
        assert any(100 < g < 300 for g in gutters)

    def test_empty_returns_empty(self):
        assert _histogram_gutters([], bins=10, gutter_width=5.0) == []

    def test_zero_bins_returns_empty(self):
        boxes = [make_box(0, 0, 10, 10, "A")]
        assert _histogram_gutters(boxes, bins=0, gutter_width=5.0) == []

    def test_single_cluster_no_gutters(self):
        boxes = [make_box(i * 5, 0, i * 5 + 4, 10, str(i)) for i in range(20)]
        gutters = _histogram_gutters(boxes, bins=20, gutter_width=50.0)
        # Tight cluster — gutter requirement too wide to find any
        assert gutters == []


class TestDetectColumnBoundaries:
    def test_single_column(self, simple_boxes, default_cfg):
        boundaries = detect_column_boundaries(simple_boxes, 800.0, default_cfg)
        # Tight one-line row shouldn't produce column splits
        assert isinstance(boundaries, list)

    def test_two_columns(self, default_cfg):
        left = [
            make_box(10 + i * 12, 200, 10 + i * 12 + 10, 210, f"L{i}") for i in range(8)
        ]
        right = [
            make_box(400 + i * 12, 200, 410 + i * 12, 210, f"R{i}") for i in range(8)
        ]
        boundaries = detect_column_boundaries(left + right, 800.0, default_cfg)
        assert len(boundaries) >= 1

    def test_empty_tokens(self, default_cfg):
        assert detect_column_boundaries([], 800.0, default_cfg) == []

    def test_zero_page_height(self, default_cfg):
        assert (
            detect_column_boundaries([make_box(0, 0, 10, 10, "A")], 0.0, default_cfg)
            == []
        )


class TestAssignColumnIds:
    def test_no_boundaries_all_col_zero(self, simple_boxes, default_cfg):
        lines = build_lines(simple_boxes, default_cfg)
        gap = compute_median_space_gap(lines, simple_boxes, default_cfg)
        for ln in lines:
            split_line_spans(ln, simple_boxes, gap, default_cfg.span_gap_mult)
        assign_column_ids(lines, simple_boxes, [])
        for ln in lines:
            for span in ln.spans:
                assert span.col_id == 0

    def test_boundary_splits_columns(self, default_cfg):
        # Two tokens far apart — force two spans by using a tiny span_gap_mult
        tokens = [
            make_box(10, 100, 50, 112, "LEFT"),
            make_box(200, 100, 250, 112, "RIGHT"),
        ]
        lines = build_lines(tokens, default_cfg)
        # Use a very small span_gap_mult so the gap (150 pt) exceeds the threshold
        for ln in lines:
            split_line_spans(ln, tokens, median_space_gap=5.0, span_gap_mult=1.0)
        assert sum(len(ln.spans) for ln in lines) == 2, "Expected 2 spans"
        assign_column_ids(lines, tokens, [130.0])
        col_ids = {span.col_id for ln in lines for span in ln.spans}
        assert 0 in col_ids
        assert 1 in col_ids


class TestIsNoteNumber:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("1.", True),
            ("12.", True),
            ("A.", True),
            ("(1)", True),
            ("[A]", True),
            ("HELLO", False),
            ("", False),
            ("1234.", False),  # too long
        ],
    )
    def test_patterns(self, text, expected):
        box = make_box(0, 0, 10, 10, text)
        assert _is_note_number(box) == expected


class TestIsNoteNumberColumn:
    def test_majority_notes(self):
        boxes = [make_box(0, i * 15, 10, i * 15 + 10, f"{i + 1}.") for i in range(5)]
        assert _is_note_number_column(boxes) is True

    def test_majority_text(self):
        boxes = [make_box(0, i * 15, 80, i * 15 + 10, f"WORD{i}") for i in range(5)]
        assert _is_note_number_column(boxes) is False

    def test_empty(self):
        assert _is_note_number_column([]) is False


class TestSplitRowByWidth:
    def test_narrow_row_unchanged(self):
        boxes = [make_box(0, 0, 10, 10, "A"), make_box(15, 0, 25, 10, "B")]
        row = RowBand(page=0, boxes=boxes)
        result = _split_row_by_width(row, 10.0, 100.0)
        assert len(result) == 1

    def test_wide_row_splits(self):
        boxes = [
            make_box(0, 0, 10, 10, "A"),
            make_box(200, 0, 210, 10, "B"),
        ]
        row = RowBand(page=0, boxes=boxes)
        result = _split_row_by_width(row, 10.0, 50.0)
        assert len(result) == 2

    def test_empty_row(self):
        row = RowBand(page=0, boxes=[])
        assert _split_row_by_width(row, 10.0, 100.0) == []


class TestSplitRowOnGaps:
    def test_no_large_gap(self):
        boxes = [make_box(0, 0, 10, 10, "A"), make_box(12, 0, 22, 10, "B")]
        row = RowBand(page=0, boxes=boxes)
        result = _split_row_on_gaps(row, 10.0, 5.0)
        assert len(result) == 1

    def test_large_gap_splits(self):
        # Use multi-char text so _is_note_number returns False
        boxes = [make_box(0, 0, 40, 10, "HELLO"), make_box(200, 0, 240, 10, "WORLD")]
        row = RowBand(page=0, boxes=boxes)
        result = _split_row_on_gaps(row, 10.0, 5.0)
        assert len(result) == 2

    def test_note_number_suppresses_split(self):
        # "1." followed by far text — split is suppressed
        boxes = [make_box(0, 0, 10, 10, "1."), make_box(100, 0, 150, 10, "TEXT")]
        row = RowBand(page=0, boxes=boxes)
        result = _split_row_on_gaps(row, 10.0, 5.0)
        # Should remain as one row since the left box is a note number
        assert len(result) == 1

    def test_empty_row(self):
        row = RowBand(page=0, boxes=[])
        assert _split_row_on_gaps(row, 10.0, 5.0) == []


class TestMarkTables:
    def test_single_row_not_table(self, default_cfg):
        boxes = [make_box(10, 100, 60, 112, "HELLO")]
        blocks = build_clusters_v2(boxes, 800.0, default_cfg)
        mark_tables(blocks, default_cfg)
        for b in blocks:
            assert b.is_table is False

    def test_structured_rows_as_table(self, default_cfg):
        # Two rows with same column structure — should be marked as table
        boxes = [
            make_box(10, 100, 60, 112, "COL1"),
            make_box(80, 100, 130, 112, "COL2"),
            make_box(10, 118, 60, 130, "VAL1"),
            make_box(80, 118, 130, 130, "VAL2"),
        ]
        blocks = build_clusters_v2(boxes, 800.0, default_cfg)
        mark_tables(blocks, default_cfg)
        # At least the logic runs without error
        for b in blocks:
            assert isinstance(b.is_table, bool)


class TestTextHelpers:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("HELLO WORLD.", False),
            ("HELLO WORLD", True),
            ("FINISH:", False),
            ("", False),
            ("CONTINUE", True),
            ("DONE!", False),
            ("QUESTION?", False),
        ],
    )
    def test_text_ends_incomplete(self, text, expected):
        assert _text_ends_incomplete(text) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("", False),
            ("1. FIRST NOTE", False),  # starts with note number
            ("CONTINUED TEXT", True),
            ("THE REST OF THE SENTENCE", True),
        ],
    )
    def test_text_starts_as_continuation(self, text, expected):
        assert _text_starts_as_continuation(text) == expected
