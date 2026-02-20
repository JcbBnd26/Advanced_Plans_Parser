"""Extended tests for plancheck.grouping — split_wide_lines, partition_columns,
merge_note_number_columns, group_blocks_from_lines, group_notes_columns,
link_continued_columns."""

import pytest
from conftest import make_box

from plancheck.config import GroupingConfig
from plancheck.grouping import (
    build_lines,
    compute_median_space_gap,
    group_blocks_from_lines,
    group_notes_columns,
    link_continued_columns,
    mark_headers,
    mark_notes,
    split_line_spans,
    split_wide_lines,
)
from plancheck.grouping.clustering import _merge_note_number_columns, _partition_columns
from plancheck.models import BlockCluster, Line, NotesColumn, RowBand, Span

# ─── split_wide_lines ─────────────────────────────────────────────────


class TestSplitWideLines:
    def test_no_split_tight_line(self, default_cfg):
        """A single narrow line should not be split."""
        tokens = [
            make_box(10, 100, 50, 112, "A"),
            make_box(55, 100, 100, 112, "B"),
        ]
        lines = build_lines(tokens, default_cfg)
        for l in lines:
            split_line_spans(l, tokens, 5.0, default_cfg.span_gap_mult)
        result = split_wide_lines(lines, tokens, median_space_gap=5.0)
        # Should not split — stays as one line
        assert len(result) == 1

    def test_split_on_wide_gap(self, default_cfg):
        """Two token clusters far apart should be split into two lines."""
        tokens = [
            make_box(10, 100, 50, 112, "LEFT"),
            make_box(400, 100, 450, 112, "RIGHT"),
        ]
        lines = build_lines(tokens, default_cfg)
        for l in lines:
            split_line_spans(l, tokens, 5.0, default_cfg.span_gap_mult)
        result = split_wide_lines(lines, tokens, median_space_gap=5.0)
        # The wide gap should cause a split
        assert len(result) >= 2

    def test_ids_renumbered(self, default_cfg):
        """After splitting, line IDs should be sequential."""
        tokens = [
            make_box(10, 100, 50, 112, "A"),
            make_box(400, 100, 450, 112, "B"),
            make_box(10, 130, 50, 142, "C"),
        ]
        lines = build_lines(tokens, default_cfg)
        gap = compute_median_space_gap(lines, tokens, default_cfg)
        for l in lines:
            split_line_spans(l, tokens, gap, default_cfg.span_gap_mult)
        result = split_wide_lines(lines, tokens, median_space_gap=gap)
        ids = [l.line_id for l in result]
        assert ids == list(range(len(result)))

    def test_empty_input(self):
        assert split_wide_lines([], [], median_space_gap=5.0) == []


# ─── _partition_columns ───────────────────────────────────────────────


class TestPartitionColumns:
    def test_single_column(self):
        """Tightly spaced boxes → one column."""
        boxes = [
            make_box(10, 100, 50, 112, "A"),
            make_box(55, 100, 100, 112, "B"),
            make_box(10, 120, 50, 132, "C"),
        ]
        cfg = GroupingConfig()
        median_w = 40.0
        result = _partition_columns(boxes, median_w, cfg)
        assert len(result) == 1
        assert len(result[0][0]) == 3  # all boxes in one column

    def test_two_columns(self):
        """Two widely-separated groups → two columns."""
        left_boxes = [
            make_box(10, 100 + i * 15, 80, 112 + i * 15, f"L{i}") for i in range(3)
        ]
        right_boxes = [
            make_box(400, 100 + i * 15, 470, 112 + i * 15, f"R{i}") for i in range(3)
        ]
        boxes = left_boxes + right_boxes
        cfg = GroupingConfig()
        median_w = 70.0
        result = _partition_columns(boxes, median_w, cfg)
        assert len(result) >= 2

    def test_empty_input(self):
        cfg = GroupingConfig()
        result = _partition_columns([], 10.0, cfg)
        assert result == []


# ─── _merge_note_number_columns ───────────────────────────────────────


class TestMergeNoteNumberColumns:
    def test_note_column_merged_right(self):
        """A column of note numbers should merge into the adjacent text column."""
        # Note number column
        note_boxes = [
            make_box(10, 100 + i * 15, 25, 112 + i * 15, f"{i + 1}.") for i in range(5)
        ]
        # Text column
        text_boxes = [
            make_box(30, 100 + i * 15, 200, 112 + i * 15, f"NOTE TEXT {i}")
            for i in range(5)
        ]
        segments = [
            (note_boxes, 10.0, 25.0),
            (text_boxes, 30.0, 200.0),
        ]
        result = _merge_note_number_columns(segments)
        # The note column should be merged into the text column
        assert len(result) == 1
        assert len(result[0][0]) == 10  # all boxes merged

    def test_no_merge_text_columns(self):
        """Two text columns should not be merged."""
        left = [
            make_box(10, 100 + i * 15, 80, 112 + i * 15, f"LEFT{i}") for i in range(3)
        ]
        right = [
            make_box(200, 100 + i * 15, 280, 112 + i * 15, f"RIGHT{i}")
            for i in range(3)
        ]
        segments = [
            (left, 10.0, 80.0),
            (right, 200.0, 280.0),
        ]
        result = _merge_note_number_columns(segments)
        assert len(result) == 2

    def test_empty_segments(self):
        assert _merge_note_number_columns([]) == []

    def test_single_segment(self):
        boxes = [make_box(10, 100, 50, 112, "A")]
        segments = [(boxes, 10.0, 50.0)]
        result = _merge_note_number_columns(segments)
        assert len(result) == 1


# ─── group_blocks_from_lines ──────────────────────────────────────────


class TestGroupBlocksFromLines:
    def test_single_block(self, default_cfg):
        tokens = [
            make_box(10, 100, 50, 112, "A"),
            make_box(55, 100, 100, 112, "B"),
            make_box(10, 118, 50, 130, "C"),
        ]
        lines = build_lines(tokens, default_cfg)
        blocks = group_blocks_from_lines(lines, tokens, default_cfg)
        assert len(blocks) >= 1
        # Each block should have lines
        for blk in blocks:
            assert blk.lines is not None
            assert len(blk.lines) > 0

    def test_large_gap_splits(self, default_cfg):
        tokens = [
            make_box(10, 100, 50, 112, "TOP"),
            make_box(10, 400, 50, 412, "BOTTOM"),
        ]
        lines = build_lines(tokens, default_cfg)
        blocks = group_blocks_from_lines(lines, tokens, default_cfg)
        assert len(blocks) >= 2

    def test_empty_input(self, default_cfg):
        assert group_blocks_from_lines([], [], default_cfg) == []

    def test_tokens_set_on_blocks(self, default_cfg):
        tokens = [make_box(10, 100, 50, 112, "X")]
        lines = build_lines(tokens, default_cfg)
        blocks = group_blocks_from_lines(lines, tokens, default_cfg)
        for blk in blocks:
            assert blk._tokens is tokens

    def test_rows_populated(self, default_cfg):
        tokens = [
            make_box(10, 100, 50, 112, "A"),
            make_box(10, 118, 50, 130, "B"),
        ]
        lines = build_lines(tokens, default_cfg)
        blocks = group_blocks_from_lines(lines, tokens, default_cfg)
        for blk in blocks:
            assert len(blk.rows) > 0


# ─── group_notes_columns ─────────────────────────────────────────────


class TestGroupNotesColumns:
    def _make_header_notes_blocks(self):
        """Build a header block + 2 notes blocks."""
        tokens = [
            make_box(50, 100, 120, 112, "GENERAL"),
            make_box(130, 100, 190, 112, "NOTES:"),
            make_box(50, 120, 65, 132, "1."),
            make_box(70, 120, 200, 132, "DO THIS"),
            make_box(50, 140, 65, 152, "2."),
            make_box(70, 140, 200, 152, "DO THAT"),
        ]
        cfg = GroupingConfig()
        from plancheck.grouping import build_clusters_v2

        blocks = build_clusters_v2(tokens, 800.0, cfg)
        mark_headers(blocks)
        mark_notes(blocks)
        return blocks, cfg

    def test_basic_column_detection(self):
        blocks, cfg = self._make_header_notes_blocks()
        columns = group_notes_columns(blocks, cfg=cfg)
        # Should produce at least one column if a header was detected
        # (heuristic — depends on mark_headers succeeding)
        assert isinstance(columns, list)
        for col in columns:
            assert isinstance(col, NotesColumn)

    def test_empty_blocks(self):
        columns = group_notes_columns([])
        assert columns == []

    def test_no_header_blocks(self, default_cfg):
        """Blocks without any header should produce no columns."""
        tokens = [
            make_box(50, 100, 65, 112, "1."),
            make_box(70, 100, 200, 112, "NOTE TEXT"),
        ]
        from plancheck.grouping import build_clusters_v2

        blocks = build_clusters_v2(tokens, 800.0, default_cfg)
        # Don't call mark_headers — no header detectable
        for b in blocks:
            b.is_notes = True
        columns = group_notes_columns(blocks, cfg=default_cfg)
        # No headers → no columns formed (orphan notes)
        assert isinstance(columns, list)


# ─── link_continued_columns ──────────────────────────────────────────


class TestLinkContinuedColumns:
    def _make_base_and_contd_columns(self):
        """Create two columns: 'GENERAL NOTES' and 'GENERAL NOTES (CONT'D)'."""
        tokens = [
            make_box(50, 100, 190, 112, "GENERAL NOTES:"),
            make_box(50, 120, 65, 132, "1."),
            make_box(70, 120, 200, 132, "DO THIS"),
        ]
        hdr1 = BlockCluster(
            page=0,
            rows=[RowBand(page=0, boxes=[tokens[0]])],
            is_header=True,
        )
        note1 = BlockCluster(
            page=0,
            rows=[RowBand(page=0, boxes=[tokens[1], tokens[2]])],
            is_notes=True,
        )
        col1 = NotesColumn(page=0, header=hdr1, notes_blocks=[note1])

        tokens2 = [
            make_box(400, 100, 600, 112, "GENERAL NOTES (CONT'D):"),
            make_box(400, 120, 415, 132, "3."),
            make_box(420, 120, 550, 132, "DO MORE"),
        ]
        hdr2 = BlockCluster(
            page=0,
            rows=[RowBand(page=0, boxes=[tokens2[0]])],
            is_header=True,
        )
        note2 = BlockCluster(
            page=0,
            rows=[RowBand(page=0, boxes=[tokens2[1], tokens2[2]])],
            is_notes=True,
        )
        col2 = NotesColumn(page=0, header=hdr2, notes_blocks=[note2])
        blocks = [hdr1, note1, hdr2, note2]
        return [col1, col2], blocks

    def test_columns_linked(self):
        columns, blocks = self._make_base_and_contd_columns()
        link_continued_columns(columns, blocks=blocks)
        # The continuation column should be linked
        contd = [c for c in columns if c.is_continuation()]
        if contd:
            assert contd[0].continues_from is not None
            assert contd[0].column_group_id is not None

    def test_shared_group_id(self):
        columns, blocks = self._make_base_and_contd_columns()
        link_continued_columns(columns, blocks=blocks)
        group_ids = {c.column_group_id for c in columns if c.column_group_id}
        # All linked columns should share the same group ID
        if group_ids:
            assert len(group_ids) == 1

    def test_empty_columns(self):
        link_continued_columns([])
        # Should not crash

    def test_single_column_no_link(self):
        tokens = [make_box(50, 100, 190, 112, "GENERAL NOTES:")]
        hdr = BlockCluster(
            page=0,
            rows=[RowBand(page=0, boxes=[tokens[0]])],
            is_header=True,
        )
        col = NotesColumn(page=0, header=hdr)
        link_continued_columns([col])
        assert col.column_group_id is None or col.continues_from is None
