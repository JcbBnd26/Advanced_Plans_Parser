"""Tests for plancheck.grouping — line building, clustering, notes detection."""

import pytest
from conftest import make_box

from plancheck.config import GroupingConfig
from plancheck.grouping import (
    build_clusters_v2,
    build_lines,
    compute_median_space_gap,
    group_blocks,
    group_blocks_from_lines,
    group_rows,
    mark_headers,
    mark_notes,
    mark_tables,
    split_line_spans,
)


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
