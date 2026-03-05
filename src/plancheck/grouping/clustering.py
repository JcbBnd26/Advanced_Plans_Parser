"""Geometry-first clustering: orchestration and re-exports.

This module provides the main entry point ``build_clusters_v2()`` and
re-exports all public functions from the domain submodules for backward
compatibility.

Submodules:
- lines: Row-truth layer (build_lines, split_line_spans, etc.)
- spatial: Spatial partitioning (group_rows, group_blocks, group_blocks_from_lines)
- labeling: Semantic labeling (mark_tables, mark_headers, mark_notes)
- notes_columns: Notes column grouping (group_notes_columns, link_continued_columns)
"""

from __future__ import annotations

import logging
from statistics import median
from typing import List

from ..config import GroupingConfig
from ..models import BlockCluster, GlyphBox, Line, Span
from ._utils import NOTE_BROAD_RE
from .labeling import flag_suspect_header_words, mark_headers, mark_notes, mark_tables

# Import from submodules
from .lines import (
    build_lines,
    compute_median_space_gap,
    split_line_spans,
    split_wide_lines,
)
from .notes_columns import group_notes_columns, link_continued_columns
from .spatial import group_blocks, group_blocks_from_lines, group_rows

log = logging.getLogger(__name__)

__all__ = [
    # Orchestration
    "build_clusters_v2",
    "_split_wide_blocks",  # Exposed for testing
    # Lines (from .lines)
    "build_lines",
    "compute_median_space_gap",
    "split_line_spans",
    "split_wide_lines",
    # Spatial (from .spatial)
    "group_rows",
    "group_blocks",
    "group_blocks_from_lines",
    # Labeling (from .labeling)
    "mark_tables",
    "mark_headers",
    "mark_notes",
    "flag_suspect_header_words",
    # Notes columns (from .notes_columns)
    "group_notes_columns",
    "link_continued_columns",
]


# ── Wide-block splitting ───────────────────────────────────────────────


def _split_wide_blocks(
    blocks: List[BlockCluster],
    tokens: List[GlyphBox],
    settings: GroupingConfig,
    median_space_gap: float,
) -> List[BlockCluster]:
    """Split blocks that span multiple visual columns.

    When ``build_lines`` merges words from adjacent columns onto the same
    Line (because their y-positions are within tolerance), and the inter-
    column x-gap is smaller than ``span_gap_mult × median_space_gap``,
    the resulting Block can span two visual columns.  This function
    detects such blocks and splits them at the inter-column boundary,
    then re-groups each half into independent blocks (respecting vertical
    gaps and note-number starts).
    """
    if len(blocks) < 3:
        return blocks

    widths = sorted([b.bbox()[2] - b.bbox()[0] for b in blocks])
    med_w = widths[len(widths) // 2]
    width_thresh = med_w * 1.6
    min_half = 60.0  # each side must be at least this wide

    # Gap threshold: lighter than span_gap_mult (which missed these gaps)
    gap_thresh = max(median_space_gap * 6.0, 20.0)

    # For re-grouping split lines into blocks
    note_re = NOTE_BROAD_RE
    line_heights = []
    for blk in blocks:
        for ln in blk.lines:
            bb = ln.bbox(tokens)
            h = bb[3] - bb[1]
            if h > 0:
                line_heights.append(h)
    median_line_h = (
        sorted(line_heights)[len(line_heights) // 2] if line_heights else 12.0
    )
    block_gap = median_line_h * settings.block_gap_mult

    def _regroup(line_set: List[Line], page: int) -> List[BlockCluster]:
        """Group a set of lines into blocks, splitting on vertical gap /
        note-number start (mirrors the inner logic of group_blocks_from_lines).
        """
        if not line_set:
            return []
        line_set = sorted(line_set, key=lambda ln: ln.baseline_y)
        sub_blocks: List[BlockCluster] = []
        current: List[Line] = [line_set[0]]
        for ln in line_set[1:]:
            prev_bb = current[-1].bbox(tokens)
            cur_bb = ln.bbox(tokens)
            v_gap = cur_bb[1] - prev_bb[3]
            line_text = ln.text(tokens).strip()
            starts_note = bool(note_re.match(line_text))
            if not starts_note and ln.spans:
                for sp in ln.spans:
                    sp_text = sp.text(tokens).strip()
                    if sp_text and note_re.match(sp_text):
                        starts_note = True
                        break
            should_split = v_gap > block_gap or (starts_note and len(current) >= 1)
            if should_split:
                blk = BlockCluster(page=page, rows=[], lines=current, _tokens=tokens)
                blk.populate_rows_from_lines()
                sub_blocks.append(blk)
                current = [ln]
            else:
                current.append(ln)
        if current:
            blk = BlockCluster(page=page, rows=[], lines=current, _tokens=tokens)
            blk.populate_rows_from_lines()
            sub_blocks.append(blk)
        return sub_blocks

    result: List[BlockCluster] = []
    for blk in blocks:
        bb = blk.bbox()
        blk_w = bb[2] - bb[0]
        if blk_w <= width_thresh or len(blk.lines) < 2:
            result.append(blk)
            continue

        # Collect all inter-token gaps within each line
        gap_candidates: List[tuple] = []
        for ln in blk.lines:
            if len(ln.token_indices) < 2:
                continue
            sorted_idx = sorted(ln.token_indices, key=lambda i: tokens[i].x0)
            for j in range(len(sorted_idx) - 1):
                left_x1 = tokens[sorted_idx[j]].x1
                right_x0 = tokens[sorted_idx[j + 1]].x0
                gap = right_x0 - left_x1
                if gap >= gap_thresh:
                    gap_candidates.append((gap, left_x1, right_x0))

        if not gap_candidates:
            result.append(blk)
            continue

        # Choose the largest gap as the split boundary
        gap_candidates.sort(key=lambda t: t[0], reverse=True)
        best_left_x1, best_right_x0 = gap_candidates[0][1], gap_candidates[0][2]
        split_x = (best_left_x1 + best_right_x0) / 2.0

        # Both halves must be wide enough
        if (split_x - bb[0]) < min_half or (bb[2] - split_x) < min_half:
            result.append(blk)
            continue

        # Split each line's tokens at split_x
        left_lines: List[Line] = []
        right_lines: List[Line] = []
        lid = 0
        for ln in blk.lines:
            left_idx = [i for i in ln.token_indices if tokens[i].x0 < split_x]
            right_idx = [i for i in ln.token_indices if tokens[i].x0 >= split_x]
            if left_idx:
                left_idx = sorted(left_idx, key=lambda i: tokens[i].x0)
                yc = [(tokens[i].y0 + tokens[i].y1) / 2.0 for i in left_idx]
                left_lines.append(
                    Line(
                        line_id=lid,
                        page=ln.page,
                        token_indices=left_idx,
                        baseline_y=median(yc) if yc else ln.baseline_y,
                    )
                )
                lid += 1
            if right_idx:
                right_idx = sorted(right_idx, key=lambda i: tokens[i].x0)
                yc = [(tokens[i].y0 + tokens[i].y1) / 2.0 for i in right_idx]
                right_lines.append(
                    Line(
                        line_id=lid,
                        page=ln.page,
                        token_indices=right_idx,
                        baseline_y=median(yc) if yc else ln.baseline_y,
                    )
                )
                lid += 1

        result.extend(_regroup(left_lines, blk.page))
        result.extend(_regroup(right_lines, blk.page))

    return result


# ── Main entry point ───────────────────────────────────────────────────


def build_clusters_v2(
    tokens: List[GlyphBox],
    page_height: float,
    settings: GroupingConfig,
) -> List[BlockCluster]:
    """Build clusters using the row-truth pipeline.

    Pipeline:
    1. build_lines(tokens) → Lines (row truth)
    2. compute_median_space_gap() → true space width
    3. split_line_spans() → populate spans using space-based gap detection
    4. group_blocks_from_lines() → BlockClusters via spatial proximity
    5. mark_tables/headers/notes → semantic labeling

    Args:
        tokens: GlyphBox tokens from PDF extraction
        page_height: Page height in points (unused, kept for API compat)
        settings: GroupingConfig

    Returns:
        List of BlockCluster with lines populated
    """
    if not tokens:
        return []

    # Step 1: Build canonical lines (row truth)
    lines = build_lines(tokens, settings)

    # Step 2: Compute median space gap for span splitting
    median_space_gap = compute_median_space_gap(lines, tokens, settings)

    # Step 3: Split lines into spans based on large gaps
    for line in lines:
        split_line_spans(line, tokens, median_space_gap, settings.span_gap_mult)

    # Step 3b: Split multi-span lines into sub-lines so each sub-line
    #          belongs to one visual column (pure gap-based, no histograms)
    lines = split_wide_lines(lines, tokens, median_space_gap)

    # Step 4: Group lines into blocks via spatial proximity
    blocks = group_blocks_from_lines(lines, tokens, settings, median_space_gap)

    # Step 4b: Split blocks that span multiple visual columns
    blocks = _split_wide_blocks(blocks, tokens, settings, median_space_gap)

    # Step 5: Semantic labeling (uses .rows via compat shim)
    mark_tables(blocks, settings)
    mark_headers(blocks, debug_path=None, cfg=settings)
    mark_notes(blocks, debug_path=None)

    n_headers = sum(1 for b in blocks if b.is_header)
    n_notes = sum(1 for b in blocks if b.is_notes)
    n_tables = sum(1 for b in blocks if b.is_table)
    log.info(
        "build_clusters_v2: %d tokens → %d lines → %d blocks "
        "(headers=%d, notes=%d, tables=%d)",
        len(tokens),
        len(lines),
        len(blocks),
        n_headers,
        n_notes,
        n_tables,
    )
    return blocks
