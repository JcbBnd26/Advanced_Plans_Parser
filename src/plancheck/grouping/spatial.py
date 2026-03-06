"""Spatial partitioning: rows, columns, and block grouping.

This module handles spatial layout detection and block grouping:
- Column detection via gap analysis
- Row clustering by y-position
- Block grouping from rows or lines
"""

from __future__ import annotations

import bisect
import logging
import re
from statistics import median
from typing import List

from ..config import GroupingConfig
from ..models import BlockCluster, GlyphBox, Line, RowBand
from ._utils import NOTE_BROAD_RE, NOTE_SIMPLE_RE
from .lines import _median_size

log = logging.getLogger(__name__)

__all__ = [
    "group_rows",
    "group_blocks",
    "group_blocks_from_lines",
]

# ── Pre-compiled patterns ──────────────────────────────────────────────
_NOTE_NUMBER_RE = re.compile(r"^[\(\[]?[0-9A-Za-z]{1,3}[\.\)\]]?$")


# ── Column partitioning ────────────────────────────────────────────────


def _partition_columns(
    boxes: List[GlyphBox], median_w: float, settings: GroupingConfig
) -> List[tuple[List[GlyphBox], float, float]]:
    """Split boxes into columns using gap-based detection."""
    if not boxes:
        return []

    span_min = min(b.x0 for b in boxes)
    span_max = max(b.x1 for b in boxes)
    span_width = span_max - span_min
    if median_w <= 0:
        return [(boxes, span_min, span_max)]

    # Gap-based splitting with adaptive threshold
    centers = sorted([(b.x0 + b.x1) * 0.5 for b in boxes])
    if len(centers) < 2:
        return [(boxes, span_min, span_max)]

    # Pre-sort boxes by center for efficient bisect-based partitioning
    boxes_by_center = sorted(boxes, key=lambda b: (b.x0 + b.x1) * 0.5)
    box_centers = [(b.x0 + b.x1) * 0.5 for b in boxes_by_center]

    def segments_for_thresh(thresh: float) -> List[tuple[List[GlyphBox], float, float]]:
        """Return column segments by splitting at gaps wider than *thresh*.

        Uses bisect for O(n log n) partitioning instead of O(n × num_breaks).
        """
        breaks = []
        for a, b in zip(centers[:-1], centers[1:]):
            if b - a > thresh:
                breaks.append((a + b) * 0.5)
        if not breaks:
            return []

        # Use bisect to partition boxes into segments efficiently
        segments = []
        prev_idx = 0
        for br in breaks:
            # Find index of first box with center >= break point
            end_idx = bisect.bisect_left(box_centers, br)
            seg = boxes_by_center[prev_idx:end_idx]
            if seg:
                seg_min = min(bx.x0 for bx in seg)
                seg_max = max(bx.x1 for bx in seg)
                segments.append((seg, seg_min, seg_max))
            prev_idx = end_idx

        # Last segment: all remaining boxes
        last_seg = boxes_by_center[prev_idx:]
        if last_seg:
            seg_min = min(bx.x0 for bx in last_seg)
            seg_max = max(bx.x1 for bx in last_seg)
            segments.append((last_seg, seg_min, seg_max))
        return segments

    gap_mult_cur = settings.column_gap_mult
    min_gap_mult = settings.grouping_partition_floor
    width_guard = (
        median_w * settings.grouping_partition_width_guard_mult
    )  # only lower threshold when the page is meaningfully wide

    while True:
        gap_thresh = median_w * gap_mult_cur
        segments = segments_for_thresh(gap_thresh)
        if segments:
            break
        if span_width >= width_guard and gap_mult_cur > min_gap_mult:
            gap_mult_cur = max(
                min_gap_mult, gap_mult_cur * settings.grouping_partition_decay
            )
            continue
        return [(boxes, span_min, span_max)]

    # Further split any very wide segment on its largest internal gap to avoid mega-columns.
    refined: List[tuple[List[GlyphBox], float, float]] = []
    max_seg_width = median_w * settings.max_column_width_mult
    for seg_boxes, seg_min, seg_max in segments:
        queue = [(seg_boxes, seg_min, seg_max)]
        while queue:
            cur_boxes, cur_min, cur_max = queue.pop(0)
            width = cur_max - cur_min
            if width <= max_seg_width or len(cur_boxes) < 4:
                refined.append((cur_boxes, cur_min, cur_max))
                continue
            centers_cur = sorted([(b.x0 + b.x1) * 0.5 for b in cur_boxes])
            gaps_cur = [(b - a) for a, b in zip(centers_cur[:-1], centers_cur[1:])]
            if not gaps_cur:
                refined.append((cur_boxes, cur_min, cur_max))
                continue
            max_gap = max(gaps_cur)
            gap_idx = gaps_cur.index(max_gap)
            split_center = (centers_cur[gap_idx] + centers_cur[gap_idx + 1]) * 0.5
            left = [bx for bx in cur_boxes if (bx.x0 + bx.x1) * 0.5 < split_center]
            right = [bx for bx in cur_boxes if (bx.x0 + bx.x1) * 0.5 >= split_center]
            if left:
                queue.append((left, min(b.x0 for b in left), max(b.x1 for b in left)))
            if right:
                queue.append(
                    (right, min(b.x0 for b in right), max(b.x1 for b in right))
                )
    return refined if refined else segments


def _is_note_number_column(
    boxes: List[GlyphBox],
    cfg: GroupingConfig | None = None,
) -> bool:
    """Check if a column consists primarily of note numbers (short labels like '1.', '10.', 'A.')."""
    if not boxes:
        return False
    _majority = cfg.grouping_note_majority if cfg else 0.5
    _max_rows = cfg.grouping_note_max_rows if cfg else 50
    note_count = sum(1 for b in boxes if _NOTE_NUMBER_RE.match(b.text.strip()))
    # If most boxes in this column are note numbers, it's a note number column
    return note_count >= len(boxes) * _majority and len(boxes) <= _max_rows


def _merge_note_number_columns(
    segments: List[tuple[List[GlyphBox], float, float]],
    cfg: GroupingConfig | None = None,
) -> List[tuple[List[GlyphBox], float, float]]:
    """Merge note-number-only columns with their adjacent text column."""
    if len(segments) <= 1:
        return segments

    # Sort segments by x position
    sorted_segs = sorted(segments, key=lambda s: s[1])  # sort by seg_min

    merged: List[tuple[List[GlyphBox], float, float]] = []
    i = 0
    while i < len(sorted_segs):
        seg_boxes, seg_min, seg_max = sorted_segs[i]

        # Check if this is a note number column
        if _is_note_number_column(seg_boxes, cfg) and i + 1 < len(sorted_segs):
            # Merge with the next column (the text column to the right)
            next_boxes, next_min, next_max = sorted_segs[i + 1]
            combined_boxes = seg_boxes + next_boxes
            combined_min = min(seg_min, next_min)
            combined_max = max(seg_max, next_max)
            merged.append((combined_boxes, combined_min, combined_max))
            i += 2  # Skip both columns
        else:
            merged.append((seg_boxes, seg_min, seg_max))
            i += 1

    return merged


def _split_row_by_width(
    row: RowBand, median_w: float, max_width: float
) -> List[RowBand]:
    """Split an over-wide row at its largest horizontal gap."""
    if not row.boxes:
        return []
    x0, y0, x1, y1 = row.bbox()
    width = x1 - x0
    if width <= max_width or len(row.boxes) < 2:
        return [row]
    # Split at largest gap.
    gaps = []
    for a, b in zip(row.boxes[:-1], row.boxes[1:]):
        gaps.append((b.x0 - a.x1, a, b))
    gaps_sorted = sorted(gaps, key=lambda g: g[0], reverse=True)
    if not gaps_sorted:
        return [row]
    _, left_box, right_box = gaps_sorted[0]
    left_boxes = [bx for bx in row.boxes if bx.x1 <= left_box.x1]
    right_boxes = [bx for bx in row.boxes if bx.x0 >= right_box.x0]
    middle_boxes = [
        bx for bx in row.boxes if bx not in left_boxes and bx not in right_boxes
    ]
    # Assign middle boxes to nearest side by center.
    for bx in middle_boxes:
        if abs(bx.x1 - left_box.x1) <= abs(bx.x0 - right_box.x0):
            left_boxes.append(bx)
        else:
            right_boxes.append(bx)
    out: List[RowBand] = []
    for subset in (left_boxes, right_boxes):
        if subset:
            subset_sorted = sorted(subset, key=lambda b: b.x0)
            out.append(
                RowBand(page=row.page, boxes=subset_sorted, column_id=row.column_id)
            )
    # If still too wide, recurse.
    final: List[RowBand] = []
    for r in out:
        final.extend(_split_row_by_width(r, median_w, max_width))
    return final if final else [row]


def _is_note_number(box: GlyphBox) -> bool:
    """Check if a glyph box looks like a note number (e.g., '1.', '12.', 'A.')."""
    text = box.text.strip()
    if not text:
        return False
    return bool(_NOTE_NUMBER_RE.match(text))


def _split_row_on_gaps(row: RowBand, median_w: float, gap_mult: float) -> List[RowBand]:
    """Split a row wherever horizontal gaps exceed the threshold."""
    if not row.boxes:
        return []
    if median_w <= 0:
        return [row]
    gap_thresh = median_w * gap_mult
    boxes_sorted = sorted(row.boxes, key=lambda b: b.x0)
    rows: List[RowBand] = []
    current = RowBand(page=row.page, boxes=[boxes_sorted[0]], column_id=row.column_id)
    for b in boxes_sorted[1:]:
        prev = current.boxes[-1]
        gap = b.x0 - prev.x1
        # Don't split if the previous box looks like a note number - allow larger gaps
        if gap > gap_thresh and not _is_note_number(prev):
            rows.append(current)
            current = RowBand(page=row.page, boxes=[b], column_id=row.column_id)
        else:
            current.boxes.append(b)
    rows.append(current)
    return rows


# ── Row grouping (OLD API) ─────────────────────────────────────────────


def group_rows(boxes: List[GlyphBox], settings: GroupingConfig) -> List[RowBand]:
    """Group glyph boxes into horizontal row bands per column."""
    if not boxes:
        return []

    median_w, median_h = _median_size(boxes)
    vert_tol = median_h * settings.vertical_tol_mult
    split_rows: List[RowBand] = []

    # Partition into columns to avoid spanning across big horizontal whitespace.
    columns = _partition_columns(boxes, median_w, settings)
    # Merge note-number-only columns with their adjacent text column
    columns = _merge_note_number_columns(columns, settings)

    for col_idx, (col_boxes, col_min, col_max) in enumerate(columns):
        col_width = col_max - col_min
        max_row_width = col_width * settings.max_row_width_mult
        # First pass: cluster by y-center proximity within this column.
        rows: List[RowBand] = []
        for b in sorted(col_boxes, key=lambda b: ((b.y0 + b.y1) * 0.5, b.x0)):
            y_center = (b.y0 + b.y1) * 0.5
            placed = False
            for r in rows:
                rx0, ry0, rx1, ry1 = r.bbox()
                r_center = (ry0 + ry1) * 0.5
                overlap = min(b.y1, ry1) - max(b.y0, ry0)
                min_h = min((b.y1 - b.y0), (ry1 - ry0))
                overlap_ratio = overlap / (min_h + 1e-6)
                if (
                    abs(y_center - r_center) <= vert_tol
                    and overlap_ratio > settings.grouping_line_overlap_ratio
                ):
                    r.boxes.append(b)
                    placed = True
                    break
            if not placed:
                rows.append(RowBand(page=b.page, boxes=[b], column_id=col_idx))

        # Order boxes in each row and split on large horizontal gaps.
        for r in rows:
            r.boxes.sort(key=lambda b: b.x0)
            gap_split = _split_row_on_gaps(r, median_w, settings.row_split_gap_mult)
            for gr in gap_split:
                width_split = _split_row_by_width(gr, median_w, max_row_width)
                split_rows.extend(width_split)

    # Sort final rows by y, then x for stability.
    split_rows.sort(key=lambda r: (r.bbox()[1], r.bbox()[0]))
    return split_rows


# ── Block grouping (OLD API) ───────────────────────────────────────────


def group_blocks(rows: List[RowBand], settings: GroupingConfig) -> List[BlockCluster]:
    """Merge adjacent rows into logical block clusters."""
    if not rows:
        return []
    row_heights = []
    row_widths = []
    for r in rows:
        _, y0, _, y1 = r.bbox()
        row_heights.append(y1 - y0)
        x0, _, x1, _ = r.bbox()
        row_widths.append(x1 - x0)
    median_row_h = float(median(row_heights)) if row_heights else 1.0
    median_row_w = float(median(row_widths)) if row_widths else 1.0
    block_gap = median_row_h * settings.block_gap_mult
    max_block_height = median_row_h * settings.max_block_height_mult
    note_re = NOTE_SIMPLE_RE

    if any(r.column_id is not None for r in rows):
        col_index = [r.column_id if r.column_id is not None else 0 for r in rows]
    else:
        # Fallback: infer bands by horizontal gaps between row centers.
        col_gap = median_row_w * settings.grouping_col_gap_fallback_mult
        centers = [
            (((r.bbox()[0] + r.bbox()[2]) * 0.5), idx) for idx, r in enumerate(rows)
        ]
        centers.sort(key=lambda t: t[0])
        col_index = [0] * len(rows)
        if centers:
            col_index[centers[0][1]] = 0
            prev_c = centers[0][0]
            band = 0
            for c, idx in centers[1:]:
                if c - prev_c > col_gap:
                    band += 1
                col_index[idx] = band
                prev_c = c

    def _group_band(rows_band: List[RowBand]) -> List[BlockCluster]:
        """Group vertically consecutive rows into block clusters."""
        if not rows_band:
            return []
        rows_band = sorted(rows_band, key=lambda r: r.bbox()[1])
        band_blocks: List[BlockCluster] = []
        current = BlockCluster(page=rows_band[0].page, rows=[rows_band[0]])
        cur_min_x, _, cur_max_x, _ = rows_band[0].bbox()
        for r in rows_band[1:]:
            _, prev_y0, _, prev_y1 = current.rows[-1].bbox()
            cur_y0, cur_y1 = r.bbox()[1], r.bbox()[3]
            cur_block_top = current.rows[0].bbox()[1]
            cur_block_height = prev_y1 - cur_block_top
            row_text = " ".join(
                b.text for b in sorted(r.boxes, key=lambda b: b.x0)
            ).strip()
            starts_note = bool(note_re.match(row_text))
            rx0, _, rx1, _ = r.bbox()
            note_starts_block = starts_note and len(current.rows) >= 1
            should_split = (
                (cur_y0 - prev_y1) > block_gap
                or (cur_block_height > max_block_height)
                or note_starts_block
            )
            if should_split:
                band_blocks.append(current)
                current = BlockCluster(page=r.page, rows=[r])
                cur_min_x, _, cur_max_x, _ = r.bbox()
            else:
                current.rows.append(r)
                cur_min_x = min(cur_min_x, rx0)
                cur_max_x = max(cur_max_x, rx1)
        band_blocks.append(current)
        return band_blocks

    blocks: List[BlockCluster] = []
    max_band = max(col_index) if col_index else 0
    for band in range(max_band + 1):
        band_rows = [r for idx, r in enumerate(rows) if col_index[idx] == band]
        blocks.extend(_group_band(band_rows))

    # Post-process: merge standalone note-number blocks into the following block.
    merged: List[BlockCluster] = []
    sorted_blocks = sorted(blocks, key=lambda b: b.bbox()[1])
    i = 0
    while i < len(sorted_blocks):
        blk = sorted_blocks[i]
        merge_done = False
        if len(blk.rows) == 1:
            row = blk.rows[0]
            texts = [b.text for b in sorted(row.boxes, key=lambda b: b.x0) if b.text]
            row_text = " ".join(texts).strip()
            if (
                note_re.match(row_text)
                and len(texts) <= 2
                and (i + 1) < len(sorted_blocks)
            ):
                x0_b, y0, x1_b, y1 = blk.bbox()
                nxt = sorted_blocks[i + 1]
                nx0, ny0, nx1, ny1 = nxt.bbox()
                gap = ny0 - y1
                if gap <= block_gap * settings.grouping_block_merge_mult:
                    nxt.rows = blk.rows + nxt.rows
                    merge_done = True
                    # Do not append blk; it is merged into next.
        if not merge_done:
            merged.append(blk)
        i += 1
    return merged


# ── Block grouping from Lines (NEW API) ────────────────────────────────


def group_blocks_from_lines(
    lines: List[Line],
    tokens: List[GlyphBox],
    settings: GroupingConfig,
    median_space_gap: float = 0.0,
) -> List[BlockCluster]:
    """Group Lines into BlockClusters using x0 clustering + vertical gap.

    Two-pass algorithm:
      1. **Cluster lines into visual columns** by x0 (left-edge) proximity.
         Lines are sorted by x0; a new cluster starts when the gap
         between a line's x0 and the running cluster average exceeds *col_gap*.
      2. **Within each column cluster**, sort by baseline_y and split into
         blocks on vertical gap, note-number start, or max block height.

    Args:
        lines: Lines from build_lines()/split_wide_lines() with spans populated
        tokens: Original token list
        settings: GroupingConfig
        median_space_gap: Inter-word space gap from compute_median_space_gap()

    Returns:
        List of BlockCluster with lines populated
    """
    if not lines:
        return []

    # Compute median line height for gap detection
    line_heights = []
    for line in lines:
        bbox = line.bbox(tokens)
        line_heights.append(bbox[3] - bbox[1])

    median_line_h = float(median(line_heights)) if line_heights else 1.0
    block_gap = median_line_h * settings.block_gap_mult
    max_block_height = median_line_h * settings.max_block_height_mult
    log.debug(
        "group_blocks_from_lines: %d lines, median_h=%.1f, block_gap=%.1f",
        len(lines),
        median_line_h,
        block_gap,
    )
    # Match the same note-start patterns recognised by mark_notes():
    #   1.  12.  A.  a.  (1)  (A)  (a)
    note_re = NOTE_BROAD_RE

    # Column gap: the same threshold that separated spans within a line.
    col_gap = (
        median_space_gap * settings.span_gap_mult
        if median_space_gap > 0
        else median_line_h * 5.0
    )

    # --- Pass 1: cluster lines into visual columns by x0 (left edge) ---
    # Left-edge clustering is far more stable than x-center for text
    # paragraphs: all lines in a column share a common left margin
    # regardless of line length, whereas x-center shifts dramatically
    # for short continuation lines (e.g. "NOTED." vs a full-width note).
    line_x0s = []
    for ln in lines:
        bb = ln.bbox(tokens)
        line_x0s.append((bb[0], ln))

    line_x0s.sort(key=lambda t: t[0])

    # Use incremental sum tracking instead of recalculating cluster average each time
    # This reduces O(n × avg_cluster_size) to O(n)
    x_clusters: List[List[Line]] = [[line_x0s[0][1]]]
    cluster_x0_sum = line_x0s[0][0]  # Running sum of x0 values in current cluster
    cluster_count = 1  # Count of lines in current cluster

    for x0, ln in line_x0s[1:]:
        avg_x0 = cluster_x0_sum / cluster_count
        if abs(x0 - avg_x0) > col_gap:
            # Start new cluster
            x_clusters.append([ln])
            cluster_x0_sum = x0
            cluster_count = 1
        else:
            # Add to current cluster and update running stats
            x_clusters[-1].append(ln)
            cluster_x0_sum += x0
            cluster_count += 1

    # --- Pass 2: within each cluster, group by vertical gap ---
    def _group_column(col_lines: List[Line]) -> List[BlockCluster]:
        """Group vertically consecutive lines into block clusters."""
        if not col_lines:
            return []
        col_lines = sorted(col_lines, key=lambda ln: ln.baseline_y)
        col_blocks: List[BlockCluster] = []

        current = BlockCluster(
            page=col_lines[0].page,
            rows=[],
            lines=[col_lines[0]],
            _tokens=tokens,
        )

        for ln in col_lines[1:]:
            prev_bbox = current.lines[-1].bbox(tokens)
            cur_bbox = ln.bbox(tokens)
            first_bbox = current.lines[0].bbox(tokens)
            cur_block_height = prev_bbox[3] - first_bbox[1]

            # Check if line starts with note number
            line_text = ln.text(tokens).strip()
            starts_note = bool(note_re.match(line_text))
            if not starts_note and ln.spans:
                for sp in ln.spans:
                    sp_text = sp.text(tokens).strip()
                    if sp_text and note_re.match(sp_text):
                        starts_note = True
                        break
            note_starts_block = starts_note and len(current.lines) >= 1

            v_gap = cur_bbox[1] - prev_bbox[3]

            should_split = (
                v_gap > block_gap
                or cur_block_height > max_block_height
                or note_starts_block
            )

            if should_split:
                col_blocks.append(current)
                current = BlockCluster(
                    page=ln.page,
                    rows=[],
                    lines=[ln],
                    _tokens=tokens,
                )
            else:
                current.lines.append(ln)

        col_blocks.append(current)
        return col_blocks

    blocks: List[BlockCluster] = []
    for cluster in x_clusters:
        blocks.extend(_group_column(cluster))

    # Post-process: merge standalone note-number lines into following block
    merged: List[BlockCluster] = []
    sorted_blocks = sorted(blocks, key=lambda b: b.bbox()[1])
    i = 0
    while i < len(sorted_blocks):
        blk = sorted_blocks[i]
        merge_done = False

        if len(blk.lines) == 1:
            line = blk.lines[0]
            line_text = line.text(tokens).strip()

            # Check if it's a note number line
            if (
                note_re.match(line_text)
                and len(line.token_indices) <= 2
                and (i + 1) < len(sorted_blocks)
            ):
                x0_b, y0, x1_b, y1 = blk.bbox()
                nxt = sorted_blocks[i + 1]
                nx0, ny0, nx1, ny1 = nxt.bbox()
                gap = ny0 - y1

                if gap <= block_gap * settings.grouping_block_merge_mult:
                    nxt.lines = blk.lines + nxt.lines
                    merge_done = True

        if not merge_done:
            merged.append(blk)
        i += 1

    # Populate .rows from .lines for backward compatibility with all
    # downstream code (mark_headers, mark_notes, legends, overlay, etc.)
    for blk in merged:
        blk.populate_rows_from_lines()

    return merged
