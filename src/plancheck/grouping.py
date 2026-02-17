from __future__ import annotations

import re
from statistics import mean, median, pstdev
from typing import Iterable, List, Optional, Tuple

from .config import GroupingConfig
from .models import BlockCluster, GlyphBox, Line, NotesColumn, RowBand, Span


def _median_size(boxes: Iterable[GlyphBox]) -> Tuple[float, float]:
    widths = [b.width() for b in boxes]
    heights = [b.height() for b in boxes]
    if not widths or not heights:
        return (1.0, 1.0)
    return (float(median(widths)), float(median(heights)))


# =============================================================================
# Row-Truth Layer: build_lines, compute_median_space_gap, split_line_spans
# =============================================================================


def build_lines(tokens: List[GlyphBox], settings: GroupingConfig) -> List[Line]:
    """Build canonical lines from tokens by clustering on baseline y-position.

    This is the row-truth layer. Each token belongs to exactly one Line.
    Lines are never split by subsequent column detection.

    Args:
        tokens: List of GlyphBox tokens from PDF extraction
        settings: GroupingConfig with vertical_tol_mult

    Returns:
        List of Line objects sorted by (baseline_y, min_x)
    """
    if not tokens:
        return []

    _, median_h = _median_size(tokens)
    vert_tol = median_h * settings.vertical_tol_mult

    # Build index mapping: we need to track which original token index goes where
    # Sort tokens by (y_center, x0) for stable clustering
    indexed_tokens = [(i, t, (t.y0 + t.y1) * 0.5) for i, t in enumerate(tokens)]
    indexed_tokens.sort(key=lambda x: (x[2], tokens[x[0]].x0))

    # Cluster by y-center proximity
    lines: List[Line] = []
    line_id = 0

    for idx, token, y_center in indexed_tokens:
        placed = False

        # Try to place in existing line
        for line in lines:
            # Compute current line's y-center from its tokens
            line_boxes = [tokens[i] for i in line.token_indices]
            line_y_centers = [(b.y0 + b.y1) * 0.5 for b in line_boxes]
            line_center = mean(line_y_centers)

            # Check overlap ratio for better accuracy
            line_y0 = min(b.y0 for b in line_boxes)
            line_y1 = max(b.y1 for b in line_boxes)
            overlap = min(token.y1, line_y1) - max(token.y0, line_y0)
            min_h = min(token.y1 - token.y0, line_y1 - line_y0)
            overlap_ratio = overlap / (min_h + 1e-6)

            if (
                abs(y_center - line_center) <= vert_tol
                and overlap_ratio > settings.grouping_line_overlap_ratio
            ):
                line.token_indices.append(idx)
                placed = True
                break

        if not placed:
            # Create new line
            new_line = Line(
                line_id=line_id,
                page=token.page,
                token_indices=[idx],
                baseline_y=y_center,
                spans=[],
            )
            lines.append(new_line)
            line_id += 1

    # Finalize: sort token_indices by x0, compute final baseline_y
    for line in lines:
        line.token_indices.sort(key=lambda i: tokens[i].x0)
        y_centers = [(tokens[i].y0 + tokens[i].y1) * 0.5 for i in line.token_indices]
        line.baseline_y = median(y_centers) if y_centers else 0.0

    # Sort lines by (baseline_y, min_x) for reading order
    def line_sort_key(line: Line) -> Tuple[float, float]:
        if not line.token_indices:
            return (0.0, 0.0)
        min_x = min(tokens[i].x0 for i in line.token_indices)
        return (line.baseline_y, min_x)

    lines.sort(key=line_sort_key)

    # Re-assign line_ids after sorting
    for i, line in enumerate(lines):
        line.line_id = i

    return lines


def compute_median_space_gap(
    lines: List[Line],
    tokens: List[GlyphBox],
    cfg: GroupingConfig | None = None,
) -> float:
    """Compute the median inter-word space gap from actual token spacing.

    This gives a robust estimate of the typical space width on the page,
    which is consistent across fonts (unlike median token width).

    Args:
        lines: List of Line objects
        tokens: Original token list
        cfg: GroupingConfig (optional) for fallback and percentile values

    Returns:
        Median space gap in points. Returns fallback if insufficient data.
    """
    _fallback = cfg.grouping_space_gap_fallback if cfg else 5.0
    _percentile = cfg.grouping_space_gap_percentile if cfg else 0.9

    if not lines or not tokens:
        return _fallback

    all_gaps: List[float] = []

    for line in lines:
        if len(line.token_indices) < 2:
            continue

        # Walk tokens in x-order (they should already be sorted)
        sorted_indices = sorted(line.token_indices, key=lambda i: tokens[i].x0)

        for i in range(len(sorted_indices) - 1):
            curr_token = tokens[sorted_indices[i]]
            next_token = tokens[sorted_indices[i + 1]]
            gap = next_token.x0 - curr_token.x1
            if gap > 0:  # Only positive gaps (no overlaps)
                all_gaps.append(gap)

    if not all_gaps:
        return _fallback

    # Filter to "small" gaps: exclude huge gaps (gutters) to get true space width
    sorted_gaps = sorted(all_gaps)
    cutoff_idx = int(len(sorted_gaps) * _percentile)
    small_gaps = sorted_gaps[:cutoff_idx] if cutoff_idx > 0 else sorted_gaps

    return median(small_gaps) if small_gaps else _fallback


def split_line_spans(
    line: Line,
    tokens: List[GlyphBox],
    median_space_gap: float,
    span_gap_mult: float,
) -> None:
    """Split a line into spans based on large horizontal gaps.

    Spans preserve column/table structure without breaking line integrity.
    Modifies line.spans in place.

    Args:
        line: Line object to split
        tokens: Original token list
        median_space_gap: Typical inter-word space from compute_median_space_gap()
        span_gap_mult: Multiplier for gap threshold (default ~8)
    """
    if not line.token_indices:
        line.spans = []
        return

    gap_thresh = median_space_gap * span_gap_mult

    # Sort by x0 (should already be sorted, but ensure)
    sorted_indices = sorted(line.token_indices, key=lambda i: tokens[i].x0)

    # Build spans by detecting large gaps
    spans: List[Span] = []
    current_span_indices: List[int] = [sorted_indices[0]]

    for i in range(1, len(sorted_indices)):
        prev_token = tokens[sorted_indices[i - 1]]
        curr_token = tokens[sorted_indices[i]]
        gap = curr_token.x0 - prev_token.x1

        if gap > gap_thresh:
            # Large gap: finalize current span and start new one
            spans.append(Span(token_indices=current_span_indices, col_id=None))
            current_span_indices = [sorted_indices[i]]
        else:
            current_span_indices.append(sorted_indices[i])

    # Finalize last span
    if current_span_indices:
        spans.append(Span(token_indices=current_span_indices, col_id=None))

    line.spans = spans


def split_wide_lines(
    lines: List[Line],
    tokens: List[GlyphBox],
) -> List[Line]:
    """Split lines that have multiple widely-separated spans into sub-lines.

    After ``split_line_spans`` each Line has one or more Spans.  When a
    line has ≥2 spans those spans are already separated by a large
    horizontal gap (> span_gap_mult × median_space_gap).  This function
    creates a separate Line for each span so that downstream spatial
    block grouping can treat them independently.

    Single-span lines are kept as-is.

    Returns a new list of Lines with sequential IDs.
    """
    result: List[Line] = []
    next_id = 0

    for line in lines:
        if len(line.spans) <= 1:
            line.line_id = next_id
            next_id += 1
            result.append(line)
            continue

        # Multi-span line: create a sub-line per span
        for span in line.spans:
            if not span.token_indices:
                continue

            sub_indices = sorted(span.token_indices, key=lambda i: tokens[i].x0)
            y_centers = [
                (tokens[i].y0 + tokens[i].y1) * 0.5 for i in sub_indices
            ]
            sub_baseline = median(y_centers) if y_centers else line.baseline_y

            sub_line = Line(
                line_id=next_id,
                page=line.page,
                token_indices=sub_indices,
                baseline_y=sub_baseline,
                spans=[span],
            )
            next_id += 1
            result.append(sub_line)

    # Re-sort by (baseline_y, min_x) for reading order
    def _sort_key(ln: Line) -> tuple:
        if not ln.token_indices:
            return (0.0, 0.0)
        return (ln.baseline_y, min(tokens[i].x0 for i in ln.token_indices))

    result.sort(key=_sort_key)
    for idx, ln in enumerate(result):
        ln.line_id = idx

    return result


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

    def segments_for_thresh(thresh: float) -> List[tuple[List[GlyphBox], float, float]]:
        breaks = []
        for a, b in zip(centers[:-1], centers[1:]):
            if b - a > thresh:
                breaks.append((a + b) * 0.5)
        if not breaks:
            return []
        segments = []
        prev = float("-inf")
        for br in breaks:
            seg = [bx for bx in boxes if prev <= (bx.x0 + bx.x1) * 0.5 < br]
            if seg:
                seg_min = min(bx.x0 for bx in seg)
                seg_max = max(bx.x1 for bx in seg)
                segments.append((seg, seg_min, seg_max))
            prev = br
        last_seg = [bx for bx in boxes if (bx.x0 + bx.x1) * 0.5 >= prev]
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
    import re

    if not boxes:
        return False
    _majority = cfg.grouping_note_majority if cfg else 0.5
    _max_rows = cfg.grouping_note_max_rows if cfg else 50
    note_pattern = re.compile(r"^[\(\[]?[0-9A-Za-z]{1,3}[\.\)\]]?$")
    note_count = sum(1 for b in boxes if note_pattern.match(b.text.strip()))
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
    # Pattern: digit(s) or letter followed by period, optionally with closing paren
    # Examples: "1.", "12.", "A.", "1)", "(1)"
    import re

    return bool(re.match(r"^[\(\[]?[0-9A-Za-z]{1,3}[\.\)\]]?$", text))


def _split_row_on_gaps(row: RowBand, median_w: float, gap_mult: float) -> List[RowBand]:
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


def group_rows(boxes: List[GlyphBox], settings: GroupingConfig) -> List[RowBand]:
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


def group_blocks(rows: List[RowBand], settings: GroupingConfig) -> List[BlockCluster]:
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
    note_re = re.compile(r"^\d+\.")

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
    # Match the same note-start patterns recognised by mark_notes():
    #   1.  12.  A.  a.  (1)  (A)  (a)
    note_re = re.compile(r"^(?:\d+\.|[A-Z]\.|[a-z]\.|\(\d+\)|\([A-Za-z]\))")

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

    x_clusters: List[List[Line]] = [[line_x0s[0][1]]]
    for x0, ln in line_x0s[1:]:
        prev_x0 = sum(
            l.bbox(tokens)[0] for l in x_clusters[-1]
        ) / len(x_clusters[-1])
        if abs(x0 - prev_x0) > col_gap:
            x_clusters.append([ln])
        else:
            x_clusters[-1].append(ln)

    # --- Pass 2: within each cluster, group by vertical gap ---
    def _group_column(col_lines: List[Line]) -> List[BlockCluster]:
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


def mark_tables(blocks: List[BlockCluster], settings: GroupingConfig) -> None:
    for blk in blocks:
        if len(blk.rows) < 2:
            blk.is_table = False
            continue
        col_counts: List[int] = []
        gaps_sets: List[List[float]] = []
        for row in blk.rows:
            xs = [b.x0 for b in row.boxes] + [b.x1 for b in row.boxes]
            xs_sorted = sorted(xs)
            col_counts.append(len(row.boxes))
            gaps = [xs_sorted[i + 1] - xs_sorted[i] for i in range(len(xs_sorted) - 1)]
            if gaps:
                gaps_sets.append(gaps)
        if not gaps_sets:
            blk.is_table = False
            continue
        flat_gaps = [g for gaps in gaps_sets for g in gaps if g > 0]
        if not flat_gaps:
            blk.is_table = False
            continue
        gap_mean = mean(flat_gaps)
        gap_cv = (pstdev(flat_gaps) / (gap_mean + 1e-6)) if len(flat_gaps) > 1 else 0.0
        col_mean = mean(col_counts) if col_counts else 0.0
        col_cv = (
            (pstdev(col_counts) / (col_mean + 1e-6)) if len(col_counts) > 1 else 0.0
        )
        blk.is_table = (
            gap_cv < settings.table_regular_tol and col_cv < settings.table_regular_tol
        )


def _block_first_row_text(blk: BlockCluster) -> str:
    """Get normalised text of the first row (ALL CAPS, collapsed whitespace)."""
    if not blk.rows:
        return ""
    first_row = blk.rows[0]
    texts = [b.text for b in sorted(first_row.boxes, key=lambda b: b.x0) if b.text]
    if not texts:
        return ""
    return re.sub(r"\s+", " ", " ".join(texts).strip()).upper()


def mark_headers(blocks: List[BlockCluster], debug_path: str = None) -> None:
    """Identify and label header blocks before notes grouping.

    Detection signals (any one is sufficient):
      1. ALL CAPS text ending with ":"  (classic plan header)
      2. ALL CAPS text + bold font      (bold section title)
      3. ALL CAPS text + font size > 1.3 × median  (large section title)

    Single-row blocks only.  Excluded phrases prevent false-positives.
    """
    # Classic: "GENERAL NOTES:", "EROSION CONTROL NOTES - GENERAL:", etc.
    header_colon_re = re.compile(r"^[A-Z0-9\s\-\(\)\'\.\/]+: *$", re.ASCII)
    # Broader: ALL CAPS, ≥2 words, no numbered-note start
    header_caps_re = re.compile(r"^[A-Z][A-Z0-9\s\-\(\)\'\.\/]{4,}$", re.ASCII)
    excluded_phrases = {
        "BE USED ON THIS PROJECT:",
        "SHEET",
    }
    # Patterns that are title-block / sheet-label text, not section headers
    _title_block_re = re.compile(
        r"OKLAHOMA DEPARTMENT|LEGEND AND |SHEET |^[A-Z]+-\d+$", re.ASCII
    )

    # Compute median font size across all boxes for relative-size check
    all_sizes = []
    for blk in blocks:
        for row in blk.rows:
            for b in row.boxes:
                sz = getattr(b, "font_size", 0.0)
                if sz > 0:
                    all_sizes.append(sz)
    median_size = sorted(all_sizes)[len(all_sizes) // 2] if all_sizes else 0.0

    debug_path = debug_path or "debug_headers.txt"
    with open(debug_path, "a", encoding="utf-8") as dbg:
        for i, blk in enumerate(blocks):
            blk.is_header = False
            if not blk.rows or len(blk.rows) > 3:
                continue  # Headers are short — skip tall blocks

            text = _block_first_row_text(blk)
            if not text or text in excluded_phrases:
                continue
            if _title_block_re.search(text):
                continue  # Skip title-block / sheet-label text

            # --- Signal 1: ALL CAPS + colon ---
            has_colon = bool(header_colon_re.match(text))

            # --- Signal 2: bold font ---
            is_bold = any(
                getattr(b, "fontname", "").lower().find("bold") >= 0
                for b in blk.rows[0].boxes
            )

            # --- Signal 3: larger than median ---
            avg_size = 0.0
            sizes = [getattr(b, "font_size", 0.0) for b in blk.rows[0].boxes]
            sizes = [s for s in sizes if s > 0]
            if sizes:
                avg_size = sum(sizes) / len(sizes)
            is_large = median_size > 0 and avg_size > median_size * 1.3

            # Must also be ALL CAPS (at least for signals 2 & 3)
            is_all_caps = bool(header_caps_re.match(text))

            # Accept: colon-pattern, OR (all caps + bold), OR (colon + large)
            # Large alone is NOT enough — too many title-block false positives
            if has_colon or (is_all_caps and is_bold):
                blk.is_header = True
                blk.label = "note_column_header"
                signals = []
                if has_colon:
                    signals.append("colon")
                if is_bold:
                    signals.append("bold")
                if is_large:
                    signals.append(f"large({avg_size:.1f}>{median_size:.1f})")
                dbg.write(
                    f"[DEBUG] Header block {i}: '{text}' [{', '.join(signals)}]\n"
                )


def mark_notes(blocks: List[BlockCluster], debug_path: str = None) -> None:
    """Label notes blocks, skipping any block already labeled as header.

    Patterns recognised:
      - ``1.``, ``12.``  — numbered notes
      - ``A.``, ``B.``   — lettered notes
      - ``(1)``, ``(A)`` — parenthesised numbered/lettered notes
    """
    note_re = re.compile(r"^(?:\d+\.|[A-Z]\.|[a-z]\.|\(\d+\)|\([A-Za-z]\))")
    debug_path = debug_path or "debug_headers.txt"
    with open(debug_path, "a", encoding="utf-8") as dbg:
        for i, blk in enumerate(blocks):
            blk.is_notes = False
            if getattr(blk, "is_header", False):
                continue  # Skip header blocks
            if blk.rows:
                first_row = blk.rows[0]
                texts = [
                    b.text
                    for b in sorted(first_row.boxes, key=lambda b: b.x0)
                    if b.text
                ]
                if texts:
                    row_text = " ".join(texts).strip()
                    # Check full row text first; if that fails, check
                    # individual word texts (handles margin reference
                    # numbers that precede the actual note number).
                    is_note_start = bool(note_re.match(row_text))
                    if not is_note_start:
                        for t in texts:
                            if note_re.match(t.strip()):
                                is_note_start = True
                                break
                    if is_note_start:
                        blk.is_notes = True
                        blk.label = "notes_block"


def group_notes_columns(
    blocks: List[BlockCluster],
    x_tolerance: float = 30.0,
    y_gap_max: float = 50.0,
    debug_path: str = None,
    cfg: GroupingConfig | None = None,
) -> List[NotesColumn]:
    """Group header blocks with their associated notes blocks into NotesColumn objects.

    **x0-clustering algorithm (no histogram columns):**

    1. Collect all header and notes blocks.
    2. Cluster them by x0 proximity: sort by x0, split on gaps > x_tolerance.
       Each cluster represents a visual column of notes.
    3. Within each cluster, sort blocks top-to-bottom by y0.
    4. Walk each cluster: headers open a new NotesColumn, notes append to
       the most recently opened one. Notes before any header → orphan column.

    Returns a list of :class:`NotesColumn` objects.
    """
    if cfg is not None:
        x_tolerance = cfg.grouping_notes_x_tolerance
        y_gap_max = cfg.grouping_notes_y_gap_max
    debug_path = debug_path or "debug_headers.txt"

    # Collect labeled blocks
    headers = [b for b in blocks if getattr(b, "is_header", False)]
    notes = [b for b in blocks if getattr(b, "is_notes", False)]
    labeled = set(id(b) for b in headers) | set(id(b) for b in notes)

    # Build list of all labeled blocks with their x0 positions
    labeled_blocks = [b for b in blocks if id(b) in labeled]
    if not labeled_blocks:
        return []

    # Sort by x0 for clustering
    labeled_blocks.sort(key=lambda b: b.bbox()[0])

    # Cluster by x0 proximity: split when gap between consecutive x0 > x_tolerance
    x0_clusters: List[List[BlockCluster]] = [[labeled_blocks[0]]]
    for blk in labeled_blocks[1:]:
        prev_x0 = x0_clusters[-1][-1].bbox()[0]
        cur_x0 = blk.bbox()[0]
        if abs(cur_x0 - prev_x0) > x_tolerance:
            x0_clusters.append([blk])
        else:
            x0_clusters[-1].append(blk)

    with open(debug_path, "a", encoding="utf-8") as dbg:
        dbg.write(
            f"\n[DEBUG] group_notes_columns (x0-cluster): "
            f"{len(headers)} headers, {len(notes)} notes blocks, "
            f"{len(x0_clusters)} x-clusters\n"
        )

        columns: List[NotesColumn] = []

        for ci, cluster in enumerate(x0_clusters):
            # Sort cluster blocks top-to-bottom
            cluster.sort(key=lambda b: b.bbox()[1])
            active_col: NotesColumn | None = None

            dbg.write(
                f"[DEBUG]   x-cluster {ci}: {len(cluster)} blocks, "
                f"x0 range [{cluster[0].bbox()[0]:.1f} .. {cluster[-1].bbox()[0]:.1f}]\n"
            )

            for blk in cluster:
                bb = blk.bbox()

                if getattr(blk, "is_header", False):
                    header_text = _block_first_row_text(blk)
                    dbg.write(
                        f"[DEBUG]     Header '{header_text}' "
                        f"x0={bb[0]:.1f} y={bb[1]:.1f}\n"
                    )
                    active_col = NotesColumn(
                        page=blk.page, header=blk, notes_blocks=[]
                    )
                    columns.append(active_col)

                elif getattr(blk, "is_notes", False):
                    if active_col is None:
                        active_col = NotesColumn(
                            page=blk.page, header=None, notes_blocks=[]
                        )
                        columns.append(active_col)
                        dbg.write(
                            "[DEBUG]     Orphan column opened (no header yet)\n"
                        )

                    active_col.notes_blocks.append(blk)
                    note_text = _block_first_row_text(blk)
                    dbg.write(
                        f"[DEBUG]       +note x0={bb[0]:.1f} y={bb[1]:.1f} "
                        f"'{note_text[:60]}'\n"
                    )

        # Summary
        for col in columns:
            hdr = _block_first_row_text(col.header) if col.header else "(orphan)"
            dbg.write(f"[DEBUG] Column '{hdr}': {len(col.notes_blocks)} notes\n")

    return columns


def _get_last_block_text(col: NotesColumn) -> str:
    """Get the full text of the last notes block in a column."""
    if not col.notes_blocks:
        return ""
    last_block = col.notes_blocks[-1]
    texts = []
    for row in last_block.rows:
        row_texts = [b.text for b in sorted(row.boxes, key=lambda b: b.x0) if b.text]
        texts.extend(row_texts)
    return " ".join(texts).strip()


def _get_first_block_text(col: NotesColumn) -> str:
    """Get the full text of the first notes block in a column."""
    if not col.notes_blocks:
        return ""
    first_block = col.notes_blocks[0]
    texts = []
    for row in first_block.rows:
        row_texts = [b.text for b in sorted(row.boxes, key=lambda b: b.x0) if b.text]
        texts.extend(row_texts)
    return " ".join(texts).strip()


def _extract_note_numbers(col: NotesColumn) -> list[int]:
    """Extract all note numbers from a column's notes blocks."""
    note_re = re.compile(r"^(\d+)\.")
    numbers = []
    for block in col.notes_blocks:
        if not block.rows:
            continue
        first_row = block.rows[0]
        texts = [b.text for b in sorted(first_row.boxes, key=lambda b: b.x0) if b.text]
        if texts:
            first_text = texts[0]
            match = note_re.match(first_text)
            if match:
                numbers.append(int(match.group(1)))
    return sorted(numbers)


def _text_ends_incomplete(text: str) -> bool:
    """Check if text ends without terminal punctuation (likely mid-sentence)."""
    if not text:
        return False
    text = text.strip()
    # Terminal punctuation that indicates complete sentence
    terminal = {".", "!", "?", ":", ";"}
    return text[-1] not in terminal


def _get_first_row_text(block: BlockCluster) -> str:
    """Get the text of the first row of a block."""
    if not block.rows:
        return ""
    first_row = block.rows[0]
    texts = [b.text for b in sorted(first_row.boxes, key=lambda b: b.x0) if b.text]
    return " ".join(texts)


def _text_starts_as_continuation(text: str) -> bool:
    """Check if text appears to be a sentence continuation (not a new note)."""
    if not text:
        return False
    text = text.strip().upper()
    note_re = re.compile(r"^\d+\.")
    # If it starts with a note number, it's not a continuation
    if note_re.match(text):
        return False
    # If it starts with lowercase-like words or common continuation words
    # (Since all our text is uppercase, we check for sentence-continuation patterns)
    # A continuation typically won't start with note-like patterns
    return True


def link_continued_columns(
    columns: List[NotesColumn],
    blocks: List[BlockCluster] = None,
    x_tolerance: float = 50.0,
    debug_path: str = None,
    cfg: GroupingConfig | None = None,
) -> None:
    """
    Detect and link columns that are continuations of each other.

    Handles two cases:
    1. Explicit continuations: "SITE NOTES" and "SITE NOTES (CONT'D)" headers
    2. Implicit continuations: Snake columns where text wraps to next column
       without a continuation header

    If `blocks` is provided, also finds "leading continuation text" - text blocks
    that appear above the first numbered note in a continuation column and adds
    them to that column's notes_blocks.

    This modifies columns in-place, setting:
    - column_group_id: A shared identifier for linked columns
    - continues_from: The header text of the parent column (for continuations)
    """
    if cfg is not None:
        x_tolerance = cfg.grouping_link_x_tolerance
    debug_path = debug_path or "debug_headers.txt"

    # Build a map of base header text -> list of columns
    base_to_columns: dict = {}
    for col in columns:
        if col.header is None:
            continue
        base = col.base_header_text()
        if base:
            if base not in base_to_columns:
                base_to_columns[base] = []
            base_to_columns[base].append(col)

    with open(debug_path, "a", encoding="utf-8") as dbg:
        dbg.write(
            f"\n[DEBUG] link_continued_columns: checking {len(columns)} columns\n"
        )

        group_id_counter = 0
        for base, cols in base_to_columns.items():
            if len(cols) < 2:
                # Only one column with this base name, no continuation
                continue

            # Sort by y position to determine order
            cols.sort(key=lambda c: c.bbox()[1] if c.header else float("inf"))

            # Assign a shared group ID
            group_id = f"notes_group_{group_id_counter}"
            group_id_counter += 1

            dbg.write(
                f"[DEBUG] Found column group '{base}' with {len(cols)} columns:\n"
            )

            parent_header_text = None
            for i, col in enumerate(cols):
                col.column_group_id = group_id
                header_text = col.header_text()

                if col.is_continuation():
                    # This is a continuation column
                    col.continues_from = parent_header_text
                    dbg.write(
                        f"[DEBUG]   [{i}] '{header_text}' (continues from '{parent_header_text}')\n"
                    )
                else:
                    # This is a primary column
                    parent_header_text = header_text
                    dbg.write(f"[DEBUG]   [{i}] '{header_text}' (primary)\n")

        # Log summary for explicit continuations
        linked_count = sum(1 for c in columns if c.column_group_id is not None)
        continuation_count = sum(1 for c in columns if c.continues_from is not None)
        dbg.write(
            f"[DEBUG] Explicit linking: {linked_count} columns in groups, {continuation_count} continuations\n"
        )

        # --- IMPLICIT CONTINUATION DETECTION (snake columns) ---
        # Find columns with headers whose last block ends mid-sentence
        # and orphan columns that could be their continuation
        dbg.write(f"\n[DEBUG] Checking for implicit snake-column continuations...\n")

        named_columns = [c for c in columns if c.header is not None]
        orphan_columns = [c for c in columns if c.header is None]

        for named_col in named_columns:
            if not named_col.notes_blocks:
                continue

            # Get the bounding box and text of the named column
            named_bbox = named_col.bbox()
            named_x0, named_y0, named_x1, named_y1 = named_bbox
            last_text = _get_last_block_text(named_col)

            dbg.write(
                f"[DEBUG] Checking '{named_col.header_text()}' for snake continuation\n"
            )
            dbg.write(
                f"[DEBUG]   Last text: '{last_text[-80:] if len(last_text) > 80 else last_text}'\n"
            )

            # Check if the last block ends incomplete
            if not _text_ends_incomplete(last_text):
                dbg.write(f"[DEBUG]   Last text ends complete, skipping\n")
                continue

            dbg.write(
                f"[DEBUG]   Last text ends incomplete, looking for continuation\n"
            )

            # Look for an orphan column that could be a continuation
            # Must be to the right of the named column and have text that continues
            best_orphan = None
            best_score = float("inf")

            for orphan in orphan_columns:
                if orphan.column_group_id is not None:
                    # Already linked
                    continue

                orphan_bbox = orphan.bbox()
                orphan_x0, orphan_y0, orphan_x1, orphan_y1 = orphan_bbox

                # Orphan should be to the right (its x0 > named column's x1 - tolerance)
                # or at similar x but starting at top of page (y0 is small)
                x_gap = orphan_x0 - named_x1

                # For snake columns, the orphan typically is:
                # - To the right of the named column (x_gap > -50)
                # - Starts near top of page (orphan_y0 < named_y0) OR
                # - Has similar x to named column but different y range

                is_right_of = x_gap > -50  # Allow some overlap tolerance
                is_above = orphan_y0 < named_y0

                dbg.write(
                    f"[DEBUG]   Orphan at x0={orphan_x0:.1f}, y0={orphan_y0:.1f}: "
                    f"x_gap={x_gap:.1f}, is_right={is_right_of}, is_above={is_above}\n"
                )

                if not is_right_of:
                    continue

                # Check if orphan's first block looks like a continuation
                first_text = _get_first_block_text(orphan)
                if not _text_starts_as_continuation(first_text):
                    dbg.write(
                        f"[DEBUG]   Orphan first text doesn't look like continuation: '{first_text[:50]}...'\n"
                    )
                    continue

                # Score by proximity: prefer orphans closer to the right edge
                # and that start higher on the page (snake pattern)
                score = x_gap + orphan_y0 / 10  # Weight towards close + high
                if score < best_score:
                    best_score = score
                    best_orphan = orphan

            if best_orphan is not None:
                # Link the orphan to the named column
                if named_col.column_group_id is None:
                    named_col.column_group_id = f"notes_group_{group_id_counter}"
                    group_id_counter += 1

                best_orphan.column_group_id = named_col.column_group_id
                best_orphan.continues_from = named_col.header_text()

                dbg.write(
                    f"[DEBUG]   LINKED orphan to '{named_col.header_text()}' "
                    f"(group={named_col.column_group_id})\n"
                )

        # --- NOTE NUMBERING SEQUENCE DETECTION ---
        # Link orphan columns that continue the note numbering sequence
        # Use best-match algorithm: prefer columns where orphan's min note = named's max + 1
        dbg.write(f"\n[DEBUG] Checking note numbering sequences...\n")

        # Build a list of (named_col, max_note) for columns with notes
        named_with_notes = []
        for named_col in named_columns:
            if not named_col.notes_blocks:
                continue
            named_numbers = _extract_note_numbers(named_col)
            if not named_numbers:
                continue
            named_with_notes.append((named_col, max(named_numbers), named_numbers))
            dbg.write(
                f"[DEBUG] '{named_col.header_text()}' has notes: {named_numbers}, max={max(named_numbers)}\n"
            )

        # For each orphan, find the best matching named column
        for orphan in orphan_columns:
            if orphan.column_group_id is not None:
                # Already linked
                continue

            orphan_numbers = _extract_note_numbers(orphan)
            if not orphan_numbers:
                continue

            min_orphan_note = min(orphan_numbers)
            orphan_bbox = orphan.bbox()
            orphan_x0 = orphan_bbox[0]

            dbg.write(
                f"[DEBUG]   Orphan has notes: {orphan_numbers}, min={min_orphan_note}\n"
            )

            # Find the best matching named column
            best_match = None
            best_gap = float("inf")

            for named_col, max_named_note, named_numbers in named_with_notes:
                named_bbox = named_col.bbox()
                named_x1 = named_bbox[2]

                # Orphan's min must be greater than named's max
                if min_orphan_note <= max_named_note:
                    continue

                # Calculate the gap in note numbers
                note_gap = min_orphan_note - max_named_note

                # Prefer gaps of 1 (immediate continuation) but allow up to 3
                if note_gap > 3:
                    continue

                # Check spatial relationship - orphan should be to the right or similar x
                x_gap = orphan_x0 - named_x1
                is_right_or_similar = x_gap > -100

                if not is_right_or_similar:
                    continue

                dbg.write(
                    f"[DEBUG]     Candidate '{named_col.header_text()}' max={max_named_note}, "
                    f"note_gap={note_gap}, x_gap={x_gap:.1f}\n"
                )

                # Prefer the closest match (smallest note_gap)
                if note_gap < best_gap:
                    best_gap = note_gap
                    best_match = named_col

            if best_match is not None:
                # Link them!
                if best_match.column_group_id is None:
                    best_match.column_group_id = f"notes_group_{group_id_counter}"
                    group_id_counter += 1

                orphan.column_group_id = best_match.column_group_id
                orphan.continues_from = best_match.header_text()

                dbg.write(
                    f"[DEBUG]   LINKED by sequence: orphan notes {orphan_numbers} "
                    f"continue '{best_match.header_text()}' (gap={best_gap})\n"
                )

        # --- Capture leading continuation text ---
        # For orphan columns that have been linked, look for text blocks that appear
        # ABOVE the first numbered note in the column. These are continuations of
        # the previous column's last note.
        if blocks is not None:
            dbg.write(
                f"\n[DEBUG] Checking for leading continuation text in linked orphans\n"
            )

            # Build set of blocks already assigned to columns
            assigned_block_ids = set()
            for col in columns:
                for blk in col.notes_blocks:
                    assigned_block_ids.add(id(blk))
                if col.header is not None:
                    assigned_block_ids.add(id(col.header))

            # Find orphan columns that were just linked
            for orphan in orphan_columns:
                if orphan.continues_from is None:
                    continue  # Not linked
                if not orphan.notes_blocks:
                    continue

                # Get the first notes block (numbered note) in the orphan
                first_notes_block = orphan.notes_blocks[0]
                fnb_x0, fnb_y0, fnb_x1, fnb_y1 = first_notes_block.bbox()
                orphan_bbox = orphan.bbox()
                orphan_x0 = orphan_bbox[0]

                dbg.write(
                    f"[DEBUG]   Orphan continuing '{orphan.continues_from}': "
                    f"first note at y={fnb_y0:.1f}, x={orphan_x0:.1f}\n"
                )

                # Look for text blocks above the first numbered note
                leading_blocks = []
                for blk in blocks:
                    if id(blk) in assigned_block_ids:
                        continue
                    if getattr(blk, "is_header", False):
                        continue
                    if blk.is_table:
                        continue

                    bx0, by0, bx1, by1 = blk.bbox()

                    # Must be above the first notes block
                    if by1 >= fnb_y0:
                        continue

                    # Must be x-aligned with the orphan column
                    if abs(bx0 - orphan_x0) > x_tolerance:
                        continue

                    # Check that the block has some text content
                    if not blk.rows:
                        continue

                    dbg.write(
                        f"[DEBUG]     Found leading text at y={by0:.1f}-{by1:.1f}: "
                        f"'{_get_first_row_text(blk)[:50]}...'\n"
                    )
                    leading_blocks.append(blk)

                # Add leading blocks to the orphan's notes_blocks (at the beginning)
                if leading_blocks:
                    # Sort by y position
                    leading_blocks.sort(key=lambda b: b.bbox()[1])
                    orphan.notes_blocks = leading_blocks + orphan.notes_blocks
                    dbg.write(
                        f"[DEBUG]     Added {len(leading_blocks)} leading text block(s) to orphan\n"
                    )

        # Final summary
        linked_count = sum(1 for c in columns if c.column_group_id is not None)
        continuation_count = sum(1 for c in columns if c.continues_from is not None)
        dbg.write(
            f"[DEBUG] Final: {linked_count} columns in groups, {continuation_count} continuations\n"
        )


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
    lines = split_wide_lines(lines, tokens)

    # Step 4: Group lines into blocks via spatial proximity
    blocks = group_blocks_from_lines(lines, tokens, settings, median_space_gap)

    # Step 5: Semantic labeling (uses .rows via compat shim)
    mark_tables(blocks, settings)
    mark_headers(blocks, debug_path=None)
    mark_notes(blocks, debug_path=None)

    return blocks
