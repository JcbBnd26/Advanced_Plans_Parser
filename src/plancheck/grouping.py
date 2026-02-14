from __future__ import annotations

import re
from statistics import mean, median, pstdev
from typing import Iterable, List, Optional, Tuple

from .config import GroupingConfig
from .models import BlockCluster, GlyphBox, Line, NotesColumn, RowBand, Span


def _histogram_gutters(
    boxes: List[GlyphBox],
    bins: int,
    gutter_width: float,
    density_threshold: float = 0.08,
) -> List[float]:
    """Find gutters via x-density histogram; return split positions (x)."""
    if not boxes or bins <= 0:
        return []
    span_min = min(b.x0 for b in boxes)
    span_max = max(b.x1 for b in boxes)
    span_width = span_max - span_min
    if span_width <= 0:
        return []
    bucket_w = span_width / bins
    hist = [0] * bins
    for b in boxes:
        c = (b.x0 + b.x1) * 0.5
        idx = int((c - span_min) / bucket_w)
        if idx < 0:
            idx = 0
        if idx >= bins:
            idx = bins - 1
        hist[idx] += 1
    # Identify low-density runs below a small percentile of the max.
    max_count = max(hist) if hist else 0
    if max_count == 0:
        return []
    threshold = max_count * density_threshold  # adjustable via config
    gutters = []
    run_start = None
    for i, val in enumerate(hist):
        if val <= threshold:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_end = i - 1
                run_width = (run_end - run_start + 1) * bucket_w
                if run_width >= gutter_width:
                    mid = span_min + (run_start + run_end + 1) * 0.5 * bucket_w
                    gutters.append(mid)
                run_start = None
    if run_start is not None:
        run_end = len(hist) - 1
        run_width = (run_end - run_start + 1) * bucket_w
        if run_width >= gutter_width:
            mid = span_min + (run_start + run_end + 1) * 0.5 * bucket_w
            gutters.append(mid)
    return gutters


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


def detect_column_boundaries(
    tokens: List[GlyphBox],
    page_height: float,
    settings: GroupingConfig,
) -> List[float]:
    """Detect column boundary x-positions using only the main content band.

    This avoids pollution from headers, footers, and legends which often
    have different column structures than the main content area.

    Args:
        tokens: All tokens on the page
        page_height: Page height in points
        settings: GroupingConfig with content_band_top/bottom

    Returns:
        List of column boundary x-positions (sorted). Empty list means single column.
    """
    if not tokens or page_height <= 0:
        return []

    # Filter to main content band (middle 70% by default)
    y_min = page_height * settings.content_band_top
    y_max = page_height * settings.content_band_bottom

    content_tokens = [t for t in tokens if y_min < (t.y0 + t.y1) * 0.5 < y_max]

    if len(content_tokens) < 4:
        # Not enough tokens in content band, fall back to all tokens
        content_tokens = tokens

    median_w, _ = _median_size(content_tokens)
    if median_w <= 0:
        return []

    # Use histogram-based gutter detection
    if settings.use_hist_gutter:
        gutter_width = median_w * settings.gutter_width_mult
        gutters = _histogram_gutters(
            content_tokens,
            bins=settings.grouping_histogram_bins,
            gutter_width=gutter_width,
            density_threshold=settings.grouping_histogram_density,
        )
        if gutters:
            return sorted(gutters)

    # Fallback: gap-based detection
    centers = sorted([(t.x0 + t.x1) * 0.5 for t in content_tokens])
    if len(centers) < 2:
        return []

    # Find significant gaps
    gap_thresh = median_w * settings.column_gap_mult
    boundaries: List[float] = []

    for a, b in zip(centers[:-1], centers[1:]):
        if b - a > gap_thresh:
            boundaries.append((a + b) * 0.5)

    return sorted(boundaries)


def assign_column_ids(
    lines: List[Line],
    tokens: List[GlyphBox],
    col_boundaries: List[float],
) -> None:
    """Assign column IDs to spans based on column boundary positions.

    This is non-destructive: it only labels spans with col_id, never splits lines.

    Args:
        lines: List of Line objects with populated spans
        tokens: Original token list
        col_boundaries: Column boundary x-positions from detect_column_boundaries()
    """
    if not col_boundaries:
        # Single column: all spans get col_id=0
        for line in lines:
            for span in line.spans:
                span.col_id = 0
        return

    # Boundaries define regions: (-inf, b0), [b0, b1), [b1, b2), ..., [bn, +inf)
    # So n boundaries create n+1 columns
    for line in lines:
        for span in line.spans:
            if not span.token_indices:
                span.col_id = 0
                continue

            # Use span's x-center to determine column
            span_boxes = [tokens[i] for i in span.token_indices]
            span_x_center = (
                min(b.x0 for b in span_boxes) + max(b.x1 for b in span_boxes)
            ) / 2

            # Find which column this span belongs to
            col_id = 0
            for boundary in col_boundaries:
                if span_x_center >= boundary:
                    col_id += 1
                else:
                    break

            span.col_id = col_id


def _partition_columns(
    boxes: List[GlyphBox], median_w: float, settings: GroupingConfig
) -> List[tuple[List[GlyphBox], float, float]]:
    """Split boxes into columns using histogram gutters (optional), then gap-based fallback."""
    if not boxes:
        return []

    span_min = min(b.x0 for b in boxes)
    span_max = max(b.x1 for b in boxes)
    span_width = span_max - span_min
    if median_w <= 0:
        return [(boxes, span_min, span_max)]

    # Histogram-based gutter detection (if enabled)
    if settings.use_hist_gutter:
        gutter_width = median_w * settings.gutter_width_mult
        gutters = _histogram_gutters(
            boxes,
            bins=settings.grouping_histogram_bins,
            gutter_width=gutter_width,
            density_threshold=settings.grouping_histogram_density,
        )
        if gutters:
            cuts = [span_min] + gutters + [span_max]
            segments: List[tuple[List[GlyphBox], float, float]] = []
            for a, b in zip(cuts[:-1], cuts[1:]):
                seg = [
                    bx
                    for bx in boxes
                    if (bx.x0 + bx.x1) * 0.5 >= a and (bx.x0 + bx.x1) * 0.5 < b
                ]
                if seg:
                    seg_min = min(bx.x0 for bx in seg)
                    seg_max = max(bx.x1 for bx in seg)
                    segments.append((seg, seg_min, seg_max))
            if segments:
                return segments

    # Fallback: gap-based splitting with adaptive threshold
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
) -> List[BlockCluster]:
    """Group Lines into BlockClusters using the new row-truth pipeline.

    Unlike the old group_blocks() which uses RowBands, this function:
    - Takes Lines as input (never splits them)
    - Uses span col_id for horizontal banding
    - Uses baseline_y for vertical gap detection
    - Populates BlockCluster.lines instead of .rows

    Args:
        lines: Lines from build_lines() with spans populated
        tokens: Original token list
        settings: GroupingConfig

    Returns:
        List of BlockCluster with lines populated (rows will be empty)
    """
    if not lines:
        return []

    # Compute median line height for gap detection
    line_heights = []
    line_widths = []
    for line in lines:
        bbox = line.bbox(tokens)
        line_heights.append(bbox[3] - bbox[1])
        line_widths.append(bbox[2] - bbox[0])

    median_line_h = float(median(line_heights)) if line_heights else 1.0
    median_line_w = float(median(line_widths)) if line_widths else 1.0
    block_gap = median_line_h * settings.block_gap_mult
    max_block_height = median_line_h * settings.max_block_height_mult
    note_re = re.compile(r"^\d+\.")

    # Determine column index for each line based on its leftmost span's col_id
    def get_line_col_id(line: Line) -> int:
        if line.spans:
            # Use the leftmost span's col_id
            leftmost_span = min(
                line.spans,
                key=lambda s: (
                    tokens[s.token_indices[0]].x0 if s.token_indices else float("inf")
                ),
            )
            return leftmost_span.col_id if leftmost_span.col_id is not None else 0
        return 0

    col_index = [get_line_col_id(line) for line in lines]

    # If no col_ids are set, infer from horizontal positions
    if all(c == 0 for c in col_index):
        col_gap = median_line_w * settings.grouping_col_gap_fallback_mult
        centers = [
            ((line.bbox(tokens)[0] + line.bbox(tokens)[2]) * 0.5, idx)
            for idx, line in enumerate(lines)
        ]
        centers.sort(key=lambda t: t[0])
        if centers:
            col_index[centers[0][1]] = 0
            prev_c = centers[0][0]
            band = 0
            for c, idx in centers[1:]:
                if c - prev_c > col_gap:
                    band += 1
                col_index[idx] = band
                prev_c = c

    def _group_band(band_lines: List[Line]) -> List[BlockCluster]:
        if not band_lines:
            return []
        # Sort by baseline_y
        band_lines = sorted(band_lines, key=lambda ln: ln.baseline_y)
        band_blocks: List[BlockCluster] = []

        current = BlockCluster(
            page=band_lines[0].page,
            rows=[],  # Empty for new pipeline
            lines=[band_lines[0]],
            _tokens=tokens,
        )

        for ln in band_lines[1:]:
            # Get previous line's bbox
            prev_bbox = current.lines[-1].bbox(tokens)
            prev_y1 = prev_bbox[3]

            # Get current line's bbox
            cur_bbox = ln.bbox(tokens)
            cur_y0 = cur_bbox[1]

            # Calculate block height
            first_bbox = current.lines[0].bbox(tokens)
            cur_block_height = prev_y1 - first_bbox[1]

            # Check if line starts with note number
            line_text = ln.text(tokens).strip()
            starts_note = bool(note_re.match(line_text))
            note_starts_block = starts_note and len(current.lines) >= 1

            # Decide whether to split
            should_split = (
                (cur_y0 - prev_y1) > block_gap
                or (cur_block_height > max_block_height)
                or note_starts_block
            )

            if should_split:
                band_blocks.append(current)
                current = BlockCluster(
                    page=ln.page,
                    rows=[],
                    lines=[ln],
                    _tokens=tokens,
                )
            else:
                current.lines.append(ln)

        band_blocks.append(current)
        return band_blocks

    # Group by column band
    blocks: List[BlockCluster] = []
    max_band = max(col_index) if col_index else 0
    for band in range(max_band + 1):
        band_lines = [ln for idx, ln in enumerate(lines) if col_index[idx] == band]
        blocks.extend(_group_band(band_lines))

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


def mark_headers(blocks: List[BlockCluster], debug_path: str = None) -> None:
    """Identify and label header blocks before notes grouping."""
    header_re = re.compile(r"^[A-Z0-9\s\-\(\)\'\.]+: *$", re.ASCII)
    # Phrases that look like headers but are actually subheader fragments
    # (e.g., "BE USED ON THIS PROJECT:" is part of the Standard Details subheader)
    excluded_phrases = {
        "BE USED ON THIS PROJECT:",
    }
    debug_path = debug_path or "debug_headers.txt"
    with open(debug_path, "a", encoding="utf-8") as dbg:
        for i, blk in enumerate(blocks):
            blk.is_header = False
            if blk.rows:
                first_row = blk.rows[0]
                texts = [
                    b.text
                    for b in sorted(first_row.boxes, key=lambda b: b.x0)
                    if b.text
                ]
                if texts:
                    row_text = " ".join(texts).strip()
                    row_text_norm = re.sub(r"\s+", " ", row_text).upper()
                    is_bold = any(
                        getattr(b, "fontname", "").lower().find("bold") >= 0
                        for b in first_row.boxes
                    )
                    if (
                        header_re.match(row_text_norm)
                        and row_text_norm not in excluded_phrases
                    ):
                        blk.is_header = True
                        blk.label = "note_column_header"
                        dbg.write(f"[DEBUG] Header block {i}: '{row_text_norm}'\n")


def mark_notes(blocks: List[BlockCluster], debug_path: str = None) -> None:
    """Label notes blocks, skipping any block already labeled as header."""
    note_re = re.compile(r"^\d+\.")
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
                    row_text_norm = re.sub(r"\s+", " ", row_text).upper()
                    if note_re.match(row_text_norm) and len(blk.rows) >= 2:
                        blk.is_notes = True
                        blk.label = "notes_block"


def group_notes_columns(
    blocks: List[BlockCluster],
    x_tolerance: float = 30.0,
    y_gap_max: float = 50.0,
    debug_path: str = None,
    cfg: GroupingConfig | None = None,
) -> List[NotesColumn]:
    """
    Group header blocks with their associated notes blocks into NotesColumn objects.

    For each notes block, find the nearest header above it that is x-aligned.
    This ensures that when multiple headers share the same x position, notes
    are assigned to the closest header above them.

    Returns a list of NotesColumn objects. Each header gets one NotesColumn.
    Notes blocks without a header are grouped into a NotesColumn with header=None.
    """
    if cfg is not None:
        x_tolerance = cfg.grouping_notes_x_tolerance
        y_gap_max = cfg.grouping_notes_y_gap_max
    debug_path = debug_path or "debug_headers.txt"

    # Find all header and notes blocks
    headers = [b for b in blocks if getattr(b, "is_header", False)]
    notes = [b for b in blocks if getattr(b, "is_notes", False)]

    # For each notes block, find the best matching header
    # (the closest header above it that is x-aligned)
    notes_to_header: dict = {}  # notes block id -> header

    with open(debug_path, "a", encoding="utf-8") as dbg:
        dbg.write(
            f"\n[DEBUG] group_notes_columns: {len(headers)} headers, {len(notes)} notes blocks\n"
        )

        for n in notes:
            nx0, ny0, nx1, ny1 = n.bbox()
            best_header = None
            best_gap = float("inf")

            for header in headers:
                hx0, hy0, hx1, hy1 = header.bbox()

                # Header must be above the notes block
                if hy1 > ny0:
                    continue

                # Left edge must be aligned (within tolerance)
                if abs(nx0 - hx0) > x_tolerance:
                    continue

                # Calculate vertical gap
                gap = ny0 - hy1

                # Pick the closest header above
                if gap < best_gap:
                    best_gap = gap
                    best_header = header

            if best_header is not None:
                notes_to_header[id(n)] = best_header

        # Now group notes by their assigned header
        header_to_notes: dict = {id(h): [] for h in headers}
        for n in notes:
            if id(n) in notes_to_header:
                header_to_notes[id(notes_to_header[id(n)])].append(n)

        # Build columns, applying vertical chaining within each column
        columns: List[NotesColumn] = []

        for header in headers:
            hx0, hy0, hx1, hy1 = header.bbox()
            header_label = ""
            if header.rows:
                texts = [b.text for b in header.rows[0].boxes if b.text]
                header_label = " ".join(texts).strip()
            dbg.write(
                f"[DEBUG] Header '{header_label}' bbox: x0={hx0:.1f}, y0={hy0:.1f}, x1={hx1:.1f}, y1={hy1:.1f}\n"
            )

            column = NotesColumn(page=header.page, header=header, notes_blocks=[])

            # Get candidate notes for this header, sorted by y position
            candidates = header_to_notes[id(header)]
            candidates.sort(key=lambda b: b.bbox()[1])  # Sort by y0

            # Apply vertical chaining - each block must be close to the previous
            last_y1 = hy1
            for n in candidates:
                nx0, ny0, nx1, ny1 = n.bbox()
                gap = ny0 - last_y1

                # Gap must be positive and within threshold
                _first_mult = cfg.grouping_notes_first_gap_mult if cfg else 2.0
                max_gap = (
                    y_gap_max * _first_mult if not column.notes_blocks else y_gap_max
                )

                if 0 <= gap <= max_gap:
                    column.notes_blocks.append(n)
                    last_y1 = ny1
                    dbg.write(
                        f"[DEBUG]   Added notes block at y={ny0:.1f}, gap={gap:.1f}\n"
                    )
                else:
                    dbg.write(
                        f"[DEBUG]   Skipped notes block at y={ny0:.1f}, gap={gap:.1f} > max={max_gap:.1f}\n"
                    )

            dbg.write(f"[DEBUG] Column has {len(column.notes_blocks)} notes blocks\n")
            columns.append(column)

        # Create a column for orphan notes (notes without a header or not chained)
        assigned_notes = set()
        for col in columns:
            for n in col.notes_blocks:
                assigned_notes.add(id(n))

        orphan_notes = [n for n in notes if id(n) not in assigned_notes]
        if orphan_notes:
            dbg.write(f"[DEBUG] {len(orphan_notes)} orphan notes blocks\n")
            # Group orphans by horizontal position into separate columns
            orphan_notes.sort(key=lambda b: b.bbox()[0])  # Sort by x0

            current_orphan_col = None
            for n in orphan_notes:
                nx0, ny0, nx1, ny1 = n.bbox()
                if current_orphan_col is None:
                    current_orphan_col = NotesColumn(
                        page=n.page, header=None, notes_blocks=[n]
                    )
                else:
                    # Check if this orphan is horizontally aligned with current column
                    col_bbox = current_orphan_col.bbox()
                    if abs(nx0 - col_bbox[0]) <= x_tolerance:
                        current_orphan_col.notes_blocks.append(n)
                    else:
                        # Sort current column's blocks by y position before starting new column
                        current_orphan_col.notes_blocks.sort(key=lambda b: b.bbox()[1])
                        columns.append(current_orphan_col)
                        current_orphan_col = NotesColumn(
                            page=n.page, header=None, notes_blocks=[n]
                        )

            if current_orphan_col:
                # Sort final column's blocks by y position
                current_orphan_col.notes_blocks.sort(key=lambda b: b.bbox()[1])
                columns.append(current_orphan_col)

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
    """Build clusters using the new row-truth pipeline.

    This pipeline guarantees that all words on the same baseline stay together.
    Lines are never split by column detection - columns only label spans.

    Pipeline:
    1. build_lines(tokens) → Lines (row truth)
    2. compute_median_space_gap() → true space width
    3. split_line_spans() → populate spans using space-based gap detection
    4. detect_column_boundaries() → zone-aware column detection (content band only)
    5. assign_column_ids() → label spans with col_id (non-destructive)
    6. group_blocks_from_lines() → BlockClusters with lines (not rows)
    7. mark_tables/headers/notes → semantic labeling

    Args:
        tokens: GlyphBox tokens from PDF extraction
        page_height: Page height in points (for zone-aware column detection)
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

    # Step 4: Detect column boundaries from content band
    col_boundaries = detect_column_boundaries(tokens, page_height, settings)

    # Step 5: Assign column IDs to spans (non-destructive)
    assign_column_ids(lines, tokens, col_boundaries)

    # Step 6: Group lines into blocks (also populates .rows for compat)
    blocks = group_blocks_from_lines(lines, tokens, settings)

    # Step 7: Semantic labeling (uses .rows via compat shim)
    mark_tables(blocks, settings)
    mark_headers(blocks, debug_path=None)
    mark_notes(blocks, debug_path=None)

    return blocks
