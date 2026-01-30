from __future__ import annotations

import re
from statistics import mean, median, pstdev
from typing import Iterable, List, Tuple

from .config import GroupingConfig
from .models import BlockCluster, GlyphBox, RowBand


def _histogram_gutters(
    boxes: List[GlyphBox], bins: int, gutter_width: float
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
    threshold = max_count * 0.08  # adjustable if needed
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
        gutters = _histogram_gutters(boxes, bins=80, gutter_width=gutter_width)
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
    min_gap_mult = 1.0
    width_guard = (
        median_w * 30
    )  # only lower threshold when the page is meaningfully wide

    while True:
        gap_thresh = median_w * gap_mult_cur
        segments = segments_for_thresh(gap_thresh)
        if segments:
            break
        if span_width >= width_guard and gap_mult_cur > min_gap_mult:
            gap_mult_cur = max(min_gap_mult, gap_mult_cur * 0.7)
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
        if gap > gap_thresh:
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
    for col_idx, (col_boxes, col_min, col_max) in enumerate(
        _partition_columns(boxes, median_w, settings)
    ):
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
                if abs(y_center - r_center) <= vert_tol and overlap_ratio > 0.3:
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
        col_gap = median_row_w * 0.6
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
                if gap <= block_gap * 1.5:
                    nxt.rows = blk.rows + nxt.rows
                    merge_done = True
                    # Do not append blk; it is merged into next.
        if not merge_done:
            merged.append(blk)
        i += 1
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


def mark_notes(blocks: List[BlockCluster]) -> None:
    note_re = re.compile(r"^\d+\.")
    for blk in blocks:
        blk.is_notes = False
        if len(blk.rows) < 2:
            continue
        first_row = blk.rows[0]
        texts = [b.text for b in sorted(first_row.boxes, key=lambda b: b.x0) if b.text]
        if not texts:
            continue
        row_text = " ".join(texts).strip()
        if note_re.match(row_text):
            blk.is_notes = True
            blk.label = "notes_block"


def build_clusters(
    boxes: List[GlyphBox], settings: GroupingConfig
) -> List[BlockCluster]:
    rows = group_rows(boxes, settings)
    blocks = group_blocks(rows, settings)
    mark_tables(blocks, settings)
    mark_notes(blocks)
    return blocks
