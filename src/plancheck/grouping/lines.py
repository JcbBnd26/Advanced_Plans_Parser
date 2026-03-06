"""Row-truth layer: build canonical lines from tokens.

This module handles the first stage of the clustering pipeline:
converting raw GlyphBox tokens into Line objects by clustering based
on baseline y-position.
"""

from __future__ import annotations

import logging
import re
from statistics import median
from typing import Iterable, List, Tuple

from ..config import GroupingConfig
from ..models import GlyphBox, Line, Span
from ._utils import _open_debug

log = logging.getLogger(__name__)

__all__ = [
    "build_lines",
    "compute_median_space_gap",
    "split_line_spans",
    "split_wide_lines",
]


# ── Shared utilities ───────────────────────────────────────────────────


def _median_size(boxes: Iterable[GlyphBox]) -> Tuple[float, float]:
    """Return (median_width, median_height) of the given glyph boxes."""
    widths = [b.width() for b in boxes]
    heights = [b.height() for b in boxes]
    if not widths or not heights:
        return (1.0, 1.0)
    return (float(median(widths)), float(median(heights)))


# ── Line building ──────────────────────────────────────────────────────


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
            # Use cached accumulators for O(1) access instead of recomputing
            line_center = line.y_center
            line_y0 = line._y0_min
            line_y1 = line._y1_max

            # Check overlap ratio for better accuracy
            overlap = min(token.y1, line_y1) - max(token.y0, line_y0)
            min_h = min(token.y1 - token.y0, line_y1 - line_y0)
            overlap_ratio = overlap / (min_h + 1e-6)

            if (
                abs(y_center - line_center) <= vert_tol
                and overlap_ratio > settings.grouping_line_overlap_ratio
            ):
                line.token_indices.append(idx)
                line.update_bounds(token)  # O(1) incremental update
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
            new_line.init_bounds(token)  # Initialize accumulators
            lines.append(new_line)
            line_id += 1

    # Finalize: sort token_indices by x0, compute final baseline_y
    for line in lines:
        line.token_indices.sort(key=lambda i: tokens[i].x0)
        y_centers = [(tokens[i].y0 + tokens[i].y1) * 0.5 for i in line.token_indices]
        line.baseline_y = median(y_centers) if y_centers else 0.0

    # Sort lines by (baseline_y, min_x) for reading order
    def line_sort_key(line: Line) -> Tuple[float, float]:
        """Sort key: (baseline_y, min_x) for reading order."""
        if not line.token_indices:
            return (0.0, 0.0)
        min_x = min(tokens[i].x0 for i in line.token_indices)
        return (line.baseline_y, min_x)

    lines.sort(key=line_sort_key)

    # Re-assign line_ids after sorting
    for i, line in enumerate(lines):
        line.line_id = i

    log.debug(
        "build_lines: %d tokens → %d lines (median_h=%.1f, vert_tol=%.1f)",
        len(tokens),
        len(lines),
        median_h,
        vert_tol,
    )
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
    median_space_gap: float = 3.5,
) -> List[Line]:
    """Split lines that have multiple widely-separated spans into sub-lines.

    After ``split_line_spans`` each Line has one or more Spans.  When a
    line has ≥2 spans those spans are already separated by a large
    horizontal gap (> span_gap_mult × median_space_gap).  This function
    creates a separate Line for each span so that downstream spatial
    block grouping can treat them independently.

    Additionally, any resulting single-span sub-line wider than 1.5× the
    median span width is checked for a large inter-token gap (>
    ``_wide_line_gap_mult × median_space_gap``).  If found, the sub-line
    is split at that gap.  This catches cross-column merges where the
    inter-column gap is smaller than the span threshold but still
    significantly larger than a normal word space.

    Returns a new list of Lines with sequential IDs.
    """
    _wide_line_gap_mult = 4.0  # gap must exceed this × median_space_gap

    # Compute median *span* width from multi-span lines so we have a
    # single-column width reference.  Single-span lines at this stage may
    # span the entire page, inflating the median if we used raw line widths.
    span_widths: List[float] = []
    for ln in lines:
        if len(ln.spans) >= 2:
            for sp in ln.spans:
                if sp.token_indices:
                    xs = [tokens[i].x0 for i in sp.token_indices]
                    xe = [tokens[i].x1 for i in sp.token_indices]
                    span_widths.append(max(xe) - min(xs))
    if not span_widths:
        # Fallback: use line widths directly
        span_widths = [
            ln.bbox(tokens)[2] - ln.bbox(tokens)[0] for ln in lines if ln.token_indices
        ]
    med_span_width = median(span_widths) if span_widths else 0.0
    wide_thresh = med_span_width * 1.5
    gap_thresh = max(median_space_gap * _wide_line_gap_mult, 12.0)

    # --- helper: try to split a single-span line at its largest gap ---
    def _try_gap_split(line: Line, next_id_val: int) -> tuple[list[Line], int] | None:
        """Return (sub_lines, updated_next_id) or None if no split needed."""
        bb = line.bbox(tokens)
        line_w = bb[2] - bb[0]
        if len(line.token_indices) < 2 or line_w <= wide_thresh:
            return None
        sorted_idx = sorted(line.token_indices, key=lambda i: tokens[i].x0)
        best_gap = 0.0
        best_pos = -1
        for j in range(len(sorted_idx) - 1):
            g = tokens[sorted_idx[j + 1]].x0 - tokens[sorted_idx[j]].x1
            if g > best_gap:
                best_gap = g
                best_pos = j
        if best_gap < gap_thresh or best_pos < 0:
            return None
        # Split into two sub-lines at the gap
        parts = [sorted_idx[: best_pos + 1], sorted_idx[best_pos + 1 :]]
        subs: list[Line] = []
        nid = next_id_val
        for sub_idx in parts:
            if not sub_idx:
                continue
            yc = [(tokens[i].y0 + tokens[i].y1) * 0.5 for i in sub_idx]
            sub_bl = median(yc) if yc else line.baseline_y
            sub_span = Span(token_indices=list(sub_idx), col_id=None)
            subs.append(
                Line(
                    line_id=nid,
                    page=line.page,
                    token_indices=list(sub_idx),
                    baseline_y=sub_bl,
                    spans=[sub_span],
                )
            )
            nid += 1
        return subs, nid

    # --- Phase 1: split multi-span lines into sub-lines per span,
    #              keep single-span lines as-is. ---
    phase1: List[Line] = []
    next_id = 0

    for line in lines:
        if len(line.spans) <= 1:
            line.line_id = next_id
            next_id += 1
            phase1.append(line)
            continue

        # Multi-span line: create a sub-line per span
        for span in line.spans:
            if not span.token_indices:
                continue
            sub_indices = sorted(span.token_indices, key=lambda i: tokens[i].x0)
            y_centers = [(tokens[i].y0 + tokens[i].y1) * 0.5 for i in sub_indices]
            sub_baseline = median(y_centers) if y_centers else line.baseline_y
            sub_line = Line(
                line_id=next_id,
                page=line.page,
                token_indices=sub_indices,
                baseline_y=sub_baseline,
                spans=[span],
            )
            next_id += 1
            phase1.append(sub_line)

    # --- Phase 2: split any wide single-span (sub-)lines at their
    #              largest inter-token gap.  This catches cross-column
    #              merges that survived the span threshold. ---
    result: List[Line] = []
    next_id = 0
    for line in phase1:
        split = _try_gap_split(line, next_id)
        if split is not None:
            subs, next_id = split
            result.extend(subs)
        else:
            line.line_id = next_id
            next_id += 1
            result.append(line)

    # Re-sort by (baseline_y, min_x) for reading order
    def _sort_key(ln: Line) -> tuple:
        """Sort key: (baseline_y, min_x) for reading order."""
        if not ln.token_indices:
            return (0.0, 0.0)
        return (ln.baseline_y, min(tokens[i].x0 for i in ln.token_indices))

    result.sort(key=_sort_key)
    for idx, ln in enumerate(result):
        ln.line_id = idx

    log.debug(
        "split_wide_lines: %d lines → %d lines (gap_thresh=%.1f)",
        len(lines),
        len(result),
        gap_thresh,
    )
    return result
