"""Shared geometry and detection helpers used by multiple legend-family detectors."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from .models import BlockCluster, GlyphBox, GraphicElement

logger = logging.getLogger("plancheck.legends")


# ---------------------------------------------------------------------------
# Bounding-box overlap
# ---------------------------------------------------------------------------


def _bboxes_overlap(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float],
) -> bool:
    """Check if two bounding boxes overlap."""
    x0a, y0a, x1a, y1a = bbox1
    x0b, y0b, x1b, y1b = bbox2
    return not (x1a < x0b or x0a > x1b or y1a < y0b or y0a > y1b)


# ---------------------------------------------------------------------------
# Enclosing rectangle
# ---------------------------------------------------------------------------


def _find_enclosing_rect(
    header_bbox: Tuple[float, float, float, float],
    graphics: List[GraphicElement],
    tolerance: float = 20.0,
) -> Optional[GraphicElement]:
    """Find a rectangle that encloses the legend header (boxed legend)."""
    hx0, hy0, hx1, hy1 = header_bbox

    for g in graphics:
        if g.element_type != "rect":
            continue

        # Check if this rect could enclose the header
        gx0, gy0, gx1, gy1 = g.bbox()

        # Rect should be larger than header and contain it
        if (
            gx0 <= hx0 + tolerance
            and gy0 <= hy0 + tolerance
            and gx1 >= hx1 - tolerance
            and gy1 >= hy1 - tolerance
        ):

            # Should be substantially larger (a real enclosing box)
            if g.width() > 100 and g.height() > 50:
                return g

    return None


# ---------------------------------------------------------------------------
# Symbols in region
# ---------------------------------------------------------------------------


def _find_symbols_in_region(
    region_bbox: Tuple[float, float, float, float],
    graphics: List[GraphicElement],
    max_symbol_size: float = 50.0,
) -> List[GraphicElement]:
    """Find small graphical elements within a region that could be legend symbols."""
    rx0, ry0, rx1, ry1 = region_bbox
    symbols = []

    for g in graphics:
        gx0, gy0, gx1, gy1 = g.bbox()

        # Check if within region
        if gx0 < rx0 or gx1 > rx1 or gy0 < ry0 or gy1 > ry1:
            continue

        # Check if small enough to be a symbol
        if g.is_small_symbol(max_symbol_size):
            symbols.append(g)

    return symbols


# ---------------------------------------------------------------------------
# Text blocks in region
# ---------------------------------------------------------------------------


def _find_text_blocks_in_region(
    region_bbox: Tuple[float, float, float, float],
    blocks: List[BlockCluster],
    exclude_header: Optional[BlockCluster] = None,
) -> List[BlockCluster]:
    """Find text blocks that overlap with a region."""
    rx0, ry0, rx1, ry1 = region_bbox
    result = []

    for blk in blocks:
        if blk is exclude_header:
            continue
        if getattr(blk, "is_header", False):
            continue
        if blk.is_table:
            continue

        bx0, by0, bx1, by1 = blk.bbox()

        # Check if block overlaps with region (not fully contained)
        # Overlap means: not (separated horizontally or vertically)
        if not (bx1 < rx0 or bx0 > rx1 or by1 < ry0 or by0 > ry1):
            result.append(blk)

    return result


# ---------------------------------------------------------------------------
# Extract text rows
# ---------------------------------------------------------------------------


def _extract_text_rows_from_blocks(
    blocks: List[BlockCluster],
) -> List[Tuple[str, Tuple[float, float, float, float], BlockCluster]]:
    """
    Extract individual text rows from blocks.
    Returns list of (text, row_bbox, parent_block).
    """
    rows = []
    for blk in blocks:
        for row in blk.rows:
            row_text = " ".join(
                b.text for b in sorted(row.boxes, key=lambda b: b.x0) if b.text
            ).strip()
            if row_text:
                rows.append((row_text, row.bbox(), blk))
    return rows


# ---------------------------------------------------------------------------
# Merge same-line rows
# ---------------------------------------------------------------------------


def _merge_same_line_rows(
    header: BlockCluster,
    blocks: List[BlockCluster],
) -> Tuple[str, Tuple[float, float, float, float]]:
    """
    Find blocks on the same line as the header's rows and merge their text.

    This handles cases where column partitioning splits a single line of text
    into multiple blocks (e.g., "THE FOLLOWING ... SHALL" in one block and
    "BE USED ON THIS PROJECT:" in another).

    Returns merged subheader text and bbox.
    """
    if len(header.rows) < 2:
        return None, None

    # Get header row 1 (the potential subheader row)
    row1 = header.rows[1]
    if not row1.boxes:
        return None, None

    row1_bbox = row1.bbox()
    row1_x0, row1_y0, row1_x1, row1_y1 = row1_bbox
    row1_y_center = (row1_y0 + row1_y1) / 2
    row1_height = row1_y1 - row1_y0

    # Get header x-range to constrain merging
    hx0, _, hx1, _ = header.bbox()

    # Collect all text boxes on the same line (from header and other blocks)
    line_boxes = list(row1.boxes)
    merged_blocks = []

    for blk in blocks:
        if blk is header:
            continue
        bx0, by0, bx1, by1 = blk.bbox()
        blk_y_center = (by0 + by1) / 2

        # Check if block is on the same line (y-center within half row height)
        # AND is to the right of header row (continuation, not unrelated content)
        # AND is within reasonable horizontal distance (within 300 pts of row end)
        if (
            abs(blk_y_center - row1_y_center) < row1_height * 0.6
            and bx0 >= row1_x1 - 20  # Block starts at or after row1 ends
            and bx0 <= row1_x1 + 300
        ):  # But not too far away
            # Add all glyph boxes from this block's rows that overlap the line
            for row in blk.rows:
                ry0, ry1 = row.bbox()[1], row.bbox()[3]
                ry_center = (ry0 + ry1) / 2
                if abs(ry_center - row1_y_center) < row1_height * 0.6:
                    line_boxes.extend(row.boxes)
                    merged_blocks.append(blk)
                    logger.debug("  Merging same-line block at x=%.1f", bx0)

    if not line_boxes:
        return None, None

    # Sort by x-position and build merged text
    line_boxes.sort(key=lambda b: b.x0)
    merged_text = " ".join(b.text for b in line_boxes if b.text).strip()

    # Calculate merged bbox
    merged_x0 = min(b.x0 for b in line_boxes)
    merged_y0 = min(b.y0 for b in line_boxes)
    merged_x1 = max(b.x1 for b in line_boxes)
    merged_y1 = max(b.y1 for b in line_boxes)
    merged_bbox = (merged_x0, merged_y0, merged_x1, merged_y1)

    return merged_text, merged_bbox


# ---------------------------------------------------------------------------
# Filter graphics outside regions
# ---------------------------------------------------------------------------


def filter_graphics_outside_regions(
    graphics: List[GraphicElement],
    exclusion_regions: List[Tuple[float, float, float, float]],
) -> List[GraphicElement]:
    """Filter out graphics that fall within exclusion regions."""
    result = []
    for g in graphics:
        g_bbox = g.bbox()
        is_excluded = False
        for region in exclusion_regions:
            if _bboxes_overlap(g_bbox, region):
                is_excluded = True
                break
        if not is_excluded:
            result.append(g)
    return result
