"""Legend detection and extraction from PDF pages."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

from ._abbreviation_detect import detect_abbreviation_regions  # noqa: F401

# --- Sub-module re-exports (public API) -----------------------------------
from ._graphics import extract_graphics  # noqa: F401
from ._misc_title_detect import detect_misc_title_regions  # noqa: F401

# --- Internal helpers used by this file only ------------------------------
from ._region_helpers import filter_graphics_outside_regions  # noqa: F401
from ._region_helpers import (
    _bboxes_overlap,
    _extract_text_rows_from_blocks,
    _find_enclosing_rect,
    _find_symbols_in_region,
    _find_text_blocks_in_region,
)
from ._revision_detect import detect_revision_regions  # noqa: F401
from ._standard_detail_detect import detect_standard_detail_regions  # noqa: F401
from .config import GroupingConfig
from .models import BlockCluster, GraphicElement, LegendEntry, LegendRegion


def _is_legend_header(block: BlockCluster) -> bool:
    """Check if a block is a legend header."""
    if not getattr(block, "is_header", False):
        return False
    if not block.rows:
        return False

    texts = [b.text for b in block.rows[0].boxes if b.text]
    header_text = " ".join(texts).strip().upper()

    # Check for common legend patterns
    return "LEGEND" in header_text


def _detect_legend_columns(
    symbols: List[GraphicElement],
    x_tolerance: float = 30.0,
) -> List[List[GraphicElement]]:
    """
    Group symbols into columns based on x-alignment.
    Returns a list of columns, each containing symbols sorted by y position.
    """
    if not symbols:
        return []

    # Sort by x position
    sorted_symbols = sorted(symbols, key=lambda s: s.x0)

    columns: List[List[GraphicElement]] = []
    current_col: List[GraphicElement] = [sorted_symbols[0]]

    for sym in sorted_symbols[1:]:
        # Check if aligned with current column
        col_x0 = current_col[0].x0
        if abs(sym.x0 - col_x0) <= x_tolerance:
            current_col.append(sym)
        else:
            # Start new column
            columns.append(current_col)
            current_col = [sym]

    if current_col:
        columns.append(current_col)

    # Sort each column by y position
    for col in columns:
        col.sort(key=lambda s: s.y0)

    return columns


def _pair_symbols_with_text(
    symbols: List[GraphicElement],
    blocks: List[BlockCluster],
    page: int,
    y_tolerance: float = 20.0,
    x_gap_max: float = 300.0,
) -> List[LegendEntry]:
    """
    Pair legend symbols with their text descriptions.
    Text should be to the right of the symbol and vertically aligned.

    This matches symbols to individual text rows, not whole blocks,
    since legend descriptions often appear as rows in a larger block.
    """
    entries = []
    used_rows = set()  # Track used (text, row_bbox) pairs

    # Extract all text rows from blocks
    text_rows = _extract_text_rows_from_blocks(blocks)

    for sym in symbols:
        sx0, sy0, sx1, sy1 = sym.bbox()
        sym_center_y = (sy0 + sy1) / 2

        best_row = None
        best_distance = float("inf")

        for row_text, row_bbox, parent_blk in text_rows:
            row_key = (row_text, row_bbox)
            if row_key in used_rows:
                continue

            rx0, ry0, rx1, ry1 = row_bbox
            row_center_y = (ry0 + ry1) / 2

            # Text should be to the right of symbol
            if rx0 < sx1:
                continue

            # Check x gap
            x_gap = rx0 - sx1
            if x_gap > x_gap_max:
                continue

            # Check y alignment
            y_diff = abs(row_center_y - sym_center_y)
            if y_diff > y_tolerance:
                continue

            # Find closest text row
            distance = x_gap + y_diff
            if distance < best_distance:
                best_distance = distance
                best_row = (row_text, row_bbox)

        if best_row:
            used_rows.add(best_row)
            row_text, row_bbox = best_row

            entries.append(
                LegendEntry(
                    page=page,
                    symbol=sym,
                    symbol_bbox=sym.bbox(),
                    description=row_text,
                    description_bbox=row_bbox,
                )
            )
        else:
            # Symbol without matching text
            entries.append(
                LegendEntry(
                    page=page,
                    symbol=sym,
                    symbol_bbox=sym.bbox(),
                )
            )

    return entries


def detect_legend_regions(
    blocks: List[BlockCluster],
    graphics: List[GraphicElement],
    page_width: float,
    page_height: float,
    exclusion_zones: List[Tuple[float, float, float, float]] = None,
    cfg: GroupingConfig | None = None,
) -> List[LegendRegion]:
    """
    Detect legend regions on a page.

    Process:
    1. Find blocks with "LEGEND" in header text
    2. Check if legend is boxed (enclosed in rectangle)
    3. Find symbols (small graphics) below/around the header
    4. Detect column structure in symbols
    5. Pair symbols with text descriptions

    Args:
        exclusion_zones: List of (x0, y0, x1, y1) bboxes to exclude from legend detection
                        (e.g., abbreviation regions)
    """
    exclusion_zones = exclusion_zones or []
    if cfg is None:
        cfg = GroupingConfig()
    legends: List[LegendRegion] = []

    # Find legend headers (skip those in exclusion zones)
    legend_headers = []
    for blk in blocks:
        if _is_legend_header(blk):
            blk_bbox = blk.bbox()
            # Check if header is in an exclusion zone
            in_exclusion = False
            for zone in exclusion_zones:
                if _bboxes_overlap(blk_bbox, zone):
                    in_exclusion = True
                    break
            if not in_exclusion:
                legend_headers.append(blk)

    logger.debug("detect_legend_regions: found %d legend headers", len(legend_headers))
    logger.debug("Total graphics: %d (lines/rects/curves)", len(graphics))

    for header in legend_headers:
        hx0, hy0, hx1, hy1 = header.bbox()
        header_text = ""
        if header.rows:
            header_text = " ".join(b.text for b in header.rows[0].boxes if b.text)

        logger.debug(
            "Processing legend '%s' at bbox=(%.1f, %.1f, %.1f, %.1f)",
            header_text,
            hx0,
            hy0,
            hx1,
            hy1,
        )

        page = header.page

        # Check for enclosing box
        enclosing_rect = _find_enclosing_rect(
            (hx0, hy0, hx1, hy1), graphics, tolerance=cfg.legend_enclosure_tolerance
        )
        is_boxed = enclosing_rect is not None
        box_bbox = enclosing_rect.bbox() if enclosing_rect else None

        if is_boxed:
            logger.debug("  Boxed legend, box bbox=%s", box_bbox)
            region_bbox = box_bbox
        else:
            # Define search region below header
            # Extend down and to the right until we hit another header or edge
            region_x0 = (
                hx0 - cfg.legend_unboxed_x_margin
            )  # Allow left margin for symbols
            region_y0 = hy0
            region_x1 = min(
                hx0 + cfg.legend_unboxed_x_extent, page_width
            )  # Extend significantly right for text descriptions
            region_y1 = min(
                hy1 + cfg.legend_unboxed_y_extent, page_height
            )  # Extend down

            # Try to find natural boundary (another header below)
            for blk in blocks:
                if blk is header:
                    continue
                if getattr(blk, "is_header", False):
                    bx0, by0, bx1, by1 = blk.bbox()
                    # If another header is below and overlaps in x
                    if by0 > hy1 and by0 < region_y1:
                        if not (bx1 < region_x0 or bx0 > region_x1):
                            region_y1 = by0 - 10

            region_bbox = (region_x0, region_y0, region_x1, region_y1)
            logger.debug(
                "  Unboxed legend, search region=(%.1f, %.1f, %.1f, %.1f)",
                region_x0,
                region_y0,
                region_x1,
                region_y1,
            )

        # Find symbols in region
        symbols = _find_symbols_in_region(
            region_bbox, graphics, max_symbol_size=cfg.legend_max_symbol_size
        )
        logger.debug("  Found %d potential symbols in region", len(symbols))

        # Filter out very tiny symbols (noise) and very large ones
        symbols = [
            s
            for s in symbols
            if s.area() > cfg.legend_symbol_min_area
            and s.area() < cfg.legend_symbol_max_area
        ]
        logger.debug("  After size filter: %d symbols", len(symbols))

        # Detect column structure
        symbol_columns = _detect_legend_columns(
            symbols, x_tolerance=cfg.legend_column_x_tolerance
        )
        logger.debug("  Detected %d symbol columns", len(symbol_columns))
        for i, col in enumerate(symbol_columns):
            logger.debug(
                "    Column %d: %d symbols at xâ‰ˆ%.1f",
                i,
                len(col),
                col[0].x0,
            )

        # Find text blocks in region, excluding those in exclusion zones
        text_blocks = _find_text_blocks_in_region(region_bbox, blocks, header)
        # Filter out text blocks that overlap with exclusion zones
        filtered_text_blocks = []
        for blk in text_blocks:
            blk_bbox = blk.bbox()
            in_exclusion = False
            for zone in exclusion_zones:
                if _bboxes_overlap(blk_bbox, zone):
                    in_exclusion = True
                    break
            if not in_exclusion:
                filtered_text_blocks.append(blk)
        text_blocks = filtered_text_blocks
        logger.debug(
            "  Found %d text blocks in region (after exclusion filter)",
            len(text_blocks),
        )

        # Pair symbols with text
        all_symbols = [s for col in symbol_columns for s in col]
        entries = _pair_symbols_with_text(
            all_symbols,
            text_blocks,
            page,
            y_tolerance=cfg.legend_text_y_tolerance,
            x_gap_max=cfg.legend_text_x_gap_max,
        )
        logger.debug("  Created %d legend entries", len(entries))
        for entry in entries:
            if entry.description:
                logger.debug(
                    "    Entry: '%s...' at %s",
                    entry.description[:50],
                    entry.symbol_bbox,
                )
            else:
                logger.debug("    Entry: (no text) at %s", entry.symbol_bbox)

        legend = LegendRegion(
            page=page,
            header=header,
            entries=entries,
            is_boxed=is_boxed,
            box_bbox=box_bbox,
        )
        legends.append(legend)

    return legends
