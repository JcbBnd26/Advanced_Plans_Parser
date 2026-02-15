"""Miscellaneous title region detection (department names, agency titles, etc.)."""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

from ._region_helpers import _bboxes_overlap
from .config import GroupingConfig
from .models import BlockCluster, GraphicElement, MiscTitleRegion

logger = logging.getLogger("plancheck.legends")


def _is_misc_title_text(text: str) -> bool:
    """Check if text looks like a misc title (department name, agency, etc.)."""
    text_upper = text.upper().strip()

    # Must be relatively short (title block items, not construction notes)
    if len(text_upper) > 100:
        return False

    # Common patterns for title block text that should be excluded
    # These are complete phrases that indicate government/agency titles
    patterns = [
        r"DEPARTMENT\s+OF\s+TRANSPORTATION",
        r"OKLAHOMA\s+DEPARTMENT\s+OF",
        r"STATE\s+OF\s+OKLAHOMA",
        r"COUNTY\s+OF\s+\w+",
        r"CITY\s+OF\s+\w+",
        r"PUBLIC\s+WORKS\s+DEPARTMENT",
        r"DIVISION\s+OF\s+HIGHWAYS",
        r"BUREAU\s+OF\s+\w+",
    ]
    for pattern in patterns:
        if re.search(pattern, text_upper):
            return True
    return False


def detect_misc_title_regions(
    blocks: List[BlockCluster],
    graphics: List[GraphicElement],
    page_width: float,
    page_height: float,
    exclusion_zones: List[Tuple[float, float, float, float]] = None,
    cfg: GroupingConfig | None = None,
) -> List[MiscTitleRegion]:
    """
    Detect miscellaneous title boxes (e.g., 'OKLAHOMA DEPARTMENT OF TRANSPORTATION').

    These are typically boxed text elements in the title block area that should
    be excluded from other detection (legends, abbreviations, etc.).
    """
    misc_titles: List[MiscTitleRegion] = []
    exclusion_zones = exclusion_zones or []

    logger.debug("detect_misc_title_regions: checking %d blocks", len(blocks))

    for blk in blocks:
        # Skip large blocks (more than 2 rows = probably notes, not title)
        if len(blk.rows) > 2:
            continue

        # Get all text from the block
        all_text = ""
        for row in blk.rows:
            row_text = " ".join(b.text for b in row.boxes if b.text).strip()
            all_text += " " + row_text
        all_text = all_text.strip()

        if not all_text:
            continue

        # Skip if in exclusion zone
        blk_bbox = blk.bbox()
        in_exclusion = False
        for ez in exclusion_zones:
            if _bboxes_overlap(blk_bbox, ez):
                in_exclusion = True
                break
        if in_exclusion:
            continue

        # Check if text matches misc title patterns
        if not _is_misc_title_text(all_text):
            continue

        bx0, by0, bx1, by1 = blk_bbox
        text_area = (bx1 - bx0) * (by1 - by0)
        logger.debug(
            "  Found misc title candidate: '%s...' at bbox=(%.1f, %.1f, %.1f, %.1f)",
            all_text[:50],
            bx0,
            by0,
            bx1,
            by1,
        )

        # Look for tightest enclosing box
        is_boxed = False
        box_bbox = None

        # Check for box formed by horizontal lines (common for rounded rect titles)
        # Find horizontal lines above and below the text
        # Note: Don't require lines to span the text - text may be clipped to page edge
        # but the visual box is smaller. Instead, find lines that overlap the text x-range.
        top_lines = []
        bottom_lines = []
        for g in graphics:
            if g.element_type == "line":
                line_width = abs(g.x1 - g.x0)
                line_height = abs(g.y1 - g.y0)
                # Horizontal line that overlaps text x-range (not necessarily spanning it)
                if line_width > 50 and line_height < 3:
                    # Check for x-overlap with text
                    if g.x1 > bx0 - 10 and g.x0 < bx1 + 10:
                        # Line above text (within 25 pts)
                        if g.y0 < by0 and g.y0 > by0 - 25:
                            top_lines.append(g)
                        # Line below text (within 25 pts)
                        elif g.y0 > by1 and g.y0 < by1 + 25:
                            bottom_lines.append(g)

        if top_lines and bottom_lines:
            # Use the closest lines to form the box
            top_line = min(top_lines, key=lambda g: abs(g.y0 - by0))
            bottom_line = min(bottom_lines, key=lambda g: abs(g.y0 - by1))
            lx0 = min(top_line.x0, bottom_line.x0)
            ly0 = top_line.y0
            lx1 = max(top_line.x1, bottom_line.x1)
            ly1 = bottom_line.y1

            # Look for rounded corner curves at the edges
            # These extend slightly beyond the horizontal lines
            for g in graphics:
                if g.element_type == "curve":
                    cx0, cy0, cx1, cy1 = g.bbox()
                    curve_height = cy1 - cy0
                    curve_width = cx1 - cx0
                    # Corner curve: small, near edge, spans the line height
                    if curve_width < 20 and curve_height > (ly1 - ly0) * 0.5:
                        # Left corner: near left edge of lines
                        if abs(cx1 - lx0) < 15 and cy0 <= ly0 + 5 and cy1 >= ly1 - 5:
                            lx0 = min(lx0, cx0)
                            logger.debug(
                                "    Extended left to corner curve at x=%.1f",
                                cx0,
                            )
                        # Right corner: near right edge of lines
                        if abs(cx0 - lx1) < 15 and cy0 <= ly0 + 5 and cy1 >= ly1 - 5:
                            lx1 = max(lx1, cx1)
                            logger.debug(
                                "    Extended right to corner curve at x=%.1f",
                                cx1,
                            )

            area = (lx1 - lx0) * (ly1 - ly0)
            if area < text_area * 15:  # Reasonably tight
                is_boxed = True
                box_bbox = (lx0, ly0, lx1, ly1)
                logger.debug(
                    "    Boxed by horizontal lines, box bbox=%s",
                    box_bbox,
                )

        # If not found by lines, check for curves (rounded rectangles)
        if not is_boxed:
            best_curve = None
            best_curve_area = float("inf")
            for g in graphics:
                if g.element_type == "curve":
                    gx0, gy0, gx1, gy1 = g.bbox()
                    # Check if curve encloses the text block with reasonable padding
                    if (
                        gx0 <= bx0 + 10
                        and gy0 <= by0 + 10
                        and gx1 >= bx1 - 10
                        and gy1 >= by1 - 10
                    ):
                        area = (gx1 - gx0) * (gy1 - gy0)
                        if area < best_curve_area and area < text_area * 10:
                            best_curve = g
                            best_curve_area = area

            if best_curve:
                is_boxed = True
                box_bbox = best_curve.bbox()
                logger.debug("    Boxed by curve, box bbox=%s", box_bbox)

        # If still not found, check for tight enclosing rectangles
        if not is_boxed:
            best_rect = None
            best_rect_area = float("inf")
            for g in graphics:
                if g.element_type == "rect":
                    gx0, gy0, gx1, gy1 = g.bbox()
                    if (
                        gx0 <= bx0 + 5
                        and gy0 <= by0 + 5
                        and gx1 >= bx1 - 5
                        and gy1 >= by1 - 5
                    ):
                        area = (gx1 - gx0) * (gy1 - gy0)
                        if area < best_rect_area and area < text_area * 10:
                            best_rect = g
                            best_rect_area = area

            if best_rect:
                is_boxed = True
                box_bbox = best_rect.bbox()
                logger.debug("    Boxed by rect, box bbox=%s", box_bbox)

        # Note: Don't extend bbox to include adjacent blocks - keep it at the actual box bounds
        # The overlay will handle combining text visually

        misc_title = MiscTitleRegion(
            page=blk.page,
            text=all_text,
            text_block=blk,
            is_boxed=is_boxed,
            box_bbox=box_bbox,
        )
        misc_titles.append(misc_title)

    logger.debug(
        "detect_misc_title_regions: found %d misc titles",
        len(misc_titles),
    )

    return misc_titles
