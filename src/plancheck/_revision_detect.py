"""Revision region detection and parsing."""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from ._region_helpers import _find_enclosing_rect, _find_text_blocks_in_region
from .config import GroupingConfig
from .models import (
    BlockCluster,
    GlyphBox,
    GraphicElement,
    RevisionEntry,
    RevisionRegion,
)

logger = logging.getLogger("plancheck.legends")


def _is_revision_header(blk: BlockCluster) -> bool:
    """Check if a block is a REVISIONS header."""
    if not blk.rows:
        return False
    first_row_text = (
        " ".join(b.text for b in blk.rows[0].boxes if b.text).strip().upper()
    )
    # Match "REVISIONS" or "REVISION" (with or without colon)
    return bool(re.match(r"^REVISIONS?:?\s*$", first_row_text))


def _has_revision_column_headers(blk: BlockCluster) -> bool:
    """Check if a block contains revision column headers (NO., DESCRIPTION, DATE)."""
    all_text = ""
    for row in blk.rows:
        row_text = " ".join(b.text for b in row.boxes if b.text).strip().upper()
        all_text += " " + row_text

    # Look for typical column header patterns
    has_no = "NO" in all_text or "NO." in all_text or "#" in all_text
    has_description = "DESCRIPTION" in all_text or "DESC" in all_text
    has_date = "DATE" in all_text

    return has_description or (has_no and has_date)


def _is_column_header_row(text: str) -> bool:
    """Check if text looks like a column header row."""
    text_upper = text.upper()
    return ("NO" in text_upper and "DESCRIPTION" in text_upper) or (
        "NO." in text_upper and "DATE" in text_upper
    )


def _parse_revision_row(
    boxes: List[GlyphBox],
    page: int,
    row_bbox: Tuple[float, float, float, float],
) -> Optional[RevisionEntry]:
    """
    Parse a single revision row into an entry.

    Typical format: NUMBER (narrow) | DESCRIPTION (wide) | DATE (medium)
    """
    if not boxes:
        return None

    # Sort boxes by x-position
    sorted_boxes = sorted(boxes, key=lambda b: b.x0)

    # Simple heuristic: first box is number, last box or boxes are date, middle is description
    if len(sorted_boxes) == 0:
        return None

    # Get texts
    texts = [b.text for b in sorted_boxes if b.text]
    if not texts:
        return None

    # First element is usually the revision number
    number = texts[0] if texts else ""

    # Last element(s) might be date - look for date pattern
    date = ""
    description_texts = texts[1:] if len(texts) > 1 else []

    # Check if last text looks like a date (contains / or - with numbers)
    if description_texts:
        last_text = description_texts[-1]
        if re.match(r".*\d+[/\-\.]\d+.*", last_text):
            date = last_text
            description_texts = description_texts[:-1]

    description = " ".join(description_texts)

    # Only return if we have at least a number
    if number:
        return RevisionEntry(
            page=page,
            number=number,
            description=description,
            date=date,
            row_bbox=row_bbox,
        )

    return None


def _parse_revision_entries(
    blocks: List[BlockCluster],
    page: int,
    region_bbox: Tuple[float, float, float, float],
) -> List[RevisionEntry]:
    """
    Parse revision entries from text blocks in the revision table.

    Looks for rows with NO., DESCRIPTION, DATE pattern.
    Many revision boxes are empty, so we may find 0 entries.
    """
    entries = []

    # Look for the column header row (NO., DESCRIPTION, DATE)
    column_header_y = None
    for blk in blocks:
        for row in blk.rows:
            row_text = " ".join(b.text for b in row.boxes if b.text).strip().upper()
            # Look for column headers
            if "NO" in row_text and ("DESCRIPTION" in row_text or "DATE" in row_text):
                row_bbox = row.bbox()
                column_header_y = row_bbox[3]  # Bottom of column header row
                logger.debug(
                    "  Found column headers at y=%.1f: '%s'",
                    column_header_y,
                    row_text,
                )
                break
        if column_header_y:
            break

    if not column_header_y:
        logger.debug("  No column headers found, checking for data rows directly")
        # Still try to find any data rows below the region start
        column_header_y = region_bbox[1] + 30  # Assume header takes ~30 pts

    # Collect all rows below the column header
    data_rows = []
    for blk in blocks:
        for row in blk.rows:
            row_bbox = row.bbox()
            # Row must be below column headers
            if row_bbox[1] >= column_header_y - 5:
                row_text = " ".join(b.text for b in row.boxes if b.text).strip()
                if row_text and not _is_column_header_row(row_text):
                    data_rows.append((row_bbox, row_text, row.boxes))

    # Sort by y-position
    data_rows.sort(key=lambda x: x[0][1])

    for row_bbox, row_text, boxes in data_rows:
        # Try to parse as revision entry
        # Format is typically: NUMBER  DESCRIPTION  DATE
        # We'll extract based on position
        entry = _parse_revision_row(boxes, page, row_bbox)
        if entry:
            entries.append(entry)
            logger.debug(
                "    Revision: No=%s, Desc=%s..., Date=%s",
                entry.number,
                entry.description[:30],
                entry.date,
            )

    return entries


def detect_revision_regions(
    blocks: List[BlockCluster],
    graphics: List[GraphicElement],
    page_width: float,
    page_height: float,
    exclusion_zones: List[Tuple[float, float, float, float]] = None,
    cfg: GroupingConfig | None = None,
) -> List[RevisionRegion]:
    """
    Detect revision boxes on a page.

    Revisions boxes typically have:
    - Header: "REVISIONS"
    - Columns: NO., DESCRIPTION, DATE
    - Usually enclosed in a rectangle/box
    - Often found in title block area (corner of page)

    Args:
        exclusion_zones: List of bboxes to exclude (e.g., abbreviation regions)
    """
    revisions: List[RevisionRegion] = []
    exclusion_zones = exclusion_zones or []

    # Find revision headers
    revision_headers = [blk for blk in blocks if _is_revision_header(blk)]

    logger.debug(
        "detect_revision_regions: found %d revision headers",
        len(revision_headers),
    )

    for header in revision_headers:
        hx0, hy0, hx1, hy1 = header.bbox()
        header_text = ""
        if header.rows:
            header_text = " ".join(b.text for b in header.rows[0].boxes if b.text)

        logger.debug(
            "Processing revision section '%s' at bbox=(%.1f, %.1f, %.1f, %.1f)",
            header_text,
            hx0,
            hy0,
            hx1,
            hy1,
        )

        page = header.page

        # Check if the header block itself contains column headers (common pattern)
        has_column_headers = _has_revision_column_headers(header)
        logger.debug(
            "  Has column headers in header block: %s",
            has_column_headers,
        )

        # Check for enclosing box - revision tables are almost always boxed
        enclosing_rect = _find_enclosing_rect((hx0, hy0, hx1, hy1), graphics)
        is_boxed = enclosing_rect is not None
        box_bbox = enclosing_rect.bbox() if enclosing_rect else None

        if is_boxed:
            logger.debug("  Boxed revision table, box bbox=%s", box_bbox)
            region_bbox = box_bbox
        else:
            # For unboxed revision tables, look for horizontal lines that define the table
            # Search for lines below or at the header that might define table rows
            # But stop before any exclusion zones (e.g., abbreviations)

            # Find the closest exclusion zone below the header
            max_y = hy1 + 150  # Default max search range
            for ez in exclusion_zones:
                ez_y0 = ez[1]  # top of exclusion zone
                if ez_y0 > hy1 and ez_y0 < max_y:
                    # Check if exclusion zone is roughly in same x-range
                    if ez[0] < hx1 + 100 and ez[2] > hx0 - 100:
                        max_y = ez_y0 - 5  # Stop just before exclusion zone
                        logger.debug(
                            "  Limiting search to y=%.1f due to exclusion zone at y=%.1f",
                            max_y,
                            ez_y0,
                        )

            table_lines = []
            for g in graphics:
                if g.element_type == "line":
                    # Horizontal line at or below header, but before exclusion zones?
                    if g.y0 >= hy0 - 5 and g.y0 < max_y:
                        line_width = abs(g.x1 - g.x0)
                        line_height = abs(g.y1 - g.y0)
                        # Horizontal line (much wider than tall) and near header x-range
                        if line_width > 50 and line_height < 3:
                            if g.x0 < hx1 + 50 and g.x1 > hx0 - 50:
                                table_lines.append(g)

            if table_lines:
                # Use the table lines to define the box boundary
                lowest_line_y = max(g.y1 for g in table_lines)
                highest_line_y = min(g.y0 for g in table_lines)
                region_x0 = min(g.x0 for g in table_lines)
                region_x1 = max(g.x1 for g in table_lines)
                region_y0 = highest_line_y
                region_y1 = lowest_line_y + 2
                logger.debug(
                    "  Found %d table lines, region=(%.1f, %.1f, %.1f, %.1f)",
                    len(table_lines),
                    region_x0,
                    region_y0,
                    region_x1,
                    region_y1,
                )
                # Treat table lines as defining a box
                is_boxed = True
                box_bbox = (region_x0, region_y0, region_x1, region_y1)
            else:
                # No table lines found - if header block contains column headers,
                # the revision table is likely empty. Use header bbox only.
                if has_column_headers:
                    region_x0 = hx0 - 5
                    region_y0 = hy0
                    region_x1 = hx1 + 5
                    region_y1 = hy1 + 5  # Just the header, no extension
                    logger.debug(
                        "  No table lines, header has column headers = empty table",
                    )
                else:
                    region_x0 = hx0 - 20
                    region_y0 = hy0
                    region_x1 = hx0 + 300
                    region_y1 = hy1 + 150
                    logger.debug("  No table lines, using estimated region")

            region_bbox = (region_x0, region_y0, region_x1, region_y1)
            logger.debug(
                "  Revision region=(%.1f, %.1f, %.1f, %.1f)",
                region_bbox[0],
                region_bbox[1],
                region_bbox[2],
                region_bbox[3],
            )

        # Find text blocks in region (skip the header itself)
        # If header already contains column headers, don't look for extra blocks (table is empty)
        if has_column_headers:
            text_blocks = []
            logger.debug("  Empty revision table (header has column headers)")
        else:
            text_blocks = _find_text_blocks_in_region(region_bbox, blocks, header)
        logger.debug(
            "  Found %d text blocks in revision region",
            len(text_blocks),
        )

        # Parse revision entries
        entries = _parse_revision_entries(text_blocks, page, region_bbox)
        logger.debug("  Parsed %d revision entries", len(entries))

        revision = RevisionRegion(
            page=page,
            header=header,
            entries=entries,
            is_boxed=is_boxed,
            box_bbox=box_bbox,
        )
        revisions.append(revision)

    return revisions
