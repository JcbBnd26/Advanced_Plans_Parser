"""Standard detail region detection and parsing."""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from ..config import GroupingConfig
from ..models import (
    BlockCluster,
    GraphicElement,
    StandardDetailEntry,
    StandardDetailRegion,
)
from .region_helpers import (
    _bboxes_overlap,
    _find_enclosing_rect,
    _find_text_blocks_in_region,
    _merge_same_line_rows,
    collect_rows_with_y,
    match_rows_by_y,
)

logger = logging.getLogger("plancheck.standard_details")


def _has_inline_entries(block: BlockCluster) -> bool:
    """Check if a multi-row header block has inline entries (vs subtitle rows).

    Returns True if row 1+ starts with a sheet number pattern like "BMPR-0", "SS-1".
    Returns False if row 1 looks like a subtitle ("THE FOLLOWING...", etc.)
    """
    if len(block.rows) < 2:
        return False

    # Check row 1 (first row after header)
    row1 = block.rows[1]
    if not row1.boxes:
        return False

    row1_text = " ".join(b.text for b in row1.boxes if b.text).strip()

    # Sheet number pattern: requires dash+digit OR just digits after letters
    # Examples: "BMPR-0", "SS-1", "SPI-5-2", "621-1"
    # Must have: letters followed by (dash+digit) or just digits
    sheet_pattern = re.compile(r"^[A-Z]{1,6}[-]\d")  # Requires dash then digit
    sheet_pattern2 = re.compile(r"^\d+[-]\d")  # Numeric like "621-1"

    first_word = row1_text.split()[0] if row1_text else ""

    if sheet_pattern.match(first_word) or sheet_pattern2.match(first_word):
        return True

    return False


def _is_standard_detail_header(block: BlockCluster) -> bool:
    """Check if a block is a Standard Details header.

    Matches patterns like:
    - "ODOT STANDARD DETAILS:"
    - "STANDARD DETAILS:"
    - "ODOT STANDARDS DETAILS:"
    - "ODOT STANDARDS" (common format)
    """
    if not block.rows:
        return False

    # Check the first row text
    texts = [b.text for b in block.rows[0].boxes if b.text]
    first_row_text = " ".join(texts).strip().upper()

    # Check for standard detail patterns
    # "ODOT STANDARDS" or "STANDARD DETAILS" or variations
    if "ODOT" in first_row_text and "STANDARD" in first_row_text:
        return True
    if "STANDARD" in first_row_text and "DETAIL" in first_row_text:
        return True

    return False


def _parse_standard_detail_entries(
    blocks: List[BlockCluster],
    page: int,
) -> List[StandardDetailEntry]:
    """Parse standard detail entries from blocks within a boxed region."""
    entries = []

    for blk in blocks:
        for row in blk.rows:
            boxes = sorted(row.boxes, key=lambda b: b.x0)
            row_text = " ".join(b.text for b in boxes if b.text).strip()

            if not row_text:
                continue

            # Try to parse "SHEET_NUM DESCRIPTION" format
            # Sheet numbers are typically short (e.g., "SS-1", "621-1")
            match = re.match(r"^([A-Z0-9]{1,5}[-]?\d*)\s+(.+)$", row_text)
            if match:
                sheet_num = match.group(1)
                description = match.group(2)

                sheet_box = boxes[0] if boxes else None
                desc_box = boxes[-1] if len(boxes) > 1 else None

                sheet_bbox_val = (
                    (sheet_box.x0, sheet_box.y0, sheet_box.x1, sheet_box.y1)
                    if sheet_box
                    else None
                )
                desc_bbox_val = (
                    (desc_box.x0, desc_box.y0, desc_box.x1, desc_box.y1)
                    if desc_box
                    else None
                )

                entries.append(
                    StandardDetailEntry(
                        page=page,
                        sheet_number=sheet_num,
                        description=description,
                        sheet_bbox=sheet_bbox_val,
                        description_bbox=desc_bbox_val,
                    )
                )

    return entries


def _parse_standard_detail_entries_from_inline_blocks(
    inline_blocks: List[BlockCluster],
    page: int,
) -> List[StandardDetailEntry]:
    """
    Parse standard detail entries from inline entry blocks.

    Each row in the block has format: "SHEET-NUM DESCRIPTION WORDS"
    (e.g., "BMPR-0 BEST MANAGEMENT PRACTICE REFERENCE MATRIX")
    """
    entries = []

    # Sheet number pattern at start of row
    # Allow alphanumeric (letters+digits) before dash, e.g., "CET6D-4-2"
    sheet_pattern = re.compile(
        r"^([A-Z0-9]{1,8}[-]\d[A-Z0-9\-]*)\s+(.*)", re.IGNORECASE
    )

    for blk in inline_blocks:
        for row in blk.rows:
            if not row.boxes:
                continue

            boxes = sorted(row.boxes, key=lambda b: b.x0)
            row_text = " ".join(b.text for b in boxes if b.text).strip()

            if not row_text:
                continue

            match = sheet_pattern.match(row_text)
            if match:
                sheet_num = match.group(1)
                description = match.group(2)

                # Sheet bbox from first box, description from rest
                sheet_box = boxes[0] if boxes else None
                sheet_bbox_val = (
                    (sheet_box.x0, sheet_box.y0, sheet_box.x1, sheet_box.y1)
                    if sheet_box
                    else None
                )

                # Description bbox covers remaining boxes
                if len(boxes) > 1:
                    desc_boxes = boxes[1:]
                    desc_x0 = min(b.x0 for b in desc_boxes)
                    desc_y0 = min(b.y0 for b in desc_boxes)
                    desc_x1 = max(b.x1 for b in desc_boxes)
                    desc_y1 = max(b.y1 for b in desc_boxes)
                    desc_bbox_val = (desc_x0, desc_y0, desc_x1, desc_y1)
                else:
                    desc_bbox_val = sheet_bbox_val

                entries.append(
                    StandardDetailEntry(
                        page=page,
                        sheet_number=sheet_num,
                        description=description,
                        sheet_bbox=sheet_bbox_val,
                        description_bbox=desc_bbox_val,
                    )
                )
                logger.debug(
                    "    Inline entry: '%s' = '%s...'",
                    sheet_num,
                    description[:50],
                )

    return entries


def _parse_standard_detail_entries_inline(
    header_block: BlockCluster,
    page: int,
) -> List[StandardDetailEntry]:
    """
    Parse standard detail entries from a multi-row header block.

    The header block contains the header in row 0 and entries in subsequent rows.
    Each entry row typically has format: "SHEET-NUM - DESCRIPTION WORDS"
    (e.g., "PCES-5-1 - PREFABRICATED CULVERT END")
    """
    entries = []

    # Sheet number pattern: alphanumeric with optional dash and numbers
    sheet_pattern = re.compile(r"^([A-Z0-9]+-?\d*-?\d*)")

    # Skip the first row (header), process remaining rows
    for row_idx, row in enumerate(header_block.rows[1:], start=1):
        if not row.boxes:
            continue

        boxes = sorted(row.boxes, key=lambda b: b.x0)
        row_text = " ".join(b.text for b in boxes if b.text).strip()

        if not row_text:
            continue

        # Try to parse "SHEET-NUM - DESCRIPTION" format
        # Look for pattern like "PCES-5-1 - PREFABRICATED CULVERT END"
        match = sheet_pattern.match(row_text)
        if match:
            sheet_num = match.group(1)
            # Get description (everything after " - " or after the sheet number)
            remaining = row_text[len(sheet_num) :].strip()
            if remaining.startswith("-"):
                remaining = remaining[1:].strip()
            description = remaining

            # Get bounding boxes
            sheet_box = boxes[0] if boxes else None
            # Description covers remaining boxes
            if len(boxes) > 1:
                desc_boxes = boxes[1:]
                desc_x0 = min(b.x0 for b in desc_boxes)
                desc_y0 = min(b.y0 for b in desc_boxes)
                desc_x1 = max(b.x1 for b in desc_boxes)
                desc_y1 = max(b.y1 for b in desc_boxes)
                desc_bbox = (desc_x0, desc_y0, desc_x1, desc_y1)
            else:
                desc_bbox = None

            sheet_bbox_val = (
                (sheet_box.x0, sheet_box.y0, sheet_box.x1, sheet_box.y1)
                if sheet_box
                else None
            )

            entries.append(
                StandardDetailEntry(
                    page=page,
                    sheet_number=sheet_num,
                    description=description,
                    sheet_bbox=sheet_bbox_val,
                    description_bbox=desc_bbox,
                )
            )
            logger.debug("    Inline entry: '%s' = '%s'", sheet_num, description)

    return entries


def _parse_standard_detail_entries_two_column(
    sheet_blocks: List[BlockCluster],
    description_blocks: List[BlockCluster],
    page: int,
) -> List[StandardDetailEntry]:
    """
    Parse standard detail entries from paired sheet number and description blocks.

    Each sheet block should have a corresponding description block to its right.
    Rows are matched by y-position.
    """
    entries = []

    sheet_rows = collect_rows_with_y(sheet_blocks, skip_empty_boxes=True)
    desc_rows = collect_rows_with_y(description_blocks, skip_empty_boxes=True)

    # Match sheet rows to description rows by y-position (non-exclusive)
    pairs = match_rows_by_y(sheet_rows, desc_rows, y_tolerance=10.0, exclusive=False)
    matched_left = {li for li, _ in pairs}

    for li, ri in pairs:
        s_y, s_text, s_bbox, _ = sheet_rows[li]
        if not s_text:
            continue
        d_y, d_text, d_bbox, _ = desc_rows[ri]
        entries.append(
            StandardDetailEntry(
                page=page,
                sheet_number=s_text,
                description=d_text,
                sheet_bbox=s_bbox,
                description_bbox=d_bbox,
            )
        )

    # Unmatched sheet rows — add with empty description
    for li, (s_y, s_text, s_bbox, _) in enumerate(sheet_rows):
        if li in matched_left or not s_text:
            continue
        entries.append(
            StandardDetailEntry(
                page=page,
                sheet_number=s_text,
                description="",
                sheet_bbox=s_bbox,
                description_bbox=None,
            )
        )

    return entries


def detect_standard_detail_regions(
    blocks: List[BlockCluster],
    graphics: List[GraphicElement],
    page_width: float,
    page_height: float,
    exclusion_zones: Optional[List[Tuple[float, float, float, float]]] = None,
    cfg: GroupingConfig | None = None,
) -> List[StandardDetailRegion]:
    """
    Detect standard detail regions on a page.

    Standard details are two-column text (sheet number + description), similar to abbreviations.
    Format example:
        SS-1        PAVEMENT MARKING STANDARD
        621-1       STORM SEWER DETAILS

    This uses the same detection logic as abbreviations.
    """
    exclusion_zones = exclusion_zones or []
    details: List[StandardDetailRegion] = []

    # Find standard detail headers
    detail_headers = [blk for blk in blocks if _is_standard_detail_header(blk)]

    logger.debug(
        "detect_standard_detail_regions: found %d standard detail headers",
        len(detail_headers),
    )

    for header in detail_headers:
        hx0, hy0, hx1, hy1 = header.bbox()

        # Check if header is in an exclusion zone
        header_excluded = False
        for zone in exclusion_zones:
            if _bboxes_overlap((hx0, hy0, hx1, hy1), zone):
                header_excluded = True
                break
        if header_excluded:
            logger.debug(
                "Skipping standard detail header in exclusion zone at (%.1f, %.1f)",
                hx0,
                hy0,
            )
            continue

        header_text = ""
        if header.rows:
            header_text = " ".join(b.text for b in header.rows[0].boxes if b.text)

        logger.debug(
            "Processing standard detail section '%s' at bbox=(%.1f, %.1f, %.1f, %.1f)",
            header_text,
            hx0,
            hy0,
            hx1,
            hy1,
        )

        page = header.page

        # Check for enclosing box
        enclosing_rect = _find_enclosing_rect((hx0, hy0, hx1, hy1), graphics)
        is_boxed = enclosing_rect is not None
        box_bbox = enclosing_rect.bbox() if enclosing_rect else None

        if is_boxed:
            logger.debug("  Boxed standard detail, box bbox=%s", box_bbox)
            region_bbox = box_bbox
            text_blocks = _find_text_blocks_in_region(region_bbox, blocks, header)
            entries = _parse_standard_detail_entries(text_blocks, page)
            subheader = None
            subheader_bbox = None
        elif len(header.rows) > 1 and _has_inline_entries(header):
            # Multi-row header block with actual entries (row 1+ starts with sheet number)
            logger.debug(
                "  Multi-row header block with inline entries (%d rows)",
                len(header.rows),
            )
            entries = _parse_standard_detail_entries_inline(header, page)
            box_bbox = header.bbox()
            subheader = None
            subheader_bbox = None
        else:
            # Header may have subtitle row, entries are in separate blocks below
            # Use the new merge function to handle split rows
            subheader, subheader_bbox = _merge_same_line_rows(header, blocks)

            if subheader:
                logger.debug("  Subheader detected: '%s'", subheader)

            # For unboxed standard details with separate blocks,
            # find sheet number blocks and description blocks
            # Similar to abbreviation detection

            # Search range: below header, within reasonable x-range
            search_x0 = hx0 - 50  # Tighter range - entries should be near header
            search_x1 = (
                hx0 + 400
            )  # Wider than abbreviations since descriptions are longer

            # Pattern for sheet numbers (e.g., "SS-1", "621-1", "BMPR-0", "TESCA-0", etc.)
            sheet_pattern = re.compile(r"^[A-Z0-9]{1,8}[-]?\d*[-]?\d*$")
            # Pattern for inline entries: "SHEET-NUM DESCRIPTION" - requires dash+digit
            # Examples: "BMPR-0 BEST...", "SSS-2-1 SOLID...", "CET6D-4-2 CULVERT..."
            # Allow alphanumeric (letters+digits) before dash
            inline_entry_pattern = re.compile(r"^[A-Z0-9]{1,8}[-]\d")

            # First pass: find entry blocks below header
            # These can be either:
            # 1. Sheet number column blocks (>50% rows are just sheet numbers)
            # 2. Inline entry blocks (rows start with sheet number pattern)
            sheet_blocks = []
            inline_entry_blocks = []

            for blk in blocks:
                if blk is header:
                    continue
                bx0, by0, bx1, by1 = blk.bbox()

                # Must be below header and in search x-range
                if by0 < hy1 or bx1 < search_x0 or bx0 > search_x1:
                    continue

                # Count sheet-number-like rows and inline entry rows
                if len(blk.rows) == 0:
                    continue
                sheet_only_count = 0
                inline_entry_count = 0
                for row in blk.rows:
                    row_text = " ".join(b.text for b in row.boxes if b.text).strip()
                    if sheet_pattern.match(row_text):
                        sheet_only_count += 1
                    elif inline_entry_pattern.match(row_text):
                        inline_entry_count += 1

                # Check for sheet-only block (majority sheet numbers)
                if sheet_only_count > len(blk.rows) * 0.5 and len(blk.rows) >= 2:
                    sheet_blocks.append(blk)
                    logger.debug(
                        "  Sheet number block found: x0=%.1f, y0=%.1f, rows=%d",
                        bx0,
                        by0,
                        len(blk.rows),
                    )
                # Check for inline entry block (majority inline entries)
                elif inline_entry_count > len(blk.rows) * 0.5 and len(blk.rows) >= 2:
                    inline_entry_blocks.append(blk)
                    logger.debug(
                        "  Inline entry block found: x0=%.1f, y0=%.1f, rows=%d",
                        bx0,
                        by0,
                        len(blk.rows),
                    )

            # Second pass: find description blocks (to the right of sheet blocks, similar y-range)
            description_blocks = []
            for blk in blocks:
                if blk is header or blk in sheet_blocks or blk in inline_entry_blocks:
                    continue
                bx0, by0, bx1, by1 = blk.bbox()

                # Must be below header
                if by0 < hy1:
                    continue

                # Check if this block is to the right of any sheet block with similar y-range
                for sb in sheet_blocks:
                    sbx0, sby0, sbx1, sby1 = sb.bbox()

                    # Must be to the right of sheet block (within 150 pts)
                    # and have similar y-position (start within 20 pts)
                    if bx0 > sbx0 and bx0 < sbx1 + 150:
                        if abs(by0 - sby0) < 20:
                            # Row count should be similar (within 30%)
                            row_ratio = (
                                len(blk.rows) / len(sb.rows) if len(sb.rows) > 0 else 0
                            )
                            if 0.6 < row_ratio < 1.5:
                                description_blocks.append(blk)
                                logger.debug(
                                    "  Description block found: x0=%.1f, y0=%.1f, rows=%d",
                                    bx0,
                                    by0,
                                    len(blk.rows),
                                )
                                break

            logger.debug(
                "  Total: %d sheet blocks, %d desc blocks, %d inline blocks",
                len(sheet_blocks),
                len(description_blocks),
                len(inline_entry_blocks),
            )

            # Calculate region from all found blocks
            all_detail_blocks = sheet_blocks + description_blocks + inline_entry_blocks
            if all_detail_blocks:
                all_x0 = [blk.bbox()[0] for blk in all_detail_blocks]
                all_y0 = [blk.bbox()[1] for blk in all_detail_blocks]
                all_x1 = [blk.bbox()[2] for blk in all_detail_blocks]
                all_y1 = [blk.bbox()[3] for blk in all_detail_blocks]

                # Include subheader bbox if present
                sub_x1 = subheader_bbox[2] if subheader_bbox else hx1

                region_x0 = min(min(all_x0), hx0) - 10
                region_y0 = hy0
                region_x1 = max(max(all_x1), hx1, sub_x1) + 10
                region_y1 = max(all_y1) + 10
            else:
                # Fallback to fixed extension
                sub_x1 = subheader_bbox[2] if subheader_bbox else hx1
                region_x0 = hx0 - 50
                region_y0 = hy0
                region_x1 = max(min(hx0 + 500, page_width), sub_x1 + 10)
                region_y1 = min(hy1 + 600, page_height)

            region_bbox = (region_x0, region_y0, region_x1, region_y1)
            box_bbox = region_bbox  # Use calculated region as box
            logger.debug(
                "  Region bbox=(%.1f, %.1f, %.1f, %.1f)",
                region_x0,
                region_y0,
                region_x1,
                region_y1,
            )

            # Parse entries - use inline if we have inline blocks, else use two-column
            if inline_entry_blocks:
                entries = _parse_standard_detail_entries_from_inline_blocks(
                    inline_entry_blocks,
                    page,
                )
            else:
                entries = _parse_standard_detail_entries_two_column(
                    sheet_blocks,
                    description_blocks,
                    page,
                )

        logger.debug("  Parsed %d standard detail entries", len(entries))
        for entry in entries[:10]:  # Show first 10
            logger.debug("    '%s' = '%s'", entry.sheet_number, entry.description)
        if len(entries) > 10:
            logger.debug("    ... and %d more", len(entries) - 10)

        # Compute detection confidence:
        #   - boxed regions are more reliable (+0.3)
        #   - entries confirm structure (+0.3 scaled by count)
        #   - subheader confirms context (+0.1)
        #   - header pattern match is baseline (+0.3)
        conf = 0.3  # baseline: header matched
        if is_boxed:
            conf += 0.3
        if subheader:
            conf += 0.1
        if entries:
            conf += min(0.3, 0.1 * len(entries))  # up to +0.3 for >=3 entries
        conf = round(min(1.0, conf), 2)

        detail_region = StandardDetailRegion(
            page=page,
            header=header,
            subheader=subheader,
            subheader_bbox=subheader_bbox,
            entries=entries,
            is_boxed=is_boxed,
            box_bbox=box_bbox,
            confidence=conf,
        )
        details.append(detail_region)

    return details
