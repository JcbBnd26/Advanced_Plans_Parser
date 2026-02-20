"""Abbreviation region detection and parsing."""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from ..config import GroupingConfig
from ..models import AbbreviationEntry, AbbreviationRegion, BlockCluster, GraphicElement
from .region_helpers import (
    _find_enclosing_rect,
    _find_text_blocks_in_region,
    collect_rows_with_y,
    match_rows_by_y,
)

logger = logging.getLogger("plancheck.abbreviations")


def _is_abbreviation_header(block: BlockCluster) -> bool:
    """Check if a block is an abbreviation header.

    Note: Unlike legends, abbreviation headers often don't have a colon,
    so we don't require the block to be marked as is_header.
    """
    if not block.rows:
        return False

    texts = [b.text for b in block.rows[0].boxes if b.text]
    header_text = " ".join(texts).strip().upper()

    # Check for abbreviation patterns - single word "ABBREVIATIONS" or "ABBREVIATION"
    # It may or may not have a colon
    return "ABBREVIATION" in header_text


def _parse_abbreviation_entries_two_column(
    code_blocks: List[BlockCluster],
    meaning_blocks: List[BlockCluster],
    page: int,
) -> List[AbbreviationEntry]:
    """
    Parse abbreviation entries from paired code and meaning blocks.

    Each code block should have a corresponding meaning block to its right.
    Rows are matched by y-position.
    """
    entries = []

    code_rows = collect_rows_with_y(code_blocks)
    meaning_rows = collect_rows_with_y(meaning_blocks)

    code_rows.sort(key=lambda x: x[0])
    meaning_rows.sort(key=lambda x: x[0])

    for li, ri in match_rows_by_y(
        code_rows, meaning_rows, y_tolerance=5.0, exclusive=True
    ):
        c_y, c_text, c_bbox, _ = code_rows[li]
        m_y, m_text, m_bbox, _ = meaning_rows[ri]
        entries.append(
            AbbreviationEntry(
                page=page,
                code=c_text,
                meaning=m_text,
                code_bbox=c_bbox,
                meaning_bbox=m_bbox,
            )
        )

    return entries


def _parse_abbreviation_entries(
    blocks: List[BlockCluster],
    page: int,
) -> List[AbbreviationEntry]:
    """
    Parse abbreviation entries from text blocks (for boxed abbreviation regions).

    Handles two common layouts:
    1. Single-column: "AI = AREA INLET" or "AI  AREA INLET"
    2. Two-column: Left block has codes (AI, BOC), right block has meanings (AREA INLET, BACK OF CURB)

    For two-column layouts, we match rows by y-position.
    """
    entries = []

    # Collect all text rows with their y-positions and x-positions
    all_rows = []  # List of (y_center, x0, row_text, row_bbox, parent_block)
    for blk in blocks:
        for row in blk.rows:
            boxes = sorted(row.boxes, key=lambda b: b.x0)
            row_text = " ".join(b.text for b in boxes if b.text).strip()
            if not row_text:
                continue
            row_bbox = row.bbox()
            y_center = (row_bbox[1] + row_bbox[3]) / 2
            x0 = row_bbox[0]
            all_rows.append((y_center, x0, row_text, row_bbox, blk))

    if not all_rows:
        return entries

    # Determine if this is a two-column layout
    # Group by x-position to find columns
    x_positions = sorted(set(r[1] for r in all_rows))

    # Check if we have two distinct x-groups (columns)
    left_rows = []
    right_rows = []

    if len(x_positions) >= 2:
        # Find the gap between columns
        min_x = min(x_positions)
        max_x = max(x_positions)
        x_threshold = (min_x + max_x) / 2

        for y_center, x0, row_text, row_bbox, blk in all_rows:
            if x0 < x_threshold:
                left_rows.append((y_center, row_text, row_bbox))
            else:
                right_rows.append((y_center, row_text, row_bbox))

    # If we have two columns, match by y-position
    if left_rows and right_rows:
        # Check if left column looks like abbreviation codes (short, uppercase)
        code_pattern = re.compile(r"^[A-Z0-9]{1,6}$")
        left_looks_like_codes = (
            sum(1 for _, txt, _ in left_rows if code_pattern.match(txt))
            > len(left_rows) / 2
        )

        if left_looks_like_codes:
            # Match left (codes) with right (meanings) by y-position
            for li, ri in match_rows_by_y(
                left_rows, right_rows, y_tolerance=5.0, exclusive=True
            ):
                _, code_text, code_bbox = left_rows[li]
                _, meaning_text, meaning_bbox = right_rows[ri]
                entries.append(
                    AbbreviationEntry(
                        page=page,
                        code=code_text,
                        meaning=meaning_text,
                        code_bbox=code_bbox,
                        meaning_bbox=meaning_bbox,
                    )
                )

            return entries

    # Fallback: Single-column parsing (original logic)
    for blk in blocks:
        for row in blk.rows:
            boxes = sorted(row.boxes, key=lambda b: b.x0)
            row_text = " ".join(b.text for b in boxes if b.text).strip()

            if not row_text:
                continue

            # Try to parse "CODE = MEANING" or "CODE MEANING"
            code = None
            meaning = None

            # Look for = sign
            if "=" in row_text:
                parts = row_text.split("=", 1)
                code = parts[0].strip()
                meaning = parts[1].strip() if len(parts) > 1 else ""
            else:
                # Try to find short code at start (1-5 chars, uppercase)
                match = re.match(r"^([A-Z0-9]{1,5})\s+(.+)$", row_text)
                if match:
                    code = match.group(1)
                    meaning = match.group(2)

            if code and meaning:
                # Get bounding boxes
                code_box = boxes[0] if boxes else None
                meaning_box = boxes[-1] if len(boxes) > 1 else None

                code_bbox_val = (
                    (code_box.x0, code_box.y0, code_box.x1, code_box.y1)
                    if code_box
                    else None
                )
                meaning_bbox_val = (
                    (meaning_box.x0, meaning_box.y0, meaning_box.x1, meaning_box.y1)
                    if meaning_box
                    else None
                )

                entries.append(
                    AbbreviationEntry(
                        page=page,
                        code=code,
                        meaning=meaning,
                        code_bbox=code_bbox_val,
                        meaning_bbox=meaning_bbox_val,
                    )
                )

    return entries


def detect_abbreviation_regions(
    blocks: List[BlockCluster],
    graphics: List[GraphicElement],
    page_width: float,
    page_height: float,
    cfg: GroupingConfig | None = None,
) -> List[AbbreviationRegion]:
    """
    Detect abbreviation regions on a page.

    Abbreviations are pure text (code + meaning), NO graphics.
    They typically have format like:
        AI = AREA INLET
        BOC = BACK OF CURB
    or with equal sign or just spacing between code and meaning.

    This function identifies these regions so they can be EXCLUDED from legend detection.
    """
    abbreviations: List[AbbreviationRegion] = []

    # Find abbreviation headers
    abbrev_headers = [blk for blk in blocks if _is_abbreviation_header(blk)]

    logger.debug(
        "detect_abbreviation_regions: found %d abbreviation headers",
        len(abbrev_headers),
    )

    for header in abbrev_headers:
        hx0, hy0, hx1, hy1 = header.bbox()
        header_text = ""
        if header.rows:
            header_text = " ".join(b.text for b in header.rows[0].boxes if b.text)

        logger.debug(
            "Processing abbreviation section '%s' at bbox=(%.1f, %.1f, %.1f, %.1f)",
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
            logger.debug("  Boxed abbreviation, box bbox=%s", box_bbox)
            region_bbox = box_bbox
            text_blocks = _find_text_blocks_in_region(region_bbox, blocks, header)
            entries = _parse_abbreviation_entries(text_blocks, page)
        else:
            # For unboxed abbreviations, find code blocks (short uppercase codes)
            # and their paired meaning blocks to the right

            # Search range: below header, within reasonable x-range
            search_x0 = hx0 - 100
            search_x1 = hx0 + 300

            # Pattern for short abbreviation codes (1-6 uppercase letters/digits)
            code_pattern = re.compile(r"^[A-Z0-9\.]{1,6}$")

            # First pass: find all code blocks (blocks where >50% of rows are short codes)
            code_blocks = []
            for blk in blocks:
                if blk is header:
                    continue
                bx0, by0, bx1, by1 = blk.bbox()

                # Must be below header and in search x-range
                if by0 < hy1 or bx1 < search_x0 or bx0 > search_x1:
                    continue

                # Count code-like rows
                if len(blk.rows) == 0:
                    continue
                code_row_count = 0
                for row in blk.rows:
                    row_text = " ".join(b.text for b in row.boxes if b.text).strip()
                    if code_pattern.match(row_text):
                        code_row_count += 1

                # Must have majority code rows and at least 3 rows
                if code_row_count > len(blk.rows) * 0.5 and len(blk.rows) >= 3:
                    code_blocks.append(blk)
                    logger.debug(
                        "  Code block found: x0=%.1f, y0=%.1f, rows=%d",
                        bx0,
                        by0,
                        len(blk.rows),
                    )

            # Second pass: find meaning blocks (to the right of code blocks, similar y-range)
            meaning_blocks = []
            for blk in blocks:
                if blk is header or blk in code_blocks:
                    continue
                bx0, by0, bx1, by1 = blk.bbox()

                # Must be below header
                if by0 < hy1:
                    continue

                # Check if this block is to the right of any code block with similar y-range
                for cb in code_blocks:
                    cbx0, cby0, cbx1, cby1 = cb.bbox()

                    # Must be to the right of code block (within 100 pts)
                    # and have similar y-position (start within 20 pts)
                    if bx0 > cbx0 and bx0 < cbx1 + 100:
                        if abs(by0 - cby0) < 20:
                            # Row count should be similar (within 20%)
                            row_ratio = (
                                len(blk.rows) / len(cb.rows) if len(cb.rows) > 0 else 0
                            )
                            if 0.7 < row_ratio < 1.3:
                                meaning_blocks.append(blk)
                                logger.debug(
                                    "  Meaning block found: x0=%.1f, y0=%.1f, rows=%d",
                                    bx0,
                                    by0,
                                    len(blk.rows),
                                )
                                break

            logger.debug(
                "  Total: %d code blocks, %d meaning blocks",
                len(code_blocks),
                len(meaning_blocks),
            )

            # Calculate region from all found blocks
            all_abbrev_blocks = code_blocks + meaning_blocks
            if all_abbrev_blocks:
                all_x0 = [blk.bbox()[0] for blk in all_abbrev_blocks]
                all_y0 = [blk.bbox()[1] for blk in all_abbrev_blocks]
                all_x1 = [blk.bbox()[2] for blk in all_abbrev_blocks]
                all_y1 = [blk.bbox()[3] for blk in all_abbrev_blocks]

                region_x0 = min(min(all_x0), hx0) - 10
                region_y0 = hy0
                region_x1 = max(max(all_x1), hx1) + 10
                region_y1 = max(all_y1) + 10
            else:
                # Fallback to fixed extension
                region_x0 = hx0 - 50
                region_y0 = hy0
                region_x1 = min(hx0 + 300, page_width)
                region_y1 = min(hy1 + 600, page_height)

            region_bbox = (region_x0, region_y0, region_x1, region_y1)
            logger.debug(
                "  Region bbox=(%.1f, %.1f, %.1f, %.1f)",
                region_x0,
                region_y0,
                region_x1,
                region_y1,
            )

            # Parse entries from the specific code and meaning blocks
            entries = _parse_abbreviation_entries_two_column(
                code_blocks,
                meaning_blocks,
                page,
            )

        logger.debug("  Parsed %d abbreviation entries", len(entries))
        for entry in entries[:10]:  # Show first 10
            logger.debug("    '%s' = '%s'", entry.code, entry.meaning)
        if len(entries) > 10:
            logger.debug("    ... and %d more", len(entries) - 10)

        # Compute detection confidence:
        #   - boxed regions are more reliable (+0.3)
        #   - having entries confirms the structure (+0.4 scaled by count)
        #   - header pattern match is baseline (+0.3)
        conf = 0.3  # baseline: header matched
        if is_boxed:
            conf += 0.3
        if entries:
            conf += min(0.4, 0.1 * len(entries))  # up to +0.4 for >=4 entries
        conf = round(min(1.0, conf), 2)

        abbrev = AbbreviationRegion(
            page=page,
            header=header,
            entries=entries,
            is_boxed=is_boxed,
            box_bbox=box_bbox,
            confidence=conf,
        )
        abbreviations.append(abbrev)

    return abbreviations
