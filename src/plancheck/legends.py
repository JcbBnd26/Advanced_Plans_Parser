"""Legend detection and extraction from PDF pages."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import pdfplumber

from .models import (
    AbbreviationEntry,
    AbbreviationRegion,
    BlockCluster,
    GlyphBox,
    GraphicElement,
    LegendEntry,
    LegendRegion,
    MiscTitleRegion,
    RevisionEntry,
    RevisionRegion,
    StandardDetailEntry,
    StandardDetailRegion,
)


def extract_graphics(pdf_path: str, page_num: int) -> List[GraphicElement]:
    """
    Extract graphical elements (lines, rects, curves) from a PDF page.

    Returns a list of GraphicElement objects with their bounding boxes and colors.
    """
    graphics: List[GraphicElement] = []

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]

        # Extract lines
        for line in page.lines:
            x0 = float(line.get("x0", 0))
            y0 = float(line.get("top", 0))
            x1 = float(line.get("x1", 0))
            y1 = float(line.get("bottom", 0))
            stroke = line.get("stroking_color")
            linewidth = float(line.get("linewidth", 1.0))

            graphics.append(
                GraphicElement(
                    page=page_num,
                    element_type="line",
                    x0=min(x0, x1),
                    y0=min(y0, y1),
                    x1=max(x0, x1),
                    y1=max(y0, y1),
                    stroke_color=stroke,
                    linewidth=linewidth,
                )
            )

        # Extract rectangles
        for rect in page.rects:
            x0 = float(rect.get("x0", 0))
            y0 = float(rect.get("top", 0))
            x1 = float(rect.get("x1", 0))
            y1 = float(rect.get("bottom", 0))
            stroke = rect.get("stroking_color")
            fill = rect.get("non_stroking_color")
            linewidth = float(rect.get("linewidth", 1.0))

            graphics.append(
                GraphicElement(
                    page=page_num,
                    element_type="rect",
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    stroke_color=stroke,
                    fill_color=fill,
                    linewidth=linewidth,
                )
            )

        # Extract curves
        for curve in page.curves:
            pts = curve.get("pts", [])
            if not pts:
                continue

            # Get bounding box from points
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)

            stroke = curve.get("stroking_color")
            fill = curve.get("non_stroking_color")
            linewidth = float(curve.get("linewidth", 1.0))

            graphics.append(
                GraphicElement(
                    page=page_num,
                    element_type="curve",
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    stroke_color=stroke,
                    fill_color=fill,
                    linewidth=linewidth,
                    pts=pts,
                )
            )

    return graphics


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
    debug_path: str = None,
    exclusion_zones: List[Tuple[float, float, float, float]] = None,
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
    debug_path = debug_path or "debug_headers.txt"
    exclusion_zones = exclusion_zones or []
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

    with open(debug_path, "a", encoding="utf-8") as dbg:
        dbg.write(
            f"\n[DEBUG] detect_legend_regions: found {len(legend_headers)} legend headers\n"
        )
        dbg.write(f"[DEBUG] Total graphics: {len(graphics)} (lines/rects/curves)\n")

        for header in legend_headers:
            hx0, hy0, hx1, hy1 = header.bbox()
            header_text = ""
            if header.rows:
                header_text = " ".join(b.text for b in header.rows[0].boxes if b.text)

            dbg.write(
                f"\n[DEBUG] Processing legend '{header_text}' at bbox=({hx0:.1f}, {hy0:.1f}, {hx1:.1f}, {hy1:.1f})\n"
            )

            page = header.page

            # Check for enclosing box
            enclosing_rect = _find_enclosing_rect((hx0, hy0, hx1, hy1), graphics)
            is_boxed = enclosing_rect is not None
            box_bbox = enclosing_rect.bbox() if enclosing_rect else None

            if is_boxed:
                dbg.write(f"[DEBUG]   Boxed legend, box bbox={box_bbox}\n")
                region_bbox = box_bbox
            else:
                # Define search region below header
                # Extend down and to the right until we hit another header or edge
                region_x0 = hx0 - 100  # Allow left margin for symbols
                region_y0 = hy0
                region_x1 = min(
                    hx0 + 600, page_width
                )  # Extend significantly right for text descriptions
                region_y1 = min(hy1 + 500, page_height)  # Extend down

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
                dbg.write(
                    f"[DEBUG]   Unboxed legend, search region=({region_x0:.1f}, {region_y0:.1f}, {region_x1:.1f}, {region_y1:.1f})\n"
                )

            # Find symbols in region
            symbols = _find_symbols_in_region(region_bbox, graphics)
            dbg.write(f"[DEBUG]   Found {len(symbols)} potential symbols in region\n")

            # Filter out very tiny symbols (noise) and very large ones
            symbols = [s for s in symbols if s.area() > 10 and s.area() < 2500]
            dbg.write(f"[DEBUG]   After size filter: {len(symbols)} symbols\n")

            # Detect column structure
            symbol_columns = _detect_legend_columns(symbols)
            dbg.write(f"[DEBUG]   Detected {len(symbol_columns)} symbol columns\n")
            for i, col in enumerate(symbol_columns):
                dbg.write(
                    f"[DEBUG]     Column {i}: {len(col)} symbols at x≈{col[0].x0:.1f}\n"
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
            dbg.write(
                f"[DEBUG]   Found {len(text_blocks)} text blocks in region (after exclusion filter)\n"
            )

            # Pair symbols with text
            all_symbols = [s for col in symbol_columns for s in col]
            entries = _pair_symbols_with_text(all_symbols, text_blocks, page)
            dbg.write(f"[DEBUG]   Created {len(entries)} legend entries\n")
            for entry in entries:
                dbg.write(
                    f"[DEBUG]     Entry: '{entry.description[:50]}...' at {entry.symbol_bbox}\n"
                    if entry.description
                    else f"[DEBUG]     Entry: (no text) at {entry.symbol_bbox}\n"
                )

            legend = LegendRegion(
                page=page,
                header=header,
                entries=entries,
                is_boxed=is_boxed,
                box_bbox=box_bbox,
            )
            legends.append(legend)

    return legends


def detect_abbreviation_regions(
    blocks: List[BlockCluster],
    graphics: List[GraphicElement],
    page_width: float,
    page_height: float,
    debug_path: str = None,
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
    debug_path = debug_path or "debug_headers.txt"
    abbreviations: List[AbbreviationRegion] = []

    # Find abbreviation headers
    abbrev_headers = [blk for blk in blocks if _is_abbreviation_header(blk)]

    with open(debug_path, "a", encoding="utf-8") as dbg:
        dbg.write(
            f"\n[DEBUG] detect_abbreviation_regions: found {len(abbrev_headers)} abbreviation headers\n"
        )

        for header in abbrev_headers:
            hx0, hy0, hx1, hy1 = header.bbox()
            header_text = ""
            if header.rows:
                header_text = " ".join(b.text for b in header.rows[0].boxes if b.text)

            dbg.write(
                f"\n[DEBUG] Processing abbreviation section '{header_text}' at bbox=({hx0:.1f}, {hy0:.1f}, {hx1:.1f}, {hy1:.1f})\n"
            )

            page = header.page

            # Check for enclosing box
            enclosing_rect = _find_enclosing_rect((hx0, hy0, hx1, hy1), graphics)
            is_boxed = enclosing_rect is not None
            box_bbox = enclosing_rect.bbox() if enclosing_rect else None

            if is_boxed:
                dbg.write(f"[DEBUG]   Boxed abbreviation, box bbox={box_bbox}\n")
                region_bbox = box_bbox
                text_blocks = _find_text_blocks_in_region(region_bbox, blocks, header)
                entries = _parse_abbreviation_entries(text_blocks, page, dbg)
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
                        dbg.write(
                            f"[DEBUG]   Code block found: x0={bx0:.1f}, y0={by0:.1f}, rows={len(blk.rows)}\n"
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
                                    len(blk.rows) / len(cb.rows)
                                    if len(cb.rows) > 0
                                    else 0
                                )
                                if 0.7 < row_ratio < 1.3:
                                    meaning_blocks.append(blk)
                                    dbg.write(
                                        f"[DEBUG]   Meaning block found: x0={bx0:.1f}, y0={by0:.1f}, rows={len(blk.rows)}\n"
                                    )
                                    break

                dbg.write(
                    f"[DEBUG]   Total: {len(code_blocks)} code blocks, {len(meaning_blocks)} meaning blocks\n"
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
                dbg.write(
                    f"[DEBUG]   Region bbox=({region_x0:.1f}, {region_y0:.1f}, {region_x1:.1f}, {region_y1:.1f})\n"
                )

                # Parse entries from the specific code and meaning blocks
                entries = _parse_abbreviation_entries_two_column(
                    code_blocks, meaning_blocks, page, dbg
                )

            dbg.write(f"[DEBUG]   Parsed {len(entries)} abbreviation entries\n")
            for entry in entries[:10]:  # Show first 10
                dbg.write(f"[DEBUG]     '{entry.code}' = '{entry.meaning}'\n")
            if len(entries) > 10:
                dbg.write(f"[DEBUG]     ... and {len(entries) - 10} more\n")

            abbrev = AbbreviationRegion(
                page=page,
                header=header,
                entries=entries,
                is_boxed=is_boxed,
                box_bbox=box_bbox,
            )
            abbreviations.append(abbrev)

    return abbreviations


def _parse_abbreviation_entries_two_column(
    code_blocks: List[BlockCluster],
    meaning_blocks: List[BlockCluster],
    page: int,
    dbg,
) -> List[AbbreviationEntry]:
    """
    Parse abbreviation entries from paired code and meaning blocks.

    Each code block should have a corresponding meaning block to its right.
    Rows are matched by y-position.
    """
    entries = []

    # Collect all code rows with y-positions
    code_rows = []  # (y_center, row_text, row_bbox, block)
    for blk in code_blocks:
        for row in blk.rows:
            row_bbox = row.bbox()
            y_center = (row_bbox[1] + row_bbox[3]) / 2
            row_text = " ".join(b.text for b in row.boxes if b.text).strip()
            if row_text:
                code_rows.append((y_center, row_text, row_bbox, blk))

    # Collect all meaning rows with y-positions
    meaning_rows = []  # (y_center, row_text, row_bbox, block)
    for blk in meaning_blocks:
        for row in blk.rows:
            row_bbox = row.bbox()
            y_center = (row_bbox[1] + row_bbox[3]) / 2
            row_text = " ".join(b.text for b in row.boxes if b.text).strip()
            if row_text:
                meaning_rows.append((y_center, row_text, row_bbox, blk))

    # Sort both by y-position
    code_rows.sort(key=lambda x: x[0])
    meaning_rows.sort(key=lambda x: x[0])

    # Match code rows to meaning rows by y-position
    y_tolerance = 5.0
    used_meaning_indices = set()

    for c_y, c_text, c_bbox, c_blk in code_rows:
        best_meaning = None
        best_y_diff = float("inf")
        best_idx = -1

        for idx, (m_y, m_text, m_bbox, m_blk) in enumerate(meaning_rows):
            if idx in used_meaning_indices:
                continue
            y_diff = abs(m_y - c_y)
            if y_diff < y_tolerance and y_diff < best_y_diff:
                best_y_diff = y_diff
                best_meaning = (m_text, m_bbox)
                best_idx = idx

        if best_meaning:
            used_meaning_indices.add(best_idx)
            meaning_text, meaning_bbox = best_meaning
            entries.append(
                AbbreviationEntry(
                    page=page,
                    code=c_text,
                    meaning=meaning_text,
                    code_bbox=c_bbox,
                    meaning_bbox=meaning_bbox,
                )
            )

    return entries


def _parse_abbreviation_entries(
    blocks: List[BlockCluster],
    page: int,
    dbg=None,
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
            y_tolerance = 5.0
            used_right = set()

            for y_center, code_text, code_bbox in left_rows:
                best_meaning = None
                best_y_diff = float("inf")
                best_idx = -1

                for idx, (r_y, r_text, r_bbox) in enumerate(right_rows):
                    if idx in used_right:
                        continue
                    y_diff = abs(r_y - y_center)
                    if y_diff < y_tolerance and y_diff < best_y_diff:
                        best_y_diff = y_diff
                        best_meaning = (r_text, r_bbox)
                        best_idx = idx

                if best_meaning:
                    used_right.add(best_idx)
                    meaning_text, meaning_bbox = best_meaning
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


def detect_standard_detail_regions(
    blocks: List[BlockCluster],
    graphics: List[GraphicElement],
    page_width: float,
    page_height: float,
    debug_path: str = None,
    exclusion_zones: List[Tuple[float, float, float, float]] = None,
) -> List[StandardDetailRegion]:
    """
    Detect standard detail regions on a page.

    Standard details are two-column text (sheet number + description), similar to abbreviations.
    Format example:
        SS-1        PAVEMENT MARKING STANDARD
        621-1       STORM SEWER DETAILS

    This uses the same detection logic as abbreviations.
    """
    debug_path = debug_path or "debug_headers.txt"
    exclusion_zones = exclusion_zones or []
    details: List[StandardDetailRegion] = []

    # Find standard detail headers
    detail_headers = [blk for blk in blocks if _is_standard_detail_header(blk)]

    with open(debug_path, "a", encoding="utf-8") as dbg:
        dbg.write(
            f"\n[DEBUG] detect_standard_detail_regions: found {len(detail_headers)} standard detail headers\n"
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
                dbg.write(
                    f"[DEBUG] Skipping standard detail header in exclusion zone at ({hx0:.1f}, {hy0:.1f})\n"
                )
                continue

            header_text = ""
            if header.rows:
                header_text = " ".join(b.text for b in header.rows[0].boxes if b.text)

            dbg.write(
                f"\n[DEBUG] Processing standard detail section '{header_text}' at bbox=({hx0:.1f}, {hy0:.1f}, {hx1:.1f}, {hy1:.1f})\n"
            )

            page = header.page

            # Check for enclosing box
            enclosing_rect = _find_enclosing_rect((hx0, hy0, hx1, hy1), graphics)
            is_boxed = enclosing_rect is not None
            box_bbox = enclosing_rect.bbox() if enclosing_rect else None

            if is_boxed:
                dbg.write(f"[DEBUG]   Boxed standard detail, box bbox={box_bbox}\n")
                region_bbox = box_bbox
                text_blocks = _find_text_blocks_in_region(region_bbox, blocks, header)
                entries = _parse_standard_detail_entries(text_blocks, page, dbg)
                subheader = None
                subheader_bbox = None
            elif len(header.rows) > 1 and _has_inline_entries(header):
                # Multi-row header block with actual entries (row 1+ starts with sheet number)
                dbg.write(
                    f"[DEBUG]   Multi-row header block with inline entries ({len(header.rows)} rows)\n"
                )
                entries = _parse_standard_detail_entries_inline(header, page, dbg)
                box_bbox = header.bbox()
                subheader = None
                subheader_bbox = None
            else:
                # Header may have subtitle row, entries are in separate blocks below
                subheader = None
                subheader_bbox = None

                # Check if row 1 is a subheader (not a sheet number entry)
                if len(header.rows) > 1:
                    row1 = header.rows[1]
                    if row1.boxes:
                        subheader = " ".join(
                            b.text for b in row1.boxes if b.text
                        ).strip()
                        row1_bbox = row1.bbox()
                        subheader_bbox = row1_bbox

                        # Look for continuation blocks on same line (to the right)
                        row1_y0, row1_y1 = row1_bbox[1], row1_bbox[3]
                        row1_x1 = row1_bbox[2]
                        for blk in blocks:
                            if blk is header:
                                continue
                            bx0, by0, bx1, by1 = blk.bbox()
                            # Block is to the right and overlaps y-range
                            if (
                                bx0 > row1_x1 - 10
                                and by0 < row1_y1 + 5
                                and by1 > row1_y0 - 5
                            ):
                                # Single row block on same line = continuation
                                if len(blk.rows) == 1:
                                    cont_text = " ".join(
                                        b.text for b in blk.rows[0].boxes if b.text
                                    ).strip()
                                    subheader = subheader + " " + cont_text
                                    # Expand subheader bbox
                                    subheader_bbox = (
                                        row1_bbox[0],
                                        min(row1_bbox[1], by0),
                                        bx1,
                                        max(row1_bbox[3], by1),
                                    )
                                    dbg.write(
                                        f"[DEBUG]   Subheader continuation: '{cont_text}'\n"
                                    )

                        dbg.write(f"[DEBUG]   Subheader detected: '{subheader}'\n")

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
                        dbg.write(
                            f"[DEBUG]   Sheet number block found: x0={bx0:.1f}, y0={by0:.1f}, rows={len(blk.rows)}\n"
                        )
                    # Check for inline entry block (majority inline entries)
                    elif (
                        inline_entry_count > len(blk.rows) * 0.5 and len(blk.rows) >= 2
                    ):
                        inline_entry_blocks.append(blk)
                        dbg.write(
                            f"[DEBUG]   Inline entry block found: x0={bx0:.1f}, y0={by0:.1f}, rows={len(blk.rows)}\n"
                        )

                # Second pass: find description blocks (to the right of sheet blocks, similar y-range)
                description_blocks = []
                for blk in blocks:
                    if (
                        blk is header
                        or blk in sheet_blocks
                        or blk in inline_entry_blocks
                    ):
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
                                    len(blk.rows) / len(sb.rows)
                                    if len(sb.rows) > 0
                                    else 0
                                )
                                if 0.6 < row_ratio < 1.5:
                                    description_blocks.append(blk)
                                    dbg.write(
                                        f"[DEBUG]   Description block found: x0={bx0:.1f}, y0={by0:.1f}, rows={len(blk.rows)}\n"
                                    )
                                    break

                dbg.write(
                    f"[DEBUG]   Total: {len(sheet_blocks)} sheet blocks, {len(description_blocks)} desc blocks, {len(inline_entry_blocks)} inline blocks\n"
                )

                # Calculate region from all found blocks
                all_detail_blocks = (
                    sheet_blocks + description_blocks + inline_entry_blocks
                )
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
                dbg.write(
                    f"[DEBUG]   Region bbox=({region_x0:.1f}, {region_y0:.1f}, {region_x1:.1f}, {region_y1:.1f})\n"
                )

                # Parse entries - use inline if we have inline blocks, else use two-column
                if inline_entry_blocks:
                    entries = _parse_standard_detail_entries_from_inline_blocks(
                        inline_entry_blocks, page, dbg
                    )
                else:
                    entries = _parse_standard_detail_entries_two_column(
                        sheet_blocks, description_blocks, page, dbg
                    )

            dbg.write(f"[DEBUG]   Parsed {len(entries)} standard detail entries\n")
            for entry in entries[:10]:  # Show first 10
                dbg.write(
                    f"[DEBUG]     '{entry.sheet_number}' = '{entry.description}'\n"
                )
            if len(entries) > 10:
                dbg.write(f"[DEBUG]     ... and {len(entries) - 10} more\n")

            detail_region = StandardDetailRegion(
                page=page,
                header=header,
                subheader=subheader,
                subheader_bbox=subheader_bbox,
                entries=entries,
                is_boxed=is_boxed,
                box_bbox=box_bbox,
            )
            details.append(detail_region)

    return details


def _parse_standard_detail_entries(
    blocks: List[BlockCluster],
    page: int,
    dbg,
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
    dbg,
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
                dbg.write(
                    f"[DEBUG]     Inline entry: '{sheet_num}' = '{description[:50]}...'\n"
                )

    return entries


def _parse_standard_detail_entries_inline(
    header_block: BlockCluster,
    page: int,
    dbg,
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
            dbg.write(f"[DEBUG]     Inline entry: '{sheet_num}' = '{description}'\n")

    return entries


def _parse_standard_detail_entries_two_column(
    sheet_blocks: List[BlockCluster],
    description_blocks: List[BlockCluster],
    page: int,
    dbg,
) -> List[StandardDetailEntry]:
    """
    Parse standard detail entries from paired sheet number and description blocks.

    Each sheet block should have a corresponding description block to its right.
    Rows are matched by y-position.
    """
    entries = []

    # Collect all sheet rows with y-positions
    sheet_rows = []  # (y_center, row_text, row_bbox, block)
    for blk in sheet_blocks:
        for row in blk.rows:
            if not row.boxes:
                continue
            y0, y1 = row.bbox()[1], row.bbox()[3]
            y_center = (y0 + y1) / 2
            row_text = " ".join(b.text for b in row.boxes if b.text).strip()
            row_bbox = row.bbox()
            sheet_rows.append((y_center, row_text, row_bbox, blk))

    # Collect all description rows with y-positions
    desc_rows = []  # (y_center, row_text, row_bbox, block)
    for blk in description_blocks:
        for row in blk.rows:
            if not row.boxes:
                continue
            y0, y1 = row.bbox()[1], row.bbox()[3]
            y_center = (y0 + y1) / 2
            row_text = " ".join(b.text for b in row.boxes if b.text).strip()
            row_bbox = row.bbox()
            desc_rows.append((y_center, row_text, row_bbox, blk))

    # Match sheet rows to description rows by y-position
    y_tolerance = 10.0  # pts

    for sheet_y, sheet_text, sheet_bbox, _ in sheet_rows:
        if not sheet_text:
            continue

        # Find best matching description row
        best_match = None
        best_dist = float("inf")

        for desc_y, desc_text, desc_bbox, _ in desc_rows:
            dist = abs(sheet_y - desc_y)
            if dist < y_tolerance and dist < best_dist:
                best_dist = dist
                best_match = (desc_text, desc_bbox)

        if best_match:
            desc_text, desc_bbox = best_match
            entries.append(
                StandardDetailEntry(
                    page=page,
                    sheet_number=sheet_text,
                    description=desc_text,
                    sheet_bbox=sheet_bbox,
                    description_bbox=desc_bbox,
                )
            )
        else:
            # No matching description found, just add sheet number
            entries.append(
                StandardDetailEntry(
                    page=page,
                    sheet_number=sheet_text,
                    description="",
                    sheet_bbox=sheet_bbox,
                    description_bbox=None,
                )
            )

    return entries


def _bboxes_overlap(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float],
) -> bool:
    """Check if two bounding boxes overlap."""
    x0a, y0a, x1a, y1a = bbox1
    x0b, y0b, x1b, y1b = bbox2
    return not (x1a < x0b or x0a > x1b or y1a < y0b or y0a > y1b)


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


def detect_revision_regions(
    blocks: List[BlockCluster],
    graphics: List[GraphicElement],
    page_width: float,
    page_height: float,
    debug_path: str = None,
    exclusion_zones: List[Tuple[float, float, float, float]] = None,
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
    debug_path = debug_path or "debug_headers.txt"
    revisions: List[RevisionRegion] = []
    exclusion_zones = exclusion_zones or []

    # Find revision headers
    revision_headers = [blk for blk in blocks if _is_revision_header(blk)]

    with open(debug_path, "a", encoding="utf-8") as dbg:
        dbg.write(
            f"\n[DEBUG] detect_revision_regions: found {len(revision_headers)} revision headers\n"
        )

        for header in revision_headers:
            hx0, hy0, hx1, hy1 = header.bbox()
            header_text = ""
            if header.rows:
                header_text = " ".join(b.text for b in header.rows[0].boxes if b.text)

            dbg.write(
                f"\n[DEBUG] Processing revision section '{header_text}' at bbox=({hx0:.1f}, {hy0:.1f}, {hx1:.1f}, {hy1:.1f})\n"
            )

            page = header.page

            # Check if the header block itself contains column headers (common pattern)
            has_column_headers = _has_revision_column_headers(header)
            dbg.write(
                f"[DEBUG]   Has column headers in header block: {has_column_headers}\n"
            )

            # Check for enclosing box - revision tables are almost always boxed
            enclosing_rect = _find_enclosing_rect((hx0, hy0, hx1, hy1), graphics)
            is_boxed = enclosing_rect is not None
            box_bbox = enclosing_rect.bbox() if enclosing_rect else None

            if is_boxed:
                dbg.write(f"[DEBUG]   Boxed revision table, box bbox={box_bbox}\n")
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
                            dbg.write(
                                f"[DEBUG]   Limiting search to y={max_y:.1f} due to exclusion zone at y={ez_y0:.1f}\n"
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
                    dbg.write(
                        f"[DEBUG]   Found {len(table_lines)} table lines, region=({region_x0:.1f}, {region_y0:.1f}, {region_x1:.1f}, {region_y1:.1f})\n"
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
                        dbg.write(
                            f"[DEBUG]   No table lines, header has column headers = empty table\n"
                        )
                    else:
                        region_x0 = hx0 - 20
                        region_y0 = hy0
                        region_x1 = hx0 + 300
                        region_y1 = hy1 + 150
                        dbg.write(f"[DEBUG]   No table lines, using estimated region\n")

                region_bbox = (region_x0, region_y0, region_x1, region_y1)
                dbg.write(
                    f"[DEBUG]   Revision region=({region_bbox[0]:.1f}, {region_bbox[1]:.1f}, {region_bbox[2]:.1f}, {region_bbox[3]:.1f})\n"
                )

            # Find text blocks in region (skip the header itself)
            # If header already contains column headers, don't look for extra blocks (table is empty)
            if has_column_headers:
                text_blocks = []
                dbg.write(
                    f"[DEBUG]   Empty revision table (header has column headers)\n"
                )
            else:
                text_blocks = _find_text_blocks_in_region(region_bbox, blocks, header)
            dbg.write(
                f"[DEBUG]   Found {len(text_blocks)} text blocks in revision region\n"
            )

            # Parse revision entries
            entries = _parse_revision_entries(text_blocks, page, region_bbox, dbg)
            dbg.write(f"[DEBUG]   Parsed {len(entries)} revision entries\n")

            revision = RevisionRegion(
                page=page,
                header=header,
                entries=entries,
                is_boxed=is_boxed,
                box_bbox=box_bbox,
            )
            revisions.append(revision)

    return revisions


def _parse_revision_entries(
    blocks: List[BlockCluster],
    page: int,
    region_bbox: Tuple[float, float, float, float],
    dbg,
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
                dbg.write(
                    f"[DEBUG]   Found column headers at y={column_header_y:.1f}: '{row_text}'\n"
                )
                break
        if column_header_y:
            break

    if not column_header_y:
        dbg.write(
            f"[DEBUG]   No column headers found, checking for data rows directly\n"
        )
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
            dbg.write(
                f"[DEBUG]     Revision: No={entry.number}, Desc={entry.description[:30]}..., Date={entry.date}\n"
            )

    return entries


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
    debug_path: str = None,
    exclusion_zones: List[Tuple[float, float, float, float]] = None,
) -> List[MiscTitleRegion]:
    """
    Detect miscellaneous title boxes (e.g., 'OKLAHOMA DEPARTMENT OF TRANSPORTATION').

    These are typically boxed text elements in the title block area that should
    be excluded from other detection (legends, abbreviations, etc.).
    """
    debug_path = debug_path or "debug_headers.txt"
    misc_titles: List[MiscTitleRegion] = []
    exclusion_zones = exclusion_zones or []

    with open(debug_path, "a", encoding="utf-8") as dbg:
        dbg.write(
            f"\n[DEBUG] detect_misc_title_regions: checking {len(blocks)} blocks\n"
        )

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
            dbg.write(
                f"[DEBUG]   Found misc title candidate: '{all_text[:50]}...' at bbox=({bx0:.1f}, {by0:.1f}, {bx1:.1f}, {by1:.1f})\n"
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
                            if (
                                abs(cx1 - lx0) < 15
                                and cy0 <= ly0 + 5
                                and cy1 >= ly1 - 5
                            ):
                                lx0 = min(lx0, cx0)
                                dbg.write(
                                    f"[DEBUG]     Extended left to corner curve at x={cx0:.1f}\n"
                                )
                            # Right corner: near right edge of lines
                            if (
                                abs(cx0 - lx1) < 15
                                and cy0 <= ly0 + 5
                                and cy1 >= ly1 - 5
                            ):
                                lx1 = max(lx1, cx1)
                                dbg.write(
                                    f"[DEBUG]     Extended right to corner curve at x={cx1:.1f}\n"
                                )

                area = (lx1 - lx0) * (ly1 - ly0)
                if area < text_area * 15:  # Reasonably tight
                    is_boxed = True
                    box_bbox = (lx0, ly0, lx1, ly1)
                    dbg.write(
                        f"[DEBUG]     Boxed by horizontal lines, box bbox={box_bbox}\n"
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
                    dbg.write(f"[DEBUG]     Boxed by curve, box bbox={box_bbox}\n")

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
                    dbg.write(f"[DEBUG]     Boxed by rect, box bbox={box_bbox}\n")

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

        dbg.write(
            f"[DEBUG] detect_misc_title_regions: found {len(misc_titles)} misc titles\n"
        )

    return misc_titles
