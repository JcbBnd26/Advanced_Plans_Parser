"""Structural box detection, classification, and semantic region growth.

This module re-introduces three layers that were present in an earlier
architecture and are essential for correct semantic grouping:

1. **Box detection** – find drawn rectangles/frames on the page from PDF
   graphic elements (lines & rects extracted by ``_graphics.py``).
2. **Box classification** – classify each detected box by its text content
   and geometric properties (title_block, notes_box, legend, etc.).
3. **Anchor-based region growth** – find semantic text anchors (e.g.
   "GENERAL NOTES:", "LEGEND", "ABBREVIATIONS") and grow bounded regions
   from them using nearby text blocks and (optionally) enclosing drawn
   rectangles.

The results feed into the grouping pipeline so that blocks can be
assigned to semantic regions *before* column detection, rather than
relying on pure full-page geometric column bands.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .models import BlockCluster, GlyphBox, GraphicElement

logger = logging.getLogger("plancheck.structural")

# ── Constants ──────────────────────────────────────────────────────────

# Minimum area fraction of page for a rectangle to be considered structural
MIN_AREA_FRAC = 0.0005

# Minimum pixel/point dimension to be a real box (not a dot or hairline)
MIN_SIZE_PTS = 8.0

# Area fraction above which a box is likely the page border
PAGE_BORDER_AREA_FRAC = 0.80

# Title-block heuristics
TITLE_BLOCK_MIN_HEIGHT_FRAC = 0.40  # At least 40% page height
TITLE_BLOCK_MAX_ASPECT = 0.6  # w/h ≤ 0.6 (tall & skinny)
TITLE_BLOCK_RIGHT_EDGE_FRAC = 0.90  # Right edge > 90% of page

# Header anchor patterns (section headers that start semantic regions)
HEADER_ANCHORS_RE = re.compile(
    r"^(?:"
    r"(?:GENERAL\s+)?NOTES"
    r"|ABBREVIATIONS"
    r"|LEGEND"
    r"|UTILITY\s+LEGEND"
    r"|DEMOLITION\s+LEGEND"
    r"|PAVEMENT\s+LEGEND"
    r"|STANDARD\s+DETAILS"
    r"|STANDARD\s+DRAWINGS"
    r"|REVISIONS?"
    r"|SHEET\s+(?:NAME|NO|NUMBER)"
    r"|LOCATION\s+MAP"
    r"|DATA\s+TABLE"
    r")(?:\s*\(.*?\))?[:.]?\s*$",
    re.IGNORECASE,
)

# Note-number leader (for anchor growth)
NOTE_NUMBER_RE = re.compile(r"^\s*\d+\s*[\.:)]\s*")


# ── Box type enum ──────────────────────────────────────────────────────


class BoxType(str, Enum):
    """Semantic classification of a detected structural box."""

    page_border = "page_border"
    title_block = "title_block"
    sheet_info = "sheet_info"
    legend = "legend"
    notes_box = "notes_box"
    location_map = "location_map"
    data_table = "data_table"
    callout = "callout"
    unknown = "unknown"


# ── Data model ─────────────────────────────────────────────────────────


@dataclass
class StructuralBox:
    """A detected rectangular frame on the page with semantic classification."""

    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    box_type: BoxType = BoxType.unknown
    confidence: float = 0.0
    # Indices of blocks whose centres fall inside this box
    contained_block_indices: List[int] = field(default_factory=list)
    # Text of blocks inside (for classification)
    contained_text: str = ""
    # The source graphic element (if from a drawn rectangle)
    source: Optional[GraphicElement] = None
    # True if this box was synthesised from a text anchor rather than
    # detected as a drawn rectangle
    is_synthetic: bool = False

    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def width(self) -> float:
        return self.x1 - self.x0

    def height(self) -> float:
        return self.y1 - self.y0

    def area(self) -> float:
        return max(0.0, self.width()) * max(0.0, self.height())

    def contains_point(self, x: float, y: float, pad: float = 0.0) -> bool:
        return (
            self.x0 - pad <= x <= self.x1 + pad and self.y0 - pad <= y <= self.y1 + pad
        )


@dataclass
class SemanticRegion:
    """A bounded semantic region grown from a text anchor.

    Unlike a pure geometric column band, a SemanticRegion has a specific
    bounded rectangle, a semantic label (what kind of content it is), and
    references to the text blocks it contains.
    """

    page: int
    label: str  # e.g. "GENERAL NOTES", "LEGEND", "ABBREVIATIONS"
    x0: float
    y0: float
    x1: float
    y1: float
    anchor_block: Optional[BlockCluster] = None
    child_blocks: List[BlockCluster] = field(default_factory=list)
    enclosing_box: Optional[StructuralBox] = None
    confidence: float = 0.0

    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)


# ── 1. Box Detection ──────────────────────────────────────────────────


def detect_structural_boxes(
    graphics: List[GraphicElement],
    page_width: float,
    page_height: float,
    min_area_frac: float = MIN_AREA_FRAC,
    min_size: float = MIN_SIZE_PTS,
) -> List[StructuralBox]:
    """Detect rectangular frames from PDF graphic elements.

    Filters out tiny decorations and keeps only boxes large enough to be
    structural (section borders, title-block frames, legend enclosures,
    etc.).

    Parameters
    ----------
    graphics : list[GraphicElement]
        All graphic elements on the page (from ``extract_graphics``).
    page_width, page_height : float
        Page dimensions in points.
    min_area_frac : float
        Minimum fraction of page area for a box to be kept.
    min_size : float
        Minimum width AND height in points.

    Returns
    -------
    list[StructuralBox]
        Detected boxes sorted by area (largest first).
    """
    page_area = page_width * page_height
    if page_area <= 0:
        return []

    boxes: List[StructuralBox] = []

    for g in graphics:
        if g.element_type != "rect":
            continue

        w = g.width()
        h = g.height()

        # Skip tiny elements (symbols, hatching, decoration)
        if w < min_size or h < min_size:
            continue

        area = w * h
        area_frac = area / page_area

        if area_frac < min_area_frac:
            continue

        boxes.append(
            StructuralBox(
                page=g.page,
                x0=g.x0,
                y0=g.y0,
                x1=g.x1,
                y1=g.y1,
                source=g,
            )
        )

    # Also look for boxes formed by intersecting lines.
    # Construction PDFs often draw frames as four separate lines rather
    # than a single rect.  We use a conservative heuristic: find axis-
    # aligned horizontal + vertical line pairs that form closed rectangles.
    line_boxes = _detect_boxes_from_lines(graphics, page_width, page_height, min_size)
    for lb in line_boxes:
        area_frac = lb.area() / page_area
        if area_frac >= min_area_frac:
            boxes.append(lb)

    # De-duplicate near-identical boxes (same rect drawn with stroke + fill)
    boxes = _dedup_boxes(boxes, tolerance=4.0)

    # Sort by area descending
    boxes.sort(key=lambda b: b.area(), reverse=True)

    logger.debug(
        "detect_structural_boxes: %d rects, %d from-lines → %d total after dedup",
        sum(1 for g in graphics if g.element_type == "rect"),
        len(line_boxes),
        len(boxes),
    )

    return boxes


def _detect_boxes_from_lines(
    graphics: List[GraphicElement],
    page_width: float,
    page_height: float,
    min_size: float,
    snap_tolerance: float = 3.0,
) -> List[StructuralBox]:
    """Detect rectangular frames formed by four axis-aligned lines.

    Construction drawings often draw section borders as four separate
    line segments.  This function finds sets of two horizontal + two
    vertical lines that form a closed rectangle (within snap tolerance).
    """
    # Separate horizontal and vertical lines
    h_lines: List[GraphicElement] = []  # near-horizontal
    v_lines: List[GraphicElement] = []  # near-vertical

    for g in graphics:
        if g.element_type != "line":
            continue
        dx = abs(g.x1 - g.x0)
        dy = abs(g.y1 - g.y0)
        length = max(dx, dy)
        if length < min_size:
            continue

        if dy <= snap_tolerance and dx >= min_size:
            h_lines.append(g)
        elif dx <= snap_tolerance and dy >= min_size:
            v_lines.append(g)

    if len(h_lines) < 2 or len(v_lines) < 2:
        return []

    # Sort horizontal by y, vertical by x
    h_lines.sort(key=lambda g: (g.y0 + g.y1) / 2)
    v_lines.sort(key=lambda g: (g.x0 + g.x1) / 2)

    boxes: List[StructuralBox] = []

    # Try pairs of horizontal lines as top/bottom edges
    for i, h_top in enumerate(h_lines):
        ht_y = (h_top.y0 + h_top.y1) / 2
        ht_xmin = min(h_top.x0, h_top.x1)
        ht_xmax = max(h_top.x0, h_top.x1)

        for h_bot in h_lines[i + 1 :]:
            hb_y = (h_bot.y0 + h_bot.y1) / 2
            hb_xmin = min(h_bot.x0, h_bot.x1)
            hb_xmax = max(h_bot.x0, h_bot.x1)

            height = hb_y - ht_y
            if height < min_size:
                continue

            # Horizontal lines must overlap in x
            overlap_x0 = max(ht_xmin, hb_xmin)
            overlap_x1 = min(ht_xmax, hb_xmax)
            if overlap_x1 - overlap_x0 < min_size:
                continue

            # Look for two vertical lines connecting them
            for j, v_left in enumerate(v_lines):
                vl_x = (v_left.x0 + v_left.x1) / 2
                vl_ymin = min(v_left.y0, v_left.y1)
                vl_ymax = max(v_left.y0, v_left.y1)

                # Left vert must be near the left ends of horizontals
                if abs(vl_x - overlap_x0) > snap_tolerance:
                    continue
                if vl_ymin > ht_y + snap_tolerance:
                    continue
                if vl_ymax < hb_y - snap_tolerance:
                    continue

                for v_right in v_lines[j + 1 :]:
                    vr_x = (v_right.x0 + v_right.x1) / 2
                    vr_ymin = min(v_right.y0, v_right.y1)
                    vr_ymax = max(v_right.y0, v_right.y1)

                    if abs(vr_x - overlap_x1) > snap_tolerance:
                        continue
                    if vr_ymin > ht_y + snap_tolerance:
                        continue
                    if vr_ymax < hb_y - snap_tolerance:
                        continue

                    # Found a rectangle!
                    boxes.append(
                        StructuralBox(
                            page=h_top.page,
                            x0=overlap_x0,
                            y0=ht_y,
                            x1=overlap_x1,
                            y1=hb_y,
                        )
                    )

    return boxes


def _dedup_boxes(
    boxes: List[StructuralBox], tolerance: float = 4.0
) -> List[StructuralBox]:
    """Remove near-duplicate boxes (same rect drawn twice, stroke + fill, etc.)."""
    if not boxes:
        return []

    keep: List[StructuralBox] = []
    for b in boxes:
        is_dup = False
        for k in keep:
            if (
                abs(b.x0 - k.x0) <= tolerance
                and abs(b.y0 - k.y0) <= tolerance
                and abs(b.x1 - k.x1) <= tolerance
                and abs(b.y1 - k.y1) <= tolerance
            ):
                is_dup = True
                break
        if not is_dup:
            keep.append(b)
    return keep


# ── 2. Box Classification ─────────────────────────────────────────────

# Keywords for text-based classification
_LEGEND_KEYWORDS = {
    "legend",
    "utility legend",
    "demolition legend",
    "pavement legend",
    "abbreviations",
}
_NOTES_KEYWORDS = {
    "notes",
    "general notes",
    "construction notes",
    "drainage notes",
    "grading notes",
    "electrical notes",
    "plumbing notes",
    "structural notes",
    "mechanical notes",
}
_SHEET_INFO_KEYWORDS = {
    "sheet name",
    "sheet no",
    "sheet number",
    "project name",
    "project no",
}
_TABLE_KEYWORDS = {"table", "qty", "quantity", "schedule"}
_LOCATION_KEYWORDS = {"location map", "vicinity map", "state of"}


def classify_structural_boxes(
    boxes: List[StructuralBox],
    blocks: List[BlockCluster],
    page_width: float,
    page_height: float,
) -> List[StructuralBox]:
    """Classify each structural box by its geometric properties and text content.

    Modifies boxes in-place (sets ``box_type``, ``confidence``,
    ``contained_block_indices``, ``contained_text``).

    Classification priority (highest first):
    1. page_border – area ≥ 80% of page
    2. title_block – tall, skinny, near right edge
    3. sheet_info – contains "SHEET NAME" + right side
    4. legend – text contains legend keywords
    5. notes_box – header text contains "NOTES"
    6. location_map – text contains "LOCATION MAP" etc.
    7. data_table – text contains "TABLE", "QTY", etc.
    8. callout – very small box with short all-caps text
    9. unknown – fallback
    """
    page_area = page_width * page_height
    if page_area <= 0:
        return boxes

    for box in boxes:
        # Collect text from blocks whose centre falls inside the box
        contained_indices = []
        text_parts = []
        for idx, blk in enumerate(blocks):
            bx0, by0, bx1, by1 = blk.bbox()
            cx = (bx0 + bx1) / 2
            cy = (by0 + by1) / 2
            if box.contains_point(cx, cy, pad=4.0):
                contained_indices.append(idx)
                for row in blk.rows:
                    for b in row.boxes:
                        if b.text:
                            text_parts.append(b.text)

        box.contained_block_indices = contained_indices
        box.contained_text = " ".join(text_parts)
        text_lower = box.contained_text.lower()

        area_frac = box.area() / page_area
        right_frac = box.x1 / page_width if page_width > 0 else 0
        height_frac = box.height() / page_height if page_height > 0 else 0
        aspect = box.width() / box.height() if box.height() > 0 else 999

        # 1. Page border
        if area_frac >= PAGE_BORDER_AREA_FRAC:
            box.box_type = BoxType.page_border
            box.confidence = 0.95
            continue

        # 2. Title block – tall, skinny, near right edge
        if (
            height_frac >= TITLE_BLOCK_MIN_HEIGHT_FRAC
            and aspect <= TITLE_BLOCK_MAX_ASPECT
            and right_frac >= TITLE_BLOCK_RIGHT_EDGE_FRAC
        ):
            box.box_type = BoxType.title_block
            box.confidence = 0.85
            continue

        # 3. Sheet info
        if any(kw in text_lower for kw in _SHEET_INFO_KEYWORDS) and right_frac >= 0.6:
            box.box_type = BoxType.sheet_info
            box.confidence = 0.75
            continue

        # 4. Legend
        if any(kw in text_lower for kw in _LEGEND_KEYWORDS):
            box.box_type = BoxType.legend
            box.confidence = 0.80
            continue

        # 5. Notes box
        if any(kw in text_lower for kw in _NOTES_KEYWORDS):
            box.box_type = BoxType.notes_box
            box.confidence = 0.80
            continue

        # 6. Location map
        if any(kw in text_lower for kw in _LOCATION_KEYWORDS):
            box.box_type = BoxType.location_map
            box.confidence = 0.70
            continue

        # 7. Data table
        if any(kw in text_lower for kw in _TABLE_KEYWORDS):
            box.box_type = BoxType.data_table
            box.confidence = 0.65
            continue

        # 8. Callout – very small, all-caps, short text
        if area_frac < 0.01 and box.contained_text.isupper() and len(text_parts) < 6:
            box.box_type = BoxType.callout
            box.confidence = 0.50
            continue

        # 9. Unknown
        box.box_type = BoxType.unknown
        box.confidence = 0.30

    # Second pass: promote unknown boxes that are directly below a classified
    # header box of the same type (e.g., legend body below legend header).
    _promote_sub_boxes(boxes, page_width)

    logger.debug(
        "classify_structural_boxes: %s",
        {bt.value: sum(1 for b in boxes if b.box_type == bt) for bt in BoxType},
    )

    return boxes


def _promote_sub_boxes(boxes: List[StructuralBox], page_width: float) -> None:
    """Promote unknown boxes below a classified box with >60% x-overlap."""
    classified = [
        b for b in boxes if b.box_type not in (BoxType.unknown, BoxType.page_border)
    ]
    unknowns = [b for b in boxes if b.box_type == BoxType.unknown]

    for unk in unknowns:
        for cls in classified:
            # Must be below
            if unk.y0 < cls.y1 - 5:
                continue
            # Must be close vertically
            if unk.y0 - cls.y1 > 50:
                continue
            # Horizontal overlap
            overlap_x0 = max(unk.x0, cls.x0)
            overlap_x1 = min(unk.x1, cls.x1)
            unk_w = unk.width()
            if unk_w <= 0:
                continue
            h_overlap_frac = max(0, overlap_x1 - overlap_x0) / unk_w
            if h_overlap_frac >= 0.60:
                unk.box_type = cls.box_type
                unk.confidence = cls.confidence * 0.7
                break


# ── 3. Synthetic Region Generation ────────────────────────────────────


def create_synthetic_regions(
    blocks: List[BlockCluster],
    structural_boxes: List[StructuralBox],
    page_width: float,
    page_height: float,
) -> List[StructuralBox]:
    """Create synthetic structural boxes from text anchors when no drawn
    rectangle was detected.

    Scans blocks for header text matching known section patterns
    (LEGEND, NOTES, ABBREVIATIONS, etc.) and creates a synthetic
    ``StructuralBox`` around the anchor + its nearby content.  Skips
    any anchor that already falls inside a classified structural box.

    Returns new synthetic boxes (does NOT modify the input list).
    """
    synthetic: List[StructuralBox] = []

    # Build set of block indices already covered by classified boxes
    covered_indices: set[int] = set()
    for sb in structural_boxes:
        if sb.box_type != BoxType.unknown:
            covered_indices.update(sb.contained_block_indices)

    for idx, blk in enumerate(blocks):
        if idx in covered_indices:
            continue

        # Check first row text for header anchor pattern
        first_text = _first_row_text(blk).strip()
        if not first_text:
            continue

        m = HEADER_ANCHORS_RE.match(first_text.upper())
        if not m:
            continue

        # Determine the semantic type
        anchor_upper = first_text.upper()
        if "LEGEND" in anchor_upper or "ABBREVIATION" in anchor_upper:
            box_type = BoxType.legend
        elif "NOTE" in anchor_upper:
            box_type = BoxType.notes_box
        elif "REVISION" in anchor_upper:
            box_type = BoxType.sheet_info
        elif "SHEET" in anchor_upper:
            box_type = BoxType.sheet_info
        elif "LOCATION" in anchor_upper:
            box_type = BoxType.location_map
        elif (
            "TABLE" in anchor_upper
            or "DETAIL" in anchor_upper
            or "DRAWING" in anchor_upper
        ):
            box_type = BoxType.data_table
        else:
            box_type = BoxType.unknown

        # Grow a region from this anchor
        region_bbox = _grow_region_from_anchor(
            anchor_idx=idx,
            blocks=blocks,
            structural_boxes=structural_boxes,
            page_width=page_width,
            page_height=page_height,
        )

        sb = StructuralBox(
            page=blk.page,
            x0=region_bbox[0],
            y0=region_bbox[1],
            x1=region_bbox[2],
            y1=region_bbox[3],
            box_type=box_type,
            confidence=0.65,
            is_synthetic=True,
        )
        synthetic.append(sb)

        logger.debug(
            "Synthetic %s region from anchor '%s' → bbox (%.0f, %.0f, %.0f, %.0f)",
            box_type.value,
            first_text[:40],
            *region_bbox,
        )

    return synthetic


# ── 4. Anchor-Based Region Growth ─────────────────────────────────────


def _grow_region_from_anchor(
    anchor_idx: int,
    blocks: List[BlockCluster],
    structural_boxes: List[StructuralBox],
    page_width: float,
    page_height: float,
    max_gap: float = 40.0,
    x_tolerance: float = 80.0,
) -> Tuple[float, float, float, float]:
    """Grow a bounded region downward from a header anchor block.

    Starting from the anchor block, we expand the region to include
    nearby blocks below it that are:
    - Within ``x_tolerance`` of the anchor's x-range (same column)
    - Within ``max_gap`` vertical distance of the current region bottom
    - Not already inside a classified structural box
    - Not a header for a *different* section

    The region stops at:
    - A large vertical gap (> max_gap)
    - Another header block
    - A classified structural box of a different type
    - The bottom of the page (or a title block boundary)

    Returns the bounding box ``(x0, y0, x1, y1)`` of the grown region.
    """
    anchor = blocks[anchor_idx]
    ax0, ay0, ax1, ay1 = anchor.bbox()

    # Determine the stop boundary (don't grow into title blocks)
    stop_y = page_height
    for sb in structural_boxes:
        if sb.box_type == BoxType.title_block:
            stop_y = min(stop_y, sb.y0)

    # Build set of indices already inside other structural boxes
    other_box_indices: set[int] = set()
    for sb in structural_boxes:
        if sb.box_type in (BoxType.page_border, BoxType.unknown):
            continue
        other_box_indices.update(sb.contained_block_indices)

    # Collect candidate blocks below the anchor
    region_blocks = [anchor]
    region_bottom = ay1

    # Sort remaining blocks by y-position
    candidates = []
    for i, blk in enumerate(blocks):
        if i == anchor_idx:
            continue
        bx0, by0, bx1, by1 = blk.bbox()
        if by0 < ay0:
            # Skip blocks above anchor
            continue
        candidates.append((i, blk, bx0, by0, bx1, by1))

    candidates.sort(key=lambda c: c[3])  # sort by y0

    for i, blk, bx0, by0, bx1, by1 in candidates:
        # Stop if we hit a different section's classified box
        if i in other_box_indices:
            continue

        # Stop if below the title block
        if by0 >= stop_y:
            break

        # Stop if there's a large vertical gap
        if by0 - region_bottom > max_gap:
            break

        # Check horizontal alignment with anchor
        # Block should overlap the anchor's x-range within tolerance
        x_overlap_left = max(ax0 - x_tolerance, 0)
        x_overlap_right = min(ax1 + x_tolerance, page_width)
        if bx1 < x_overlap_left or bx0 > x_overlap_right:
            continue

        # Stop if this block is a *different* header anchor
        ft = _first_row_text(blk).strip().upper()
        if ft and HEADER_ANCHORS_RE.match(ft):
            # Different header — stop growing
            break

        region_blocks.append(blk)
        region_bottom = max(region_bottom, by1)

    # Compute bounding box of all collected blocks
    r_x0 = min(blk.bbox()[0] for blk in region_blocks)
    r_y0 = min(blk.bbox()[1] for blk in region_blocks)
    r_x1 = max(blk.bbox()[2] for blk in region_blocks)
    r_y1 = max(blk.bbox()[3] for blk in region_blocks)

    return (r_x0, r_y0, r_x1, r_y1)


# ── 5. Full Semantic Region Pipeline ──────────────────────────────────


def detect_semantic_regions(
    blocks: List[BlockCluster],
    graphics: List[GraphicElement],
    page_width: float,
    page_height: float,
) -> Tuple[List[StructuralBox], List[SemanticRegion]]:
    """Run the full structural detection + semantic region pipeline.

    This is the main entry point.  It:

    1. Detects structural boxes from graphics.
    2. Classifies them by text + geometry.
    3. Creates synthetic regions from text anchors.
    4. Grows semantic regions from anchors.
    5. Returns both the classified structural boxes and the semantic regions.

    Parameters
    ----------
    blocks : list[BlockCluster]
        All text blocks on the page (from grouping).
    graphics : list[GraphicElement]
        Graphic elements on the page (from ``extract_graphics``).
    page_width, page_height : float
        Page dimensions in points.

    Returns
    -------
    (structural_boxes, semantic_regions)
        structural_boxes : list[StructuralBox]
            All detected + classified boxes (drawn + synthetic).
        semantic_regions : list[SemanticRegion]
            Bounded semantic regions grown from anchors.
    """
    # Step 1: Detect drawn rectangles
    struct_boxes = detect_structural_boxes(graphics, page_width, page_height)

    # Step 2: Classify by text + geometry
    classify_structural_boxes(struct_boxes, blocks, page_width, page_height)

    # Step 3: Synthetic regions from text anchors
    synthetic = create_synthetic_regions(blocks, struct_boxes, page_width, page_height)
    # Classify the synthetic boxes (they need text collection too)
    for sb in synthetic:
        # Populate contained blocks
        for idx, blk in enumerate(blocks):
            bx0, by0, bx1, by1 = blk.bbox()
            cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
            if sb.contains_point(cx, cy, pad=4.0):
                sb.contained_block_indices.append(idx)
                for row in blk.rows:
                    for b in row.boxes:
                        if b.text:
                            sb.contained_text += " " + b.text

    all_boxes = struct_boxes + synthetic

    # Step 4: Build semantic regions
    regions = _build_semantic_regions(blocks, all_boxes, page_width, page_height)

    logger.debug(
        "detect_semantic_regions: %d structural boxes (%d drawn, %d synthetic), "
        "%d semantic regions",
        len(all_boxes),
        len(struct_boxes),
        len(synthetic),
        len(regions),
    )

    return all_boxes, regions


def _build_semantic_regions(
    blocks: List[BlockCluster],
    structural_boxes: List[StructuralBox],
    page_width: float,
    page_height: float,
) -> List[SemanticRegion]:
    """Build semantic regions from header anchors + structural boxes.

    For each header anchor block, create a SemanticRegion that includes
    the header + all child blocks.  If the anchor falls inside a
    classified structural box, use that box as the region boundary.
    Otherwise, grow the region from the anchor.
    """
    regions: List[SemanticRegion] = []
    used_anchors: set[int] = set()

    # First pass: regions from classified drawn boxes with notes/legend content
    for sb in structural_boxes:
        if sb.box_type in (BoxType.page_border, BoxType.unknown, BoxType.callout):
            continue

        # Find header anchor inside this box
        anchor_blk = None
        anchor_idx = -1
        for idx in sb.contained_block_indices:
            if idx >= len(blocks):
                continue
            blk = blocks[idx]
            ft = _first_row_text(blk).strip().upper()
            if ft and HEADER_ANCHORS_RE.match(ft):
                anchor_blk = blk
                anchor_idx = idx
                break

        child_blocks = [
            blocks[i]
            for i in sb.contained_block_indices
            if i < len(blocks) and blocks[i] is not anchor_blk
        ]

        label = _classify_region_label(sb, anchor_blk)

        region = SemanticRegion(
            page=sb.page,
            label=label,
            x0=sb.x0,
            y0=sb.y0,
            x1=sb.x1,
            y1=sb.y1,
            anchor_block=anchor_blk,
            child_blocks=child_blocks,
            enclosing_box=sb,
            confidence=sb.confidence,
        )
        regions.append(region)

        if anchor_idx >= 0:
            used_anchors.add(anchor_idx)

    # Second pass: anchors not yet covered → grow regions
    for idx, blk in enumerate(blocks):
        if idx in used_anchors:
            continue

        ft = _first_row_text(blk).strip().upper()
        if not ft or not HEADER_ANCHORS_RE.match(ft):
            continue

        # Check if already inside a classified box
        inside_box = False
        for sb in structural_boxes:
            if sb.box_type in (BoxType.page_border, BoxType.unknown):
                continue
            bx0, by0, bx1, by1 = blk.bbox()
            cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
            if sb.contains_point(cx, cy, pad=4.0):
                inside_box = True
                break
        if inside_box:
            continue

        # Grow region from anchor
        region_bbox = _grow_region_from_anchor(
            anchor_idx=idx,
            blocks=blocks,
            structural_boxes=structural_boxes,
            page_width=page_width,
            page_height=page_height,
        )

        # Collect child blocks inside the grown region
        child_blocks = []
        for ci, cblk in enumerate(blocks):
            if ci == idx:
                continue
            cbx0, cby0, cbx1, cby1 = cblk.bbox()
            ccx, ccy = (cbx0 + cbx1) / 2, (cby0 + cby1) / 2
            rx0, ry0, rx1, ry1 = region_bbox
            if rx0 <= ccx <= rx1 and ry0 <= ccy <= ry1:
                child_blocks.append(cblk)

        label = _label_from_header_text(ft)

        region = SemanticRegion(
            page=blk.page,
            label=label,
            x0=region_bbox[0],
            y0=region_bbox[1],
            x1=region_bbox[2],
            y1=region_bbox[3],
            anchor_block=blk,
            child_blocks=child_blocks,
            confidence=0.60,
        )
        regions.append(region)

    return regions


# ── Masking ────────────────────────────────────────────────────────────


def mask_blocks_by_structural_boxes(
    blocks: List[BlockCluster],
    structural_boxes: List[StructuralBox],
    exclude_types: Optional[List[BoxType]] = None,
) -> List[int]:
    """Return indices of blocks that fall inside excluded box types.

    Useful for filtering notes blocks that actually belong to the legend
    or title block.

    Parameters
    ----------
    blocks : list[BlockCluster]
        All blocks.
    structural_boxes : list[StructuralBox]
        Classified structural boxes.
    exclude_types : list[BoxType], optional
        Box types to exclude. Default: ``[title_block, legend, location_map]``.

    Returns
    -------
    list[int]
        Indices of blocks to exclude.
    """
    if exclude_types is None:
        exclude_types = [BoxType.title_block, BoxType.legend, BoxType.location_map]

    exclude_set = set(exclude_types)
    masked_indices: set[int] = set()

    for sb in structural_boxes:
        if sb.box_type not in exclude_set:
            continue
        masked_indices.update(sb.contained_block_indices)

    return sorted(masked_indices)


# ── Helpers ────────────────────────────────────────────────────────────


def _first_row_text(block: BlockCluster) -> str:
    """Get the text of the first row in a block."""
    if block.rows and block.rows[0].boxes:
        return " ".join(
            b.text for b in sorted(block.rows[0].boxes, key=lambda b: b.x0) if b.text
        ).strip()
    return ""


def _classify_region_label(sb: StructuralBox, anchor: Optional[BlockCluster]) -> str:
    """Derive a human-readable label from box type + anchor text."""
    if anchor is not None:
        ft = _first_row_text(anchor).strip()
        if ft:
            return ft

    return sb.box_type.value.upper().replace("_", " ")


def _label_from_header_text(header_upper: str) -> str:
    """Clean up header text into a region label."""
    # Remove trailing colon/period and extra spaces
    label = re.sub(r"[:\.\s]+$", "", header_upper).strip()
    return label if label else "UNKNOWN"
