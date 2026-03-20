"""Semantic region detection pipeline.

Implements the full structural detection + semantic region pipeline:

1. Detect drawn structural boxes (delegated to :mod:`structural_boxes`).
2. Classify boxes by text + geometry (delegated to :mod:`box_classifier`).
3. Create synthetic boxes from text anchors.
4. Optionally merge overlapping same-type boxes.
5. Grow bounded :class:`~plancheck.analysis.structural_boxes.SemanticRegion`
   objects from header anchors.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ..models import BlockCluster, GraphicElement

if TYPE_CHECKING:
    from ..config.subconfigs import AnalysisConfig

from .box_classifier import classify_structural_boxes
from .box_merge import find_overlap_clusters, merge_boxes, polygon_bbox
from .structural_boxes import (
    BoxType,
    SemanticRegion,
    StructuralBox,
    detect_structural_boxes,
)

logger = logging.getLogger("plancheck.structural")

# ── Header anchor pattern ──────────────────────────────────────────────

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


# ── Box merging ────────────────────────────────────────────────────────


def merge_overlapping_structural_boxes(
    boxes: List[StructuralBox],
) -> List[StructuralBox]:
    """Merge overlapping structural boxes of the same type into union polygons.

    For each :class:`BoxType`, finds overlap clusters (transitive) and
    merges each cluster into a single :class:`StructuralBox` whose
    ``polygon`` is the Shapely union of all source rectangles.

    Boxes of different types never merge.  Clusters of size 1 are
    returned unchanged.

    Parameters
    ----------
    boxes : list[StructuralBox]
        The input boxes (may be mutated in-place via replacement).

    Returns
    -------
    list[StructuralBox]
        A new list with merged boxes replacing their source clusters.
    """
    if not boxes:
        return []

    # Group by box_type
    by_type: Dict[BoxType, List[int]] = {}
    for i, b in enumerate(boxes):
        by_type.setdefault(b.box_type, []).append(i)

    merged_indices: set = set()  # indices consumed by merging
    new_boxes: List[StructuralBox] = []

    for btype, indices in by_type.items():
        bboxes = [boxes[i].bbox() for i in indices]
        clusters = find_overlap_clusters(bboxes)

        for cluster in clusters:
            if len(cluster) == 1:
                continue  # leave singletons alone

            global_indices = [indices[c] for c in cluster]
            merged_indices.update(global_indices)

            cluster_bboxes = [boxes[gi].bbox() for gi in global_indices]
            poly_coords = merge_boxes(cluster_bboxes)
            bbox = polygon_bbox(poly_coords)

            # Survivor inherits properties from the largest source box
            largest_gi = max(global_indices, key=lambda gi: boxes[gi].area())
            src = boxes[largest_gi]

            merged = StructuralBox(
                page=src.page,
                x0=bbox[0],
                y0=bbox[1],
                x1=bbox[2],
                y1=bbox[3],
                box_type=btype,
                confidence=max(boxes[gi].confidence for gi in global_indices),
                contained_block_indices=sum(
                    (boxes[gi].contained_block_indices for gi in global_indices), []
                ),
                contained_text="\n".join(
                    boxes[gi].contained_text
                    for gi in global_indices
                    if boxes[gi].contained_text
                ),
                source=src.source,
                is_synthetic=any(boxes[gi].is_synthetic for gi in global_indices),
                polygon=poly_coords,
            )
            new_boxes.append(merged)
            logger.debug(
                "Merged %d %s boxes → polygon with %d vertices",
                len(global_indices),
                btype.value,
                len(poly_coords),
            )

    # Combine: keep un-merged originals + new merged boxes
    result = [b for i, b in enumerate(boxes) if i not in merged_indices]
    result.extend(new_boxes)
    result.sort(key=lambda b: b.area(), reverse=True)
    return result


# ── Synthetic region generation ────────────────────────────────────────


def create_synthetic_regions(
    blocks: List[BlockCluster],
    structural_boxes: List[StructuralBox],
    page_width: float,
    page_height: float,
    max_gap: float = 0.0,
    x_tolerance: float = 80.0,
    font_size_ratio: float = 1.8,
    adaptive_gap_mult: float = 3.0,
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
            max_gap=max_gap,
            x_tolerance=x_tolerance,
            font_size_ratio=font_size_ratio,
            adaptive_gap_mult=adaptive_gap_mult,
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


# ── Anchor-based region growth ─────────────────────────────────────────


def _grow_region_from_anchor(
    anchor_idx: int,
    blocks: List[BlockCluster],
    structural_boxes: List[StructuralBox],
    page_width: float,
    page_height: float,
    max_gap: float = 0.0,
    x_tolerance: float = 80.0,
    font_size_ratio: float = 1.8,
    adaptive_gap_mult: float = 3.0,
) -> Tuple[float, float, float, float]:
    """Grow a bounded region downward from a header anchor block.

    Starting from the anchor block, we expand the region to include
    nearby blocks below it that are:
    - Within ``x_tolerance`` of the anchor's x-range (same column)
    - Within an adaptive vertical gap of the current region bottom
    - Not already inside a classified structural box
    - Not a header for a *different* section
    - Within ``font_size_ratio`` of the section's median font size
      (prevents vacuuming title-block labels into note regions)

    The adaptive gap is computed as median_line_spacing * adaptive_gap_mult
    from the blocks already in the region.  If ``max_gap > 0``, it is used
    as a hard cap instead.

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

    # Track y-positions of block tops for adaptive gap computation
    block_bottoms = [ay1]

    # Collect font sizes in the region for glyph-style continuity
    region_font_sizes: List[float] = []
    for row in anchor.rows:
        for b in row.boxes:
            sz = getattr(b, "font_size", 0.0)
            if sz > 0:
                region_font_sizes.append(sz)

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

        # Compute adaptive gap threshold from blocks already in region
        if max_gap > 0:
            effective_gap = max_gap
        else:
            # Adaptive: use median line spacing * multiplier
            if len(block_bottoms) >= 2:
                spacings = [
                    block_bottoms[j] - block_bottoms[j - 1]
                    for j in range(1, len(block_bottoms))
                    if block_bottoms[j] > block_bottoms[j - 1]
                ]
                if spacings:
                    spacings.sort()
                    median_spacing = spacings[len(spacings) // 2]
                    effective_gap = median_spacing * adaptive_gap_mult
                else:
                    effective_gap = 40.0 * adaptive_gap_mult
            else:
                # Fallback for first block after anchor
                effective_gap = 40.0 * adaptive_gap_mult

            # Clamp to a reasonable range
            effective_gap = max(20.0, min(effective_gap, 200.0))

        # Stop if there's a large vertical gap
        if by0 - region_bottom > effective_gap:
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

        # Font-size continuity check: reject blocks whose average font
        # size deviates too far from the section's median.
        if font_size_ratio > 0 and region_font_sizes:
            blk_sizes = []
            for row in blk.rows:
                for b in row.boxes:
                    sz = getattr(b, "font_size", 0.0)
                    if sz > 0:
                        blk_sizes.append(sz)
            if blk_sizes:
                region_median_sz = sorted(region_font_sizes)[
                    len(region_font_sizes) // 2
                ]
                blk_avg_sz = sum(blk_sizes) / len(blk_sizes)
                if region_median_sz > 0:
                    ratio = max(
                        blk_avg_sz / region_median_sz,
                        region_median_sz / blk_avg_sz,
                    )
                    if ratio > font_size_ratio:
                        # Font size too different — skip this block
                        continue

        region_blocks.append(blk)
        region_bottom = max(region_bottom, by1)
        block_bottoms.append(by1)

        # Update font-size tracking
        for row in blk.rows:
            for b in row.boxes:
                sz = getattr(b, "font_size", 0.0)
                if sz > 0:
                    region_font_sizes.append(sz)

    # Compute bounding box of all collected blocks
    r_x0 = min(blk.bbox()[0] for blk in region_blocks)
    r_y0 = min(blk.bbox()[1] for blk in region_blocks)
    r_x1 = max(blk.bbox()[2] for blk in region_blocks)
    r_y1 = max(blk.bbox()[3] for blk in region_blocks)

    return (r_x0, r_y0, r_x1, r_y1)


# ── Full semantic region pipeline ──────────────────────────────────────


def detect_semantic_regions(
    blocks: List[BlockCluster],
    graphics: List[GraphicElement],
    page_width: float,
    page_height: float,
    max_gap: float = 0.0,
    x_tolerance: float = 80.0,
    font_size_ratio: float = 1.8,
    adaptive_gap_mult: float = 3.0,
    merge_overlapping: bool = False,
    config: Optional[AnalysisConfig] = None,
) -> Tuple[List[StructuralBox], List[SemanticRegion]]:
    """Run the full structural detection + semantic region pipeline.

    This is the main entry point.  It:

    1. Detects structural boxes from graphics.
    2. Classifies them by text + geometry.
    3. Creates synthetic regions from text anchors.
    4. Optionally merges overlapping boxes of the same type (when
       *merge_overlapping* is ``True``).
    5. Grows semantic regions from anchors.
    6. Returns both the classified structural boxes and the semantic regions.

    Parameters
    ----------
    blocks : list[BlockCluster]
        All text blocks on the page (from grouping).
    graphics : list[GraphicElement]
        Graphic elements on the page (from ``extract_graphics``).
    page_width, page_height : float
        Page dimensions in points.
    max_gap : float
        Max vertical gap for region growth (0 = adaptive).
    x_tolerance : float
        Horizontal tolerance for column membership.
    font_size_ratio : float
        Max font-size ratio before excluding a block (0 = disabled).
    adaptive_gap_mult : float
        Multiplier for adaptive gap computation.

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
    classify_structural_boxes(
        struct_boxes, blocks, page_width, page_height, config=config
    )

    # Step 3: Synthetic regions from text anchors
    synthetic = create_synthetic_regions(
        blocks,
        struct_boxes,
        page_width,
        page_height,
        max_gap=max_gap,
        x_tolerance=x_tolerance,
        font_size_ratio=font_size_ratio,
        adaptive_gap_mult=adaptive_gap_mult,
    )
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

    # Step 3b: Optionally merge overlapping boxes of the same type
    if merge_overlapping:
        all_boxes = merge_overlapping_structural_boxes(all_boxes)

    # Step 4: Build semantic regions
    regions = _build_semantic_regions(
        blocks,
        all_boxes,
        page_width,
        page_height,
        max_gap=max_gap,
        x_tolerance=x_tolerance,
        font_size_ratio=font_size_ratio,
        adaptive_gap_mult=adaptive_gap_mult,
    )

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
    max_gap: float = 0.0,
    x_tolerance: float = 80.0,
    font_size_ratio: float = 1.8,
    adaptive_gap_mult: float = 3.0,
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
            max_gap=max_gap,
            x_tolerance=x_tolerance,
            font_size_ratio=font_size_ratio,
            adaptive_gap_mult=adaptive_gap_mult,
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


# ── Private helpers ────────────────────────────────────────────────────


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
