"""Structural box classification.

Classifies detected :class:`~plancheck.analysis.structural_boxes.StructuralBox`
objects by their geometric properties and enclosed text content.
"""

from __future__ import annotations

import logging
from typing import List

from ..models import BlockCluster
from .structural_boxes import (
    PAGE_BORDER_AREA_FRAC,
    TITLE_BLOCK_MAX_ASPECT,
    TITLE_BLOCK_MIN_HEIGHT_FRAC,
    TITLE_BLOCK_RIGHT_EDGE_FRAC,
    BoxType,
    StructuralBox,
)

logger = logging.getLogger("plancheck.structural")

# ── Classification keyword sets ────────────────────────────────────────

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
