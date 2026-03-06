"""Structural box detection — data model and box detection.

This module provides:

1. **Data model** – :class:`BoxType`, :class:`StructuralBox`,
   :class:`SemanticRegion`.
2. **Box detection** – :func:`detect_structural_boxes` finds drawn
   rectangles/frames on the page from PDF graphic elements.
3. **Masking** – :func:`mask_blocks_by_structural_boxes` filters blocks
   that fall inside excluded box types.

Classification is handled by :mod:`~plancheck.analysis.box_classifier`.
Semantic region growth is handled by
:mod:`~plancheck.analysis.semantic_regions`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from ..models import BlockCluster, GraphicElement

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
    # Union polygon vertices (set when this box was merged from overlapping
    # boxes). None means the box is a simple axis-aligned rectangle.
    polygon: Optional[List[Tuple[float, float]]] = None

    def bbox(self) -> Tuple[float, float, float, float]:
        """Bounding box as ``(x0, y0, x1, y1)``."""
        return (self.x0, self.y0, self.x1, self.y1)

    def width(self) -> float:
        """Horizontal extent in points."""
        return self.x1 - self.x0

    def height(self) -> float:
        """Vertical extent in points."""
        return self.y1 - self.y0

    def area(self) -> float:
        """Area in square points, clamped to zero."""
        return max(0.0, self.width()) * max(0.0, self.height())

    def contains_point(self, x: float, y: float, pad: float = 0.0) -> bool:
        """Return True if (x, y) lies inside the box, with optional padding."""
        return (
            self.x0 - pad <= x <= self.x1 + pad and self.y0 - pad <= y <= self.y1 + pad
        )

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "page": self.page,
            "bbox": [
                round(self.x0, 3),
                round(self.y0, 3),
                round(self.x1, 3),
                round(self.y1, 3),
            ],
            "box_type": (
                self.box_type.value
                if isinstance(self.box_type, BoxType)
                else str(self.box_type)
            ),
            "confidence": round(self.confidence, 4),
            "contained_block_indices": list(self.contained_block_indices),
            "contained_text": self.contained_text,
            "is_synthetic": self.is_synthetic,
            "polygon": (
                [[round(x, 3), round(y, 3)] for x, y in self.polygon]
                if self.polygon
                else None
            ),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StructuralBox":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        bbox = d.get("bbox", [0, 0, 0, 0])
        bt = d.get("box_type", "unknown")
        try:
            box_type = BoxType(bt)
        except ValueError:
            box_type = BoxType.unknown
        return cls(
            page=d.get("page", 0),
            x0=bbox[0],
            y0=bbox[1],
            x1=bbox[2],
            y1=bbox[3],
            box_type=box_type,
            confidence=d.get("confidence", 0.0),
            contained_block_indices=d.get("contained_block_indices", []),
            contained_text=d.get("contained_text", ""),
            is_synthetic=d.get("is_synthetic", False),
            polygon=[(p[0], p[1]) for p in d["polygon"]] if d.get("polygon") else None,
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
        """Bounding box as ``(x0, y0, x1, y1)``."""
        return (self.x0, self.y0, self.x1, self.y1)

    def area(self) -> float:
        """Area in square points, clamped to zero."""
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def to_dict(self, blocks: Optional[List["BlockCluster"]] = None) -> dict:
        """Serialize to a JSON-compatible dict."""
        anchor_idx = None
        if blocks is not None and self.anchor_block is not None:
            try:
                anchor_idx = blocks.index(self.anchor_block)
            except ValueError:
                pass
        child_indices = []
        if blocks is not None:
            for cb in self.child_blocks:
                try:
                    child_indices.append(blocks.index(cb))
                except ValueError:
                    pass
        return {
            "page": self.page,
            "label": self.label,
            "bbox": [
                round(self.x0, 3),
                round(self.y0, 3),
                round(self.x1, 3),
                round(self.y1, 3),
            ],
            "anchor_block_index": anchor_idx,
            "child_block_indices": child_indices,
            "confidence": round(self.confidence, 4),
        }

    @classmethod
    def from_dict(
        cls, d: dict, blocks: Optional[List["BlockCluster"]] = None
    ) -> "SemanticRegion":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        bbox = d.get("bbox", [0, 0, 0, 0])
        anchor_block = None
        ai = d.get("anchor_block_index")
        if blocks is not None and ai is not None and 0 <= ai < len(blocks):
            anchor_block = blocks[ai]
        child_blocks = []
        if blocks is not None:
            for ci in d.get("child_block_indices", []):
                if 0 <= ci < len(blocks):
                    child_blocks.append(blocks[ci])
        return cls(
            page=d.get("page", 0),
            label=d.get("label", ""),
            x0=bbox[0],
            y0=bbox[1],
            x1=bbox[2],
            y1=bbox[3],
            anchor_block=anchor_block,
            child_blocks=child_blocks,
            confidence=d.get("confidence", 0.0),
        )


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
