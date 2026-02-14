"""Page zoning: partition a plan sheet into semantic regions.

Construction/engineering plan sheets have a predictable layout:

    ┌─────────────────────────────────────────────┐
    │  BORDER / MARGIN                            │
    │  ┌──────────────────────────────┬──────────┐│
    │  │                              │          ││
    │  │       DRAWING AREA           │  NOTES   ││
    │  │                              │  COLUMN  ││
    │  │                              │          ││
    │  ├──────────────────────────────┴──────────┤│
    │  │            TITLE BLOCK                  ││
    │  └─────────────────────────────────────────┘│
    └─────────────────────────────────────────────┘

This module provides:

- :class:`PageZone` – an axis-aligned region with a semantic tag.
- :func:`detect_zones` – heuristic zone detection from blocks, regions,
  and page geometry.
- :func:`classify_blocks` – assign each :class:`BlockCluster` to its
  most-overlapping zone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from .config import GroupingConfig
from .models import BlockCluster, GlyphBox, NotesColumn

# ── Zone tags ──────────────────────────────────────────────────────────


class ZoneTag(str, Enum):
    """Semantic tag for a page zone."""

    page = "page"  # Whole page (root)
    border = "border"  # Sheet border / margin
    drawing = "drawing"  # Main drawing area
    notes = "notes"  # Notes column(s)
    title_block = "title_block"  # Title block (bottom strip)
    legend = "legend"  # Legend region
    abbreviations = "abbreviations"  # Abbreviation table
    revisions = "revisions"  # Revision table / block
    details = "details"  # Standard details region
    unknown = "unknown"  # Unclassified


# ── PageZone dataclass ─────────────────────────────────────────────────


@dataclass
class PageZone:
    """An axis-aligned rectangular zone on a page.

    Attributes
    ----------
    tag : ZoneTag
        Semantic meaning of this zone.
    x0, y0, x1, y1 : float
        Bounding box in page coordinates (pts, origin top-left).
    confidence : float
        0–1 heuristic confidence.
    children : list[PageZone]
        Nested sub-zones (e.g., a notes zone inside the page zone).
    """

    tag: ZoneTag
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float = 1.0
    children: List[PageZone] = field(default_factory=list)

    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def contains_point(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def overlap_area(self, other_bbox: Tuple[float, float, float, float]) -> float:
        """Axis-aligned intersection area with another bbox."""
        ox0, oy0, ox1, oy1 = other_bbox
        ix0 = max(self.x0, ox0)
        iy0 = max(self.y0, oy0)
        ix1 = min(self.x1, ox1)
        iy1 = min(self.y1, oy1)
        return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)

    def overlap_fraction(self, other_bbox: Tuple[float, float, float, float]) -> float:
        """Fraction of *other_bbox* that falls inside this zone."""
        ox0, oy0, ox1, oy1 = other_bbox
        other_area = max(0.0, ox1 - ox0) * max(0.0, oy1 - oy0)
        if other_area <= 0:
            return 0.0
        return self.overlap_area(other_bbox) / other_area


# ── Zone detection ─────────────────────────────────────────────────────


def _detect_title_block_zone(
    page_width: float,
    page_height: float,
    blocks: List[BlockCluster],
    cfg: GroupingConfig,
) -> Optional[PageZone]:
    """Detect the title block at the bottom of the sheet.

    Heuristic: title blocks on ODOT / standard CAD sheets occupy the
    bottom ~15% of the page and span most of the page width.  We look for
    a cluster of blocks in that region.
    """
    title_top = page_height * 0.85
    title_bottom = page_height

    title_blocks = []
    for blk in blocks:
        bx0, by0, bx1, by1 = blk.bbox()
        cy = (by0 + by1) / 2.0
        if cy >= title_top:
            title_blocks.append(blk)

    if not title_blocks:
        return None

    # Compute bounding box of all title-area blocks
    x0 = min(blk.bbox()[0] for blk in title_blocks)
    y0 = min(blk.bbox()[1] for blk in title_blocks)
    x1 = max(blk.bbox()[2] for blk in title_blocks)
    y1 = max(blk.bbox()[3] for blk in title_blocks)

    # Confidence: higher if this zone is wide (spans > 50% page width)
    width_frac = (x1 - x0) / page_width if page_width > 0 else 0
    confidence = min(1.0, width_frac * 1.5)  # 67% page width → full confidence

    return PageZone(
        tag=ZoneTag.title_block,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        confidence=confidence,
    )


def _detect_notes_zone(
    page_width: float,
    page_height: float,
    notes_columns: List[NotesColumn],
    title_zone: Optional[PageZone],
) -> Optional[PageZone]:
    """Detect the notes column zone from grouped notes columns.

    Notes columns are typically on the right side of the page, above the
    title block.
    """
    if not notes_columns:
        return None

    all_bboxes = [col.bbox() for col in notes_columns]
    x0 = min(bb[0] for bb in all_bboxes)
    y0 = min(bb[1] for bb in all_bboxes)
    x1 = max(bb[2] for bb in all_bboxes)
    y1 = max(bb[3] for bb in all_bboxes)

    # Extend to page edges where appropriate
    # Notes column typically extends to right edge
    if x1 > page_width * 0.6:
        x1 = page_width

    # Notes column typically starts near the top of the content band
    # and ends at the title block (or bottom of page)
    if title_zone is not None:
        y1 = max(y1, title_zone.y0)
    else:
        y1 = max(y1, page_height * 0.85)

    # Confidence from how many notes columns we found
    confidence = min(1.0, 0.5 + len(notes_columns) * 0.15)

    return PageZone(
        tag=ZoneTag.notes,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        confidence=confidence,
    )


def _detect_border_zone(
    page_width: float,
    page_height: float,
    blocks: List[BlockCluster],
    cfg: GroupingConfig,
) -> PageZone:
    """Detect the border / margin zone.

    The border is the area between the page edge and the outermost content.
    On CAD sheets this is typically 0.25"–0.5" on each side.
    """
    if not blocks:
        # Default border: 0.5" = 36pts on all sides
        margin = 36.0
        return PageZone(
            tag=ZoneTag.border,
            x0=0,
            y0=0,
            x1=page_width,
            y1=page_height,
            confidence=0.5,
            children=[
                PageZone(
                    tag=ZoneTag.drawing,
                    x0=margin,
                    y0=margin,
                    x1=page_width - margin,
                    y1=page_height - margin,
                    confidence=0.5,
                )
            ],
        )

    # Detect actual content bounds
    content_x0 = min(blk.bbox()[0] for blk in blocks)
    content_y0 = min(blk.bbox()[1] for blk in blocks)
    content_x1 = max(blk.bbox()[2] for blk in blocks)
    content_y1 = max(blk.bbox()[3] for blk in blocks)

    # Inset slightly to exclude edge-touching content
    margin_left = max(0, content_x0 - 5)
    margin_top = max(0, content_y0 - 5)
    margin_right = min(page_width, content_x1 + 5)
    margin_bottom = min(page_height, content_y1 + 5)

    return PageZone(
        tag=ZoneTag.border,
        x0=0,
        y0=0,
        x1=page_width,
        y1=page_height,
        confidence=0.8,
        children=[
            PageZone(
                tag=ZoneTag.drawing,
                x0=margin_left,
                y0=margin_top,
                x1=margin_right,
                y1=margin_bottom,
                confidence=0.8,
            )
        ],
    )


def detect_zones(
    page_width: float,
    page_height: float,
    blocks: List[BlockCluster],
    notes_columns: Optional[List[NotesColumn]] = None,
    legend_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
    abbreviation_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
    revision_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
    detail_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
    cfg: Optional[GroupingConfig] = None,
) -> List[PageZone]:
    """Detect semantic zones on a plan sheet page.

    Parameters
    ----------
    page_width, page_height : float
        Page dimensions in points.
    blocks : list[BlockCluster]
        All detected blocks on the page.
    notes_columns : list[NotesColumn], optional
        Grouped notes columns (from ``group_notes_columns``).
    legend_bboxes : list of bbox tuples, optional
        Bounding boxes of detected legend regions.
    abbreviation_bboxes : list of bbox tuples, optional
        Bounding boxes of detected abbreviation regions.
    revision_bboxes : list of bbox tuples, optional
        Bounding boxes of detected revision regions.
    detail_bboxes : list of bbox tuples, optional
        Bounding boxes of detected standard detail regions.
    cfg : GroupingConfig, optional
        Configuration (uses defaults if None).

    Returns
    -------
    list[PageZone]
        Flat list of detected zones, sorted by area (largest first).
        The ``page`` zone is always first.
    """
    if cfg is None:
        cfg = GroupingConfig()

    zones: List[PageZone] = []

    # Root zone: entire page
    page_zone = PageZone(
        tag=ZoneTag.page,
        x0=0,
        y0=0,
        x1=page_width,
        y1=page_height,
        confidence=1.0,
    )
    zones.append(page_zone)

    # Title block
    title_zone = _detect_title_block_zone(page_width, page_height, blocks, cfg)
    if title_zone is not None:
        zones.append(title_zone)
        page_zone.children.append(title_zone)

    # Notes column
    notes_zone = _detect_notes_zone(
        page_width, page_height, notes_columns or [], title_zone
    )
    if notes_zone is not None:
        zones.append(notes_zone)
        page_zone.children.append(notes_zone)

    # Region zones from detected bboxes
    def _add_region_zones(
        bboxes: Optional[List[Tuple[float, float, float, float]]],
        tag: ZoneTag,
    ) -> None:
        if not bboxes:
            return
        for bb in bboxes:
            zone = PageZone(
                tag=tag,
                x0=bb[0],
                y0=bb[1],
                x1=bb[2],
                y1=bb[3],
                confidence=0.7,
            )
            zones.append(zone)
            page_zone.children.append(zone)

    _add_region_zones(legend_bboxes, ZoneTag.legend)
    _add_region_zones(abbreviation_bboxes, ZoneTag.abbreviations)
    _add_region_zones(revision_bboxes, ZoneTag.revisions)
    _add_region_zones(detail_bboxes, ZoneTag.details)

    # Drawing area: everything not covered by notes, title, or other regions
    non_drawing_zones = [z for z in zones if z.tag not in (ZoneTag.page,)]
    if non_drawing_zones:
        # Drawing area bounds: shrink from page, exclude title block
        draw_y1 = title_zone.y0 if title_zone else page_height * 0.85
        draw_x1 = notes_zone.x0 if notes_zone else page_width

        border_zone = _detect_border_zone(page_width, page_height, blocks, cfg)
        drawing_content = border_zone.children[0] if border_zone.children else None
        if drawing_content:
            draw_x0 = drawing_content.x0
            draw_y0 = drawing_content.y0
        else:
            draw_x0 = 0
            draw_y0 = 0

        drawing_zone = PageZone(
            tag=ZoneTag.drawing,
            x0=draw_x0,
            y0=draw_y0,
            x1=draw_x1,
            y1=draw_y1,
            confidence=0.6,
        )
        zones.append(drawing_zone)
        page_zone.children.append(drawing_zone)

    return zones


# ── Block classification ───────────────────────────────────────────────


def classify_blocks(
    blocks: List[BlockCluster],
    zones: List[PageZone],
) -> dict[int, ZoneTag]:
    """Assign each block to its best-matching zone.

    Parameters
    ----------
    blocks : list[BlockCluster]
        Blocks to classify.
    zones : list[PageZone]
        Zones from :func:`detect_zones`.

    Returns
    -------
    dict[int, ZoneTag]
        Mapping of block index → best zone tag.  If a block doesn't
        overlap any non-page zone significantly, it gets ``ZoneTag.unknown``.
    """
    # Exclude the page-level zone from matching (everything is in it)
    candidate_zones = [z for z in zones if z.tag != ZoneTag.page]

    result: dict[int, ZoneTag] = {}
    for idx, blk in enumerate(blocks):
        bb = blk.bbox()
        best_zone: Optional[PageZone] = None
        best_overlap = 0.0

        for zone in candidate_zones:
            overlap = zone.overlap_fraction(bb)
            if overlap > best_overlap:
                best_overlap = overlap
                best_zone = zone

        if best_zone is not None and best_overlap > 0.3:
            result[idx] = best_zone.tag
        else:
            result[idx] = ZoneTag.unknown

    return result


def zone_summary(zones: List[PageZone]) -> dict:
    """Produce a JSON-serializable summary of detected zones.

    Useful for manifest output and debugging.
    """
    return {
        "zones": [
            {
                "tag": z.tag.value,
                "bbox": [z.x0, z.y0, z.x1, z.y1],
                "area": round(z.area(), 1),
                "confidence": round(z.confidence, 3),
                "children": [c.tag.value for c in z.children],
            }
            for z in zones
        ]
    }
