"""Geometry utility functions for bounding box calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from .blocks import BlockCluster


def _region_bbox(
    header: Optional["BlockCluster"],
    entries: list,
    box_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[float, float, float, float]:
    """Compute bbox for a region with an optional header and iterable entries.

    If *box_bbox* is pre-set (e.g. from a detected enclosing rectangle)
    it is returned directly.  Otherwise the bbox is the union of the
    header bbox and every entry bbox.  Returns ``(0, 0, 0, 0)`` when
    there are no components.
    """
    if box_bbox:
        return box_bbox
    xs0: list[float] = []
    ys0: list[float] = []
    xs1: list[float] = []
    ys1: list[float] = []
    if header:
        x0, y0, x1, y1 = header.bbox()
        xs0.append(x0)
        ys0.append(y0)
        xs1.append(x1)
        ys1.append(y1)
    for entry in entries:
        x0, y0, x1, y1 = entry.bbox()
        xs0.append(x0)
        ys0.append(y0)
        xs1.append(x1)
        ys1.append(y1)
    if not xs0:
        return (0, 0, 0, 0)
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def _multi_bbox(
    bboxes: List[Optional[Tuple[float, float, float, float]]],
) -> Tuple[float, float, float, float]:
    """Compute the union bbox from a list of optional bounding boxes.

    ``None`` entries are skipped.  Returns ``(0, 0, 0, 0)`` when no
    valid bounding boxes are supplied.
    """
    xs0: list[float] = []
    ys0: list[float] = []
    xs1: list[float] = []
    ys1: list[float] = []
    for bb in bboxes:
        if bb is not None:
            xs0.append(bb[0])
            ys0.append(bb[1])
            xs1.append(bb[2])
            ys1.append(bb[3])
    if not xs0:
        return (0, 0, 0, 0)
    return (min(xs0), min(ys0), max(xs1), max(ys1))


# ── Bounding Box Intersection / IoU Utilities ──────────────────────────


Bbox = Tuple[float, float, float, float]


def bbox_intersection_area(a: Bbox, b: Bbox) -> float:
    """Compute the intersection area of two axis-aligned bounding boxes.

    Parameters
    ----------
    a, b : tuple[float, float, float, float]
        Bounding boxes as ``(x0, y0, x1, y1)``.

    Returns
    -------
    float
        Intersection area (≥ 0). Zero if boxes don't overlap.
    """
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def bbox_iou(a: Bbox, b: Bbox) -> float:
    """Compute Intersection-over-Union (IoU) of two bounding boxes.

    Parameters
    ----------
    a, b : tuple[float, float, float, float]
        Bounding boxes as ``(x0, y0, x1, y1)``.

    Returns
    -------
    float
        IoU value in [0.0, 1.0]. Returns 0.0 if union area is zero.
    """
    inter = bbox_intersection_area(a, b)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bboxes_overlap(a: Bbox, b: Bbox) -> bool:
    """Return True if two bounding boxes overlap (share any interior area).

    Parameters
    ----------
    a, b : tuple[float, float, float, float]
        Bounding boxes as ``(x0, y0, x1, y1)``.

    Returns
    -------
    bool
        True if boxes have non-zero intersection area.
    """
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def glyph_iou(a: "GlyphBox", b: "GlyphBox") -> float:
    """Compute IoU between two GlyphBox objects.

    Convenience wrapper around :func:`bbox_iou` for token comparisons.
    """
    return bbox_iou(a.bbox(), b.bbox())


if TYPE_CHECKING:
    from .tokens import GlyphBox
