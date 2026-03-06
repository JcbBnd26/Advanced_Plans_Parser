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
