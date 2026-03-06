"""Polygon-based box merging for overlapping axis-aligned rectangles.

Given a set of ``(x0, y0, x1, y1)`` bounding boxes, this module
computes the *union polygon* via Shapely, producing complex
outlines — L-shapes, T-shapes, crosses, etc. — instead of simple
bounding-rectangle envelopes.

All public functions are pure and stateless.
"""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union

logger = logging.getLogger("plancheck.box_merge")

# Type aliases
Bbox = Tuple[float, float, float, float]  # (x0, y0, x1, y1)
PolygonCoords = List[Tuple[float, float]]  # exterior ring vertices


# ---------------------------------------------------------------------------
# Core merge
# ---------------------------------------------------------------------------


def merge_boxes(bboxes: Sequence[Bbox]) -> PolygonCoords:
    """Merge overlapping boxes into a single union polygon.

    Parameters
    ----------
    bboxes : sequence of (x0, y0, x1, y1)
        Axis-aligned bounding boxes.

    Returns
    -------
    list[(x, y)]
        Exterior ring vertices of the merged polygon.
        For non-overlapping inputs the result is the *largest* polygon.
        The ring is always **closed** (first == last vertex) and uses
        coordinate-order as returned by Shapely (counter-clockwise).

    Raises
    ------
    ValueError
        If *bboxes* is empty.
    """
    if not bboxes:
        raise ValueError("merge_boxes requires at least one bbox")

    polys = [shapely_box(x0, y0, x1, y1) for x0, y0, x1, y1 in bboxes]
    merged = unary_union(polys)

    if isinstance(merged, MultiPolygon):
        # Take the largest polygon by area
        merged = max(merged.geoms, key=lambda g: g.area)

    return list(merged.exterior.coords)


def merge_boxes_multi(bboxes: Sequence[Bbox]) -> List[PolygonCoords]:
    """Merge overlapping boxes, returning **all** resulting polygons.

    Unlike :func:`merge_boxes`, if the inputs form disjoint groups
    each group is returned as a separate polygon.

    Returns
    -------
    list[list[(x, y)]]
        One exterior ring per resulting polygon.
    """
    if not bboxes:
        return []

    polys = [shapely_box(x0, y0, x1, y1) for x0, y0, x1, y1 in bboxes]
    merged = unary_union(polys)

    if isinstance(merged, MultiPolygon):
        return [list(g.exterior.coords) for g in merged.geoms]
    return [list(merged.exterior.coords)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def polygon_bbox(coords: PolygonCoords) -> Bbox:
    """Return the axis-aligned bounding box of a polygon.

    Parameters
    ----------
    coords : list[(x, y)]
        Exterior ring vertices.

    Returns
    -------
    (x0, y0, x1, y1)
    """
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return (min(xs), min(ys), max(xs), max(ys))


def simplify_polygon(
    coords: PolygonCoords,
    tolerance: float = 1.0,
) -> PolygonCoords:
    """Simplify a polygon by removing near-collinear vertices.

    Uses Shapely's :meth:`Polygon.simplify` (Douglas–Peucker) with
    *preserve_topology=True*.

    Parameters
    ----------
    coords : list[(x, y)]
        Exterior ring vertices.
    tolerance : float
        Maximum deviation from the simplified line, in the same units
        as the coordinates (PDF points).

    Returns
    -------
    list[(x, y)]
        Simplified exterior ring.
    """
    poly = Polygon(coords)
    simplified = poly.simplify(tolerance, preserve_topology=True)
    return list(simplified.exterior.coords)


def boxes_overlap(a: Bbox, b: Bbox) -> bool:
    """Return True if two bounding boxes overlap (share any interior area)."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def find_overlap_clusters(bboxes: Sequence[Bbox]) -> List[List[int]]:
    """Group bbox indices into clusters of mutually overlapping boxes.

    Uses simple flood-fill: two boxes are in the same cluster if they
    overlap directly **or** transitively via other boxes.

    Parameters
    ----------
    bboxes : sequence of (x0, y0, x1, y1)

    Returns
    -------
    list[list[int]]
        Each inner list contains the *indices* (into *bboxes*) of boxes
        that belong to the same overlap cluster.  Clusters of size 1
        (isolated boxes) are included.
    """
    n = len(bboxes)
    visited = [False] * n

    clusters: List[List[int]] = []
    for i in range(n):
        if visited[i]:
            continue
        # BFS/DFS flood-fill
        stack = [i]
        cluster: List[int] = []
        while stack:
            idx = stack.pop()
            if visited[idx]:
                continue
            visited[idx] = True
            cluster.append(idx)
            for j in range(n):
                if not visited[j] and boxes_overlap(bboxes[idx], bboxes[j]):
                    stack.append(j)
        clusters.append(sorted(cluster))

    return clusters
