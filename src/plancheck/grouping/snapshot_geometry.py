"""Pure geometry utilities for group snapshot capture.

Computes the three geometry buckets required by the IsGroup snapshot schema:

- **group_geometry**   — overall shape and spatial properties of the group
- **normalized_geom**  — scale-invariant box positions within the group
- **page_context**     — where the group sits on the page

These functions are deliberately pure (no I/O, no database access) so they
can be unit-tested in isolation and called from pipeline or GUI code without
side effects.

Sits in ``plancheck.grouping`` because it reasons about the same spatial
concepts as :mod:`.clustering` and :mod:`.spatial`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from plancheck.models.tokens import GlyphBox

if TYPE_CHECKING:
    from plancheck.models.blocks import BlockCluster

# ── Zone classification thresholds ───────────────────────────────────────────
# Fraction of page width/height that counts as "near an edge".
_MARGIN_FRAC: float = 0.15
# Half-width of the "center" zone around the page midpoint.
_CENTER_HALF: float = 0.20


def compute_group_geometry(
    boxes: List[GlyphBox],
    page_w: float,
    page_h: float,
) -> dict:
    """Compute overall shape and spatial properties of a group.

    Parameters
    ----------
    boxes:
        The ``GlyphBox`` instances that make up the group.
    page_w, page_h:
        Page dimensions in PDF points (used only if extended properties
        need page-relative values in the future; kept for API stability).

    Returns
    -------
    dict
        Keys:

        ``bbox``
            ``[x0, y0, x1, y1]`` enclosing all boxes.
        ``n_boxes``
            Count of boxes in the group.
        ``dominant_axis``
            ``"horizontal"`` if group width ≥ height, else ``"vertical"``.
        ``density``
            Ratio of total box area to group bounding-box area (clamped 0–1).
            Low density means the boxes are spread out; high means tightly packed.
        ``aspect_ratio``
            ``group_width / group_height``; large = wide, small = tall.
        ``has_nearby_vector``
            ``True`` if any box has ``origin == "vector_symbol"``.
    """
    if not boxes:
        return {
            "bbox": [0.0, 0.0, 0.0, 0.0],
            "n_boxes": 0,
            "dominant_axis": "horizontal",
            "density": 0.0,
            "aspect_ratio": 1.0,
            "has_nearby_vector": False,
        }

    gx0 = min(b.x0 for b in boxes)
    gy0 = min(b.y0 for b in boxes)
    gx1 = max(b.x1 for b in boxes)
    gy1 = max(b.y1 for b in boxes)

    gw = max(gx1 - gx0, 1e-6)
    gh = max(gy1 - gy0, 1e-6)

    total_box_area = sum(b.area() for b in boxes)
    group_bbox_area = gw * gh

    return {
        "bbox": [round(gx0, 3), round(gy0, 3), round(gx1, 3), round(gy1, 3)],
        "n_boxes": len(boxes),
        "dominant_axis": "horizontal" if gw >= gh else "vertical",
        "density": round(min(total_box_area / group_bbox_area, 1.0), 4),
        "aspect_ratio": round(gw / gh, 4),
        "has_nearby_vector": any(b.origin == "vector_symbol" for b in boxes),
    }


def compute_normalized_geometry(boxes: List[GlyphBox]) -> dict:
    """Compute scale-invariant box positions within the group.

    Translates each box center relative to the group centroid, then scales
    by the group bounding-box dimensions so output coordinates sit in
    roughly ``[-1, 1]``.  This lets the GNN recognize the same spatial
    pattern at any scale across different plan sheets.

    Parameters
    ----------
    boxes:
        The ``GlyphBox`` instances that make up the group.

    Returns
    -------
    dict
        Key ``normalized_boxes``: list of dicts, one per box, each with:

        ``cx``, ``cy``
            Centroid-relative, scale-normalized position.
        ``w``, ``h``
            Width and height relative to group bounding-box dims.
        ``text``
            Text content of the box.
        ``font_size``
            Raw font size in PDF points.
    """
    if not boxes:
        return {"normalized_boxes": []}

    gx0 = min(b.x0 for b in boxes)
    gy0 = min(b.y0 for b in boxes)
    gx1 = max(b.x1 for b in boxes)
    gy1 = max(b.y1 for b in boxes)

    gcx = (gx0 + gx1) / 2.0
    gcy = (gy0 + gy1) / 2.0
    gw = max(gx1 - gx0, 1e-6)
    gh = max(gy1 - gy0, 1e-6)

    normalized: List[dict] = []
    for b in boxes:
        bcx = (b.x0 + b.x1) / 2.0
        bcy = (b.y0 + b.y1) / 2.0
        normalized.append(
            {
                "cx": round((bcx - gcx) / gw, 4),
                "cy": round((bcy - gcy) / gh, 4),
                "w": round(b.width() / gw, 4),
                "h": round(b.height() / gh, 4),
                "text": b.text,
                "font_size": round(b.font_size, 3),
            }
        )
    return {"normalized_boxes": normalized}


def compute_page_context(
    boxes: List[GlyphBox],
    page_num: int,
    page_w: float,
    page_h: float,
    all_blocks: "List[BlockCluster]",
) -> dict:
    """Compute where the group sits on the page and what surrounds it.

    Parameters
    ----------
    boxes:
        The ``GlyphBox`` instances that make up the group.
    page_num:
        0-based page number.
    page_w, page_h:
        Page dimensions in PDF points.
    all_blocks:
        All ``BlockCluster`` instances on the page (used for nearby-group
        count).

    Returns
    -------
    dict
        Keys:

        ``page_number``
            0-based page index (echoed for self-contained records).
        ``zone``
            One of ``"corner"``, ``"margin"``, ``"field"``, ``"center"``.
        ``edge_distances``
            Sub-dict with ``top``, ``right``, ``bottom``, ``left`` — each
            expressed as a fraction of the corresponding page dimension.
        ``nearby_group_count``
            Number of blocks whose center is within 2× the group's width.
        ``near_title_block``
            ``True`` if the group center is in the bottom 20 % of the page
            (PDF y-coordinate increases downward, so y > 0.8 * page_h).
    """
    if not boxes:
        return {
            "page_number": page_num,
            "zone": "field",
            "edge_distances": {"top": 0.5, "right": 0.5, "bottom": 0.5, "left": 0.5},
            "nearby_group_count": 0,
            "near_title_block": False,
        }

    gx0 = min(b.x0 for b in boxes)
    gy0 = min(b.y0 for b in boxes)
    gx1 = max(b.x1 for b in boxes)
    gy1 = max(b.y1 for b in boxes)

    gcx = (gx0 + gx1) / 2.0
    gcy = (gy0 + gy1) / 2.0
    gw = max(gx1 - gx0, 1e-6)

    pw = max(page_w, 1e-6)
    ph = max(page_h, 1e-6)

    # Normalized group center (fractions of page dimensions)
    fx = gcx / pw
    fy = gcy / ph

    edge_distances = {
        "top": round(gy0 / ph, 4),
        "right": round((pw - gx1) / pw, 4),
        "bottom": round((ph - gy1) / ph, 4),
        "left": round(gx0 / pw, 4),
    }

    zone = _classify_zone(fx, fy)

    # Nearby groups: blocks whose bbox center is within 2× the group width
    search_radius = 2.0 * gw
    nearby = sum(
        1
        for block in all_blocks
        if _block_center_within(block, gcx, gcy, search_radius)
    )

    return {
        "page_number": page_num,
        "zone": zone,
        "edge_distances": edge_distances,
        "nearby_group_count": nearby,
        "near_title_block": gcy > 0.8 * ph,
    }


# ── Private helpers ───────────────────────────────────────────────────────────


def _classify_zone(fx: float, fy: float) -> str:
    """Map a normalised page position to a zone name.

    The page is divided by proximity to edges and the page centre:

    - ``"corner"``  — within ``_MARGIN_FRAC`` of both an x-edge and a y-edge
    - ``"margin"``  — within ``_MARGIN_FRAC`` of any single edge (not a corner)
    - ``"center"``  — within ``_CENTER_HALF`` of the page midpoint in both axes
    - ``"field"``   — everything else
    """
    near_left = fx < _MARGIN_FRAC
    near_right = fx > 1.0 - _MARGIN_FRAC
    near_top = fy < _MARGIN_FRAC
    near_bottom = fy > 1.0 - _MARGIN_FRAC

    in_x_margin = near_left or near_right
    in_y_margin = near_top or near_bottom

    if in_x_margin and in_y_margin:
        return "corner"
    if in_x_margin or in_y_margin:
        return "margin"

    center_min = 0.5 - _CENTER_HALF
    center_max = 0.5 + _CENTER_HALF
    if center_min < fx < center_max and center_min < fy < center_max:
        return "center"

    return "field"


def _block_center_within(
    block: "BlockCluster",
    gcx: float,
    gcy: float,
    radius: float,
) -> bool:
    """Return ``True`` if *block*'s bbox center is within *radius* of (gcx, gcy)."""
    try:
        bx0, by0, bx1, by1 = block.bbox()
    except Exception:  # noqa: BLE001 — defensive; bbox() should not raise
        return False
    bcx = (bx0 + bx1) / 2.0
    bcy = (by0 + by1) / 2.0
    return abs(bcx - gcx) < radius and abs(bcy - gcy) < radius
