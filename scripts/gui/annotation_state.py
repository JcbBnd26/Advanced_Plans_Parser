"""Shared state primitives: CanvasBox dataclass, handle constants, and
pure coordinate-transform helper functions for the annotation tab.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _reshape_bbox_from_handle(
    orig_bbox: tuple[float, float, float, float],
    handle: str,
    px: float,
    py: float,
    *,
    min_size: float = 1.0,
) -> tuple[float, float, float, float]:
    """Compute a resized bbox when dragging a named handle.

    Coordinates are in PDF space.
    """
    ox0, oy0, ox1, oy1 = orig_bbox
    nx0, ny0, nx1, ny1 = ox0, oy0, ox1, oy1

    if "w" in handle:
        nx0 = min(px, ox1 - min_size)
    if "e" in handle:
        nx1 = max(px, ox0 + min_size)
    if "n" in handle:
        ny0 = min(py, oy1 - min_size)
    if "s" in handle:
        ny1 = max(py, oy0 + min_size)

    return (nx0, ny0, nx1, ny1)


def _scale_polygon_to_bbox(
    orig_bbox: tuple[float, float, float, float],
    polygon: list[tuple[float, float]],
    new_bbox: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    """Scale polygon points from orig_bbox into new_bbox.

    This keeps each point's relative position within the bbox.
    """
    ox0, oy0, ox1, oy1 = orig_bbox
    nx0, ny0, nx1, ny1 = new_bbox

    ow = max(ox1 - ox0, 1e-6)
    oh = max(oy1 - oy0, 1e-6)
    nw = nx1 - nx0
    nh = ny1 - ny0

    scaled: list[tuple[float, float]] = []
    for px, py in polygon:
        u = (px - ox0) / ow
        v = (py - oy0) / oh
        scaled.append((nx0 + u * nw, ny0 + v * nh))
    return scaled


# ── CanvasBox ──────────────────────────────────────────────────────────


@dataclass
class CanvasBox:
    """Tracks a single detection drawn on the canvas."""

    detection_id: str
    element_type: str
    confidence: float | None
    text_content: str
    features: dict
    # bbox in PDF points (x0, y0, x1, y1)
    pdf_bbox: tuple[float, float, float, float]
    # Canvas item IDs (set after drawing)
    rect_id: int = 0
    label_id: int = 0
    conf_dot_id: int = 0
    handle_ids: list[int] = field(default_factory=list)
    # State
    selected: bool = False
    corrected: bool = False
    # Polygon outline for merged boxes: list[(x, y)] in PDF points, or None
    polygon: list[tuple[float, float]] | None = None
    # Detection IDs that were merged into this box (for undo)
    merged_from: list[str] | None = None
    # Group membership
    group_id: str | None = None
    is_group_root: bool = False


# ── Handle positions ───────────────────────────────────────────────────

HANDLE_POSITIONS = ("nw", "n", "ne", "e", "se", "s", "sw", "w")
HANDLE_SIZE = 5  # half-size in pixels
