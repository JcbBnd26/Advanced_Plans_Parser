from __future__ import annotations

import math
from typing import Iterable, List

from .models import GlyphBox


def intersection_over_union(a: GlyphBox, b: GlyphBox) -> float:
    ax0, ay0, ax1, ay1 = a.bbox()
    bx0, by0, bx1, by1 = b.bbox()
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.area() + b.area() - inter
    return inter / union if union > 0 else 0.0


def nms_prune(boxes: Iterable[GlyphBox], iou_threshold: float) -> List[GlyphBox]:
    remaining = sorted(boxes, key=lambda b: b.area(), reverse=True)
    kept: List[GlyphBox] = []
    while remaining:
        current = remaining.pop(0)
        kept.append(current)
        remaining = [
            b for b in remaining if intersection_over_union(current, b) < iou_threshold
        ]
    return kept


def estimate_skew_degrees(boxes: Iterable[GlyphBox], max_degrees: float) -> float:
    centers_x = [(b.x0 + b.x1) * 0.5 for b in boxes]
    tops_y = [b.y0 for b in boxes]
    if len(centers_x) < 2:
        return 0.0
    x_mean = sum(centers_x) / len(centers_x)
    y_mean = sum(tops_y) / len(tops_y)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(centers_x, tops_y))
    den = sum((x - x_mean) ** 2 for x in centers_x)
    if den == 0:
        return 0.0
    slope = num / den
    deg = math.degrees(math.atan(slope))
    return float(max(-max_degrees, min(max_degrees, deg)))


def rotate_boxes(
    boxes: Iterable[GlyphBox], degrees: float, page_width: float, page_height: float
) -> List[GlyphBox]:
    if abs(degrees) < 0.01:
        return list(boxes)
    radians = math.radians(degrees)
    cos_t = math.cos(radians)
    sin_t = math.sin(radians)
    cx = page_width * 0.5
    cy = page_height * 0.5
    rotated: List[GlyphBox] = []
    for b in boxes:
        corners = [
            (b.x0, b.y0),
            (b.x1, b.y0),
            (b.x1, b.y1),
            (b.x0, b.y1),
        ]
        rotated_pts = []
        for x, y in corners:
            # Translate to center, rotate, translate back.
            dx = x - cx
            dy = y - cy
            rx = dx * cos_t - dy * sin_t
            ry = dx * sin_t + dy * cos_t
            rotated_pts.append((rx + cx, ry + cy))
        xs = [p[0] for p in rotated_pts]
        ys = [p[1] for p in rotated_pts]
        x0 = min(xs)
        x1 = max(xs)
        y0 = min(ys)
        y1 = max(ys)
        rotated.append(
            GlyphBox(
                page=b.page,
                x0=float(x0),
                y0=float(y0),
                x1=float(x1),
                y1=float(y1),
                text=b.text,
                origin=b.origin,
            )
        )
    return rotated
