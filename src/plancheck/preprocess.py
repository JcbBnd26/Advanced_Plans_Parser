from __future__ import annotations

import math
import random
from typing import Iterable, List, Optional

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


# ── Line-aware skew helpers ────────────────────────────────────────────


def _cluster_boxes_into_lines(
    boxes: List[GlyphBox],
    y_tolerance: float = 4.0,
) -> List[List[GlyphBox]]:
    """Group boxes into approximate text lines by y-midpoint proximity.

    Sorts boxes by vertical centre, then sweeps downward, starting a
    new line whenever the y-midpoint jumps more than *y_tolerance*.
    Returns a list of lines (each a list of boxes sorted by x).
    """
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda b: (b.y0 + b.y1) * 0.5)
    lines: List[List[GlyphBox]] = [[sorted_boxes[0]]]
    for b in sorted_boxes[1:]:
        prev_mid = sum((g.y0 + g.y1) * 0.5 for g in lines[-1]) / len(lines[-1])
        cur_mid = (b.y0 + b.y1) * 0.5
        if abs(cur_mid - prev_mid) <= y_tolerance:
            lines[-1].append(b)
        else:
            lines.append([b])
    # Sort each line by x position
    for line in lines:
        line.sort(key=lambda b: b.x0)
    return lines


def _line_angle(line: List[GlyphBox]) -> Optional[float]:
    """Compute the angle (degrees) of a text line via linear regression.

    Uses box centres within the line.  Returns ``None`` when the
    line has fewer than 3 boxes or is effectively vertical.
    """
    if len(line) < 3:
        return None
    xs = [(b.x0 + b.x1) * 0.5 for b in line]
    ys = [(b.y0 + b.y1) * 0.5 for b in line]
    n = len(xs)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    if den < 1e-9:
        return None
    slope = num / den
    return math.degrees(math.atan(slope))


def _weighted_median(values: List[float], weights: List[float]) -> float:
    """Weighted median: sorts values and picks the one where cumulative
    weight crosses 50 %."""
    if not values:
        return 0.0
    pairs = sorted(zip(values, weights), key=lambda p: p[0])
    total = sum(weights)
    if total == 0:
        return 0.0
    cumulative = 0.0
    for val, w in pairs:
        cumulative += w
        if cumulative >= total * 0.5:
            return val
    return pairs[-1][0]  # pragma: no cover


def estimate_skew_degrees(
    boxes: Iterable[GlyphBox],
    max_degrees: float,
    *,
    method: str = "auto",
    y_tolerance: float = 4.0,
    ransac_iterations: int = 200,
    ransac_inlier_threshold: float = 0.5,
    seed: Optional[int] = None,
) -> float:
    """Estimate page skew in degrees.

    **Methods:**

    ``"auto"``  (default)
        Uses ``"line_median"`` when ≥ 5 lines with 3+ boxes each are
        found, otherwise falls back to ``"regression"``.

    ``"regression"``
        Simple OLS regression on *all* box centres (original behaviour).
        Cheap but susceptible to outlier regions (stamps, logos, etc.).

    ``"line_median"``
        Clusters boxes into approximate text lines, computes per-line
        angles, and returns the weighted median angle (weight = line
        length in boxes).  Much more robust against isolated outliers.

    ``"ransac"``
        RANSAC-based: randomly samples pairs of box centres, fits slopes
        and selects the angle with the most inlier support.  Best when
        the page mixes text in many orientations but is slow on large
        box counts.  *ransac_iterations* controls the number of trials.

    All methods clamp the result to  [-*max_degrees*, *max_degrees*].
    """
    box_list = list(boxes)
    if len(box_list) < 2:
        return 0.0

    if method == "auto":
        lines = _cluster_boxes_into_lines(box_list, y_tolerance)
        # Lines with enough boxes to be useful
        good_lines = [ln for ln in lines if len(ln) >= 3]
        method = "line_median" if len(good_lines) >= 5 else "regression"

    if method == "line_median":
        return _estimate_skew_line_median(box_list, max_degrees, y_tolerance)
    if method == "ransac":
        return _estimate_skew_ransac(
            box_list, max_degrees, ransac_iterations, ransac_inlier_threshold, seed
        )
    # Default: regression
    return _estimate_skew_regression(box_list, max_degrees)


def _estimate_skew_regression(boxes: List[GlyphBox], max_degrees: float) -> float:
    """OLS regression on box centres → skew angle."""
    centers_x = [(b.x0 + b.x1) * 0.5 for b in boxes]
    tops_y = [b.y0 for b in boxes]
    x_mean = sum(centers_x) / len(centers_x)
    y_mean = sum(tops_y) / len(tops_y)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(centers_x, tops_y))
    den = sum((x - x_mean) ** 2 for x in centers_x)
    if den == 0:
        return 0.0
    slope = num / den
    deg = math.degrees(math.atan(slope))
    return float(max(-max_degrees, min(max_degrees, deg)))


def _estimate_skew_line_median(
    boxes: List[GlyphBox], max_degrees: float, y_tolerance: float
) -> float:
    """Weighted-median of per-line angles."""
    lines = _cluster_boxes_into_lines(boxes, y_tolerance)
    angles: List[float] = []
    weights: List[float] = []
    for line in lines:
        a = _line_angle(line)
        if a is not None:
            angles.append(a)
            weights.append(float(len(line)))
    if not angles:
        # Fallback to regression
        return _estimate_skew_regression(boxes, max_degrees)
    deg = _weighted_median(angles, weights)
    return float(max(-max_degrees, min(max_degrees, deg)))


def _estimate_skew_ransac(
    boxes: List[GlyphBox],
    max_degrees: float,
    iterations: int,
    inlier_threshold: float,
    seed: Optional[int],
) -> float:
    """RANSAC: sample pairs, fit slopes, pick best-supported angle."""
    rng = random.Random(seed)

    # Work with centres
    centres = [((b.x0 + b.x1) * 0.5, b.y0) for b in boxes]
    n = len(centres)

    best_angle = 0.0
    best_inliers = 0

    for _ in range(min(iterations, n * (n - 1) // 2)):
        i, j = rng.sample(range(n), 2)
        x1, y1 = centres[i]
        x2, y2 = centres[j]
        dx = x2 - x1
        if abs(dx) < 1e-9:
            continue
        slope = (y2 - y1) / dx
        candidate = math.degrees(math.atan(slope))
        if abs(candidate) > max_degrees:
            continue

        # Count inliers
        inliers = 0
        for cx, cy in centres:
            predicted_y = y1 + slope * (cx - x1)
            if abs(cy - predicted_y) <= inlier_threshold:
                inliers += 1

        if inliers > best_inliers:
            best_inliers = inliers
            best_angle = candidate

    return float(max(-max_degrees, min(max_degrees, best_angle)))


def rotate_boxes(
    boxes: Iterable[GlyphBox],
    degrees: float,
    page_width: float,
    page_height: float,
    min_rotation: float = 0.01,
) -> List[GlyphBox]:
    if abs(degrees) < min_rotation:
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
