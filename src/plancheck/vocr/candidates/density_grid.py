"""Grid-based blank cell detection methods.

Detects conspicuous blank cells in otherwise dense 2D regions.
"""

from __future__ import annotations

import math
from typing import List

from plancheck.models import GlyphBox, VocrCandidate

from .helpers import _pad_bbox


def _detect_dense_cluster_holes(
    tokens: List[GlyphBox],
    page_num: int,
    page_w: float,
    page_h: float,
    margin: float,
    grid_size: float,
) -> List[VocrCandidate]:
    """Signal #5: conspicuous blank cells in otherwise dense 2D regions."""
    if not tokens or grid_size <= 0:
        return []

    # Build density grid
    cols = max(1, int(math.ceil(page_w / grid_size)))
    rows = max(1, int(math.ceil(page_h / grid_size)))
    grid = [[0] * cols for _ in range(rows)]
    for t in tokens:
        c0 = max(0, int(t.x0 / grid_size))
        c1 = min(cols - 1, int(t.x1 / grid_size))
        r0 = max(0, int(t.y0 / grid_size))
        r1 = min(rows - 1, int(t.y1 / grid_size))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                grid[r][c] += 1

    candidates: List[VocrCandidate] = []
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if grid[r][c] > 0:
                continue
            # Check if surrounded by populated cells (at least 5 of 8 neighbors)
            neighbors = sum(
                1
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0) and grid[r + dr][c + dc] > 0
            )
            if neighbors >= 5:
                gx0 = c * grid_size
                gy0 = r * grid_size
                gx1 = gx0 + grid_size
                gy1 = gy0 + grid_size
                bx0, by0, bx1, by1 = _pad_bbox(
                    gx0, gy0, gx1, gy1, margin, page_w, page_h
                )
                candidates.append(
                    VocrCandidate(
                        page=page_num,
                        x0=bx0,
                        y0=by0,
                        x1=bx1,
                        y1=by1,
                        trigger_methods=["dense_cluster_hole"],
                        predicted_symbol="",
                        confidence=0.45,
                        context={"grid_r": r, "grid_c": c, "neighbor_count": neighbors},
                    )
                )
    return candidates
