"""Confidence scoring and spatial alignment for OCR reconciliation.

Public API
----------
_build_match_index   – spatially align OCR tokens against PDF tokens
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..config import GroupingConfig
from ..models import GlyphBox
from ..vocr.extract import iou
from .helpers import (
    MatchRecord,
    _DEFAULT_COVERAGE_THRESHOLD,
    _DEFAULT_IOU_THRESHOLD,
    center,
)


def _overlap_ratio(a: GlyphBox, b: GlyphBox) -> float:
    """Fraction of *a*'s area covered by *b*."""
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    a_area = a.area()
    return inter / a_area if a_area > 0 else 0.0


def _overlaps_existing(
    candidate: GlyphBox,
    tokens: List[GlyphBox],
    iou_thresh: float = _DEFAULT_IOU_THRESHOLD,
    cov_thresh: float = _DEFAULT_COVERAGE_THRESHOLD,
) -> bool:
    """Return True if *candidate* overlaps any existing token too much."""
    for t in tokens:
        if iou(candidate, t) > iou_thresh:
            return True
        if _overlap_ratio(candidate, t) > cov_thresh:
            return True
    return False


def _build_spatial_grid(
    tokens: List[GlyphBox], cell_size: float = 50.0
) -> dict[tuple[int, int], list[int]]:
    """Build a grid index mapping (col, row) cells to token indices.

    Each token is inserted into every cell its bounding box overlaps.
    Querying with a specific bbox then only needs to scan adjacent cells.
    """
    grid: dict[tuple[int, int], list[int]] = {}
    for idx, t in enumerate(tokens):
        x0, y0, x1, y1 = t.x0, t.y0, t.x1, t.y1
        c0 = int(x0 // cell_size)
        r0 = int(y0 // cell_size)
        c1 = int(x1 // cell_size)
        r1 = int(y1 // cell_size)
        for c in range(c0, c1 + 1):
            for r in range(r0, r1 + 1):
                grid.setdefault((c, r), []).append(idx)
    return grid


def _grid_candidates(
    grid: dict[tuple[int, int], list[int]],
    bbox: tuple[float, float, float, float],
    cell_size: float = 50.0,
) -> set[int]:
    """Return indices of tokens whose cells overlap *bbox*."""
    x0, y0, x1, y1 = bbox
    c0 = int(x0 // cell_size)
    r0 = int(y0 // cell_size)
    c1 = int(x1 // cell_size)
    r1 = int(y1 // cell_size)
    result: set[int] = set()
    for c in range(c0, c1 + 1):
        for r in range(r0, r1 + 1):
            result.update(grid.get((c, r), ()))
    return result


def _find_best_match(
    ocr_box: GlyphBox,
    pdf_tokens: List[GlyphBox],
    cfg: GroupingConfig,
    candidate_indices: set[int] | None = None,
) -> Tuple[Optional[GlyphBox], str]:
    """Find the best-matching PDF token for an OCR token.

    When *candidate_indices* is provided (from a spatial grid), only those
    pdf_tokens entries are compared — reducing cost from O(m) to O(k)
    where k is the number of nearby tokens.

    Returns (matched_pdf_box_or_None, match_type).
    """
    best_iou = 0.0
    best_box: Optional[GlyphBox] = None

    ocr_cx, ocr_cy = center(ocr_box)

    indices = (
        candidate_indices if candidate_indices is not None else range(len(pdf_tokens))
    )
    for idx in indices:
        pdf_box = pdf_tokens[idx]
        score = iou(ocr_box, pdf_box)
        if score > best_iou:
            best_iou = score
            best_box = pdf_box

    if best_iou >= cfg.ocr_reconcile_iou_threshold:
        return best_box, "iou"

    # Fallback: center-to-center proximity
    tol_x = cfg.ocr_reconcile_center_tol_x
    tol_y = cfg.ocr_reconcile_center_tol_y
    closest_dist = float("inf")
    closest_box: Optional[GlyphBox] = None

    for idx in indices:
        pdf_box = pdf_tokens[idx]
        pcx, pcy = center(pdf_box)
        dx = abs(ocr_cx - pcx)
        dy = abs(ocr_cy - pcy)
        if dx <= tol_x and dy <= tol_y:
            dist = dx + dy
            if dist < closest_dist:
                closest_dist = dist
                closest_box = pdf_box

    if closest_box is not None:
        return closest_box, "center"

    return None, "unmatched"


def _build_match_index(
    ocr_tokens: List[GlyphBox],
    ocr_confidences: List[float],
    pdf_tokens: List[GlyphBox],
    cfg: GroupingConfig,
) -> List[MatchRecord]:
    """Spatially align every OCR token against the PDF token set.

    Uses a grid-based spatial index so that each OCR token only compares
    against nearby PDF tokens instead of the full list.
    """
    # Build spatial grid on PDF tokens for fast candidate lookup
    grid = _build_spatial_grid(pdf_tokens)

    matches: List[MatchRecord] = []
    for ocr_box, conf in zip(ocr_tokens, ocr_confidences):
        candidates = _grid_candidates(grid, ocr_box.bbox())
        pdf_box, match_type = _find_best_match(
            ocr_box, pdf_tokens, cfg, candidate_indices=candidates
        )
        matches.append(
            MatchRecord(
                ocr_box=ocr_box,
                pdf_box=pdf_box,
                match_type=match_type,
                ocr_confidence=conf,
            )
        )
    return matches
