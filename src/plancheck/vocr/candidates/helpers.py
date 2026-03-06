"""Geometry and line grouping helpers for VOCR candidate detection."""

from __future__ import annotations

import re
from typing import List, Tuple

from plancheck.models import GlyphBox
from plancheck.models.geometry import bbox_iou as _iou_bbox  # backward-compat alias

# ── Geometry helpers ───────────────────────────────────────────────────


def _y_overlap_frac(a: GlyphBox, b: GlyphBox) -> float:
    """Fraction of the shorter box's height that overlaps vertically."""
    ov = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    shorter = min(a.y1 - a.y0, b.y1 - b.y0)
    return ov / shorter if shorter > 0 else 0.0


def _x_gap(left: GlyphBox, right: GlyphBox) -> float:
    """Horizontal gap between right edge of *left* and left edge of *right*."""
    return right.x0 - left.x1


def _pad_bbox(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    margin: float,
    page_w: float,
    page_h: float,
) -> Tuple[float, float, float, float]:
    return (
        max(0.0, x0 - margin),
        max(0.0, y0 - margin),
        min(page_w, x1 + margin),
        min(page_h, y1 + margin),
    )


# ── Regex patterns for digit detection ────────────────────────────────

_RE_DIGIT = re.compile(r"\d")
_RE_DIGIT_ONLY = re.compile(r"^\d[\d.,]*$")
_RE_FRACTION = re.compile(r"\d+/\d+")


def _is_digit_token(t: GlyphBox) -> bool:
    return bool(_RE_DIGIT.search(t.text))


def _token_text_upper(t: GlyphBox) -> str:
    return t.text.strip().upper()


# ── Line grouping helper ──────────────────────────────────────────────


def _group_by_baseline(
    tokens: List[GlyphBox], y_overlap_thresh: float = 0.5
) -> List[List[GlyphBox]]:
    """Group tokens into lines based on vertical overlap, sorted by x."""
    if not tokens:
        return []

    sorted_tokens = sorted(tokens, key=lambda t: (t.y0, t.x0))
    lines: List[List[GlyphBox]] = []

    for t in sorted_tokens:
        placed = False
        for line in lines:
            rep = line[0]
            if _y_overlap_frac(t, rep) >= y_overlap_thresh:
                line.append(t)
                placed = True
                break
        if not placed:
            lines.append([t])

    # Sort each line by x
    for line in lines:
        line.sort(key=lambda t: t.x0)

    return lines
