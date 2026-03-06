"""Unit and geometry detection methods.

Detects candidates based on dimension-like numbers and geometry relationships.
"""

from __future__ import annotations

from typing import List, Tuple

from plancheck.models import GlyphBox, VocrCandidate

from .helpers import _RE_DIGIT_ONLY, _is_digit_token, _pad_bbox, _x_gap


def _detect_semantic_no_units(
    tokens: List[GlyphBox],
    lines: List[List[GlyphBox]],
    page_num: int,
    page_w: float,
    page_h: float,
    margin: float,
) -> List[VocrCandidate]:
    """Signal #17: dimension-like numbers without any unit on the line."""
    _UNIT_TOKENS = {'"', "'", "MM", "CM", "M", "IN", "FT", "INCH", "INCHES", "FEET"}
    candidates: List[VocrCandidate] = []

    for line in lines:
        digit_tokens = [t for t in line if _RE_DIGIT_ONLY.match(t.text.strip())]
        if len(digit_tokens) < 2:
            continue

        # Check if any unit token or unit char exists on the line
        has_unit = False
        for t in line:
            txt = t.text.strip()
            if txt.upper() in _UNIT_TOKENS:
                has_unit = True
                break
            if any(c in "\"'" for c in txt) and _is_digit_token(t):
                has_unit = True
                break
        if has_unit:
            continue

        # Flag the position right after each digit token
        for dt in digit_tokens:
            gx0 = dt.x1
            gx1 = dt.x1 + 6.0  # small patch for unit mark
            gy0 = dt.y0
            gy1 = dt.y1
            bx0, by0, bx1, by1 = _pad_bbox(gx0, gy0, gx1, gy1, margin, page_w, page_h)
            candidates.append(
                VocrCandidate(
                    page=page_num,
                    x0=bx0,
                    y0=by0,
                    x1=bx1,
                    y1=by1,
                    trigger_methods=["semantic_no_units"],
                    predicted_symbol='"',
                    confidence=0.4,
                    context={
                        "digit_text": dt.text,
                        "line_length": len(line),
                    },
                )
            )
    return candidates


def _detect_dimension_geometry(
    page_lines: List[dict],
    tokens: List[GlyphBox],
    lines: List[List[GlyphBox]],
    page_num: int,
    page_w: float,
    page_h: float,
    margin: float,
) -> List[VocrCandidate]:
    """Signal #18: gaps overlapping with dimension/leader lines."""
    if not page_lines:
        return []

    # Build set of token-gap rects first
    gap_rects: List[Tuple[float, float, float, float, GlyphBox, GlyphBox]] = []
    for line in lines:
        for i in range(len(line) - 1):
            left, right = line[i], line[i + 1]
            gap = _x_gap(left, right)
            if gap > 2.0:
                gx0 = left.x1
                gx1 = right.x0
                gy0 = min(left.y0, right.y0)
                gy1 = max(left.y1, right.y1)
                gap_rects.append((gx0, gy0, gx1, gy1, left, right))

    if not gap_rects:
        return []

    candidates: List[VocrCandidate] = []

    for pl in page_lines:
        lx0 = pl.get("x0", 0.0)
        ly0 = pl.get("top", pl.get("y0", 0.0))
        lx1 = pl.get("x1", 0.0)
        ly1 = pl.get("bottom", pl.get("y1", 0.0))

        for gx0, gy0, gx1, gy1, left, right in gap_rects:
            # Check if the dimension line is near this gap
            # Line overlaps or is within ~10 pts of the gap
            near_x = lx0 < gx1 + 10 and lx1 > gx0 - 10
            near_y = ly0 < gy1 + 10 and ly1 > gy0 - 10
            if near_x and near_y:
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
                        trigger_methods=["dimension_geometry_proximity"],
                        predicted_symbol="",
                        confidence=0.5,
                        context={
                            "left_text": left.text,
                            "right_text": right.text,
                        },
                    )
                )
    return candidates
