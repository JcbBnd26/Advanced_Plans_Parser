"""Gap-based symbol prediction methods.

Detects candidates based on horizontal gaps within text lines.
"""

from __future__ import annotations

from statistics import median
from typing import List

from plancheck.models import GlyphBox, VocrCandidate

from .constants import _DIAMETER_KEYWORDS, _TEMPLATE_KEYWORDS
from .helpers import (
    _RE_DIGIT_ONLY,
    _RE_FRACTION,
    _is_digit_token,
    _pad_bbox,
    _token_text_upper,
    _x_gap,
)


def _detect_intraline_gaps(
    tokens: List[GlyphBox],
    lines: List[List[GlyphBox]],
    page_num: int,
    page_w: float,
    page_h: float,
    margin: float,
    gap_mult: float,
) -> List[VocrCandidate]:
    """Signal #4: abnormally large horizontal gaps within a text line."""
    candidates: List[VocrCandidate] = []

    for line in lines:
        if len(line) < 2:
            continue
        # Compute inter-token gaps
        gaps: List[float] = []
        for i in range(len(line) - 1):
            g = _x_gap(line[i], line[i + 1])
            if g > 0:
                gaps.append(g)
        if not gaps:
            continue
        med_gap = median(gaps)
        if med_gap <= 0:
            continue
        threshold = med_gap * gap_mult

        for i in range(len(line) - 1):
            g = _x_gap(line[i], line[i + 1])
            if g > threshold and g > 1.0:
                # Build gap bbox
                gx0 = line[i].x1
                gx1 = line[i + 1].x0
                gy0 = min(line[i].y0, line[i + 1].y0)
                gy1 = max(line[i].y1, line[i + 1].y1)
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
                        trigger_methods=["intraline_gap"],
                        predicted_symbol="",
                        confidence=0.6,
                        context={
                            "gap_pts": round(g, 2),
                            "median_gap": round(med_gap, 2),
                            "ratio": round(g / med_gap, 2),
                            "left_text": line[i].text,
                            "right_text": line[i + 1].text,
                        },
                    )
                )
    return candidates


def _detect_baseline_style_gaps(
    tokens: List[GlyphBox],
    lines: List[List[GlyphBox]],
    page_num: int,
    page_w: float,
    page_h: float,
    margin: float,
    gap_mult: float,
) -> List[VocrCandidate]:
    """Signal #6: intraline gap where neighbors share font + size."""
    candidates: List[VocrCandidate] = []

    for line in lines:
        if len(line) < 2:
            continue
        gaps: List[float] = []
        for i in range(len(line) - 1):
            g = _x_gap(line[i], line[i + 1])
            if g > 0:
                gaps.append(g)
        if not gaps:
            continue
        med_gap = median(gaps)
        if med_gap <= 0:
            continue
        threshold = med_gap * gap_mult

        for i in range(len(line) - 1):
            left, right = line[i], line[i + 1]
            g = _x_gap(left, right)
            if g <= threshold or g <= 1.0:
                continue
            # Require same font and similar size
            if not left.fontname or not right.fontname:
                continue
            if left.fontname != right.fontname:
                continue
            if left.font_size > 0 and right.font_size > 0:
                ratio = min(left.font_size, right.font_size) / max(
                    left.font_size, right.font_size
                )
                if ratio < 0.85:
                    continue

            gx0 = left.x1
            gx1 = right.x0
            gy0 = min(left.y0, right.y0)
            gy1 = max(left.y1, right.y1)
            bx0, by0, bx1, by1 = _pad_bbox(gx0, gy0, gx1, gy1, margin, page_w, page_h)
            candidates.append(
                VocrCandidate(
                    page=page_num,
                    x0=bx0,
                    y0=by0,
                    x1=bx1,
                    y1=by1,
                    trigger_methods=["baseline_style_gap"],
                    predicted_symbol="",
                    confidence=0.7,
                    context={
                        "gap_pts": round(g, 2),
                        "fontname": left.fontname,
                        "font_size": left.font_size,
                        "left_text": left.text,
                        "right_text": right.text,
                    },
                )
            )
    return candidates


def _detect_template_adjacency(
    tokens: List[GlyphBox],
    lines: List[List[GlyphBox]],
    page_num: int,
    page_w: float,
    page_h: float,
    margin: float,
) -> List[VocrCandidate]:
    """Signal #7: NUMBER + gap + keyword (TYP, DIA, TOL, etc.)."""
    candidates: List[VocrCandidate] = []

    for line in lines:
        for i in range(len(line) - 1):
            left, right = line[i], line[i + 1]
            right_up = _token_text_upper(right)

            # Check: digit token followed by a template keyword
            if _is_digit_token(left) and right_up in _TEMPLATE_KEYWORDS:
                gap = _x_gap(left, right)
                if gap < 0.5:
                    continue
                expected = _TEMPLATE_KEYWORDS[right_up]
                gx0 = left.x1
                gx1 = right.x0
                gy0 = min(left.y0, right.y0)
                gy1 = max(left.y1, right.y1)
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
                        trigger_methods=["template_adjacency"],
                        predicted_symbol=expected,
                        confidence=0.85,
                        context={
                            "left_text": left.text,
                            "right_text": right.text,
                            "gap_pts": round(gap, 2),
                        },
                    )
                )

            # Check: diameter keyword followed by number
            left_up = _token_text_upper(left)
            if left_up in _DIAMETER_KEYWORDS and _is_digit_token(right):
                gap = _x_gap(left, right)
                if gap < 0.5:
                    continue
                gx0 = left.x1
                gx1 = right.x0
                gy0 = min(left.y0, right.y0)
                gy1 = max(left.y1, right.y1)
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
                        trigger_methods=["template_adjacency"],
                        predicted_symbol="Ø",
                        confidence=0.8,
                        context={
                            "left_text": left.text,
                            "right_text": right.text,
                            "gap_pts": round(gap, 2),
                        },
                    )
                )
    return candidates


def _detect_regex_digit_patterns(
    tokens: List[GlyphBox],
    lines: List[List[GlyphBox]],
    page_num: int,
    page_w: float,
    page_h: float,
    margin: float,
) -> List[VocrCandidate]:
    """Signal #8: digit-gap-digit where a separator is expected."""
    candidates: List[VocrCandidate] = []

    # Pattern: two decimal numbers with a gap → likely ± or ×
    for line in lines:
        for i in range(len(line) - 1):
            left, right = line[i], line[i + 1]
            lt = left.text.strip()
            rt = right.text.strip()

            # Both look numeric
            if not _RE_DIGIT_ONLY.match(lt) or not _RE_DIGIT_ONLY.match(rt):
                continue
            gap = _x_gap(left, right)
            if gap < 1.0:
                continue

            # Heuristic: if both have decimals → likely tolerance (±)
            # If both are integers → likely dimension (×)
            if "." in lt and "." in rt:
                predicted = "±"
            else:
                predicted = "×"

            gx0 = left.x1
            gx1 = right.x0
            gy0 = min(left.y0, right.y0)
            gy1 = max(left.y1, right.y1)
            bx0, by0, bx1, by1 = _pad_bbox(gx0, gy0, gx1, gy1, margin, page_w, page_h)
            candidates.append(
                VocrCandidate(
                    page=page_num,
                    x0=bx0,
                    y0=by0,
                    x1=bx1,
                    y1=by1,
                    trigger_methods=["regex_digit_pattern"],
                    predicted_symbol=predicted,
                    confidence=0.55,
                    context={
                        "left_text": lt,
                        "right_text": rt,
                        "gap_pts": round(gap, 2),
                    },
                )
            )

        # Pattern: digit + fraction on same line (missing " inch mark)
        for i, t in enumerate(line):
            if _RE_FRACTION.search(t.text):
                # Check if preceded by a digit token with a gap
                if i > 0 and _is_digit_token(line[i - 1]):
                    gap = _x_gap(line[i - 1], t)
                    if gap > 1.0:
                        gx0 = line[i - 1].x1
                        gx1 = t.x0
                        gy0 = min(line[i - 1].y0, t.y0)
                        gy1 = max(line[i - 1].y1, t.y1)
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
                                trigger_methods=["regex_digit_pattern"],
                                predicted_symbol='"',
                                confidence=0.5,
                                context={
                                    "left_text": line[i - 1].text,
                                    "right_text": t.text,
                                    "pattern": "digit_fraction",
                                },
                            )
                        )
    return candidates


def _detect_impossible_sequences(
    tokens: List[GlyphBox],
    lines: List[List[GlyphBox]],
    page_num: int,
    page_w: float,
    page_h: float,
    margin: float,
) -> List[VocrCandidate]:
    """Signal #9: back-to-back numeric tokens with no separator."""
    candidates: List[VocrCandidate] = []

    for line in lines:
        for i in range(len(line) - 1):
            left, right = line[i], line[i + 1]
            if not _is_digit_token(left) or not _is_digit_token(right):
                continue
            lt = left.text.strip()
            rt = right.text.strip()
            if not _RE_DIGIT_ONLY.match(lt) or not _RE_DIGIT_ONLY.match(rt):
                continue

            gap = _x_gap(left, right)
            # Very small gap — almost touching — suspicious
            if gap < 0.2:
                continue
            # Already caught by regex_digit_pattern if gap is big.
            # Here we catch the "touching but separate tokens" case.
            if gap > 20.0:
                continue

            gx0 = left.x1
            gx1 = right.x0
            gy0 = min(left.y0, right.y0)
            gy1 = max(left.y1, right.y1)
            # Ensure at least a minimum patch width
            if gx1 - gx0 < 2.0:
                mid = (gx0 + gx1) / 2
                gx0 = mid - 3.0
                gx1 = mid + 3.0
            bx0, by0, bx1, by1 = _pad_bbox(gx0, gy0, gx1, gy1, margin, page_w, page_h)
            candidates.append(
                VocrCandidate(
                    page=page_num,
                    x0=bx0,
                    y0=by0,
                    x1=bx1,
                    y1=by1,
                    trigger_methods=["impossible_sequence"],
                    predicted_symbol="",
                    confidence=0.5,
                    context={
                        "left_text": lt,
                        "right_text": rt,
                        "gap_pts": round(gap, 2),
                    },
                )
            )
    return candidates
