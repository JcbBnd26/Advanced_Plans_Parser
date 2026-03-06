"""Font, width, and vector rendering analysis methods.

Detects candidates based on font encoding failures, width anomalies,
and vector graphics patterns.
"""

from __future__ import annotations

from collections import Counter
from typing import List

from plancheck.models import GlyphBox, VocrCandidate

from .helpers import _is_digit_token, _pad_bbox


def _detect_font_subset_correlation(
    tokens: List[GlyphBox],
    lines: List[List[GlyphBox]],
    page_chars: List[dict],
    page_num: int,
    page_w: float,
    page_h: float,
    margin: float,
) -> List[VocrCandidate]:
    """Signal #14: fonts with high encoding-failure rate → flag all number-adjacent uses."""
    # Count failures per fontname
    fail_counts: Counter = Counter()
    total_counts: Counter = Counter()
    for ch in page_chars:
        fn = ch.get("fontname", "")
        if not fn:
            continue
        total_counts[fn] += 1
        text = ch.get("text", "")
        if (
            not text
            or text == "\ufffd"
            or text == "\x00"
            or (isinstance(text, str) and text.startswith("(cid:"))
        ):
            fail_counts[fn] += 1

    # Identify problematic fonts (>5% failure rate, min 3 failures)
    bad_fonts = set()
    for fn, fails in fail_counts.items():
        total = total_counts[fn]
        if fails >= 3 and total > 0 and fails / total > 0.05:
            bad_fonts.add(fn)

    if not bad_fonts:
        return []

    candidates: List[VocrCandidate] = []
    for line in lines:
        for i, t in enumerate(line):
            if t.fontname not in bad_fonts:
                continue
            # Only flag if near a digit
            near_digit = False
            for j in (i - 1, i + 1):
                if 0 <= j < len(line) and _is_digit_token(line[j]):
                    near_digit = True
                    break
            if not near_digit:
                continue
            bx0, by0, bx1, by1 = _pad_bbox(
                t.x0, t.y0, t.x1, t.y1, margin, page_w, page_h
            )
            candidates.append(
                VocrCandidate(
                    page=page_num,
                    x0=bx0,
                    y0=by0,
                    x1=bx1,
                    y1=by1,
                    trigger_methods=["font_subset_correlation"],
                    predicted_symbol="",
                    confidence=0.5,
                    context={
                        "fontname": t.fontname,
                        "fail_rate": round(
                            fail_counts[t.fontname] / total_counts[t.fontname], 3
                        ),
                    },
                )
            )
    return candidates


def _detect_token_width_anomaly(
    tokens: List[GlyphBox],
    page_num: int,
    page_w: float,
    page_h: float,
    margin: float,
    char_width_ratio: float,
) -> List[VocrCandidate]:
    """Signal #15: token bbox too narrow for its text length given font size."""
    candidates: List[VocrCandidate] = []
    for t in tokens:
        if not t.text.strip() or t.font_size <= 0:
            continue
        expected_width = len(t.text) * t.font_size * 0.55  # rough heuristic
        actual_width = t.x1 - t.x0
        if expected_width <= 0:
            continue
        ratio = actual_width / expected_width
        if ratio < char_width_ratio:
            bx0, by0, bx1, by1 = _pad_bbox(
                t.x0, t.y0, t.x1, t.y1, margin, page_w, page_h
            )
            candidates.append(
                VocrCandidate(
                    page=page_num,
                    x0=bx0,
                    y0=by0,
                    x1=bx1,
                    y1=by1,
                    trigger_methods=["token_width_anomaly"],
                    predicted_symbol="",
                    confidence=0.45,
                    context={
                        "text": t.text,
                        "actual_width": round(actual_width, 2),
                        "expected_width": round(expected_width, 2),
                        "ratio": round(ratio, 3),
                    },
                )
            )
    return candidates


def _detect_vector_circles(
    page_curves: List[dict],
    tokens: List[GlyphBox],
    page_num: int,
    page_w: float,
    page_h: float,
    margin: float,
    max_diameter: float,
) -> List[VocrCandidate]:
    """Signal #16: small circles near numeric tokens → ° or ⌀."""
    candidates: List[VocrCandidate] = []

    for curve in page_curves:
        pts = curve.get("pts", [])
        if len(pts) < 3:
            continue
        # Approximate: check if points form a small closed shape
        xs = [p[0] for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]
        ys = [p[1] for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]
        if not xs or not ys:
            continue
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        # Small and roughly circular
        if w > max_diameter or h > max_diameter:
            continue
        if w < 1.0 or h < 1.0:
            continue
        aspect = min(w, h) / max(w, h)
        if aspect < 0.6:
            continue

        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2

        # Check proximity to a numeric token
        near_digit = False
        for t in tokens:
            if not _is_digit_token(t):
                continue
            dx = abs(cx - (t.x0 + t.x1) / 2)
            dy = abs(cy - (t.y0 + t.y1) / 2)
            if dx < max_diameter * 3 and dy < max_diameter * 3:
                near_digit = True
                break

        if not near_digit:
            continue

        gx0 = min(xs)
        gy0 = min(ys)
        gx1 = max(xs)
        gy1 = max(ys)
        bx0, by0, bx1, by1 = _pad_bbox(gx0, gy0, gx1, gy1, margin, page_w, page_h)
        candidates.append(
            VocrCandidate(
                page=page_num,
                x0=bx0,
                y0=by0,
                x1=bx1,
                y1=by1,
                trigger_methods=["vector_circle_near_number"],
                predicted_symbol="°",
                confidence=0.55,
                context={"diameter": round(max(w, h), 2)},
            )
        )
    return candidates
