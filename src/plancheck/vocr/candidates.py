"""VOCR candidate detection — identify small page regions likely missing a symbol.

This module analyses TOCR tokens and PDF metadata (chars, lines, curves)
to produce a list of :class:`~plancheck.models.VocrCandidate` objects.
Each candidate carries one or more ``trigger_methods`` so that per-method
hit-rate statistics can be computed after targeted VOCR confirms or
rejects the candidate.

Public API
----------
detect_vocr_candidates  — run all 18 detection methods; returns merged list
compute_candidate_stats — aggregate hit/miss stats per trigger method
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from ..config import GroupingConfig
from ..models import GlyphBox, VocrCandidate

log = logging.getLogger(__name__)

# ── Symbol expectations by keyword ─────────────────────────────────────

_ANGLE_KEYWORDS = {"ANGLE", "BEND", "DEG", "SLOPE", "TYP", "TYPICAL"}
_TOLERANCE_KEYWORDS = {"ELEV", "EL", "TOL", "TOLERANCE"}
_DIAMETER_KEYWORDS = {"DIA", "DIAM", "DIAMETER"}
_CENTERLINE_KEYWORDS = {"CENTER", "CL", "C/L", "CENTERLINE"}

_KEYWORD_SYMBOL_MAP: Dict[str, str] = {}
for _kw in _ANGLE_KEYWORDS:
    _KEYWORD_SYMBOL_MAP[_kw] = "°"
for _kw in _TOLERANCE_KEYWORDS:
    _KEYWORD_SYMBOL_MAP[_kw] = "±"
for _kw in _DIAMETER_KEYWORDS:
    _KEYWORD_SYMBOL_MAP[_kw] = "Ø"

# Template adjacency: digit + gap + keyword → expected symbol
_TEMPLATE_KEYWORDS = {
    **{k: "°" for k in ("TYP", "TYP.", "TYP)")},
    **{k: "Ø" for k in ("DIA", "DIAM", "DIAM.", "DIAMETER")},
    **{k: "±" for k in ("TOL", "TOLERANCE")},
}

# Co-occurrence pairs: keyword → expected partner symbol
_COOCCURRENCE = {
    "OC": "@",
    "O.C.": "@",
    "O.C": "@",
    "BAR": "#",
    "REBAR": "#",
}

# Expanded symbol set we care about
_TARGET_SYMBOLS = set("%/°±Ø×'\"#@⌀")

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


def _iou_bbox(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter == 0:
        return 0.0
    a_area = (a[2] - a[0]) * (a[3] - a[1])
    b_area = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (a_area + b_area - inter) if (a_area + b_area - inter) > 0 else 0.0


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


# ── Individual detection methods ──────────────────────────────────────
# Each returns a list of VocrCandidate with populated trigger_methods.


def _detect_char_encoding_failures(
    page_chars: List[dict],
    page_num: int,
    page_w: float,
    page_h: float,
    margin: float,
) -> List[VocrCandidate]:
    """Signals #1 & #2: chars with empty/unmapped Unicode but valid bbox."""
    candidates: List[VocrCandidate] = []
    for ch in page_chars:
        text = ch.get("text", "")
        # Flag empty, replacement char, cid-style, or NUL
        if (
            text
            and text != "\ufffd"
            and text != "\x00"
            and not text.startswith("(cid:")
        ):
            continue
        x0 = ch.get("x0", 0.0)
        y0 = ch.get("top", ch.get("y0", 0.0))
        x1 = ch.get("x1", 0.0)
        y1 = ch.get("bottom", ch.get("y1", 0.0))
        if x1 - x0 < 0.5 or y1 - y0 < 0.5:
            continue

        bx0, by0, bx1, by1 = _pad_bbox(x0, y0, x1, y1, margin, page_w, page_h)
        method = "char_encoding_failure" if not text else "unmapped_glyph"
        candidates.append(
            VocrCandidate(
                page=page_num,
                x0=bx0,
                y0=by0,
                x1=bx1,
                y1=by1,
                trigger_methods=[method],
                predicted_symbol="",
                confidence=0.9,
                context={
                    "char_text": repr(text),
                    "fontname": ch.get("fontname", ""),
                },
            )
        )
    return candidates


def _detect_placeholder_tokens(
    tokens: List[GlyphBox],
    page_num: int,
    page_w: float,
    page_h: float,
    margin: float,
) -> List[VocrCandidate]:
    """Signal #3: TOCR tokens with placeholder/garbage text but valid bbox."""
    candidates: List[VocrCandidate] = []
    for t in tokens:
        txt = t.text.strip()
        is_placeholder = (
            not txt or "\ufffd" in txt or "\x00" in txt or txt.startswith("(cid:")
        )
        if not is_placeholder:
            continue
        if t.x1 - t.x0 < 0.5 or t.y1 - t.y0 < 0.5:
            continue
        bx0, by0, bx1, by1 = _pad_bbox(t.x0, t.y0, t.x1, t.y1, margin, page_w, page_h)
        candidates.append(
            VocrCandidate(
                page=page_num,
                x0=bx0,
                y0=by0,
                x1=bx1,
                y1=by1,
                trigger_methods=["placeholder_token"],
                predicted_symbol="",
                confidence=0.85,
                context={"token_text": repr(t.text)},
            )
        )
    return candidates


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


def _detect_vocab_triggers(
    tokens: List[GlyphBox],
    lines: List[List[GlyphBox]],
    page_num: int,
    page_w: float,
    page_h: float,
    margin: float,
) -> List[VocrCandidate]:
    """Signal #10: keywords suggesting a specific symbol nearby."""
    candidates: List[VocrCandidate] = []

    for line in lines:
        for i, t in enumerate(line):
            up = _token_text_upper(t)
            if up not in _KEYWORD_SYMBOL_MAP:
                continue
            expected = _KEYWORD_SYMBOL_MAP[up]
            # Find nearest numeric neighbor
            for j in (i - 1, i + 1):
                if 0 <= j < len(line) and _is_digit_token(line[j]):
                    neighbor = line[j]
                    # Check that there's a gap (no symbol already present)
                    if j < i:
                        gap = _x_gap(neighbor, t)
                    else:
                        gap = _x_gap(t, neighbor)
                    if gap < 0.5:
                        continue
                    # Build bbox in the gap
                    if j < i:
                        gx0, gx1 = neighbor.x1, t.x0
                    else:
                        gx0, gx1 = t.x1, neighbor.x0
                    gy0 = min(t.y0, neighbor.y0)
                    gy1 = max(t.y1, neighbor.y1)
                    if gx1 - gx0 < 1.0:
                        mid = (gx0 + gx1) / 2
                        gx0 = mid - 3.0
                        gx1 = mid + 3.0
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
                            trigger_methods=["vocab_trigger"],
                            predicted_symbol=expected,
                            confidence=0.65,
                            context={
                                "keyword": up,
                                "neighbor_text": neighbor.text,
                                "gap_pts": round(abs(gap), 2),
                            },
                        )
                    )
                    break  # one candidate per keyword
    return candidates


def _detect_keyword_cooccurrence(
    tokens: List[GlyphBox],
    lines: List[List[GlyphBox]],
    page_num: int,
    page_w: float,
    page_h: float,
    margin: float,
) -> List[VocrCandidate]:
    """Signal #11: keyword without expected partner symbol (OC→@, BAR→#)."""
    candidates: List[VocrCandidate] = []

    for line in lines:
        line_text_set = {_token_text_upper(t) for t in line}
        line_char_set = set("".join(t.text for t in line))

        for i, t in enumerate(line):
            up = _token_text_upper(t)
            if up not in _COOCCURRENCE:
                continue
            expected_sym = _COOCCURRENCE[up]
            # Check if the symbol already exists on this line
            if expected_sym in line_char_set:
                continue

            # Find the best position (near a digit neighbor)
            for j in (i - 1, i + 1):
                if 0 <= j < len(line) and _is_digit_token(line[j]):
                    neighbor = line[j]
                    if j < i:
                        gx0, gx1 = neighbor.x1, t.x0
                    else:
                        gx0, gx1 = t.x1, neighbor.x0
                    gy0 = min(t.y0, neighbor.y0)
                    gy1 = max(t.y1, neighbor.y1)
                    if gx1 - gx0 < 1.0:
                        mid = (gx0 + gx1) / 2
                        gx0 = mid - 3.0
                        gx1 = mid + 3.0
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
                            trigger_methods=["keyword_cooccurrence"],
                            predicted_symbol=expected_sym,
                            confidence=0.6,
                            context={
                                "keyword": up,
                                "expected_symbol": expected_sym,
                            },
                        )
                    )
                    break
    return candidates


def _detect_cross_ref_phrases(
    tokens: List[GlyphBox],
    lines: List[List[GlyphBox]],
    page_num: int,
    page_w: float,
    page_h: float,
    margin: float,
) -> List[VocrCandidate]:
    """Signal #12: same dimension phrase appears elsewhere with a symbol."""
    candidates: List[VocrCandidate] = []

    # Build a map of "canonical numeric phrases" → occurrences
    # A numeric phrase = sequence of digit tokens on a line
    phrase_map: Dict[str, List[Tuple[List[GlyphBox], int]]] = defaultdict(list)
    for line_idx, line in enumerate(lines):
        numeric_run: List[GlyphBox] = []
        for t in line:
            if _is_digit_token(t):
                numeric_run.append(t)
            else:
                if len(numeric_run) >= 2:
                    key = " ".join(tk.text.strip() for tk in numeric_run)
                    phrase_map[key].append((list(numeric_run), line_idx))
                numeric_run = []
        if len(numeric_run) >= 2:
            key = " ".join(tk.text.strip() for tk in numeric_run)
            phrase_map[key].append((list(numeric_run), line_idx))

    # For each phrase group, check if some have symbols between digits
    # and some don't. The ones without are candidates.
    for key, occurrences in phrase_map.items():
        if len(occurrences) < 2:
            continue
        # Check each pair of adjacent tokens for symbol presence
        has_symbol: List[bool] = []
        for toks, _ in occurrences:
            found_sym = False
            line_idx = _
            the_line = lines[line_idx]
            for t in the_line:
                if any(c in _TARGET_SYMBOLS for c in t.text):
                    found_sym = True
                    break
            has_symbol.append(found_sym)

        if any(has_symbol) and not all(has_symbol):
            # Flag occurrences missing the symbol
            for idx, (toks, line_idx) in enumerate(occurrences):
                if has_symbol[idx]:
                    continue
                # Candidate: the gap between the first two digit tokens
                if len(toks) >= 2:
                    gx0 = toks[0].x1
                    gx1 = toks[1].x0
                    gy0 = min(toks[0].y0, toks[1].y0)
                    gy1 = max(toks[0].y1, toks[1].y1)
                    if gx1 - gx0 < 1.0:
                        mid = (gx0 + gx1) / 2
                        gx0 = mid - 3.0
                        gx1 = mid + 3.0
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
                            trigger_methods=["cross_ref_phrase"],
                            predicted_symbol="",
                            confidence=0.75,
                            context={
                                "phrase": key,
                                "total_occurrences": len(occurrences),
                                "with_symbol": sum(has_symbol),
                            },
                        )
                    )
    return candidates


def _detect_near_duplicate_lines(
    tokens: List[GlyphBox],
    lines: List[List[GlyphBox]],
    page_num: int,
    page_w: float,
    page_h: float,
    margin: float,
) -> List[VocrCandidate]:
    """Signal #13: repeated patterns where one instance lacks a symbol."""
    candidates: List[VocrCandidate] = []

    # Fingerprint each line by its stripped text sequence
    fingerprints: Dict[str, List[int]] = defaultdict(list)
    for idx, line in enumerate(lines):
        # Normalize: replace digit runs with 'N', strip spaces
        normalized = []
        for t in line:
            txt = t.text.strip()
            norm = re.sub(r"\d+\.?\d*", "N", txt)
            normalized.append(norm)
        fp = "|".join(normalized)
        fingerprints[fp].append(idx)

    # For duplicates, check if some lines have symbols and some don't
    for fp, line_indices in fingerprints.items():
        if len(line_indices) < 2:
            continue
        has_sym: List[bool] = []
        for li in line_indices:
            line = lines[li]
            has = any(any(c in _TARGET_SYMBOLS for c in t.text) for t in line)
            has_sym.append(has)

        if any(has_sym) and not all(has_sym):
            for i, li in enumerate(line_indices):
                if has_sym[i]:
                    continue
                line = lines[li]
                # Find gaps between digit tokens
                for j in range(len(line) - 1):
                    if _is_digit_token(line[j]):
                        gap = _x_gap(line[j], line[j + 1])
                        if gap > 1.0:
                            gx0 = line[j].x1
                            gx1 = line[j + 1].x0
                            gy0 = min(line[j].y0, line[j + 1].y0)
                            gy1 = max(line[j].y1, line[j + 1].y1)
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
                                    trigger_methods=["near_duplicate_line"],
                                    predicted_symbol="",
                                    confidence=0.7,
                                    context={
                                        "fingerprint": fp,
                                        "duplicate_count": len(line_indices),
                                    },
                                )
                            )
                            break  # one per line
    return candidates


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


# ── Merging / dedup ───────────────────────────────────────────────────


def _merge_overlapping_candidates(
    candidates: List[VocrCandidate],
    iou_threshold: float = 0.5,
) -> List[VocrCandidate]:
    """Merge candidates whose bboxes overlap above *iou_threshold*.

    When two candidates overlap, the one with higher confidence survives
    and inherits all ``trigger_methods`` from the absorbed candidate.
    """
    if len(candidates) <= 1:
        return candidates

    # Sort by confidence descending so the best candidate absorbs others
    cands = sorted(candidates, key=lambda c: c.confidence, reverse=True)
    merged: List[VocrCandidate] = []
    absorbed = [False] * len(cands)

    for i in range(len(cands)):
        if absorbed[i]:
            continue
        keeper = cands[i]
        keeper_bbox = keeper.bbox()
        for j in range(i + 1, len(cands)):
            if absorbed[j]:
                continue
            other_bbox = cands[j].bbox()
            if _iou_bbox(keeper_bbox, other_bbox) >= iou_threshold:
                # Merge trigger methods
                for m in cands[j].trigger_methods:
                    if m not in keeper.trigger_methods:
                        keeper.trigger_methods.append(m)
                # Keep the higher confidence
                keeper.confidence = max(keeper.confidence, cands[j].confidence)
                # Prefer a predicted symbol if one has it and the other doesn't
                if not keeper.predicted_symbol and cands[j].predicted_symbol:
                    keeper.predicted_symbol = cands[j].predicted_symbol
                absorbed[j] = True
        merged.append(keeper)

    return merged


# ── Public API ─────────────────────────────────────────────────────────


def detect_vocr_candidates(
    tokens: List[GlyphBox],
    page_chars: List[dict],
    page_lines: List[dict],
    page_curves: List[dict],
    page_rects: List[dict],
    page_width: float,
    page_height: float,
    page_num: int,
    cfg: GroupingConfig,
    method_stats: Optional[Dict[str, Any]] = None,
    producer_stats: Optional[Dict[str, Any]] = None,
    producer_id: str = "",
) -> List[VocrCandidate]:
    """Run all detection methods and return a merged, capped candidate list.

    Parameters
    ----------
    tokens : list[GlyphBox]
        TOCR-extracted tokens (origin="text").
    page_chars : list[dict]
        Character-level dicts from ``page.chars`` (pdfplumber).
    page_lines, page_curves, page_rects : list[dict]
        Vector graphic elements from pdfplumber.
    page_width, page_height : float
        Page dimensions in PDF points.
    page_num : int
        Zero-based page index.
    cfg : GroupingConfig
        Pipeline configuration.
    method_stats : dict | None
        Persistent per-method hit/miss stats loaded via
        :func:`~plancheck.vocr.method_stats.load_method_stats`.
        When provided, each candidate's confidence is overridden with
        data-driven adaptive confidence.
    producer_stats : dict | None
        Per-producer method stats (Level 3).  When provided together
        with a non-empty *producer_id*, producer-specific hit rates
        override the global adaptive confidence.
    producer_id : str
        PDF ``/Producer`` metadata string.

    Returns
    -------
    list[VocrCandidate]
        Merged candidates sorted by confidence descending, capped at
        ``cfg.vocr_cand_max_candidates``.
    """
    m = cfg.vocr_cand_patch_margin
    pw, ph = page_width, page_height
    pn = page_num

    # Cache baseline-grouped lines once for all detection methods (efficiency fix)
    baseline_lines = _group_by_baseline(tokens)

    all_candidates: List[VocrCandidate] = []

    # --- Tier 1: exact-location signals ---
    all_candidates.extend(_detect_char_encoding_failures(page_chars, pn, pw, ph, m))
    all_candidates.extend(_detect_placeholder_tokens(tokens, pn, pw, ph, m))

    # --- Tier 2: spatial gap signals ---
    all_candidates.extend(
        _detect_intraline_gaps(
            tokens, baseline_lines, pn, pw, ph, m, cfg.vocr_cand_gap_multiplier
        )
    )
    all_candidates.extend(
        _detect_dense_cluster_holes(
            tokens, pn, pw, ph, m, cfg.vocr_cand_density_grid_size
        )
    )
    all_candidates.extend(
        _detect_baseline_style_gaps(
            tokens, baseline_lines, pn, pw, ph, m, cfg.vocr_cand_gap_multiplier
        )
    )

    # --- Tier 3: template / token-context signals ---
    all_candidates.extend(
        _detect_template_adjacency(tokens, baseline_lines, pn, pw, ph, m)
    )
    all_candidates.extend(
        _detect_regex_digit_patterns(tokens, baseline_lines, pn, pw, ph, m)
    )
    all_candidates.extend(
        _detect_impossible_sequences(tokens, baseline_lines, pn, pw, ph, m)
    )
    all_candidates.extend(_detect_vocab_triggers(tokens, baseline_lines, pn, pw, ph, m))
    all_candidates.extend(
        _detect_keyword_cooccurrence(tokens, baseline_lines, pn, pw, ph, m)
    )

    # --- Tier 4: cross-reference / consensus ---
    all_candidates.extend(
        _detect_cross_ref_phrases(tokens, baseline_lines, pn, pw, ph, m)
    )
    all_candidates.extend(
        _detect_near_duplicate_lines(tokens, baseline_lines, pn, pw, ph, m)
    )

    # --- Tier 5: structural / statistical ---
    all_candidates.extend(
        _detect_font_subset_correlation(
            tokens, baseline_lines, page_chars, pn, pw, ph, m
        )
    )
    all_candidates.extend(
        _detect_token_width_anomaly(
            tokens, pn, pw, ph, m, cfg.vocr_cand_char_width_ratio
        )
    )
    all_candidates.extend(
        _detect_vector_circles(
            page_curves,
            tokens,
            pn,
            pw,
            ph,
            m,
            cfg.vocr_cand_vector_circle_max_diameter,
        )
    )
    all_candidates.extend(
        _detect_semantic_no_units(tokens, baseline_lines, pn, pw, ph, m)
    )
    all_candidates.extend(
        _detect_dimension_geometry(page_lines, tokens, baseline_lines, pn, pw, ph, m)
    )

    # Apply adaptive confidence from accumulated stats (Level 1)
    if method_stats is not None:
        from .method_stats import get_adaptive_confidence

        for cand in all_candidates:
            if cand.trigger_methods:
                primary = cand.trigger_methods[0]
                cand.confidence = get_adaptive_confidence(
                    primary, method_stats, cand.confidence
                )

    # Apply per-producer override (Level 3) — refines Level 1 when data exists
    if producer_stats is not None and producer_id:
        from .producer_stats import get_producer_confidence

        for cand in all_candidates:
            if cand.trigger_methods:
                primary = cand.trigger_methods[0]
                cand.confidence = get_producer_confidence(
                    primary, producer_id, producer_stats, cand.confidence
                )

    # Filter by minimum confidence
    all_candidates = [
        c for c in all_candidates if c.confidence >= cfg.vocr_cand_min_confidence
    ]

    # Merge overlapping candidates
    all_candidates = _merge_overlapping_candidates(all_candidates)

    # Sort by confidence descending and cap
    all_candidates.sort(key=lambda c: c.confidence, reverse=True)
    if len(all_candidates) > cfg.vocr_cand_max_candidates:
        all_candidates = all_candidates[: cfg.vocr_cand_max_candidates]

    log.info(
        "Page %d: %d VOCR candidates from %d detection methods",
        page_num,
        len(all_candidates),
        len({m for c in all_candidates for m in c.trigger_methods}),
    )
    return all_candidates


# ── Statistics ─────────────────────────────────────────────────────────


def compute_candidate_stats(
    candidates: List[VocrCandidate],
    page_width: float = 0.0,
    page_height: float = 0.0,
) -> Dict[str, Any]:
    """Aggregate hit/miss statistics per trigger method.

    Call this **after** targeted VOCR has updated each candidate's
    ``outcome`` field (``"hit"`` or ``"miss"``).

    Returns a dict suitable for JSON serialisation and cross-run analysis.
    """
    total = len(candidates)
    hits = sum(1 for c in candidates if c.outcome == "hit")
    misses = sum(1 for c in candidates if c.outcome == "miss")
    pending = sum(1 for c in candidates if c.outcome == "pending")

    # Per-method breakdown
    by_method: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        for method in c.trigger_methods:
            entry = by_method.setdefault(method, {"flagged": 0, "hits": 0, "misses": 0})
            entry["flagged"] += 1
            if c.outcome == "hit":
                entry["hits"] += 1
            elif c.outcome == "miss":
                entry["misses"] += 1
    for entry in by_method.values():
        f = entry["flagged"]
        entry["hit_rate"] = round(entry["hits"] / f, 4) if f > 0 else 0.0

    # Area stats
    total_area = sum(c.patch_area() for c in candidates)
    page_area = page_width * page_height if page_width > 0 and page_height > 0 else 0.0

    # Predicted vs found
    predicted_vs_found: Dict[str, Dict[str, int]] = {}
    for c in candidates:
        if not c.predicted_symbol:
            continue
        entry = predicted_vs_found.setdefault(
            c.predicted_symbol,
            {"predicted": 0, "correct": 0, "wrong_symbol": 0, "miss": 0},
        )
        entry["predicted"] += 1
        if c.outcome == "hit":
            if c.found_symbol == c.predicted_symbol:
                entry["correct"] += 1
            else:
                entry["wrong_symbol"] += 1
        elif c.outcome == "miss":
            entry["miss"] += 1

    return {
        "total_candidates": total,
        "total_hits": hits,
        "total_misses": misses,
        "total_pending": pending,
        "hit_rate": round(hits / total, 4) if total > 0 else 0.0,
        "by_method": by_method,
        "area_stats": {
            "total_patch_area_pts2": round(total_area, 1),
            "mean_patch_area": round(total_area / total, 1) if total > 0 else 0.0,
            "page_coverage_pct": (
                round(100.0 * total_area / page_area, 2) if page_area > 0 else 0.0
            ),
        },
        "predicted_vs_found": predicted_vs_found,
    }
