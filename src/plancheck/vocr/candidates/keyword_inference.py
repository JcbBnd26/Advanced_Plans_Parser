"""Keyword-to-symbol mapping methods.

Detects candidates based on keywords that suggest specific symbols nearby.
"""

from __future__ import annotations

from typing import List

from plancheck.models import GlyphBox, VocrCandidate

from .constants import _COOCCURRENCE, _KEYWORD_SYMBOL_MAP
from .helpers import _is_digit_token, _pad_bbox, _token_text_upper, _x_gap


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
