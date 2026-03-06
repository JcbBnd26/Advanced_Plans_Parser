"""Line fingerprinting and comparison methods.

Detects candidates by comparing similar lines across a page.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Tuple

from plancheck.models import GlyphBox, VocrCandidate

from .constants import _TARGET_SYMBOLS
from .helpers import _is_digit_token, _pad_bbox, _x_gap


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
        for toks, line_idx in occurrences:
            found_sym = False
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
