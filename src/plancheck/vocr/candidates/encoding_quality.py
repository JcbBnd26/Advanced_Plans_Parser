"""OCR/rendering corruption detection methods.

Detects characters and tokens with encoding failures or placeholder text.
"""

from __future__ import annotations

from typing import List

from plancheck.models import GlyphBox, VocrCandidate

from .helpers import _pad_bbox


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
