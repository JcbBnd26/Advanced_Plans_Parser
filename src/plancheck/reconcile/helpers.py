"""Shared helper utilities for OCR reconciliation.

Public API
----------
center                     – (x, y) centre of a GlyphBox
has_allowed_symbol         – True if text contains an allowed symbol
has_numeric_symbol_context – True if symbol appears in a numeric context
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median
from typing import List, Optional, Tuple

from ..config import GroupingConfig
from ..models import GlyphBox

# ── Default thresholds (match GroupingConfig defaults) ─────────────────
_DEFAULT_IOU_THRESHOLD = 0.15
_DEFAULT_COVERAGE_THRESHOLD = 0.30
_DEFAULT_DIGIT_BAND_TOL_MULT = 0.5
_DEFAULT_DIGIT_OVERSHOOT = -2.0
_DEFAULT_LINE_NEIGHBOUR_TOL_MULT = 0.6
_DEFAULT_LINE_NEIGHBOUR_MIN_TOL = 3.0
_DEFAULT_DIGIT_RATIO = 0.5
_DEFAULT_ACCEPT_PROXIMITY = 4.0

# ── Data structures shared across modules ─────────────────────────────


@dataclass
class MatchRecord:
    """Result of spatially aligning one OCR token against the PDF tokens."""

    ocr_box: GlyphBox
    pdf_box: Optional[GlyphBox]
    match_type: str  # "iou", "center", "unmatched"
    ocr_confidence: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "ocr_box": self.ocr_box.to_dict(),
            "pdf_box": self.pdf_box.to_dict() if self.pdf_box else None,
            "match_type": self.match_type,
            "ocr_confidence": round(self.ocr_confidence, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MatchRecord":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        return cls(
            ocr_box=GlyphBox.from_dict(d["ocr_box"]),
            pdf_box=GlyphBox.from_dict(d["pdf_box"]) if d.get("pdf_box") else None,
            match_type=d.get("match_type", "unmatched"),
            ocr_confidence=d.get("ocr_confidence", 0.0),
        )


@dataclass
class SymbolCandidate:
    """A single symbol placement candidate from composite OCR matching."""

    symbol: str
    slot_type: str  # "between_digits" or "after_digit"
    x0: float
    y0: float
    x1: float
    y1: float
    ocr_source: GlyphBox
    anchor_left: Optional[GlyphBox] = None
    anchor_right: Optional[GlyphBox] = None
    status: str = "pending"
    reject_reason: str = ""


# ── Public helpers ─────────────────────────────────────────────────────


def center(b: GlyphBox) -> Tuple[float, float]:
    """Return the (x, y) centre of a glyph box."""
    return ((b.x0 + b.x1) / 2.0, (b.y0 + b.y1) / 2.0)


def has_allowed_symbol(text: str, allowed: str) -> bool:
    """Return True if *text* contains at least one allowed symbol character."""
    return any(ch in allowed for ch in text)


# Pre-compiled regexes for numeric-context gating (Phase 0)
_RE_SLASH_NUMERIC = re.compile(r"\d\s*/\s*\d")  # digit / digit
_RE_AFTER_DIGIT = re.compile(r"\d\s*[%°±]")  # digit then %/°/±
_RE_DIAMETER = re.compile(r"[Ø⌀]\s*\d|\d\s*[Ø⌀]")  # Ø near digit
_RE_MULTIPLY = re.compile(r"\d\s*[×x]\s*\d", re.IGNORECASE)  # digit × digit
_RE_UNIT_MARK = re.compile(r"\d\s*['\"]")  # digit then '/\"
_RE_HASH_DIGIT = re.compile(r"#\s*\d")  # # before digit (rebar)
_RE_AT_DIGIT = re.compile(r"@\s*\d|\d\s*@")  # @ near digit


def has_numeric_symbol_context(text: str, allowed: str) -> bool:
    """Return True if *text* contains an allowed symbol in a numeric context.

    Rules
    -----
    * ``/`` is valid only in digit/digit context (``1/2``, ``09/15``).
    * ``%``, ``°``, ``±`` are valid only when preceded by a digit.
    * ``Ø`` (diameter) requires a digit nearby.
    * ``×`` (multiply) requires digit/digit context.
    * ``'`` / ``"`` (foot/inch) require preceding digit.
    * ``#`` requires a following digit (rebar).
    * ``@`` requires a digit nearby (spacing notation).

    This prevents headings like ``SURFACING/MILLINGS`` or ``A/C`` from
    being treated as symbol candidates.
    """
    if "/" in allowed and "/" in text:
        if _RE_SLASH_NUMERIC.search(text):
            return True
    for ch in ("%", "°", "±"):
        if ch in allowed and ch in text:
            if _RE_AFTER_DIGIT.search(text):
                return True
    for ch in ("Ø", "⌀"):
        if ch in allowed and ch in text:
            if _RE_DIAMETER.search(text):
                return True
    if "×" in allowed and "×" in text:
        if _RE_MULTIPLY.search(text):
            return True
    for ch in ("'", '"'):
        if ch in allowed and ch in text:
            if _RE_UNIT_MARK.search(text):
                return True
    if "#" in allowed and "#" in text:
        if _RE_HASH_DIGIT.search(text):
            return True
    if "@" in allowed and "@" in text:
        if _RE_AT_DIGIT.search(text):
            return True
    return False


# ── Private helpers (shared across reconcile sub-modules) ──────────────


def _extra_symbols(ocr_text: str, pdf_text: str, allowed: str) -> str:
    """Return characters present in *ocr_text* but not *pdf_text*, filtered to allowed set."""
    return "".join(ch for ch in ocr_text if ch in allowed and ch not in pdf_text)


def _has_digit_neighbour_left(
    candidate: GlyphBox,
    pdf_tokens: List[GlyphBox],
    proximity_pts: float,
    cfg: GroupingConfig | None = None,
) -> bool:
    """Return True if a digit-bearing PDF token is within *proximity_pts* to the left."""
    _tol_mult = (
        cfg.ocr_reconcile_digit_band_tol_mult if cfg else _DEFAULT_DIGIT_BAND_TOL_MULT
    )
    _overshoot = cfg.ocr_reconcile_digit_overshoot if cfg else _DEFAULT_DIGIT_OVERSHOOT
    cy = (candidate.y0 + candidate.y1) / 2.0
    band_tol = (
        max(2.0, candidate.height() * _tol_mult) if candidate.height() > 0 else 2.0
    )

    for t in pdf_tokens:
        if t.origin != "text":
            continue
        if not any(ch.isdigit() for ch in t.text):
            continue
        ty = (t.y0 + t.y1) / 2.0
        if abs(ty - cy) > band_tol:
            continue
        # PDF token must be to the left (its right edge near candidate's left edge)
        dx = candidate.x0 - t.x1
        if _overshoot <= dx <= proximity_pts:
            return True
    return False


def _estimate_char_width(
    pdf_tokens: List[GlyphBox], cfg: GroupingConfig | None = None
) -> float:
    """Rough median character width from PDF tokens."""
    _fallback = cfg.ocr_reconcile_char_width_fallback if cfg else 5.0
    widths = []
    for t in pdf_tokens:
        if t.text and t.width() > 0:
            widths.append(t.width() / len(t.text))
    return median(widths) if widths else _fallback


def _find_line_neighbours(
    ocr_box: GlyphBox,
    pdf_tokens: List[GlyphBox],
    anchor_margin: float,
    cfg: GroupingConfig | None = None,
) -> List[GlyphBox]:
    """Find PDF tokens on the same text line within a horizontal window.

    Returns origin="text" tokens sorted by x0.
    """
    _tol_mult = (
        cfg.ocr_reconcile_line_neighbour_tol_mult
        if cfg
        else _DEFAULT_LINE_NEIGHBOUR_TOL_MULT
    )
    _min_tol = (
        cfg.ocr_reconcile_line_neighbour_min_tol
        if cfg
        else _DEFAULT_LINE_NEIGHBOUR_MIN_TOL
    )
    cy = (ocr_box.y0 + ocr_box.y1) / 2.0
    band_tol = (
        max(_min_tol, ocr_box.height() * _tol_mult)
        if ocr_box.height() > 0
        else _min_tol
    )
    x_lo = ocr_box.x0 - anchor_margin
    x_hi = ocr_box.x1 + anchor_margin

    neighbours: List[GlyphBox] = []
    for t in pdf_tokens:
        if t.origin != "text":
            continue
        ty = (t.y0 + t.y1) / 2.0
        if abs(ty - cy) > band_tol:
            continue
        # Token's x-range must overlap the search window
        if t.x1 < x_lo or t.x0 > x_hi:
            continue
        neighbours.append(t)
    neighbours.sort(key=lambda b: b.x0)
    return neighbours


def _is_digit_group(text: str, cfg: GroupingConfig | None = None) -> bool:
    """Return True if *text* qualifies as a digit-group anchor.

    A token is a digit group if it starts with a digit (catches ``"09"``,
    ``"8.33"``, ``"2A"``), OR digits plus ``'.'`` comprise >= *digit_ratio* of its
    characters.  This rejects labels like ``"SECTION 2"`` while keeping
    numeric values that symbols attach to.
    """
    if not text:
        return False
    # Fast path: starts with a digit
    if text[0].isdigit():
        return True
    _ratio = cfg.ocr_reconcile_digit_ratio if cfg else _DEFAULT_DIGIT_RATIO
    digit_dot = sum(1 for ch in text if ch.isdigit() or ch == ".")
    return digit_dot >= len(text) * _ratio
