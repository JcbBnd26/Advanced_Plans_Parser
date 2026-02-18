"""Dual-source OCR reconciliation: spatial matching and symbol injection.

Spatially aligns OCR tokens against existing PDF-text tokens, and injects
only the missing special-character tokens (%, /, °, ±) that the PDF text
layer is missing.

Public API
----------
reconcile_ocr          – run the 4-stage reconciliation pipeline
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from statistics import median
from typing import List, Optional, Tuple

from ..config import GroupingConfig
from ..models import GlyphBox
from ..vocr.extract import _extract_ocr_tokens, _iou

log = logging.getLogger("plancheck.ocr_reconcile")

# ── Data structures ────────────────────────────────────────────────────


@dataclass
class MatchRecord:
    """Result of spatially aligning one OCR token against the PDF tokens."""

    ocr_box: GlyphBox
    pdf_box: Optional[GlyphBox]
    match_type: str  # "iou", "center", "unmatched"
    ocr_confidence: float = 0.0


@dataclass
class ReconcileResult:
    """Aggregated output of the reconciliation pipeline."""

    added_tokens: List[GlyphBox] = field(default_factory=list)
    all_ocr_tokens: List[GlyphBox] = field(default_factory=list)
    matches: List[MatchRecord] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


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


# ── Geometry helpers ───────────────────────────────────────────────────
# _iou is imported from ..vocr.extract (canonical copy lives there)


def _overlap_ratio(a: GlyphBox, b: GlyphBox) -> float:
    """Fraction of *a*'s area covered by *b*."""
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    a_area = a.area()
    return inter / a_area if a_area > 0 else 0.0


def _center(b: GlyphBox) -> Tuple[float, float]:
    return ((b.x0 + b.x1) / 2.0, (b.y0 + b.y1) / 2.0)


def _overlaps_existing(
    candidate: GlyphBox,
    tokens: List[GlyphBox],
    iou_thresh: float = 0.15,
    cov_thresh: float = 0.30,
) -> bool:
    """Return True if *candidate* overlaps any existing token too much."""
    for t in tokens:
        if _iou(candidate, t) > iou_thresh:
            return True
        if _overlap_ratio(candidate, t) > cov_thresh:
            return True
    return False


# ── Stage 2: Spatial alignment ─────────────────────────────────────────


def _find_best_match(
    ocr_box: GlyphBox,
    pdf_tokens: List[GlyphBox],
    cfg: GroupingConfig,
) -> Tuple[Optional[GlyphBox], str]:
    """Find the best-matching PDF token for an OCR token.

    Returns (matched_pdf_box_or_None, match_type).
    """
    best_iou = 0.0
    best_box: Optional[GlyphBox] = None

    ocr_cx, ocr_cy = _center(ocr_box)

    for pdf_box in pdf_tokens:
        iou = _iou(ocr_box, pdf_box)
        if iou > best_iou:
            best_iou = iou
            best_box = pdf_box

    if best_iou >= cfg.ocr_reconcile_iou_threshold:
        return best_box, "iou"

    # Fallback: center-to-center proximity
    tol_x = cfg.ocr_reconcile_center_tol_x
    tol_y = cfg.ocr_reconcile_center_tol_y
    closest_dist = float("inf")
    closest_box: Optional[GlyphBox] = None

    for pdf_box in pdf_tokens:
        pcx, pcy = _center(pdf_box)
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
    """Spatially align every OCR token against the PDF token set."""
    matches: List[MatchRecord] = []
    for ocr_box, conf in zip(ocr_tokens, ocr_confidences):
        pdf_box, match_type = _find_best_match(ocr_box, pdf_tokens, cfg)
        matches.append(
            MatchRecord(
                ocr_box=ocr_box,
                pdf_box=pdf_box,
                match_type=match_type,
                ocr_confidence=conf,
            )
        )
    return matches


# ── Stage 3: Symbol-only filtering & injection ────────────────────────


def _has_allowed_symbol(text: str, allowed: str) -> bool:
    """Return True if *text* contains at least one allowed symbol character."""
    return any(ch in allowed for ch in text)


# Pre-compiled regexes for numeric-context gating (Phase 0)
_RE_SLASH_NUMERIC = re.compile(r"\d\s*/\s*\d")  # digit / digit
_RE_AFTER_DIGIT = re.compile(r"\d\s*[%°±]")  # digit then %/°/±


def _has_numeric_symbol_context(text: str, allowed: str) -> bool:
    """Return True if *text* contains an allowed symbol in a numeric context.

    Rules
    -----
    * ``/`` is valid only in digit/digit context (``1/2``, ``09/15``).
    * ``%``, ``°``, ``±`` are valid only when preceded by a digit.

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
    return False


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
    _tol_mult = cfg.ocr_reconcile_digit_band_tol_mult if cfg else 0.5
    _overshoot = cfg.ocr_reconcile_digit_overshoot if cfg else -2.0
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


# ── Stage 3b: Composite (multi-anchor) symbol injection ───────────────


def _find_line_neighbours(
    ocr_box: GlyphBox,
    pdf_tokens: List[GlyphBox],
    anchor_margin: float,
    cfg: GroupingConfig | None = None,
) -> List[GlyphBox]:
    """Find PDF tokens on the same text line within a horizontal window.

    Returns origin="text" tokens sorted by x0.
    """
    _tol_mult = cfg.ocr_reconcile_line_neighbour_tol_mult if cfg else 0.6
    _min_tol = cfg.ocr_reconcile_line_neighbour_min_tol if cfg else 3.0
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
    _ratio = cfg.ocr_reconcile_digit_ratio if cfg else 0.5
    digit_dot = sum(1 for ch in text if ch.isdigit() or ch == ".")
    return digit_dot >= len(text) * _ratio


def _generate_symbol_candidates(
    ocr_box: GlyphBox,
    pdf_tokens: List[GlyphBox],
    cfg: GroupingConfig,
) -> List[SymbolCandidate]:
    """Create placement candidates for symbols in a composite OCR token.

    Uses digit anchors on the same line + line-height-derived widths.
    Does NOT infer symbol position from the OCR string — uses anchor
    adjacency instead.
    """
    allowed = cfg.ocr_reconcile_allowed_symbols
    ocr_text = ocr_box.text

    # Find digit-bearing neighbours (tightened to digit-group tokens)
    neighbours = _find_line_neighbours(
        ocr_box, pdf_tokens, cfg.ocr_reconcile_anchor_margin, cfg
    )
    digit_anchors = [t for t in neighbours if _is_digit_group(t.text, cfg)]
    if not digit_anchors:
        return []

    # Line-height reference = median digit anchor height
    heights = [a.height() for a in digit_anchors if a.height() > 0]
    h = median(heights) if heights else 8.0

    # Y-band from anchors
    band_y0 = min(a.y0 for a in digit_anchors)
    band_y1 = max(a.y1 for a in digit_anchors)

    candidates: List[SymbolCandidate] = []

    # ── Slash (/) — between_digits slots (one per gap, up to slash count) ──
    n_slashes = ocr_text.count("/")
    if n_slashes > 0 and len(digit_anchors) >= 2:
        # Rank all consecutive digit-anchor gaps by width (ascending)
        gaps = []
        for i in range(len(digit_anchors) - 1):
            a_l = digit_anchors[i]
            a_r = digit_anchors[i + 1]
            gap = a_r.x0 - a_l.x1
            gaps.append((gap, a_l, a_r))
        gaps.sort(key=lambda g: g[0])  # smallest gap first
        for gap_w, a_l, a_r in gaps[:n_slashes]:
            mid_x = (a_l.x1 + a_r.x0) / 2.0
            slash_w = max(0.5, cfg.ocr_reconcile_slash_width_mult * h)
            # Clamp to available gap if positive
            if gap_w > 0:
                slash_w = min(slash_w, gap_w)
            candidates.append(
                SymbolCandidate(
                    symbol="/",
                    slot_type="between_digits",
                    x0=mid_x - slash_w / 2.0,
                    y0=band_y0,
                    x1=mid_x + slash_w / 2.0,
                    y1=band_y1,
                    ocr_source=ocr_box,
                    anchor_left=a_l,
                    anchor_right=a_r,
                )
            )

    # ── Percent, degree, plus-minus — after_digit slot ──
    after_symbols = [ch for ch in ocr_text if ch in allowed and ch != "/"]
    # Deduplicate while preserving order
    seen: set = set()
    unique_after: List[str] = []
    for ch in after_symbols:
        if ch not in seen:
            seen.add(ch)
            unique_after.append(ch)

    if unique_after:
        rightmost = digit_anchors[-1]  # already sorted by x0
        pad = cfg.ocr_reconcile_symbol_pad
        cursor_x = rightmost.x1 + pad

        # Per-character width from the OCR token for positional placement
        ocr_cw = ocr_box.width() / max(1, len(ocr_text)) if ocr_text else 0.0

        for sym in unique_after:
            # Prefer OCR-relative placement: use the symbol's position
            # inside the OCR token so the injected box aligns with the
            # printed glyph (even if it overlaps the PDF glyph).
            sym_idx = ocr_text.find(sym)
            if sym_idx >= 0 and ocr_cw > 0:
                sym_x0 = ocr_box.x0 + sym_idx * ocr_cw
                sym_w = max(ocr_cw, 0.5 * h)  # at least half line-height
            else:
                # Fallback: anchor-based cursor placement
                sym_x0 = cursor_x
                if sym == "%":
                    sym_w = cfg.ocr_reconcile_pct_width_mult * h
                else:  # ° ±
                    sym_w = cfg.ocr_reconcile_degree_width_mult * h

            candidates.append(
                SymbolCandidate(
                    symbol=sym,
                    slot_type="after_digit",
                    x0=sym_x0,
                    y0=band_y0,
                    x1=sym_x0 + sym_w,
                    y1=band_y1,
                    ocr_source=ocr_box,
                    anchor_left=None,
                    anchor_right=rightmost,
                )
            )
            cursor_x = max(cursor_x, sym_x0 + sym_w + pad)  # advance for next symbol

    return candidates


def _accept_candidates(
    candidates: List[SymbolCandidate],
    pdf_tokens: List[GlyphBox],
    page_width: float,
    cfg: GroupingConfig | None = None,
) -> List[SymbolCandidate]:
    """Run acceptance checks on symbol candidates.

    Overlap rules:
    - between_digits: ignore overlap with anchor_left/anchor_right, reject
      overlap with any other token.
    - after_digit: ignore overlap with anchor_right, reject overlap with
      the next token to the right.
    Checks both origin="text" and origin="ocr" for "already in PDF" guard.
    """
    _proximity = cfg.ocr_reconcile_accept_proximity if cfg else 4.0
    _iou_thr = cfg.ocr_reconcile_accept_iou if cfg else 0.15
    _cov_thr = cfg.ocr_reconcile_accept_coverage if cfg else 0.30

    # Build a combined token set for overlap checking
    all_tokens = list(pdf_tokens)

    for cand in candidates:
        # Bounds check
        if cand.x0 < 0 or cand.x1 > page_width:
            cand.status = "rejected"
            cand.reject_reason = "out_of_bounds"
            continue

        # Already-in-PDF check: any token with the same symbol text near
        # the candidate's x-center on the same y-band?
        cand_cx = (cand.x0 + cand.x1) / 2.0
        cand_cy = (cand.y0 + cand.y1) / 2.0
        already = False
        for t in all_tokens:
            if cand.symbol not in t.text:
                continue
            t_cx = (t.x0 + t.x1) / 2.0
            t_cy = (t.y0 + t.y1) / 2.0
            if abs(t_cx - cand_cx) < _proximity and abs(t_cy - cand_cy) < _proximity:
                already = True
                break
        if already:
            cand.status = "rejected"
            cand.reject_reason = "already_in_pdf"
            continue

        # Build exclusion set (anchor IDs to ignore during overlap check)
        exclude_ids: set = set()
        if cand.anchor_left is not None:
            exclude_ids.add(id(cand.anchor_left))
        if cand.anchor_right is not None:
            exclude_ids.add(id(cand.anchor_right))

        # Make a temporary GlyphBox for overlap checking
        cand_box = GlyphBox(
            page=cand.ocr_source.page,
            x0=cand.x0,
            y0=cand.y0,
            x1=cand.x1,
            y1=cand.y1,
            text=cand.symbol,
            origin="ocr",
        )

        # Overlap check against non-excluded tokens
        overlap_hit = False
        for t in all_tokens:
            if id(t) in exclude_ids:
                continue
            if _iou(cand_box, t) > _iou_thr or _overlap_ratio(cand_box, t) > _cov_thr:
                overlap_hit = True
                break

        if overlap_hit:
            if cand.slot_type == "between_digits":
                cand.status = "rejected"
                cand.reject_reason = "overlap_non_anchor"
            else:
                cand.status = "rejected"
                cand.reject_reason = "overlap_next_token"
            continue

        cand.status = "accepted"

    return candidates


def _inject_symbols(
    matches: List[MatchRecord],
    pdf_tokens: List[GlyphBox],
    cfg: GroupingConfig,
    page_width: float = 0.0,
) -> Tuple[List[GlyphBox], List[dict]]:
    """Decide which OCR findings to inject as new GlyphBox tokens.

    Case C (composite): OCR token spans multiple PDF tokens — use digit
        anchors + slot-based placement for / (between_digits) and %/°/±
        (after_digit).  Tried first.
    Case A: matched OCR token has extra symbols not in the PDF token.
    Case B: unmatched OCR token contains an allowed symbol near a digit.

    Returns
    -------
    added : list[GlyphBox]
        Tokens to inject.
    debug_log : list[dict]
        Per-OCR-token trace (only for symbol-bearing tokens, capped at 200).
    """
    allowed = cfg.ocr_reconcile_allowed_symbols
    added: List[GlyphBox] = []
    debug_log: List[dict] = []
    char_w = _estimate_char_width(pdf_tokens, cfg)
    n_filtered_non_numeric = 0
    _MAX_DEBUG = cfg.ocr_reconcile_max_debug

    _RE_COMPOSITE = re.compile(r"\d+\s*[%/°±]\s*\d+")

    for m in matches:
        ocr_text = m.ocr_box.text

        # Pre-filter: must contain at least one allowed symbol
        if not _has_allowed_symbol(ocr_text, allowed):
            continue

        # Phase 0: numeric-context gate (skip headings / random slashes)
        # Exception: Case B (unmatched, symbol-only) uses its own digit-
        # neighbour gate, so we let those through.
        is_symbol_only = all(ch in allowed or ch.isspace() for ch in ocr_text)
        if not _has_numeric_symbol_context(ocr_text, allowed):
            if not (m.match_type == "unmatched" and is_symbol_only):
                if len(debug_log) < _MAX_DEBUG:
                    debug_log.append(
                        {
                            "ocr_text": ocr_text,
                            "ocr_bbox": [
                                m.ocr_box.x0,
                                m.ocr_box.y0,
                                m.ocr_box.x1,
                                m.ocr_box.y1,
                            ],
                            "match_type": m.match_type,
                            "anchors": [],
                            "candidates": [],
                            "path": "early_reject",
                            "reason": "non_numeric_symbol_context",
                        }
                    )
                n_filtered_non_numeric += 1
                continue

        entry: dict = {
            "ocr_text": ocr_text,
            "ocr_bbox": [m.ocr_box.x0, m.ocr_box.y0, m.ocr_box.x1, m.ocr_box.y1],
            "match_type": m.match_type,
            "anchors": [],
            "candidates": [],
            "path": "",
        }

        # ── Case C: composite match (try first) ──
        candidates = _generate_symbol_candidates(m.ocr_box, pdf_tokens, cfg)

        if candidates:
            _accept_candidates(candidates, pdf_tokens, page_width, cfg)
            entry["path"] = "case_c"
            # Record anchors (from first candidate that has them)
            anchor_ids_seen: set = set()
            for c in candidates:
                for a in (c.anchor_left, c.anchor_right):
                    if a is not None and id(a) not in anchor_ids_seen:
                        anchor_ids_seen.add(id(a))
                        entry["anchors"].append(
                            {"text": a.text, "bbox": [a.x0, a.y0, a.x1, a.y1]}
                        )
                entry["candidates"].append(
                    {
                        "symbol": c.symbol,
                        "slot": c.slot_type,
                        "bbox": [c.x0, c.y0, c.x1, c.y1],
                        "status": c.status,
                        "reason": c.reject_reason,
                    }
                )
                if c.status == "accepted":
                    added.append(
                        GlyphBox(
                            page=m.ocr_box.page,
                            x0=c.x0,
                            y0=c.y0,
                            x1=c.x1,
                            y1=c.y1,
                            text=c.symbol,
                            origin="ocr",
                        )
                    )
            if len(debug_log) < _MAX_DEBUG:
                debug_log.append(entry)
            continue  # Case C handled — skip A/B for this OCR token

        # ── Case A: matched, look for extra symbols ──
        if m.match_type in ("iou", "center") and m.pdf_box is not None:
            extra = _extra_symbols(ocr_text, m.pdf_box.text, allowed)
            if not extra:
                if len(debug_log) < _MAX_DEBUG:
                    entry["path"] = "case_a_no_extra"
                    debug_log.append(entry)
                continue

            # Phase 2: composite detection — if OCR text has multiple
            # digit groups separated by symbols, defer to Case C logic
            # instead of blindly suffixing.
            if _RE_COMPOSITE.search(ocr_text):
                c_candidates = _generate_symbol_candidates(m.ocr_box, pdf_tokens, cfg)
                if c_candidates:
                    _accept_candidates(c_candidates, pdf_tokens, page_width, cfg)
                    entry["path"] = "case_a_deferred_to_c"
                    anchor_ids_seen_a: set = set()
                    for c in c_candidates:
                        for a in (c.anchor_left, c.anchor_right):
                            if a is not None and id(a) not in anchor_ids_seen_a:
                                anchor_ids_seen_a.add(id(a))
                                entry["anchors"].append(
                                    {"text": a.text, "bbox": [a.x0, a.y0, a.x1, a.y1]}
                                )
                        entry["candidates"].append(
                            {
                                "symbol": c.symbol,
                                "slot": c.slot_type,
                                "bbox": [c.x0, c.y0, c.x1, c.y1],
                                "status": c.status,
                                "reason": c.reject_reason,
                            }
                        )
                        if c.status == "accepted":
                            added.append(
                                GlyphBox(
                                    page=m.ocr_box.page,
                                    x0=c.x0,
                                    y0=c.y0,
                                    x1=c.x1,
                                    y1=c.y1,
                                    text=c.symbol,
                                    origin="ocr",
                                )
                            )
                    if len(debug_log) < _MAX_DEBUG:
                        debug_log.append(entry)
                    continue
                # If no candidates generated, fall through to suffix logic

            sym_candidate = GlyphBox(
                page=m.ocr_box.page,
                x0=m.pdf_box.x1,
                y0=m.pdf_box.y0,
                x1=m.pdf_box.x1 + char_w * len(extra),
                y1=m.pdf_box.y1,
                text=extra,
                origin="ocr",
            )
            if _overlaps_existing(sym_candidate, pdf_tokens):
                entry["path"] = "case_a"
                entry["candidates"].append(
                    {
                        "symbol": extra,
                        "slot": "suffix",
                        "bbox": [
                            sym_candidate.x0,
                            sym_candidate.y0,
                            sym_candidate.x1,
                            sym_candidate.y1,
                        ],
                        "status": "rejected",
                        "reason": "overlap_existing",
                    }
                )
                if len(debug_log) < _MAX_DEBUG:
                    debug_log.append(entry)
                continue
            entry["path"] = "case_a"
            entry["candidates"].append(
                {
                    "symbol": extra,
                    "slot": "suffix",
                    "bbox": [
                        sym_candidate.x0,
                        sym_candidate.y0,
                        sym_candidate.x1,
                        sym_candidate.y1,
                    ],
                    "status": "accepted",
                    "reason": "",
                }
            )
            added.append(sym_candidate)
            if len(debug_log) < _MAX_DEBUG:
                debug_log.append(entry)

        elif m.match_type == "unmatched":
            # ── Case B: unmatched symbol ──
            symbol_text = "".join(ch for ch in ocr_text if ch in allowed)
            if not symbol_text:
                if len(debug_log) < _MAX_DEBUG:
                    entry["path"] = "case_b_no_symbol"
                    debug_log.append(entry)
                continue

            if not _has_digit_neighbour_left(
                m.ocr_box, pdf_tokens, cfg.ocr_reconcile_proximity_pts, cfg
            ):
                entry["path"] = "case_b"
                entry["candidates"].append(
                    {
                        "symbol": symbol_text,
                        "slot": "standalone",
                        "bbox": [
                            m.ocr_box.x0,
                            m.ocr_box.y0,
                            m.ocr_box.x1,
                            m.ocr_box.y1,
                        ],
                        "status": "rejected",
                        "reason": "no_digit_neighbour",
                    }
                )
                if len(debug_log) < _MAX_DEBUG:
                    debug_log.append(entry)
                continue

            sym_candidate = GlyphBox(
                page=m.ocr_box.page,
                x0=m.ocr_box.x0,
                y0=m.ocr_box.y0,
                x1=m.ocr_box.x1,
                y1=m.ocr_box.y1,
                text=symbol_text,
                origin="ocr",
            )
            if _overlaps_existing(sym_candidate, pdf_tokens):
                entry["path"] = "case_b"
                entry["candidates"].append(
                    {
                        "symbol": symbol_text,
                        "slot": "standalone",
                        "bbox": [
                            m.ocr_box.x0,
                            m.ocr_box.y0,
                            m.ocr_box.x1,
                            m.ocr_box.y1,
                        ],
                        "status": "rejected",
                        "reason": "overlap_existing",
                    }
                )
                if len(debug_log) < _MAX_DEBUG:
                    debug_log.append(entry)
                continue
            entry["path"] = "case_b"
            entry["candidates"].append(
                {
                    "symbol": symbol_text,
                    "slot": "standalone",
                    "bbox": [m.ocr_box.x0, m.ocr_box.y0, m.ocr_box.x1, m.ocr_box.y1],
                    "status": "accepted",
                    "reason": "",
                }
            )
            added.append(sym_candidate)
            if len(debug_log) < _MAX_DEBUG:
                debug_log.append(entry)

    return added, debug_log, n_filtered_non_numeric


# ── Reconcile public entry point ──────────────────────────────────────


def reconcile_ocr(
    page_image,
    tokens: List[GlyphBox],
    page_num: int,
    page_width: float,
    page_height: float,
    cfg: GroupingConfig,
    *,
    ocr_tokens: Optional[List[GlyphBox]] = None,
    ocr_confs: Optional[List[float]] = None,
) -> ReconcileResult:
    """Run full-page OCR reconciliation against existing PDF tokens.

    Parameters
    ----------
    page_image : PIL.Image.Image
        Full-page render at ``cfg.ocr_reconcile_resolution`` DPI.
        Used for VOCR extraction when *ocr_tokens* is not supplied.
    tokens : list[GlyphBox]
        Existing PDF-extracted tokens (origin="text").
    page_num : int
        Zero-based page index (for GlyphBox.page).
    page_width, page_height : float
        Page dimensions in PDF points.
    cfg : GroupingConfig
        Configuration (OCR reconcile settings).
    ocr_tokens : list[GlyphBox], optional
        Pre-extracted VOCR tokens (from :func:`extract_vocr_tokens`).
        When provided, the internal VOCR extraction is skipped.
    ocr_confs : list[float], optional
        Parallel confidence scores for *ocr_tokens*.

    Returns
    -------
    ReconcileResult
        Contains ``added_tokens`` to extend the main token list.
    """
    result = ReconcileResult()

    # Stage 1 — full-page OCR (skipped when caller supplies tokens)
    if ocr_tokens is not None:
        if ocr_confs is None:
            ocr_confs = [1.0] * len(ocr_tokens)
    else:
        ocr_tokens, ocr_confs = _extract_ocr_tokens(
            page_image,
            page_num,
            page_width,
            page_height,
            cfg,
        )
    result.all_ocr_tokens = ocr_tokens

    if not ocr_tokens:
        result.stats = {
            "ocr_total": 0,
            "with_symbol": 0,
            "matched": 0,
            "unmatched": 0,
            "accepted": 0,
            "symbols": "",
        }
        log.info("OCR reconcile: 0 OCR tokens detected")
        return result

    # Stage 2 — spatial alignment
    pdf_tokens = [t for t in tokens if t.origin == "text"]
    matches = _build_match_index(ocr_tokens, ocr_confs, pdf_tokens, cfg)
    result.matches = matches

    # Stage 3 — symbol injection
    added, debug_log, n_filtered_non_numeric = _inject_symbols(
        matches, pdf_tokens, cfg, page_width
    )
    result.added_tokens = added

    # Stats
    allowed = cfg.ocr_reconcile_allowed_symbols
    with_symbol = sum(
        1 for m in matches if _has_allowed_symbol(m.ocr_box.text, allowed)
    )
    matched = sum(1 for m in matches if m.match_type in ("iou", "center"))
    unmatched = sum(1 for m in matches if m.match_type == "unmatched")
    symbol_summary = ", ".join(f"{t.text}" for t in added) if added else "(none)"

    # Count candidates from debug_log
    n_candidates = sum(len(e.get("candidates", [])) for e in debug_log)
    n_accepted_c = sum(
        1
        for e in debug_log
        for c in e.get("candidates", [])
        if c.get("status") == "accepted"
    )
    n_rejected_c = n_candidates - n_accepted_c

    result.stats = {
        "ocr_total": len(ocr_tokens),
        "with_symbol": with_symbol,
        "matched": matched,
        "unmatched": unmatched,
        "accepted": len(added),
        "symbols": symbol_summary,
        "candidates_generated": n_candidates,
        "candidates_accepted": n_accepted_c,
        "candidates_rejected": n_rejected_c,
        "filtered_non_numeric": n_filtered_non_numeric,
        "injection_log": debug_log,
    }

    log.info(
        "OCR reconcile: %d OCR tokens -> %d with symbol -> %d accepted (%s) "
        "[candidates: %d gen, %d ok, %d rej, %d filtered]",
        len(ocr_tokens),
        with_symbol,
        len(added),
        symbol_summary,
        n_candidates,
        n_accepted_c,
        n_rejected_c,
        n_filtered_non_numeric,
    )
    print(
        f"  OCR reconcile: {len(ocr_tokens)} OCR tokens -> "
        f"{with_symbol} with symbol -> {len(added)} accepted ({symbol_summary}) "
        f"[candidates: {n_candidates} gen, {n_accepted_c} ok, {n_rejected_c} rej, "
        f"{n_filtered_non_numeric} filtered]",
        flush=True,
    )

    return result
