"""Dual-source OCR reconciliation: full-page PaddleOCR + spatial matching.

Extracts OCR tokens from a full-page render, spatially aligns them against
existing PDF-text tokens, and injects only the missing special-character
tokens (%, /, °, ±) that the PDF text layer is missing.

Public API
----------
reconcile_ocr          – run the 4-stage reconciliation pipeline
draw_reconcile_debug   – render a debug overlay showing OCR results
draw_symbol_overlay    – render a simple overlay with green boxes around symbols
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from .config import GroupingConfig
from .models import GlyphBox

log = logging.getLogger(__name__)

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


def _iou(a: GlyphBox, b: GlyphBox) -> float:
    """Intersection-over-union of two axis-aligned boxes."""
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter == 0:
        return 0.0
    union = a.area() + b.area() - inter
    return inter / union if union > 0 else 0.0


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


# ── Stage 1: Extract OCR tokens from full page (with tiling) ──────────

# PaddleOCR's internal limit: images wider/taller than this get silently
# downscaled, destroying small CAD text.  We tile the image so each tile
# stays within this limit.
_PADDLE_MAX_SIDE = 3800  # leave 200 px headroom below the 4000 internal cap
_TILE_OVERLAP_FRAC = 0.05  # 5% overlap between adjacent tiles


def _ocr_one_tile(
    ocr,
    tile_array,
    offset_x: int,
    offset_y: int,
    sx: float,
    sy: float,
    page_num: int,
    min_conf: float,
) -> Tuple[List[GlyphBox], List[float]]:
    """Run PaddleOCR on a single tile and return tokens in PDF-point space."""
    tokens: List[GlyphBox] = []
    confidences: List[float] = []

    for page_result in ocr.predict(tile_array):
        # Dict-like OCRResult
        polys = (
            page_result.get("dt_polys")
            if hasattr(page_result, "get")
            else getattr(page_result, "dt_polys", None)
        )
        texts = (
            page_result.get("rec_texts")
            if hasattr(page_result, "get")
            else getattr(page_result, "rec_texts", None)
        )
        scores = (
            page_result.get("rec_scores")
            if hasattr(page_result, "get")
            else getattr(page_result, "rec_scores", None)
        )

        if polys is None or texts is None or scores is None:
            continue

        for poly, text, conf in zip(polys, texts, scores):
            if not text or conf < min_conf:
                continue

            # poly is [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] in tile-local pixels
            xs = [p[0] + offset_x for p in poly]
            ys = [p[1] + offset_y for p in poly]

            # Convert image pixels → PDF points
            tokens.append(
                GlyphBox(
                    page=page_num,
                    x0=min(xs) / sx,
                    y0=min(ys) / sy,
                    x1=max(xs) / sx,
                    y1=max(ys) / sy,
                    text=text,
                    origin="ocr_full",
                )
            )
            confidences.append(conf)

    return tokens, confidences


def _dedup_tiles(
    tokens: List[GlyphBox], confs: List[float]
) -> Tuple[List[GlyphBox], List[float]]:
    """Remove near-duplicate tokens from overlapping tile regions.

    Keeps the higher-confidence token when two tokens overlap significantly.
    """
    if len(tokens) <= 1:
        return tokens, confs

    keep = [True] * len(tokens)
    for i in range(len(tokens)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(tokens)):
            if not keep[j]:
                continue
            if _iou(tokens[i], tokens[j]) > 0.5:
                # Drop the lower-confidence duplicate
                if confs[i] >= confs[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break  # i is dropped, stop comparing

    out_tokens = [t for t, k in zip(tokens, keep) if k]
    out_confs = [c for c, k in zip(confs, keep) if k]
    return out_tokens, out_confs


def _extract_ocr_tokens(
    page_image: Image.Image,
    page_num: int,
    page_width: float,
    page_height: float,
    cfg: GroupingConfig,
) -> Tuple[List[GlyphBox], List[float]]:
    """Run PaddleOCR on the full page image, tiling if needed.

    Returns
    -------
    tokens : list[GlyphBox]
        All OCR-detected tokens with coords in PDF-point space.
    confidences : list[float]
        Parallel list of per-token confidence scores.
    """
    import time

    import numpy as np

    from ._ocr_engine import _get_ocr

    ocr = _get_ocr()

    # Ensure RGB mode (pdfplumber may return RGBA; PaddleOCR needs 3 channels)
    if page_image.mode != "RGB":
        page_image = page_image.convert("RGB")

    img_w, img_h = page_image.size
    sx = img_w / page_width  # pixels per PDF point (x)
    sy = img_h / page_height  # pixels per PDF point (y)
    eff_dpi = img_w / page_width * 72

    # Determine tiling grid
    need_tile = img_w > _PADDLE_MAX_SIDE or img_h > _PADDLE_MAX_SIDE
    if need_tile:
        overlap_x = int(img_w * _TILE_OVERLAP_FRAC)
        overlap_y = int(img_h * _TILE_OVERLAP_FRAC)
        step_x = _PADDLE_MAX_SIDE - overlap_x
        step_y = _PADDLE_MAX_SIDE - overlap_y

        tiles_x = []
        x = 0
        while x < img_w:
            x1 = min(x + _PADDLE_MAX_SIDE, img_w)
            tiles_x.append((x, x1))
            if x1 >= img_w:
                break
            x += step_x

        tiles_y = []
        y = 0
        while y < img_h:
            y1 = min(y + _PADDLE_MAX_SIDE, img_h)
            tiles_y.append((y, y1))
            if y1 >= img_h:
                break
            y += step_y

        n_tiles = len(tiles_x) * len(tiles_y)
    else:
        n_tiles = 1

    print(
        f"  OCR Stage 1: image {img_w}x{img_h} px "
        f"({eff_dpi:.0f} eff. DPI), "
        f"{'tiling ' + str(n_tiles) + ' tiles' if need_tile else 'single pass'}",
        flush=True,
    )

    t0 = time.perf_counter()
    all_tokens: List[GlyphBox] = []
    all_confs: List[float] = []

    def _run_ocr_with_heartbeat(func, *args, label="OCR"):
        """Run a blocking OCR call in a thread, printing heartbeat dots
        every 15 s so terminals / CI don't consider the process idle."""
        import threading

        result_box: list = []
        exc_box: list = []

        def _worker():
            try:
                result_box.append(func(*args))
            except Exception as e:
                exc_box.append(e)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        beat = 0
        while t.is_alive():
            t.join(timeout=15.0)
            if t.is_alive():
                beat += 1
                elapsed = time.perf_counter() - t0
                print(
                    f"    {label} ... {elapsed:.0f}s",
                    flush=True,
                )
        if exc_box:
            raise exc_box[0]
        return result_box[0]

    try:
        if not need_tile:
            # Single-pass: image fits within Paddle's limit
            img_array = np.array(page_image)
            all_tokens, all_confs = _run_ocr_with_heartbeat(
                _ocr_one_tile,
                ocr,
                img_array,
                0,
                0,
                sx,
                sy,
                page_num,
                cfg.ocr_reconcile_confidence,
                label="single-pass OCR",
            )
        else:
            # Tile-based OCR
            img_array = np.array(page_image)
            tile_idx = 0
            for y0, y1 in tiles_y:
                for x0, x1 in tiles_x:
                    tile_idx += 1
                    tile = img_array[y0:y1, x0:x1].copy()
                    t_tokens, t_confs = _run_ocr_with_heartbeat(
                        _ocr_one_tile,
                        ocr,
                        tile,
                        x0,
                        y0,
                        sx,
                        sy,
                        page_num,
                        cfg.ocr_reconcile_confidence,
                        label=f"tile {tile_idx}/{n_tiles}",
                    )
                    print(
                        f"    tile {tile_idx}/{n_tiles} "
                        f"({x0},{y0})-({x1},{y1}): "
                        f"{len(t_tokens)} tokens",
                        flush=True,
                    )
                    all_tokens.extend(t_tokens)
                    all_confs.extend(t_confs)

            # Deduplicate tokens from overlapping regions
            pre_dedup = len(all_tokens)
            all_tokens, all_confs = _dedup_tiles(all_tokens, all_confs)
            if pre_dedup != len(all_tokens):
                print(
                    f"    tile dedup: {pre_dedup} -> {len(all_tokens)} tokens",
                    flush=True,
                )

    except Exception as e:
        import traceback

        print(f"  OCR Stage 1: EXCEPTION during predict(): {e}", flush=True)
        traceback.print_exc()
        return [], []

    elapsed = time.perf_counter() - t0
    print(
        f"  OCR Stage 1: {len(all_tokens)} tokens extracted " f"in {elapsed:.1f}s",
        flush=True,
    )

    return all_tokens, all_confs


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
) -> bool:
    """Return True if a digit-bearing PDF token is within *proximity_pts* to the left."""
    cy = (candidate.y0 + candidate.y1) / 2.0
    band_tol = max(2.0, candidate.height() * 0.5) if candidate.height() > 0 else 2.0

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
        if -2.0 <= dx <= proximity_pts:
            return True
    return False


def _estimate_char_width(pdf_tokens: List[GlyphBox]) -> float:
    """Rough median character width from PDF tokens."""
    widths = []
    for t in pdf_tokens:
        if t.text and t.width() > 0:
            widths.append(t.width() / len(t.text))
    return median(widths) if widths else 5.0


# ── Stage 3b: Composite (multi-anchor) symbol injection ───────────────


def _find_line_neighbours(
    ocr_box: GlyphBox,
    pdf_tokens: List[GlyphBox],
    anchor_margin: float,
) -> List[GlyphBox]:
    """Find PDF tokens on the same text line within a horizontal window.

    Returns origin="text" tokens sorted by x0.
    """
    cy = (ocr_box.y0 + ocr_box.y1) / 2.0
    band_tol = max(3.0, ocr_box.height() * 0.6) if ocr_box.height() > 0 else 3.0
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


def _is_digit_group(text: str) -> bool:
    """Return True if *text* qualifies as a digit-group anchor.

    A token is a digit group if it starts with a digit (catches ``"09"``,
    ``"8.33"``, ``"2A"``), OR digits plus ``'.'`` comprise >= 50 %% of its
    characters.  This rejects labels like ``"SECTION 2"`` while keeping
    numeric values that symbols attach to.
    """
    if not text:
        return False
    # Fast path: starts with a digit
    if text[0].isdigit():
        return True
    digit_dot = sum(1 for ch in text if ch.isdigit() or ch == ".")
    return digit_dot >= len(text) * 0.5


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
        ocr_box, pdf_tokens, cfg.ocr_reconcile_anchor_margin
    )
    digit_anchors = [t for t in neighbours if _is_digit_group(t.text)]
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
            slash_w = max(0.5, 0.35 * h)
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
        for sym in unique_after:
            if sym == "%":
                sym_w = 0.95 * h
            else:  # ° ±
                sym_w = 0.5 * h
            candidates.append(
                SymbolCandidate(
                    symbol=sym,
                    slot_type="after_digit",
                    x0=cursor_x,
                    y0=band_y0,
                    x1=cursor_x + sym_w,
                    y1=band_y1,
                    ocr_source=ocr_box,
                    anchor_left=None,
                    anchor_right=rightmost,
                )
            )
            cursor_x += sym_w + pad  # advance for next symbol

    return candidates


def _accept_candidates(
    candidates: List[SymbolCandidate],
    pdf_tokens: List[GlyphBox],
    page_width: float,
) -> List[SymbolCandidate]:
    """Run acceptance checks on symbol candidates.

    Overlap rules:
    - between_digits: ignore overlap with anchor_left/anchor_right, reject
      overlap with any other token.
    - after_digit: ignore overlap with anchor_right, reject overlap with
      the next token to the right.
    Checks both origin="text" and origin="ocr" for "already in PDF" guard.
    """
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
            if abs(t_cx - cand_cx) < 4.0 and abs(t_cy - cand_cy) < 4.0:
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
            if _iou(cand_box, t) > 0.15 or _overlap_ratio(cand_box, t) > 0.30:
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
    char_w = _estimate_char_width(pdf_tokens)
    n_filtered_non_numeric = 0
    _MAX_DEBUG = 200

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
            _accept_candidates(candidates, pdf_tokens, page_width)
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
                    _accept_candidates(c_candidates, pdf_tokens, page_width)
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
                m.ocr_box, pdf_tokens, cfg.ocr_reconcile_proximity_pts
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


# ── Stage 4: Public entry point ────────────────────────────────────────


def reconcile_ocr(
    page_image: Image.Image,
    tokens: List[GlyphBox],
    page_num: int,
    page_width: float,
    page_height: float,
    cfg: GroupingConfig,
) -> ReconcileResult:
    """Run full-page OCR reconciliation against existing PDF tokens.

    Parameters
    ----------
    page_image : PIL.Image.Image
        Full-page render at ``cfg.ocr_reconcile_resolution`` DPI.
    tokens : list[GlyphBox]
        Existing PDF-extracted tokens (origin="text").
    page_num : int
        Zero-based page index (for GlyphBox.page).
    page_width, page_height : float
        Page dimensions in PDF points.
    cfg : GroupingConfig
        Configuration (OCR reconcile settings).

    Returns
    -------
    ReconcileResult
        Contains ``added_tokens`` to extend the main token list.
    """
    result = ReconcileResult()

    # Stage 1 — full-page OCR
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


# ── Debug overlay ──────────────────────────────────────────────────────


def draw_reconcile_debug(
    result: ReconcileResult,
    page_width: float,
    page_height: float,
    out_path: Path | str,
    scale: float = 1.0,
    background: Optional[Image.Image] = None,
) -> None:
    """Render a debug overlay showing OCR reconciliation results.

    Colour key
    ----------
    * **Light grey** – all OCR tokens detected on the page (raw OCR output).
    * **Orange outline** – OCR tokens containing an allowed symbol that were
      rejected (filtered out).
    * **Green box + label** – accepted / injected tokens.
    * **Blue line** – match line from OCR token to its matched PDF token.
    * **Cyan outline** – digit anchors used in Case C composite matching.
    * **Red outline + label** – rejected Case C candidates with reason.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    canvas_w = int(page_width * scale)
    canvas_h = int(page_height * scale)

    if background is not None:
        base = background.copy().convert("RGBA")
        if base.size != (canvas_w, canvas_h):
            base = base.resize((canvas_w, canvas_h), Image.LANCZOS)
    else:
        base = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))

    overlay = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("arial.ttf", max(10, int(8 * scale)))
    except OSError:
        font = ImageFont.load_default()

    allowed = "%" + "/°±"  # default; could pull from stats if wanted

    def _rect(b: GlyphBox) -> Tuple[float, float, float, float]:
        return (b.x0 * scale, b.y0 * scale, b.x1 * scale, b.y1 * scale)

    # Layer 1: all OCR tokens in light grey
    for t in result.all_ocr_tokens:
        draw.rectangle(_rect(t), outline=(180, 180, 180, 80), width=1)

    # Build sets for quick lookup
    added_set = set(id(t) for t in result.added_tokens)

    # Layer 2: symbol-bearing OCR tokens that were NOT accepted → orange
    # (only tokens that pass numeric-context filter, not headings)
    for m in result.matches:
        if (
            _has_numeric_symbol_context(m.ocr_box.text, allowed)
            and id(m.ocr_box) not in added_set
        ):
            # Check it wasn't accepted (added tokens have different identity)
            draw.rectangle(_rect(m.ocr_box), outline=(255, 140, 0, 180), width=2)

    # Layer 3: match lines (blue) for matched pairs
    for m in result.matches:
        if m.pdf_box is not None and m.match_type in ("iou", "center"):
            ocr_cx, ocr_cy = _center(m.ocr_box)
            pdf_cx, pdf_cy = _center(m.pdf_box)
            if _has_allowed_symbol(m.ocr_box.text, allowed):
                draw.line(
                    [
                        (ocr_cx * scale, ocr_cy * scale),
                        (pdf_cx * scale, pdf_cy * scale),
                    ],
                    fill=(0, 120, 255, 120),
                    width=1,
                )

    # Layer 4: accepted/injected tokens → green with label
    for t in result.added_tokens:
        r = _rect(t)
        draw.rectangle(r, outline=(0, 200, 0, 220), width=2)
        label = t.text
        draw.text((r[0], r[1] - 12 * scale), label, fill=(0, 200, 0, 255), font=font)

    # Layer 5: Case-C anchors → cyan outlines (from injection_log)
    injection_log = result.stats.get("injection_log", []) if result.stats else []
    for entry in injection_log:
        for anc in entry.get("anchors", []):
            bb = anc.get("bbox")
            if bb:
                r = (bb[0] * scale, bb[1] * scale, bb[2] * scale, bb[3] * scale)
                draw.rectangle(r, outline=(0, 220, 220, 160), width=1)

    # Layer 6: rejected candidates → red dashed outline + label
    for entry in injection_log:
        for cand in entry.get("candidates", []):
            if cand.get("status") == "rejected":
                bb = cand.get("bbox")
                if bb:
                    r = (bb[0] * scale, bb[1] * scale, bb[2] * scale, bb[3] * scale)
                    draw.rectangle(r, outline=(220, 40, 40, 160), width=1)
                    reason = cand.get("reason", "")[:20]
                    draw.text(
                        (r[0], r[3] + 1),
                        f"✗ {cand.get('symbol','')} {reason}",
                        fill=(220, 40, 40, 200),
                        font=font,
                    )

    # Composite and save
    out = Image.alpha_composite(base, overlay)
    out.convert("RGB").save(str(out_path))


def draw_symbol_overlay(
    result: ReconcileResult,
    page_width: float,
    page_height: float,
    out_path: Path | str,
    scale: float = 1.0,
    background: Optional[Image.Image] = None,
    allowed_symbols: str = "%/°±",
    show_labels: bool = True,
) -> None:
    """Render a simple overlay showing symbol-bearing OCR tokens in green boxes.

    Parameters
    ----------
    result : ReconcileResult
        Output from the reconciliation pipeline.
    page_width, page_height : float
        PDF page dimensions in points.
    out_path : Path | str
        Where to save the PNG.
    scale : float
        Render scale factor (1.0 = 72 DPI).
    background : Image, optional
        Background image (e.g. page render). If None, uses white.
    allowed_symbols : str
        Characters considered symbols. Default: "%/°±".
    show_labels : bool
        If True, draw the token text above each box.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    canvas_w = int(page_width * scale)
    canvas_h = int(page_height * scale)

    if background is not None:
        base = background.copy().convert("RGBA")
        if base.size != (canvas_w, canvas_h):
            base = base.resize((canvas_w, canvas_h), Image.LANCZOS)
    else:
        base = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))

    overlay = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("arial.ttf", max(12, int(10 * scale)))
    except OSError:
        font = ImageFont.load_default()

    def _rect(b: GlyphBox) -> Tuple[float, float, float, float]:
        return (b.x0 * scale, b.y0 * scale, b.x1 * scale, b.y1 * scale)

    # Find all symbol-bearing tokens
    symbol_tokens = [
        t for t in result.all_ocr_tokens if _has_allowed_symbol(t.text, allowed_symbols)
    ]

    # Draw green boxes around symbol tokens
    for t in symbol_tokens:
        r = _rect(t)
        # Green box outline (RGB: 0, 200, 0)
        draw.rectangle(r, outline=(0, 200, 0, 255), width=2)
        if show_labels:
            label = t.text
            draw.text(
                (r[0], r[1] - 14 * scale),
                label,
                fill=(0, 200, 0, 255),
                font=font,
            )

    # Composite and save
    out = Image.alpha_composite(base, overlay)
    out.convert("RGB").save(str(out_path))
    log.info("Symbol overlay saved: %s (%d symbols)", out_path.name, len(symbol_tokens))
