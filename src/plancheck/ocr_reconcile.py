"""Dual-source OCR reconciliation: full-page PaddleOCR + spatial matching.

Extracts OCR tokens from a full-page render, spatially aligns them against
existing PDF-text tokens, and injects only the missing special-character
tokens (%, /, °, ±) that the PDF text layer is missing.

Public API
----------
reconcile_ocr          – run the 4-stage reconciliation pipeline
draw_reconcile_debug   – render a debug overlay showing OCR results
"""

from __future__ import annotations

import logging
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


def _overlaps_existing(candidate: GlyphBox, tokens: List[GlyphBox],
                       iou_thresh: float = 0.15, cov_thresh: float = 0.30) -> bool:
    """Return True if *candidate* overlaps any existing token too much."""
    for t in tokens:
        if _iou(candidate, t) > iou_thresh:
            return True
        if _overlap_ratio(candidate, t) > cov_thresh:
            return True
    return False


# ── Stage 1: Extract OCR tokens from full page ────────────────────────


def _extract_ocr_tokens(
    page_image: Image.Image,
    page_num: int,
    page_width: float,
    page_height: float,
    cfg: GroupingConfig,
) -> Tuple[List[GlyphBox], List[float]]:
    """Run PaddleOCR on the full page image.

    Returns
    -------
    tokens : list[GlyphBox]
        All OCR-detected tokens with coords in PDF-point space.
    confidences : list[float]
        Parallel list of per-token confidence scores.
    """
    from ._ocr_engine import _get_ocr
    import numpy as np

    ocr = _get_ocr()

    # Convert PIL Image → numpy array for PaddleOCR
    img_array = np.array(page_image)

    img_w, img_h = page_image.size
    sx = img_w / page_width   # pixels per PDF point (x)
    sy = img_h / page_height  # pixels per PDF point (y)

    eff_dpi = img_w / page_width * 72
    print(
        f"  OCR Stage 1: image {img_w}x{img_h} px "
        f"({eff_dpi:.0f} eff. DPI), "
        f"page {page_width:.0f}x{page_height:.0f} pts",
        flush=True,
    )

    tokens: List[GlyphBox] = []
    confidences: List[float] = []

    # PaddleOCR 3.x uses .predict() returning a generator of result objects
    result_count = 0
    for page_result in ocr.predict(img_array):
        result_count += 1
        # Diagnostic: dump result object structure on first yield
        if result_count == 1:
            attrs = [a for a in dir(page_result) if not a.startswith("_")]
            print(
                f"  OCR Stage 1: page_result type={type(page_result).__name__}, "
                f"attrs={attrs}",
                flush=True,
            )
        # Each page_result has .dt_polys, .rec_texts, .rec_scores
        polys = getattr(page_result, "dt_polys", None)
        texts = getattr(page_result, "rec_texts", None)
        scores = getattr(page_result, "rec_scores", None)

        n_polys = len(polys) if polys is not None else -1
        n_texts = len(texts) if texts is not None else -1
        n_scores = len(scores) if scores is not None else -1
        print(
            f"  OCR Stage 1: result #{result_count}: "
            f"polys={n_polys}, texts={n_texts}, scores={n_scores}",
            flush=True,
        )

        if polys is None or texts is None or scores is None:
            continue

        for poly, text, conf in zip(polys, texts, scores):
            if not text or conf < cfg.ocr_reconcile_confidence:
                continue

            # poly is [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] in image pixels
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]

            # Convert image pixels → PDF points
            pdf_x0 = min(xs) / sx
            pdf_y0 = min(ys) / sy
            pdf_x1 = max(xs) / sx
            pdf_y1 = max(ys) / sy

            tokens.append(GlyphBox(
                page=page_num, x0=pdf_x0, y0=pdf_y0, x1=pdf_x1, y1=pdf_y1,
                text=text, origin="ocr_full",
            ))
            confidences.append(conf)

    if result_count == 0:
        print("  OCR Stage 1: predict() yielded 0 results!", flush=True)
    else:
        print(
            f"  OCR Stage 1: {len(tokens)} tokens extracted "
            f"(from {result_count} result objects)",
            flush=True,
        )

    return tokens, confidences


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
        matches.append(MatchRecord(
            ocr_box=ocr_box, pdf_box=pdf_box,
            match_type=match_type, ocr_confidence=conf,
        ))
    return matches


# ── Stage 3: Symbol-only filtering & injection ────────────────────────


def _has_allowed_symbol(text: str, allowed: str) -> bool:
    """Return True if *text* contains at least one allowed symbol character."""
    return any(ch in allowed for ch in text)


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


def _inject_symbols(
    matches: List[MatchRecord],
    pdf_tokens: List[GlyphBox],
    cfg: GroupingConfig,
) -> List[GlyphBox]:
    """Decide which OCR findings to inject as new GlyphBox tokens.

    Case A: matched OCR token has extra symbols not in the PDF token.
    Case B: unmatched OCR token contains an allowed symbol near a digit.
    """
    allowed = cfg.ocr_reconcile_allowed_symbols
    added: List[GlyphBox] = []
    char_w = _estimate_char_width(pdf_tokens)

    for m in matches:
        ocr_text = m.ocr_box.text

        # Pre-filter: must contain at least one allowed symbol
        if not _has_allowed_symbol(ocr_text, allowed):
            continue

        if m.match_type in ("iou", "center") and m.pdf_box is not None:
            # ── Case A: matched, look for extra symbols ──
            extra = _extra_symbols(ocr_text, m.pdf_box.text, allowed)
            if not extra:
                continue

            # Check if a PDF token already provides this symbol nearby
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
                continue
            added.append(sym_candidate)

        elif m.match_type == "unmatched":
            # ── Case B: unmatched symbol ──
            # Only keep the allowed-symbol characters
            symbol_text = "".join(ch for ch in ocr_text if ch in allowed)
            if not symbol_text:
                continue

            if not _has_digit_neighbour_left(
                m.ocr_box, pdf_tokens, cfg.ocr_reconcile_proximity_pts
            ):
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
                continue
            added.append(sym_candidate)

    return added


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
        page_image, page_num, page_width, page_height, cfg,
    )
    result.all_ocr_tokens = ocr_tokens

    if not ocr_tokens:
        result.stats = {
            "ocr_total": 0, "with_symbol": 0, "matched": 0,
            "unmatched": 0, "accepted": 0, "symbols": "",
        }
        log.info("OCR reconcile: 0 OCR tokens detected")
        return result

    # Stage 2 — spatial alignment
    pdf_tokens = [t for t in tokens if t.origin == "text"]
    matches = _build_match_index(ocr_tokens, ocr_confs, pdf_tokens, cfg)
    result.matches = matches

    # Stage 3 — symbol injection
    added = _inject_symbols(matches, pdf_tokens, cfg)
    result.added_tokens = added

    # Stats
    allowed = cfg.ocr_reconcile_allowed_symbols
    with_symbol = sum(1 for m in matches if _has_allowed_symbol(m.ocr_box.text, allowed))
    matched = sum(1 for m in matches if m.match_type in ("iou", "center"))
    unmatched = sum(1 for m in matches if m.match_type == "unmatched")
    symbol_summary = ", ".join(
        f"{t.text}" for t in added
    ) if added else "(none)"

    result.stats = {
        "ocr_total": len(ocr_tokens),
        "with_symbol": with_symbol,
        "matched": matched,
        "unmatched": unmatched,
        "accepted": len(added),
        "symbols": symbol_summary,
    }

    log.info(
        "OCR reconcile: %d OCR tokens → %d with symbol → %d accepted (%s)",
        len(ocr_tokens), with_symbol, len(added), symbol_summary,
    )
    print(
        f"  OCR reconcile: {len(ocr_tokens)} OCR tokens → "
        f"{with_symbol} with symbol → {len(added)} accepted ({symbol_summary})",
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
    for m in result.matches:
        if _has_allowed_symbol(m.ocr_box.text, allowed) and id(m.ocr_box) not in added_set:
            # Check it wasn't accepted (added tokens have different identity)
            draw.rectangle(_rect(m.ocr_box), outline=(255, 140, 0, 180), width=2)

    # Layer 3: match lines (blue) for matched pairs
    for m in result.matches:
        if m.pdf_box is not None and m.match_type in ("iou", "center"):
            ocr_cx, ocr_cy = _center(m.ocr_box)
            pdf_cx, pdf_cy = _center(m.pdf_box)
            if _has_allowed_symbol(m.ocr_box.text, allowed):
                draw.line(
                    [(ocr_cx * scale, ocr_cy * scale),
                     (pdf_cx * scale, pdf_cy * scale)],
                    fill=(0, 120, 255, 120), width=1,
                )

    # Layer 4: accepted/injected tokens → green with label
    for t in result.added_tokens:
        r = _rect(t)
        draw.rectangle(r, outline=(0, 200, 0, 220), width=2)
        label = t.text
        draw.text((r[0], r[1] - 12 * scale), label, fill=(0, 200, 0, 255), font=font)

    # Composite and save
    out = Image.alpha_composite(base, overlay)
    out.convert("RGB").save(str(out_path))
