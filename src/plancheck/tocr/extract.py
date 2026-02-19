"""Centralised TOCR extraction — single source of truth for PDF text-layer token extraction.

Two extraction modes share the same core loop:

``"full"``
    Config-driven normalisation, filtering, rotation tracking, font-size
    gating, control-char cleanup, text-aware dedup, mojibake detection, and
    rich diagnostics.  Used by the batch runner / manifest pipeline.

``"minimal"``
    Coordinate clipping + degenerate-box skip only; no text modification,
    no text-aware dedup, no diagnostics counters beyond token counts.
    Used by overlay / viewer / extract_page scripts where speed matters
    and full normalisation would change visual-debug output.

Both modes return a :class:`TocrPageResult` so callers never have to
know which path ran.  The legacy ``page_boxes`` tuple shape is available
via :meth:`TocrPageResult.to_legacy_tuple`.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pdfplumber

from ..config import GroupingConfig
from ..models import GlyphBox

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class TocrPageResult:
    """Output of a single-page TOCR extraction."""

    tokens: list[GlyphBox]
    page_width: float
    page_height: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    # Convenience --------------------------------------------------------

    def to_legacy_tuple(self) -> tuple[list[GlyphBox], float, float, dict]:
        """Return ``(boxes, page_width, page_height, diagnostics)``."""
        return self.tokens, self.page_width, self.page_height, self.diagnostics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Control-character regex: U+0000–U+001F (except \t \n \r) plus BOM
_RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufeff]")
# Mojibake detector: replacement char or common encoding artefacts
_RE_MOJIBAKE = re.compile(r"[\ufffd\ufffc]|Ã.|â€.")


def _empty_diagnostics(cfg: GroupingConfig) -> dict[str, Any]:
    """Return a zero-valued diagnostics dict with the full schema."""
    return {
        "extraction_params": {
            "x_tolerance": cfg.tocr_x_tolerance,
            "y_tolerance": cfg.tocr_y_tolerance,
            "extra_attrs": cfg.tocr_extra_attrs,
            "filter_control_chars": cfg.tocr_filter_control_chars,
            "dedup_iou": cfg.tocr_dedup_iou,
            "min_word_length": cfg.tocr_min_word_length,
            "min_font_size": cfg.tocr_min_font_size,
            "max_font_size": cfg.tocr_max_font_size,
            "strip_whitespace_tokens": cfg.tocr_strip_whitespace_tokens,
            "clip_to_page": cfg.tocr_clip_to_page,
            "margin_pts": cfg.tocr_margin_pts,
            "keep_rotated": cfg.tocr_keep_rotated,
            "normalize_unicode": cfg.tocr_normalize_unicode,
            "case_fold": cfg.tocr_case_fold,
            "collapse_whitespace": cfg.tocr_collapse_whitespace,
            "min_token_density": cfg.tocr_min_token_density,
            "mojibake_threshold": cfg.tocr_mojibake_threshold,
            "use_text_flow": cfg.tocr_use_text_flow,
            "keep_blank_chars": cfg.tocr_keep_blank_chars,
        },
        "tokens_total": 0,
        "tokens_raw": 0,
        "tokens_degenerate_skipped": 0,
        "tokens_control_char_cleaned": 0,
        "tokens_empty_after_clean": 0,
        "tokens_duplicate_removed": 0,
        "tokens_font_size_filtered": 0,
        "tokens_rotated_dropped": 0,
        "tokens_margin_filtered": 0,
        "tokens_short_filtered": 0,
        "tokens_whitespace_filtered": 0,
        "tokens_unicode_normalized": 0,
        "tokens_case_folded": 0,
        "tokens_whitespace_collapsed": 0,
        "font_names": {},
        "font_sizes": {},
        "has_rotated_text": False,
        "upright_count": 0,
        "non_upright_count": 0,
        "char_encoding_issues": 0,
        "mojibake_fraction": 0.0,
        "below_min_density": False,
        "page_area_sqin": 0.0,
        "token_density_per_sqin": 0.0,
        "error": None,
    }


def _build_extract_words_kwargs(
    cfg: GroupingConfig,
    mode: Literal["full", "minimal"],
) -> dict[str, Any]:
    """Build ``pdfplumber.Page.extract_words`` keyword arguments."""
    kw: dict[str, Any] = {
        "x_tolerance": cfg.tocr_x_tolerance,
        "y_tolerance": cfg.tocr_y_tolerance,
    }
    if mode == "full":
        if cfg.tocr_use_text_flow:
            kw["use_text_flow"] = True
        if cfg.tocr_keep_blank_chars:
            kw["keep_blank_chars"] = True
        if cfg.tocr_extra_attrs:
            kw["extra_attrs"] = ["fontname", "size", "upright"]
    else:
        # Minimal: still pass extra_attrs so font info is captured
        if cfg.tocr_extra_attrs:
            kw["extra_attrs"] = ["fontname", "size"]
    return kw


def _word_to_glyph_minimal(
    w: dict,
    page_num: int,
    page_w: float,
    page_h: float,
    cfg: GroupingConfig,
) -> GlyphBox | None:
    """Convert a pdfplumber word dict → GlyphBox (minimal mode).

    Only clips coordinates and skips degenerate boxes.
    """
    x0 = max(0.0, min(page_w, float(w.get("x0", 0))))
    x1 = max(0.0, min(page_w, float(w.get("x1", 0))))
    y0 = max(0.0, min(page_h, float(w.get("top", 0))))
    y1 = max(0.0, min(page_h, float(w.get("bottom", 0))))
    if x1 <= x0 or y1 <= y0:
        return None
    return GlyphBox(
        page=page_num,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        text=w.get("text", ""),
        origin="text",
        fontname=w.get("fontname", ""),
        font_size=float(w["size"]) if "size" in w else 0.0,
    )


def _word_to_glyph_full(
    w: dict,
    page_num: int,
    page_w: float,
    page_h: float,
    cfg: GroupingConfig,
    diag: dict[str, Any],
    font_name_counter: Counter,
    font_size_counter: Counter,
) -> GlyphBox | None:
    """Convert a pdfplumber word dict → GlyphBox (full mode).

    Applies all config-driven filters, normalisation, and diagnostics.
    Returns ``None`` when the word should be dropped.
    """
    # Coordinate extraction — optionally clip to page bounds
    raw_x0 = float(w.get("x0", 0.0))
    raw_x1 = float(w.get("x1", 0.0))
    raw_y0 = float(w.get("top", 0.0))
    raw_y1 = float(w.get("bottom", 0.0))
    if cfg.tocr_clip_to_page:
        x0 = max(0.0, min(page_w, raw_x0))
        x1 = max(0.0, min(page_w, raw_x1))
        y0 = max(0.0, min(page_h, raw_y0))
        y1 = max(0.0, min(page_h, raw_y1))
    else:
        x0, x1, y0, y1 = raw_x0, raw_x1, raw_y0, raw_y1
    text = w.get("text", "")

    # Skip degenerate boxes (zero-area)
    if x1 <= x0 or y1 <= y0:
        diag["tokens_degenerate_skipped"] += 1
        return None

    # Margin filter
    if cfg.tocr_margin_pts > 0:
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        m = cfg.tocr_margin_pts
        if cx < m or cx > page_w - m or cy < m or cy > page_h - m:
            diag["tokens_margin_filtered"] += 1
            return None

    # Track font info
    fsize_val: float | None = None
    fname: str = ""
    if cfg.tocr_extra_attrs:
        fname = w.get("fontname", "unknown")
        fsize = w.get("size")
        font_name_counter[fname] += 1
        if fsize is not None:
            fsize_val = float(fsize)
            font_size_counter[str(round(fsize_val, 1))] += 1
        # Track rotation
        upright = w.get("upright")
        if upright is not None:
            if upright:
                diag["upright_count"] += 1
            else:
                diag["non_upright_count"] += 1
                diag["has_rotated_text"] = True

    # Font-size filter
    if fsize_val is not None:
        if cfg.tocr_min_font_size > 0 and fsize_val < cfg.tocr_min_font_size:
            diag["tokens_font_size_filtered"] += 1
            return None
        if cfg.tocr_max_font_size > 0 and fsize_val > cfg.tocr_max_font_size:
            diag["tokens_font_size_filtered"] += 1
            return None

    # Drop rotated text if configured
    if not cfg.tocr_keep_rotated and cfg.tocr_extra_attrs:
        upright = w.get("upright")
        if upright is not None and not upright:
            diag["tokens_rotated_dropped"] += 1
            return None

    # Filter control characters
    if cfg.tocr_filter_control_chars and _RE_CONTROL.search(text):
        text = _RE_CONTROL.sub("", text)
        diag["tokens_control_char_cleaned"] += 1
        if not text.strip():
            diag["tokens_empty_after_clean"] += 1
            return None

    # Detect encoding issues
    if _RE_MOJIBAKE.search(text):
        diag["char_encoding_issues"] += 1

    # Unicode normalisation
    if cfg.tocr_normalize_unicode:
        import unicodedata

        normed = unicodedata.normalize("NFKC", text)
        if normed != text:
            diag["tokens_unicode_normalized"] += 1
            text = normed

    # Collapse internal whitespace
    if cfg.tocr_collapse_whitespace:
        collapsed = re.sub(r"\s+", " ", text)
        if collapsed != text:
            diag["tokens_whitespace_collapsed"] += 1
            text = collapsed

    # Case folding
    if cfg.tocr_case_fold:
        folded = text.lower()
        if folded != text:
            diag["tokens_case_folded"] += 1
            text = folded

    # Strip whitespace-only tokens
    if cfg.tocr_strip_whitespace_tokens and not text.strip():
        diag["tokens_whitespace_filtered"] += 1
        return None

    # Minimum word length
    if cfg.tocr_min_word_length > 0 and len(text.strip()) < cfg.tocr_min_word_length:
        diag["tokens_short_filtered"] += 1
        return None

    return GlyphBox(
        page=page_num,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        text=text,
        origin="text",
        fontname=fname,
        font_size=fsize_val or 0.0,
    )


def _dedup_identical_text_iou(
    boxes: list[GlyphBox],
    threshold: float,
    diag: dict[str, Any],
) -> list[GlyphBox]:
    """Remove duplicate boxes with identical text and IoU >= *threshold*."""
    if threshold <= 0 or len(boxes) < 2:
        return boxes
    keep = [True] * len(boxes)
    for i in range(len(boxes)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(boxes)):
            if not keep[j]:
                continue
            if boxes[i].text != boxes[j].text:
                continue
            ix0 = max(boxes[i].x0, boxes[j].x0)
            iy0 = max(boxes[i].y0, boxes[j].y0)
            ix1 = min(boxes[i].x1, boxes[j].x1)
            iy1 = min(boxes[i].y1, boxes[j].y1)
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            a1 = boxes[i].area()
            a2 = boxes[j].area()
            union = a1 + a2 - inter
            if union > 0 and inter / union >= threshold:
                keep[j] = False
                diag["tokens_duplicate_removed"] += 1
    return [b for b, k in zip(boxes, keep) if k]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_tocr_from_page(
    page: "pdfplumber.page.Page",
    page_num: int,
    cfg: GroupingConfig | None = None,
    *,
    mode: Literal["full", "minimal"] = "full",
) -> TocrPageResult:
    """Extract TOCR tokens from an already-opened pdfplumber Page.

    Parameters
    ----------
    page : pdfplumber.page.Page
        An opened page object.
    page_num : int
        Zero-based page index (stored in each GlyphBox).
    cfg : GroupingConfig, optional
        Extraction configuration.  Defaults are used when ``None``.
    mode : ``"full"`` | ``"minimal"``
        ``"full"`` — all normalisation, filtering, and diagnostics.
        ``"minimal"`` — coordinate clip + degenerate skip only.

    Returns
    -------
    TocrPageResult
    """
    if cfg is None:
        cfg = GroupingConfig()

    page_w = float(page.width)
    page_h = float(page.height)

    diag = _empty_diagnostics(cfg)
    diag["page_area_sqin"] = round((page_w / 72.0) * (page_h / 72.0), 2)

    kw = _build_extract_words_kwargs(cfg, mode)
    words = page.extract_words(**kw)
    diag["tokens_raw"] = len(words)

    if mode == "minimal":
        boxes: list[GlyphBox] = []
        for w in words:
            g = _word_to_glyph_minimal(w, page_num, page_w, page_h, cfg)
            if g is not None:
                boxes.append(g)
        diag["tokens_total"] = len(boxes)
        return TocrPageResult(
            tokens=boxes,
            page_width=page_w,
            page_height=page_h,
            diagnostics=diag,
        )

    # ── Full mode ────────────────────────────────────────────────────
    font_name_counter: Counter = Counter()
    font_size_counter: Counter = Counter()
    boxes = []
    for w in words:
        g = _word_to_glyph_full(
            w,
            page_num,
            page_w,
            page_h,
            cfg,
            diag,
            font_name_counter,
            font_size_counter,
        )
        if g is not None:
            boxes.append(g)

    # Text-aware dedup (full mode only)
    boxes = _dedup_identical_text_iou(boxes, cfg.tocr_dedup_iou, diag)

    diag["tokens_total"] = len(boxes)
    diag["font_names"] = dict(font_name_counter.most_common(20))
    diag["font_sizes"] = dict(font_size_counter.most_common(20))

    # Density
    if diag["page_area_sqin"] > 0:
        diag["token_density_per_sqin"] = round(
            len(boxes) / diag["page_area_sqin"],
            1,
        )

    # Mojibake fraction
    if diag["tokens_raw"] > 0:
        diag["mojibake_fraction"] = round(
            diag["char_encoding_issues"] / diag["tokens_raw"],
            4,
        )

    # Min-density flag
    if (
        cfg.tocr_min_token_density > 0
        and diag["token_density_per_sqin"] < cfg.tocr_min_token_density
    ):
        diag["below_min_density"] = True

    # Quality warnings
    if len(boxes) == 0:
        log.warning(
            "TOCR page %d: zero tokens extracted (blank or image-only page)",
            page_num,
        )
    if diag["char_encoding_issues"] > 0:
        log.warning(
            "TOCR page %d: %d tokens with encoding issues (mojibake)",
            page_num,
            diag["char_encoding_issues"],
        )
    if (
        cfg.tocr_mojibake_threshold > 0
        and diag["mojibake_fraction"] > cfg.tocr_mojibake_threshold
    ):
        log.warning(
            "TOCR page %d: mojibake fraction %.1f%% exceeds threshold %.1f%%",
            page_num,
            diag["mojibake_fraction"] * 100,
            cfg.tocr_mojibake_threshold * 100,
        )
    if diag["below_min_density"]:
        log.warning(
            "TOCR page %d: token density %.1f/sq-in below minimum %.1f",
            page_num,
            diag["token_density_per_sqin"],
            cfg.tocr_min_token_density,
        )
    if diag["has_rotated_text"]:
        log.info(
            "TOCR page %d: %d non-upright (rotated) words detected",
            page_num,
            diag["non_upright_count"],
        )

    return TocrPageResult(
        tokens=boxes,
        page_width=page_w,
        page_height=page_h,
        diagnostics=diag,
    )


def extract_tocr_page(
    pdf_path: Path | str,
    page_num: int,
    cfg: GroupingConfig | None = None,
    *,
    mode: Literal["full", "minimal"] = "full",
) -> TocrPageResult:
    """Extract TOCR tokens from a PDF file + page number.

    Opens the PDF, delegates to :func:`extract_tocr_from_page`, and
    returns the result.  On extraction failure (full mode), returns an
    empty result with ``diagnostics["error"]`` set.

    Parameters
    ----------
    pdf_path : Path or str
        Path to the PDF file.
    page_num : int
        Zero-based page index.
    cfg : GroupingConfig, optional
    mode : ``"full"`` | ``"minimal"``

    Returns
    -------
    TocrPageResult
    """
    if cfg is None:
        cfg = GroupingConfig()

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num]
            return extract_tocr_from_page(page, page_num, cfg, mode=mode)
    except Exception as exc:
        import traceback

        log.error("TOCR page %d: extraction failed: %s", page_num, exc)
        traceback.print_exc()
        diag = _empty_diagnostics(cfg)
        diag["error"] = str(exc)
        return TocrPageResult(
            tokens=[],
            page_width=0.0,
            page_height=0.0,
            diagnostics=diag,
        )
