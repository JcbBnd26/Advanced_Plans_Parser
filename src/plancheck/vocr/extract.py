"""VOCR Stage – full-page OCR token extraction (with tiling).

Extracts OCR tokens from a full-page render using the configured OCR
backend (Surya by default), tiling the image when it exceeds the
backend's recommended size limit.

Public API
----------
extract_vocr_tokens  – run full-page OCR and return raw tokens
"""

from __future__ import annotations

import logging
import time
from typing import List, Tuple

import numpy as np
from PIL import Image

from ..config import GroupingConfig
from ..models import GlyphBox
from ..models.geometry import glyph_iou as iou  # backward-compat alias
from .backends import get_ocr_backend

log = logging.getLogger(__name__)

# Default tiling configuration
# Images larger than this will be tiled to avoid memory issues
_DEFAULT_MAX_TILE_PX = 4096
_TILE_OVERLAP_FRAC_DEFAULT = 0.05  # 5% overlap between adjacent tiles


# ── Stage 1: Extract OCR tokens from full page (with tiling) ──────────


def _ocr_one_tile(
    backend,
    tile_array: np.ndarray,
    offset_x: int,
    offset_y: int,
    sx: float,
    sy: float,
    page_num: int,
    min_conf: float,
) -> Tuple[List[GlyphBox], List[float]]:
    """Run OCR backend on a single tile and return tokens in PDF-point space."""
    tokens: List[GlyphBox] = []
    confidences: List[float] = []

    # Use the backend's predict method which returns List[TextBox]
    text_boxes = backend.predict(tile_array)

    for box in text_boxes:
        if not box.text or box.confidence < min_conf:
            continue

        # Get polygon corners and apply tile offset
        xs = [p[0] + offset_x for p in box.polygon]
        ys = [p[1] + offset_y for p in box.polygon]

        # Convert image pixels → PDF points
        tokens.append(
            GlyphBox(
                page=page_num,
                x0=min(xs) / sx,
                y0=min(ys) / sy,
                x1=max(xs) / sx,
                y1=max(ys) / sy,
                text=box.text,
                origin="ocr_full",
                confidence=float(box.confidence),
            )
        )
        confidences.append(box.confidence)

    return tokens, confidences


def _dedup_tiles(
    tokens: List[GlyphBox],
    confs: List[float],
    dedup_iou: float = 0.5,
) -> Tuple[List[GlyphBox], List[float]]:
    """Remove near-duplicate tokens from overlapping tile regions.

    Uses a sort-and-sweep approach: tokens are sorted by x0 so that
    only tokens with overlapping x-intervals are compared, reducing
    average complexity from O(n²) to O(n log n).

    Keeps the higher-confidence token when two tokens overlap significantly.
    """
    if len(tokens) <= 1:
        return tokens, confs

    # Sort by x0 for sweep-line; track original indices
    indexed = sorted(range(len(tokens)), key=lambda i: tokens[i].x0)
    keep = [True] * len(tokens)

    for pos, i in enumerate(indexed):
        if not keep[i]:
            continue
        bx1 = tokens[i].x1
        # Only compare with subsequent tokens whose x0 is within reach
        for pos2 in range(pos + 1, len(indexed)):
            j = indexed[pos2]
            if tokens[j].x0 > bx1:
                break  # No more x-overlap possible
            if not keep[j]:
                continue
            if iou(tokens[i], tokens[j]) > dedup_iou:
                # Drop the lower-confidence duplicate
                if confs[i] >= confs[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break  # i is dropped, stop comparing

    out_tokens = [t for t, k in zip(tokens, keep) if k]
    out_confs = [c for c, k in zip(confs, keep) if k]
    return out_tokens, out_confs


def extract_ocr_tokens(
    page_image: Image.Image,
    page_num: int,
    page_width: float,
    page_height: float,
    cfg: GroupingConfig,
) -> Tuple[List[GlyphBox], List[float]]:
    """Run OCR on the full page image, tiling if needed.

    Returns
    -------
    tokens : list[GlyphBox]
        All OCR-detected tokens with coords in PDF-point space.
    confidences : list[float]
        Parallel list of per-token confidence scores.
    """
    # Get the configured OCR backend (thread-safe, no subprocess needed)
    backend = get_ocr_backend(cfg)

    # Resolve effective knobs
    max_tile_px = (
        cfg.vocr_max_tile_px if cfg.vocr_max_tile_px > 0 else _DEFAULT_MAX_TILE_PX
    )
    tile_overlap_frac = cfg.vocr_tile_overlap
    min_conf = cfg.vocr_min_confidence
    tile_dedup_iou = cfg.vocr_tile_dedup_iou
    min_text_len = cfg.vocr_min_text_length
    strip_ws = cfg.vocr_strip_whitespace

    # Ensure RGB mode (pdfplumber may return RGBA; OCR needs 3 channels)
    if page_image.mode != "RGB":
        page_image = page_image.convert("RGB")

    img_w, img_h = page_image.size
    sx = img_w / page_width  # pixels per PDF point (x)
    sy = img_h / page_height  # pixels per PDF point (y)
    eff_dpi = img_w / page_width * 72

    # Determine tiling grid
    need_tile = img_w > max_tile_px or img_h > max_tile_px
    if need_tile:
        overlap_x = int(img_w * tile_overlap_frac)
        overlap_y = int(img_h * tile_overlap_frac)
        step_x = max_tile_px - overlap_x
        step_y = max_tile_px - overlap_y

        tiles_x = []
        x = 0
        while x < img_w:
            x1 = min(x + max_tile_px, img_w)
            tiles_x.append((x, x1))
            if x1 >= img_w:
                break
            x += step_x

        tiles_y = []
        y = 0
        while y < img_h:
            y1 = min(y + max_tile_px, img_h)
            tiles_y.append((y, y1))
            if y1 >= img_h:
                break
            y += step_y

        n_tiles = len(tiles_x) * len(tiles_y)
    else:
        n_tiles = 1

    log.info(
        "  OCR Stage 1 (%s): image %dx%d px (%.0f eff. DPI), %s",
        backend.name,
        img_w,
        img_h,
        eff_dpi,
        f"tiling {n_tiles} tiles" if need_tile else "single pass",
    )

    t0 = time.perf_counter()
    all_tokens: List[GlyphBox] = []
    all_confs: List[float] = []

    try:
        if not need_tile:
            # Single-pass: image fits within tile limit
            img_array = np.array(page_image)
            all_tokens, all_confs = _ocr_one_tile(
                backend,
                img_array,
                0,
                0,
                sx,
                sy,
                page_num,
                min_conf,
            )
        else:
            # Tile-based OCR
            img_array = np.array(page_image)
            tile_idx = 0
            for y0, y1 in tiles_y:
                for x0, x1 in tiles_x:
                    tile_idx += 1
                    # Copy tile to avoid potential array mutation issues
                    tile = img_array[y0:y1, x0:x1].copy()
                    t_tokens, t_confs = _ocr_one_tile(
                        backend,
                        tile,
                        x0,
                        y0,
                        sx,
                        sy,
                        page_num,
                        min_conf,
                    )
                    log.info(
                        "    tile %d/%d (%d,%d)-(%d,%d): %d tokens",
                        tile_idx,
                        n_tiles,
                        x0,
                        y0,
                        x1,
                        y1,
                        len(t_tokens),
                    )
                    all_tokens.extend(t_tokens)
                    all_confs.extend(t_confs)

            # Deduplicate tokens from overlapping regions
            pre_dedup = len(all_tokens)
            all_tokens, all_confs = _dedup_tiles(all_tokens, all_confs, tile_dedup_iou)
            if pre_dedup != len(all_tokens):
                log.info(
                    "    tile dedup: %d -> %d tokens",
                    pre_dedup,
                    len(all_tokens),
                )

    except (RuntimeError, ValueError, MemoryError, OSError) as e:
        import traceback

        log.error("  OCR Stage 1: EXCEPTION during predict(): %s", e)
        traceback.print_exc()
        return [], []

    elapsed = time.perf_counter() - t0

    # ── Post-extraction filtering ──────────────────────────────────────
    pre_filter = len(all_tokens)
    if strip_ws or min_text_len > 0:
        keep = []
        for tok, conf in zip(all_tokens, all_confs):
            if strip_ws and not tok.text.strip():
                continue
            if min_text_len > 0 and len(tok.text) < min_text_len:
                continue
            keep.append((tok, conf))
        if keep:
            all_tokens, all_confs = zip(*keep)  # type: ignore[assignment]
            all_tokens, all_confs = list(all_tokens), list(all_confs)
        else:
            all_tokens, all_confs = [], []
        n_filtered = pre_filter - len(all_tokens)
        if n_filtered:
            log.info(
                "  OCR Stage 1: filtered %d tokens (min_len=%d, strip_ws=%s)",
                n_filtered,
                min_text_len,
                strip_ws,
            )

    log.info(
        "  OCR Stage 1: %d tokens extracted in %.1fs",
        len(all_tokens),
        elapsed,
    )

    return all_tokens, all_confs


# ── VOCR public entry point ───────────────────────────────────────────


def extract_vocr_tokens(
    page_image: Image.Image,
    page_num: int,
    page_width: float,
    page_height: float,
    cfg: GroupingConfig,
) -> tuple[List[GlyphBox], List[float]]:
    """Run full-page OCR and return raw tokens + confidences.

    This is the **VOCR stage** — visual OCR extraction only, no
    reconciliation. Uses Surya OCR by default. The returned tokens can
    be passed to :func:`reconcile_ocr` via its *ocr_tokens* / *ocr_confs*
    parameters to complete the pipeline.

    Parameters
    ----------
    page_image : PIL.Image.Image
        Full-page render (ideally preprocessed by VOCRPP).
    page_num : int
        Zero-based page index.
    page_width, page_height : float
        Page dimensions in PDF points.
    cfg : GroupingConfig
        Configuration (``vocr_*`` fields used).

    Returns
    -------
    (tokens, confidences)
        Parallel lists of :class:`GlyphBox` and float scores.
    """
    return extract_ocr_tokens(page_image, page_num, page_width, page_height, cfg)
