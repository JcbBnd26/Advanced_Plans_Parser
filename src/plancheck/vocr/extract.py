"""VOCR Stage – full-page PaddleOCR token extraction (with tiling).

Extracts OCR tokens from a full-page render using PaddleOCR, tiling the
image when it exceeds PaddleOCR's internal size limit.

Public API
----------
extract_vocr_tokens  – run full-page PaddleOCR and return raw tokens
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from PIL import Image

from ..config import GroupingConfig
from ..models import GlyphBox

log = logging.getLogger("plancheck.ocr_reconcile")

# PaddleOCR's internal limit: images wider/taller than this get silently
# downscaled, destroying small CAD text.  We tile the image so each tile
# stays within this limit.  The default is overridden by cfg.vocr_max_tile_px.
_PADDLE_MAX_SIDE_DEFAULT = 3800
_TILE_OVERLAP_FRAC_DEFAULT = 0.05  # 5 % overlap between adjacent tiles


# ── Geometry helper (used by _dedup_tiles) ─────────────────────────────


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


# ── Stage 1: Extract OCR tokens from full page (with tiling) ──────────


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
    tokens: List[GlyphBox],
    confs: List[float],
    dedup_iou: float = 0.5,
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
            if _iou(tokens[i], tokens[j]) > dedup_iou:
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

    from .engine import _get_ocr

    ocr = _get_ocr(cfg)

    # Resolve effective knobs
    max_tile_px = (
        cfg.vocr_max_tile_px if cfg.vocr_max_tile_px > 0 else _PADDLE_MAX_SIDE_DEFAULT
    )
    tile_overlap_frac = cfg.vocr_tile_overlap
    min_conf = cfg.vocr_min_confidence
    tile_dedup_iou = cfg.vocr_tile_dedup_iou
    heartbeat_sec = cfg.vocr_heartbeat_interval
    min_text_len = cfg.vocr_min_text_length
    strip_ws = cfg.vocr_strip_whitespace

    # Ensure RGB mode (pdfplumber may return RGBA; PaddleOCR needs 3 channels)
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
            t.join(timeout=heartbeat_sec)
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
                min_conf,
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
                        min_conf,
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
            all_tokens, all_confs = _dedup_tiles(all_tokens, all_confs, tile_dedup_iou)
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
            print(
                f"  OCR Stage 1: filtered {n_filtered} tokens "
                f"(min_len={min_text_len}, strip_ws={strip_ws})",
                flush=True,
            )

    print(
        f"  OCR Stage 1: {len(all_tokens)} tokens extracted " f"in {elapsed:.1f}s",
        flush=True,
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
    """Run full-page PaddleOCR and return raw tokens + confidences.

    This is the **VOCR stage** — visual OCR extraction only, no
    reconciliation.  The returned tokens can be passed to
    :func:`reconcile_ocr` via its *ocr_tokens* / *ocr_confs* parameters
    to complete the pipeline.

    Parameters
    ----------
    page_image : PIL.Image.Image
        Full-page render (ideally preprocessed by VOCRPP).
    page_num : int
        Zero-based page index.
    page_width, page_height : float
        Page dimensions in PDF points.
    cfg : GroupingConfig
        Configuration (``ocr_reconcile_*`` fields used).

    Returns
    -------
    (tokens, confidences)
        Parallel lists of :class:`GlyphBox` and float scores.
    """
    return _extract_ocr_tokens(page_image, page_num, page_width, page_height, cfg)
