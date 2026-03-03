"""Targeted VOCR — crop individual candidate patches and run PaddleOCR.

Instead of scanning the full page, this module crops the page image to
each :class:`~plancheck.models.VocrCandidate` bounding box (with padding),
runs OCR on the small crop, and updates the candidate's ``outcome``,
``found_text``, and ``found_symbol`` fields.

Public API
----------
extract_vocr_targeted  — run PaddleOCR on each candidate patch
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from ..config import GroupingConfig
from ..models import GlyphBox, VocrCandidate

log = logging.getLogger(__name__)

# Characters we consider "symbols" for hit/miss classification.
_SYMBOL_CHARS = set("%/°±Ø×'\"#@⌀∅≤≥≈")


def _crop_patch(
    image: Image.Image,
    candidate: VocrCandidate,
    page_width: float,
    page_height: float,
) -> Optional[np.ndarray]:
    """Crop *image* to the candidate's bbox, returning a numpy array.

    Coordinates are in PDF points; the image may have a different pixel
    resolution.  Returns ``None`` if the crop would be degenerate.
    """
    img_w, img_h = image.size
    sx = img_w / page_width if page_width > 0 else 1.0
    sy = img_h / page_height if page_height > 0 else 1.0

    px0 = int(max(0, candidate.x0 * sx))
    py0 = int(max(0, candidate.y0 * sy))
    px1 = int(min(img_w, candidate.x1 * sx))
    py1 = int(min(img_h, candidate.y1 * sy))

    if px1 - px0 < 4 or py1 - py0 < 4:
        return None

    crop = image.crop((px0, py0, px1, py1))
    # PaddleX text detection expects H×W×C.  If the page image is grayscale
    # (mode 'L') or palette-based, np.array(crop) becomes 2D and crashes.
    if crop.mode != "RGB":
        crop = crop.convert("RGB")
    return np.array(crop)


def _ocr_crop(
    ocr,
    crop_array: np.ndarray,
    min_conf: float,
) -> List[Tuple[str, float, List[float]]]:
    """Run PaddleOCR on a small crop; return [(text, conf, [x0,y0,x1,y1]), …].

    Coordinates are in *crop-local* pixels.
    """
    results: List[Tuple[str, float, List[float]]] = []
    # PaddleX/PaddleOCR predict() expects either a single image *or* a batch.
    # Some versions treat a NumPy array as an iterable (yielding rows), which
    # turns an H×W×C image into W×C slices and crashes downstream. Always pass
    # a 1-item batch to be unambiguous.
    for page_result in ocr.predict([crop_array]):
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
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            results.append((text, float(conf), [min(xs), min(ys), max(xs), max(ys)]))
    return results


def extract_vocr_targeted(
    page_image: Image.Image,
    candidates: List[VocrCandidate],
    page_num: int,
    page_width: float,
    page_height: float,
    cfg: GroupingConfig,
) -> Tuple[List[GlyphBox], List[float], List[VocrCandidate]]:
    """Run PaddleOCR on each candidate patch and collect results.

    Parameters
    ----------
    page_image : PIL.Image.Image
        Full-page render (at OCR resolution).
    candidates : list[VocrCandidate]
        Candidates produced by :func:`detect_vocr_candidates`.
    page_num : int
        Zero-based page index.
    page_width, page_height : float
        Page dimensions in PDF points (for coordinate mapping).
    cfg : GroupingConfig
        Pipeline configuration.

    Returns
    -------
    (tokens, confidences, updated_candidates)
        *tokens*: all OCR-detected :class:`GlyphBox` across every patch.
        *confidences*: parallel confidence list.
        *updated_candidates*: input candidates with ``outcome`` /
        ``found_text`` / ``found_symbol`` populated.
    """
    from .engine import _get_ocr  # lazy — avoids paddleocr import at module level

    if not candidates:
        return [], [], candidates

    ocr = _get_ocr(cfg)

    all_tokens: List[GlyphBox] = []
    all_confs: List[float] = []
    min_conf = cfg.vocr_min_confidence

    img_w, img_h = page_image.size
    sx = img_w / page_width if page_width > 0 else 1.0
    sy = img_h / page_height if page_height > 0 else 1.0

    for cand in candidates:
        crop = _crop_patch(page_image, cand, page_width, page_height)
        if crop is None:
            cand.outcome = "miss"
            continue

        hits = _ocr_crop(ocr, crop, min_conf)

        if not hits:
            cand.outcome = "miss"
            continue

        # Map crop-local pixel coords → PDF points
        cpx0 = int(max(0, cand.x0 * sx))
        cpy0 = int(max(0, cand.y0 * sy))

        found_texts: List[str] = []
        found_symbols: List[str] = []

        for text, conf, local_bbox in hits:
            # Convert crop-local px → page-level px → PDF pts
            abs_x0 = (local_bbox[0] + cpx0) / sx
            abs_y0 = (local_bbox[1] + cpy0) / sy
            abs_x1 = (local_bbox[2] + cpx0) / sx
            abs_y1 = (local_bbox[3] + cpy0) / sy

            token = GlyphBox(
                page=page_num,
                x0=abs_x0,
                y0=abs_y0,
                x1=abs_x1,
                y1=abs_y1,
                text=text,
                origin="ocr_targeted",
                confidence=conf,
            )
            all_tokens.append(token)
            all_confs.append(conf)
            found_texts.append(text)

            # Classify symbols
            for ch in text:
                if ch in _SYMBOL_CHARS:
                    found_symbols.append(ch)

        cand.found_text = " ".join(found_texts)
        if found_symbols:
            cand.outcome = "hit"
            cand.found_symbol = found_symbols[0]
        else:
            cand.outcome = "miss"

    hit_count = sum(1 for c in candidates if c.outcome == "hit")
    log.info(
        "Targeted VOCR page %d: %d candidates → %d hits, %d tokens",
        page_num,
        len(candidates),
        hit_count,
        len(all_tokens),
    )
    return all_tokens, all_confs, candidates
