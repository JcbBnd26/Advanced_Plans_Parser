"""PaddleOCR subprocess worker — runs OCR in an isolated process.

This script runs as a standalone subprocess to avoid PaddleOCR's threading
issues. It receives OCR requests via stdin and returns results via stdout,
using JSON for serialization.

Protocol:
    Request:  {"cmd": "ocr", "image_b64": "...", "page_num": 0, "cfg": {...}}
    Response: {"status": "ok", "tokens": [...], "confs": [...]}

    Request:  {"cmd": "shutdown"}
    Response: {"status": "shutdown"}

Usage:
    python -m plancheck.vocr.subprocess_worker
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
from typing import Any

# Suppress Paddle's connectivity check before any imports
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

log = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure minimal logging to stderr (not stdout, which is for IPC)."""
    logging.basicConfig(
        level=logging.INFO,
        format="[VOCR-Worker] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )


def _decode_image(image_b64: str):
    """Decode base64 image data to numpy array."""
    import numpy as np
    from PIL import Image

    image_bytes = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def _build_config(cfg_dict: dict[str, Any]):
    """Build a GroupingConfig from a dict of settings."""
    from ..config import GroupingConfig

    # Only pass known fields
    known_fields = {
        "vocr_model_tier",
        "vocr_device",
        "vocr_use_orientation_classify",
        "vocr_use_doc_unwarping",
        "vocr_use_textline_orientation",
        "vocr_min_confidence",
        "vocr_max_tile_px",
        "vocr_tile_overlap",
        "vocr_tile_dedup_iou",
        "vocr_min_text_length",
        "vocr_strip_whitespace",
    }
    filtered = {k: v for k, v in cfg_dict.items() if k in known_fields}
    return GroupingConfig(**filtered)


def _token_iou(t1: dict, t2: dict) -> float:
    """Compute IoU between two token dicts (x0, y0, x1, y1)."""
    ix0 = max(t1["x0"], t2["x0"])
    iy0 = max(t1["y0"], t2["y0"])
    ix1 = min(t1["x1"], t2["x1"])
    iy1 = min(t1["y1"], t2["y1"])
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    a1 = (t1["x1"] - t1["x0"]) * (t1["y1"] - t1["y0"])
    a2 = (t2["x1"] - t2["x0"]) * (t2["y1"] - t2["y0"])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _dedup_tiles(
    tokens: list, confs: list, dedup_iou: float = 0.5
) -> tuple[list, list]:
    """Remove near-duplicate tokens from overlapping tile regions.

    Uses sort-and-sweep by x0 to reduce comparisons.
    Keeps the higher-confidence token when two tokens overlap significantly.
    """
    if len(tokens) <= 1:
        return tokens, confs

    # Sort by x0 for sweep-line
    indexed = sorted(range(len(tokens)), key=lambda i: tokens[i]["x0"])
    keep = [True] * len(tokens)

    for pos, i in enumerate(indexed):
        if not keep[i]:
            continue
        bx1 = tokens[i]["x1"]
        for pos2 in range(pos + 1, len(indexed)):
            j = indexed[pos2]
            if tokens[j]["x0"] > bx1:
                break  # No more x-overlap possible
            if not keep[j]:
                continue
            if _token_iou(tokens[i], tokens[j]) > dedup_iou:
                # Drop the lower-confidence duplicate
                if confs[i] >= confs[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    out_tokens = [t for t, k in zip(tokens, keep) if k]
    out_confs = [c for c, k in zip(confs, keep) if k]
    return out_tokens, out_confs


def _run_ocr(ocr, image_array, page_num: int, cfg_dict: dict) -> tuple[list, list]:
    """Run OCR on an image, with tiling if needed.

    Returns tokens in pixel coordinates (caller handles PDF-point conversion).
    """
    import numpy as np
    from PIL import Image

    # Config values
    max_tile_px = cfg_dict.get("vocr_max_tile_px", 3800)
    if max_tile_px <= 0:
        max_tile_px = 3800
    tile_overlap_frac = cfg_dict.get("vocr_tile_overlap", 0.05)
    min_conf = cfg_dict.get("vocr_min_confidence", 0.5)
    tile_dedup_iou = cfg_dict.get("vocr_tile_dedup_iou", 0.5)

    img_h, img_w = image_array.shape[:2]

    # Determine if tiling is needed
    need_tile = img_w > max_tile_px or img_h > max_tile_px

    all_tokens = []
    all_confs = []

    if not need_tile:
        # Single pass
        log.info("  Single-pass OCR: %dx%d px", img_w, img_h)
        tokens, confs = _ocr_one_tile(ocr, image_array, 0, 0, page_num, min_conf)
        all_tokens.extend(tokens)
        all_confs.extend(confs)
    else:
        # Calculate tile grid
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
        log.info("  Tiling OCR: %dx%d px, %d tiles", img_w, img_h, n_tiles)

        tile_idx = 0
        for y0, y1 in tiles_y:
            for x0, x1 in tiles_x:
                tile_idx += 1
                # Copy is required: PaddleOCR may modify the input array
                tile = image_array[y0:y1, x0:x1].copy()
                t_tokens, t_confs = _ocr_one_tile(ocr, tile, x0, y0, page_num, min_conf)
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
        if len(all_tokens) > 1:
            pre_dedup = len(all_tokens)
            all_tokens, all_confs = _dedup_tiles(all_tokens, all_confs, tile_dedup_iou)
            if pre_dedup != len(all_tokens):
                log.info("    tile dedup: %d -> %d tokens", pre_dedup, len(all_tokens))

    log.info("  OCR complete: %d tokens", len(all_tokens))
    return all_tokens, all_confs


def _ocr_one_tile(
    ocr, tile_array, offset_x: int, offset_y: int, page_num: int, min_conf: float
) -> tuple[list, list]:
    """Run OCR on a single tile, return tokens in pixel space with offsets applied."""
    tokens = []
    confs = []

    for page_result in ocr.predict(tile_array):
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

        if not (len(polys) == len(texts) == len(scores)):
            log.warning(
                "OCR result length mismatch: polys=%d, texts=%d, scores=%d",
                len(polys),
                len(texts),
                len(scores),
            )
            continue

        for poly, text, conf in zip(polys, texts, scores):
            if not text or conf < min_conf:
                continue

            # poly is [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] in tile-local pixels
            # Add offsets to get full-image coordinates
            xs = [p[0] + offset_x for p in poly]
            ys = [p[1] + offset_y for p in poly]

            tokens.append(
                {
                    "page": page_num,
                    "x0": float(min(xs)),
                    "y0": float(min(ys)),
                    "x1": float(max(xs)),
                    "y1": float(max(ys)),
                    "text": text,
                    "confidence": float(conf),
                }
            )
            confs.append(float(conf))

    return tokens, confs


def _send_response(response: dict) -> None:
    """Send a JSON response to stdout."""
    line = json.dumps(response, ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _worker_loop() -> None:
    """Main worker loop — read requests from stdin, process, respond."""
    from .engine import _get_ocr

    _setup_logging()
    log.info("Worker starting...")

    ocr_engine = None
    current_cfg_key = None

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            _send_response({"status": "error", "error": f"Invalid JSON: {e}"})
            continue

        cmd = request.get("cmd")

        if cmd == "shutdown":
            log.info("Shutdown requested")
            _send_response({"status": "shutdown"})
            break

        elif cmd == "ping":
            _send_response({"status": "ok", "message": "pong"})

        elif cmd == "init":
            # Pre-initialize OCR engine
            try:
                cfg_dict = request.get("cfg", {})
                cfg = _build_config(cfg_dict)
                ocr_engine = _get_ocr(cfg)
                current_cfg_key = tuple(sorted(cfg_dict.items()))
                log.info("OCR engine initialized")
                _send_response({"status": "ok"})
            except Exception as e:
                log.error("Init failed: %s", e, exc_info=True)
                _send_response({"status": "error", "error": str(e)})

        elif cmd == "ocr":
            try:
                image_b64 = request["image_b64"]
                page_num = request.get("page_num", 0)
                cfg_dict = request.get("cfg", {})

                # Check if config changed, rebuild engine if needed
                cfg_key = tuple(sorted(cfg_dict.items()))
                if ocr_engine is None or cfg_key != current_cfg_key:
                    cfg = _build_config(cfg_dict)
                    ocr_engine = _get_ocr(cfg)
                    current_cfg_key = cfg_key
                    log.info("OCR engine (re)initialized for new config")

                # Decode image and run OCR (with tiling)
                image_array = _decode_image(image_b64)
                tokens, confs = _run_ocr(ocr_engine, image_array, page_num, cfg_dict)

                _send_response({"status": "ok", "tokens": tokens, "confs": confs})

            except Exception as e:
                log.error("OCR failed: %s", e, exc_info=True)
                _send_response({"status": "error", "error": str(e)})

        else:
            _send_response({"status": "error", "error": f"Unknown command: {cmd}"})

    log.info("Worker exiting")


if __name__ == "__main__":
    _worker_loop()
