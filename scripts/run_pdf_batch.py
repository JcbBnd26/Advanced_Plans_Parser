from __future__ import annotations

import re


def detect_headers_from_words(boxes: list[GlyphBox]) -> list[GlyphBox]:
    """Tag header candidates directly from word boxes (before grouping)."""
    # Improved: group words into lines by y0 and horizontal proximity
    from collections import defaultdict

    header_re = re.compile(r"^[A-Z0-9\s\-\(\)\'\.]+: *$", re.ASCII)
    # Step 1: group by y0 (row)
    y_tol = 2.0  # tolerance for y alignment in points
    rows = defaultdict(list)
    for b in boxes:
        found = False
        for y in rows:
            if abs(b.y0 - y) < y_tol:
                rows[y].append(b)
                found = True
                break
        if not found:
            rows[b.y0].append(b)
    header_boxes = []
    for row in rows.values():
        # Step 2: sort by x0 and group horizontally close words into lines
        row = sorted(row, key=lambda b: b.x0)
        line = []
        lines = []
        x_gap_tol = 20.0  # max gap between words in a line (points)
        for b in row:
            if not line:
                line.append(b)
            else:
                prev = line[-1]
                if b.x0 - prev.x1 < x_gap_tol:
                    line.append(b)
                else:
                    lines.append(line)
                    line = [b]
        if line:
            lines.append(line)
        # Step 3: apply header regex to each line
        for line_words in lines:
            line_text = " ".join(b.text for b in line_words if b.text).strip()
            line_text_norm = re.sub(r"\s+", " ", line_text).upper()
            if header_re.match(line_text_norm):
                for b in line_words:
                    b.origin = "header_candidate"
                header_boxes.extend(line_words)
    return header_boxes


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import argparse
import json
import re
import shutil
from datetime import datetime

import pdfplumber

from plancheck import (
    BlockCluster,
    GlyphBox,
    GroupingConfig,
    Line,
    StageResult,
    build_clusters_v2,
    classify_blocks,
    detect_zones,
    draw_reconcile_debug,
    draw_symbol_overlay,
    estimate_skew_degrees,
    extract_vocr_tokens,
    gate,
    mark_notes,
    mark_tables,
    nms_prune,
    reconcile_ocr,
    rotate_boxes,
    run_stage,
    zone_summary,
)
from plancheck.export import export_page_results
from plancheck.grouping import (
    compute_median_space_gap,
    group_notes_columns,
    link_continued_columns,
    mark_headers,
)
from plancheck.legends import (
    detect_abbreviation_regions,
    detect_legend_regions,
    detect_misc_title_regions,
    detect_revision_regions,
    detect_standard_detail_regions,
    extract_graphics,
    filter_graphics_outside_regions,
)
from plancheck.models import (
    AbbreviationRegion,
    LegendRegion,
    RevisionRegion,
    StandardDetailRegion,
)

# OCR image preprocessing
from plancheck.ocr_preprocess_pipeline import (
    OcrPreprocessConfig,
    preprocess_image_for_ocr,
)
from plancheck.overlay import draw_lines_overlay


def make_run_dir(base: Path, name: str) -> Path:
    run_dir = base / name
    for sub in ["artifacts", "overlays", "exports", "logs"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def cleanup_old_runs(run_root: Path, keep: int = 50) -> None:
    """Delete old run folders, keeping only the most recent `keep` runs."""
    run_dirs = sorted(
        [d for d in run_root.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old_dir in run_dirs[keep:]:
        shutil.rmtree(old_dir, ignore_errors=True)
        print(f"Cleaned up old run: {old_dir.name}")


def page_boxes(
    pdf_path: Path, page_num: int, cfg: GroupingConfig | None = None
) -> tuple[list[GlyphBox], float, float, dict]:
    """Extract word boxes from PDF text layer with full diagnostics.

    Returns
    -------
    (boxes, page_width, page_height, diagnostics)
        diagnostics dict contains font distribution, token stats, quality
        signals, and extraction parameters for the manifest.
    """
    import logging
    import re
    import traceback
    from collections import Counter

    log = logging.getLogger(__name__)

    if cfg is None:
        cfg = GroupingConfig()

    diag: dict = {
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

    # Control-character regex: U+0000–U+001F (except \t \n \r) plus BOM
    _RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufeff]")
    # Mojibake detector: replacement char or common encoding artifacts
    _RE_MOJIBAKE = re.compile(r"[\ufffd\ufffc]|Ã.|â€.")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num]
            page_w, page_h = float(page.width), float(page.height)

            # Page area in square inches (assuming pts; 72 pts/inch)
            diag["page_area_sqin"] = round((page_w / 72.0) * (page_h / 72.0), 2)

            # Build extract_words kwargs from config
            extract_kwargs: dict = {
                "x_tolerance": cfg.tocr_x_tolerance,
                "y_tolerance": cfg.tocr_y_tolerance,
            }
            if cfg.tocr_use_text_flow:
                extract_kwargs["use_text_flow"] = True
            if cfg.tocr_keep_blank_chars:
                extract_kwargs["keep_blank_chars"] = True
            if cfg.tocr_extra_attrs:
                extract_kwargs["extra_attrs"] = ["fontname", "size", "upright"]

            words = page.extract_words(**extract_kwargs)
            diag["tokens_raw"] = len(words)

            font_name_counter: Counter = Counter()
            font_size_counter: Counter = Counter()
            boxes: list[GlyphBox] = []

            for w in words:
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
                    continue

                # Margin filter — drop words whose centre is within margin of edge
                if cfg.tocr_margin_pts > 0:
                    cx = (x0 + x1) / 2.0
                    cy = (y0 + y1) / 2.0
                    m = cfg.tocr_margin_pts
                    if cx < m or cx > page_w - m or cy < m or cy > page_h - m:
                        diag["tokens_margin_filtered"] += 1
                        continue

                # Track font info if available
                fsize_val: float | None = None
                if cfg.tocr_extra_attrs:
                    fname = w.get("fontname", "unknown")
                    fsize = w.get("size")
                    font_name_counter[fname] += 1
                    if fsize is not None:
                        fsize_val = float(fsize)
                        # Round to 1 decimal for grouping
                        font_size_counter[str(round(fsize_val, 1))] += 1
                    # Track rotated text
                    upright = w.get("upright")
                    if upright is not None:
                        if upright:
                            diag["upright_count"] += 1
                        else:
                            diag["non_upright_count"] += 1
                            diag["has_rotated_text"] = True

                # Font-size filter (requires extra_attrs)
                if fsize_val is not None:
                    if (
                        cfg.tocr_min_font_size > 0
                        and fsize_val < cfg.tocr_min_font_size
                    ):
                        diag["tokens_font_size_filtered"] += 1
                        continue
                    if (
                        cfg.tocr_max_font_size > 0
                        and fsize_val > cfg.tocr_max_font_size
                    ):
                        diag["tokens_font_size_filtered"] += 1
                        continue

                # Drop rotated text if configured
                if not cfg.tocr_keep_rotated and cfg.tocr_extra_attrs:
                    upright = w.get("upright")
                    if upright is not None and not upright:
                        diag["tokens_rotated_dropped"] += 1
                        continue

                # Filter control characters
                if cfg.tocr_filter_control_chars and _RE_CONTROL.search(text):
                    text = _RE_CONTROL.sub("", text)
                    diag["tokens_control_char_cleaned"] += 1
                    if not text.strip():
                        diag["tokens_empty_after_clean"] += 1
                        continue

                # Detect encoding issues
                if _RE_MOJIBAKE.search(text):
                    diag["char_encoding_issues"] += 1

                # Unicode normalisation (NFKC: ligatures, fullwidth, etc.)
                if cfg.tocr_normalize_unicode:
                    import unicodedata

                    normed = unicodedata.normalize("NFKC", text)
                    if normed != text:
                        diag["tokens_unicode_normalized"] += 1
                        text = normed

                # Collapse internal whitespace runs to single space
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
                    continue

                # Minimum word length
                if (
                    cfg.tocr_min_word_length > 0
                    and len(text.strip()) < cfg.tocr_min_word_length
                ):
                    diag["tokens_short_filtered"] += 1
                    continue

                boxes.append(
                    GlyphBox(
                        page=page_num,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        text=text,
                        origin="text",
                    )
                )

            # Deduplicate overlapping identical-text boxes
            if cfg.tocr_dedup_iou > 0 and len(boxes) > 1:
                keep = [True] * len(boxes)
                for i in range(len(boxes)):
                    if not keep[i]:
                        continue
                    for j in range(i + 1, len(boxes)):
                        if not keep[j]:
                            continue
                        if boxes[i].text != boxes[j].text:
                            continue
                        # Compute IoU
                        ix0 = max(boxes[i].x0, boxes[j].x0)
                        iy0 = max(boxes[i].y0, boxes[j].y0)
                        ix1 = min(boxes[i].x1, boxes[j].x1)
                        iy1 = min(boxes[i].y1, boxes[j].y1)
                        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                        a1 = boxes[i].area()
                        a2 = boxes[j].area()
                        union = a1 + a2 - inter
                        if union > 0 and inter / union >= cfg.tocr_dedup_iou:
                            keep[j] = False
                            diag["tokens_duplicate_removed"] += 1
                boxes = [b for b, k in zip(boxes, keep) if k]

            diag["tokens_total"] = len(boxes)
            diag["font_names"] = dict(font_name_counter.most_common(20))
            diag["font_sizes"] = dict(font_size_counter.most_common(20))

            # Token density
            if diag["page_area_sqin"] > 0:
                diag["token_density_per_sqin"] = round(
                    len(boxes) / diag["page_area_sqin"], 1
                )

            # Mojibake fraction
            if diag["tokens_raw"] > 0:
                diag["mojibake_fraction"] = round(
                    diag["char_encoding_issues"] / diag["tokens_raw"], 4
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

    except Exception as e:
        log.error("TOCR page %d: extraction failed: %s", page_num, e)
        traceback.print_exc()
        diag["error"] = str(e)
        return [], 0.0, 0.0, diag

    return boxes, page_w, page_h, diag


def render_page_image(pdf_path: Path, page_num: int, resolution: int = 200):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        img_page = page.to_image(resolution=resolution)
        return img_page.original.copy()


def save_boxes_json(boxes: list[GlyphBox], out_path: Path) -> None:
    serializable = [
        {
            "page": b.page,
            "x0": b.x0,
            "y0": b.y0,
            "x1": b.x1,
            "y1": b.y1,
            "text": b.text,
            "origin": b.origin,
        }
        for b in boxes
    ]
    out_path.write_text(json.dumps(serializable, indent=2))


def summarize(blocks: list[BlockCluster]) -> str:
    summary = [f"Blocks: {len(blocks)}"]
    table_count = sum(1 for b in blocks if b.is_table)
    summary.append(f"Marked tables: {table_count}")
    for i, blk in enumerate(blocks, start=1):
        x0, y0, x1, y1 = blk.bbox()
        num_items = len(blk.lines) if blk.lines else len(blk.rows)
        item_label = "lines" if blk.lines else "rows"
        summary.append(
            f"Block {i}: {item_label}={num_items} table={blk.is_table} bbox=({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f})"
        )
    return "\n".join(summary)


def process_page(
    pdf: Path,
    page_num: int,
    run_dir: Path,
    resolution: int,
    color_overrides: dict | None = None,
    cfg: GroupingConfig | None = None,
) -> dict:
    """Process a single page and return page results for manifest.

    Runs the 5-stage pipeline:
        ingest → tocr ‖ vocrpp → vocr → reconcile
    with per-stage timing, gating, and fallback handling.
    """
    import time as _time
    from concurrent.futures import ThreadPoolExecutor

    pdf_stem = pdf.stem.replace(" ", "_")

    if cfg is None:
        cfg = GroupingConfig()

    stage_results: dict[str, StageResult] = {}

    # ── Stage: ingest ──────────────────────────────────────────────────
    with run_stage("ingest", cfg) as sr_ingest:
        bg_img = render_page_image(pdf, page_num, resolution=resolution)
        sr_ingest.counts = {"render_dpi": resolution}
        sr_ingest.status = "success"
    stage_results["ingest"] = sr_ingest

    # ── Parallel: tocr ‖ vocrpp ────────────────────────────────────────
    _preprocess_img = None
    _vocrpp_sr = None
    boxes = []
    page_w = page_h = 0.0

    def _run_tocr():
        nonlocal boxes, page_w, page_h
        with run_stage("tocr", cfg) as sr:
            b, pw, ph, tocr_diag = page_boxes(pdf, page_num, cfg=cfg)
            boxes, page_w, page_h = b, pw, ph
            sr.counts = {
                "tokens_total": tocr_diag["tokens_total"],
                "tokens_raw": tocr_diag["tokens_raw"],
                "tokens_degenerate_skipped": tocr_diag["tokens_degenerate_skipped"],
                "tokens_control_char_cleaned": tocr_diag["tokens_control_char_cleaned"],
                "tokens_empty_after_clean": tocr_diag["tokens_empty_after_clean"],
                "tokens_duplicate_removed": tocr_diag["tokens_duplicate_removed"],
                "char_encoding_issues": tocr_diag["char_encoding_issues"],
                "has_rotated_text": tocr_diag["has_rotated_text"],
                "upright_count": tocr_diag["upright_count"],
                "non_upright_count": tocr_diag["non_upright_count"],
                "page_area_sqin": tocr_diag["page_area_sqin"],
                "token_density_per_sqin": tocr_diag["token_density_per_sqin"],
            }
            sr.outputs = {
                "font_names": tocr_diag["font_names"],
                "font_sizes": tocr_diag["font_sizes"],
                "extraction_params": tocr_diag["extraction_params"],
            }
            if tocr_diag.get("error"):
                sr.status = "failed"
                sr.error = {"message": tocr_diag["error"]}
            else:
                sr.status = "success"
        return sr

    def _run_vocrpp():
        nonlocal _preprocess_img
        with run_stage("vocrpp", cfg) as sr:
            if sr.ran:
                ocr_res = (
                    cfg.vocr_resolution
                    if cfg.vocr_resolution > 0
                    else cfg.ocr_reconcile_resolution
                )
                raw_img = render_page_image(pdf, page_num, resolution=ocr_res)
                pp_cfg = OcrPreprocessConfig(
                    enabled=True,
                    grayscale=cfg.vocrpp_grayscale,
                    autocontrast=cfg.vocrpp_autocontrast,
                    clahe=cfg.vocrpp_clahe,
                    clahe_clip_limit=cfg.vocrpp_clahe_clip_limit,
                    clahe_tile_size=cfg.vocrpp_clahe_grid_size,
                    median_denoise=cfg.vocrpp_median_denoise,
                    median_kernel_size=cfg.vocrpp_median_kernel,
                    adaptive_binarize=cfg.vocrpp_adaptive_binarize,
                    adaptive_block_size=cfg.vocrpp_binarize_block_size,
                    adaptive_c=cfg.vocrpp_binarize_constant,
                    sharpen=cfg.vocrpp_sharpen,
                    sharpen_radius=cfg.vocrpp_sharpen_radius,
                    sharpen_percent=cfg.vocrpp_sharpen_percent,
                    save_intermediate=False,
                )
                pp_result = preprocess_image_for_ocr(raw_img, cfg=pp_cfg)
                pp_dir = run_dir / "ocr_preprocess"
                pp_dir.mkdir(parents=True, exist_ok=True)
                out_path = pp_dir / f"page_{page_num:04d}_ocr_input.png"
                pp_result.image.save(out_path)
                _preprocess_img = pp_result.image
                sr.counts = {
                    "images_out": 1,
                    "applied_steps": pp_result.applied_steps,
                    "render_dpi": ocr_res,
                }
                sr.outputs = {
                    "image_path": str(out_path),
                    "metrics": pp_result.metrics,
                }
                sr.status = "success"
                print(
                    f"    vocrpp page {page_num}: {pp_result.applied_steps}",
                    flush=True,
                )
        return sr

    with ThreadPoolExecutor(max_workers=2) as pool:
        tocr_future = pool.submit(_run_tocr)
        vocrpp_future = pool.submit(_run_vocrpp)
        sr_tocr = tocr_future.result()
        sr_vocrpp = vocrpp_future.result()

    stage_results["tocr"] = sr_tocr
    stage_results["vocrpp"] = sr_vocrpp

    boxes = nms_prune(boxes, cfg.iou_prune)
    if cfg.enable_skew:
        skew = estimate_skew_degrees(boxes, cfg.max_skew_degrees)
        boxes = rotate_boxes(
            boxes, -skew, page_w, page_h, min_rotation=cfg.preprocess_min_rotation
        )
    else:
        skew = 0.0

    # ── Stage: vocr ────────────────────────────────────────────────────
    ocr_tokens = None
    ocr_confs = None
    vocr_inputs = {
        "source": "preprocessed" if _preprocess_img is not None else "raw",
        "vocrpp_ran": sr_vocrpp.ran,
    }
    with run_stage("vocr", cfg, inputs=vocr_inputs) as sr_vocr:
        if sr_vocr.ran:
            if _preprocess_img is not None:
                ocr_img = _preprocess_img
                print(
                    f"    vocr: using preprocessed image for page {page_num}",
                    flush=True,
                )
            else:
                _vocr_res = (
                    cfg.vocr_resolution
                    if cfg.vocr_resolution > 0
                    else cfg.ocr_reconcile_resolution
                )
                ocr_img = render_page_image(pdf, page_num, resolution=_vocr_res)
            ocr_tokens, ocr_confs = extract_vocr_tokens(
                page_image=ocr_img,
                page_num=page_num,
                page_width=page_w,
                page_height=page_h,
                cfg=cfg,
            )
            sr_vocr.counts = {
                "tokens_total": len(ocr_tokens),
                "confidence_min": round(min(ocr_confs), 3) if ocr_confs else 0,
                "confidence_max": round(max(ocr_confs), 3) if ocr_confs else 0,
                "confidence_mean": (
                    round(sum(ocr_confs) / len(ocr_confs), 3) if ocr_confs else 0
                ),
                "image_source": (
                    "preprocessed" if _preprocess_img is not None else "raw"
                ),
                "render_dpi": (
                    cfg.vocr_resolution
                    if cfg.vocr_resolution > 0
                    else cfg.ocr_reconcile_resolution
                ),
            }
            sr_vocr.inputs = vocr_inputs
            sr_vocr.status = "success"
    stage_results["vocr"] = sr_vocr

    # ── Stage: reconcile ───────────────────────────────────────────────
    reconcile_result = None
    recon_inputs = {
        "vocr_failed": sr_vocr.status == "failed",
        "tocr_tokens": len(boxes),
        "vocr_tokens": len(ocr_tokens) if ocr_tokens else 0,
    }
    with run_stage("reconcile", cfg, inputs=recon_inputs) as sr_recon:
        if sr_recon.ran:
            # Determine the OCR image for reconcile (fallback if vocr had tokens)
            if _preprocess_img is not None:
                ocr_img_for_recon = _preprocess_img
            else:
                ocr_img_for_recon = render_page_image(
                    pdf, page_num, resolution=cfg.ocr_reconcile_resolution
                )
            reconcile_result = reconcile_ocr(
                page_image=ocr_img_for_recon,
                tokens=boxes,
                page_num=page_num,
                page_width=page_w,
                page_height=page_h,
                cfg=cfg,
                ocr_tokens=ocr_tokens,
                ocr_confs=ocr_confs,
            )
            if reconcile_result.added_tokens:
                boxes.extend(reconcile_result.added_tokens)
                boxes = nms_prune(boxes, cfg.iou_prune)
            sr_recon.counts = {
                "accepted": len(reconcile_result.added_tokens),
                "rejected": reconcile_result.stats.get("candidates_rejected", 0),
                "filtered_non_numeric": reconcile_result.stats.get(
                    "filtered_non_numeric", 0
                ),
                "candidates_generated": reconcile_result.stats.get(
                    "candidates_generated", 0
                ),
                "ocr_total": reconcile_result.stats.get("ocr_total", 0),
            }
            sr_recon.status = "success"
    stage_results["reconcile"] = sr_recon

    # v2 pipeline: row-truth first, then non-destructive column detection
    blocks = build_clusters_v2(boxes, page_h, cfg)

    # Block-level header detection and debug output
    debug_path = str(run_dir / "artifacts" / "debug_headers.txt")
    mark_headers(blocks, debug_path=debug_path)
    # Notes detection, will skip header blocks
    mark_notes(blocks, debug_path=debug_path)
    # Group headers with their notes blocks into columns
    notes_columns = group_notes_columns(blocks, debug_path=debug_path, cfg=cfg)
    # Link continued columns (e.g., "NOTES" and "NOTES (CONT'D)")
    link_continued_columns(notes_columns, blocks=blocks, debug_path=debug_path, cfg=cfg)

    boxes_path = run_dir / "artifacts" / f"{pdf_stem}_page_{page_num}_boxes.json"
    save_boxes_json(boxes, boxes_path)

    # Save block-level clusters with tags
    blocks_path = run_dir / "artifacts" / f"{pdf_stem}_page_{page_num}_blocks.json"

    def serialize_block(blk):
        x0, y0, x1, y1 = blk.bbox()
        result = {
            "page": blk.page,
            "bbox": [x0, y0, x1, y1],
            "rows": [
                {
                    "bbox": list(row.bbox()),
                    "texts": [b.text for b in row.boxes],
                }
                for row in blk.rows
            ],
            "label": blk.label,
            # Cast to Python bool to avoid numpy.bool_ during JSON dump
            "is_table": bool(blk.is_table),
            "is_notes": bool(blk.is_notes),
        }
        # Add line-level data if available (v2 pipeline)
        if blk.lines and blk._tokens:
            result["lines"] = [
                {
                    "line_id": line.line_id,
                    "baseline_y": line.baseline_y,
                    "bbox": list(line.bbox(blk._tokens)),
                    "text": line.text(blk._tokens),
                    "spans": [
                        {
                            "col_id": span.col_id,
                            "bbox": list(span.bbox(blk._tokens)),
                            "text": span.text(blk._tokens),
                        }
                        for span in line.spans
                    ],
                }
                for line in blk.lines
            ]
        return result

    blocks_serialized = [serialize_block(blk) for blk in blocks]
    blocks_path.write_text(json.dumps(blocks_serialized, indent=2))

    # Save notes columns with continuation info
    columns_path = run_dir / "artifacts" / f"{pdf_stem}_page_{page_num}_columns.json"

    def serialize_column(col):
        return {
            "header_text": col.header_text(),
            "base_header_text": col.base_header_text(),
            "is_continuation": col.is_continuation(),
            "column_group_id": col.column_group_id,
            "continues_from": col.continues_from,
            "notes_count": len(col.notes_blocks),
            "bbox": list(col.bbox()),
        }

    columns_serialized = [serialize_column(col) for col in notes_columns]
    columns_path.write_text(json.dumps(columns_serialized, indent=2))

    # Extract graphics and detect legends
    graphics = extract_graphics(str(pdf), page_num)

    # Set up file-based logging for legends module (mirrors old debug_path behaviour)
    import logging as _logging

    _legends_logger = _logging.getLogger("plancheck.legends")
    _legends_fh = _logging.FileHandler(debug_path, mode="a", encoding="utf-8")
    _legends_fh.setLevel(_logging.DEBUG)
    _legends_fh.setFormatter(_logging.Formatter("[DEBUG] %(message)s"))
    _legends_logger.addHandler(_legends_fh)
    _legends_logger.setLevel(_logging.DEBUG)

    try:
        # Detect abbreviation regions FIRST (pure text, no graphics)
        abbreviation_regions = detect_abbreviation_regions(
            blocks=blocks,
            graphics=graphics,
            page_width=page_w,
            page_height=page_h,
            cfg=cfg,
        )

        # Get exclusion zones from abbreviation regions
        exclusion_zones = [abbrev.bbox() for abbrev in abbreviation_regions]

        # Detect misc title regions (e.g., 'OKLAHOMA DEPARTMENT OF TRANSPORTATION')
        misc_title_regions = detect_misc_title_regions(
            blocks=blocks,
            graphics=graphics,
            page_width=page_w,
            page_height=page_h,
            exclusion_zones=exclusion_zones,
            cfg=cfg,
        )

        # Add misc title regions to exclusion zones
        for mt in misc_title_regions:
            exclusion_zones.append(mt.bbox())

        # Detect revision regions BEFORE legends (title block element)
        revision_regions = detect_revision_regions(
            blocks=blocks,
            graphics=graphics,
            page_width=page_w,
            page_height=page_h,
            exclusion_zones=exclusion_zones,
            cfg=cfg,
        )

        # Add revision regions to exclusion zones for legend detection
        for rev in revision_regions:
            exclusion_zones.append(rev.bbox())

        # Filter graphics to exclude those in abbreviation/revision regions
        filtered_graphics = filter_graphics_outside_regions(graphics, exclusion_zones)

        # Now detect legend regions with filtered graphics AND exclusion zones for text
        legend_regions = detect_legend_regions(
            blocks=blocks,
            graphics=filtered_graphics,
            page_width=page_w,
            page_height=page_h,
            exclusion_zones=exclusion_zones,
            cfg=cfg,
        )

        # Detect standard detail regions (similar to abbreviations - two-column text)
        standard_detail_regions = detect_standard_detail_regions(
            blocks=blocks,
            graphics=graphics,
            page_width=page_w,
            page_height=page_h,
            exclusion_zones=exclusion_zones,
            cfg=cfg,
        )
    finally:
        _legends_logger.removeHandler(_legends_fh)
        _legends_fh.close()

    # Save abbreviation regions
    abbrev_path = (
        run_dir / "artifacts" / f"{pdf_stem}_page_{page_num}_abbreviations.json"
    )

    def serialize_abbreviation(abbrev):
        return {
            "header_text": abbrev.header_text(),
            "is_boxed": abbrev.is_boxed,
            "box_bbox": list(abbrev.box_bbox) if abbrev.box_bbox else None,
            "bbox": list(abbrev.bbox()),
            "entries_count": len(abbrev.entries),
            "entries": [
                {
                    "code": e.code,
                    "meaning": e.meaning,
                    "code_bbox": list(e.code_bbox) if e.code_bbox else None,
                    "meaning_bbox": list(e.meaning_bbox) if e.meaning_bbox else None,
                }
                for e in abbrev.entries
            ],
        }

    abbrev_serialized = [serialize_abbreviation(ab) for ab in abbreviation_regions]
    abbrev_path.write_text(json.dumps(abbrev_serialized, indent=2))

    # Save legend regions
    legends_path = run_dir / "artifacts" / f"{pdf_stem}_page_{page_num}_legends.json"

    def serialize_legend(legend):
        return {
            "header_text": legend.header_text(),
            "is_boxed": legend.is_boxed,
            "box_bbox": list(legend.box_bbox) if legend.box_bbox else None,
            "bbox": list(legend.bbox()),
            "entries_count": len(legend.entries),
            "entries": [
                {
                    "symbol_bbox": list(e.symbol_bbox) if e.symbol_bbox else None,
                    "description": e.description,
                    "description_bbox": (
                        list(e.description_bbox) if e.description_bbox else None
                    ),
                }
                for e in legend.entries
            ],
        }

    legends_serialized = [serialize_legend(leg) for leg in legend_regions]
    legends_path.write_text(json.dumps(legends_serialized, indent=2))

    # Save revision regions
    revisions_path = (
        run_dir / "artifacts" / f"{pdf_stem}_page_{page_num}_revisions.json"
    )

    def serialize_revision(rev):
        return {
            "header_text": rev.header_text(),
            "is_boxed": rev.is_boxed,
            "box_bbox": list(rev.box_bbox) if rev.box_bbox else None,
            "bbox": list(rev.bbox()),
            "entries_count": len(rev.entries),
            "entries": [
                {
                    "number": e.number,
                    "description": e.description,
                    "date": e.date,
                    "row_bbox": list(e.row_bbox) if e.row_bbox else None,
                }
                for e in rev.entries
            ],
        }

    revisions_serialized = [serialize_revision(r) for r in revision_regions]
    revisions_path.write_text(json.dumps(revisions_serialized, indent=2))

    # Save misc title regions
    misc_titles_path = (
        run_dir / "artifacts" / f"{pdf_stem}_page_{page_num}_misc_titles.json"
    )

    def serialize_misc_title(mt):
        return {
            "text": mt.text,
            "is_boxed": mt.is_boxed,
            "box_bbox": list(mt.box_bbox) if mt.box_bbox else None,
            "bbox": list(mt.bbox()),
        }

    misc_titles_serialized = [serialize_misc_title(mt) for mt in misc_title_regions]
    misc_titles_path.write_text(json.dumps(misc_titles_serialized, indent=2))

    # Save standard detail regions
    std_details_path = (
        run_dir / "artifacts" / f"{pdf_stem}_page_{page_num}_standard_details.json"
    )

    def serialize_standard_detail(sd):
        return {
            "header_text": sd.header_text(),
            "subheader": sd.subheader,
            "subheader_bbox": list(sd.subheader_bbox) if sd.subheader_bbox else None,
            "is_boxed": sd.is_boxed,
            "box_bbox": list(sd.box_bbox) if sd.box_bbox else None,
            "bbox": list(sd.bbox()),
            "entries_count": len(sd.entries),
            "entries": [
                {
                    "sheet_number": e.sheet_number,
                    "description": e.description,
                    "sheet_bbox": list(e.sheet_bbox) if e.sheet_bbox else None,
                    "description_bbox": (
                        list(e.description_bbox) if e.description_bbox else None
                    ),
                }
                for e in sd.entries
            ],
        }

    std_details_serialized = [
        serialize_standard_detail(sd) for sd in standard_detail_regions
    ]
    std_details_path.write_text(json.dumps(std_details_serialized, indent=2))

    # ── Zoning: detect semantic page zones ─────────────────────────────
    page_zones = detect_zones(
        page_width=page_w,
        page_height=page_h,
        blocks=blocks,
        notes_columns=notes_columns,
        legend_bboxes=[leg.bbox() for leg in legend_regions],
        abbreviation_bboxes=[ab.bbox() for ab in abbreviation_regions],
        revision_bboxes=[rev.bbox() for rev in revision_regions],
        detail_bboxes=[sd.bbox() for sd in standard_detail_regions],
        cfg=cfg,
    )
    block_zone_map = classify_blocks(blocks, page_zones)

    zones_path = run_dir / "artifacts" / f"{pdf_stem}_page_{page_num}_zones.json"
    zones_path.write_text(json.dumps(zone_summary(page_zones), indent=2))

    # Build the lines/spans overlay as the primary overlay (v2 pipeline)
    all_lines = [ln for blk in blocks for ln in (blk.lines or [])]
    overlay_path = run_dir / "overlays" / f"{pdf_stem}_page_{page_num}_overlay.png"
    scale = resolution / 72.0
    draw_lines_overlay(
        page_width=page_w,
        page_height=page_h,
        lines=all_lines,
        tokens=boxes,
        out_path=overlay_path,
        scale=scale,
        background=bg_img,
        cfg=cfg,
    )

    # OCR reconcile debug overlay
    ocr_reconcile_debug_path = None
    if reconcile_result is not None and draw_reconcile_debug is not None:
        if reconcile_result.added_tokens or cfg.ocr_reconcile_debug:
            ocr_reconcile_debug_path = (
                run_dir / "overlays" / f"{pdf_stem}_page_{page_num}_ocr_reconcile.png"
            )
            draw_reconcile_debug(
                result=reconcile_result,
                page_width=page_w,
                page_height=page_h,
                out_path=ocr_reconcile_debug_path,
                scale=scale,
                background=bg_img,
            )

    # OCR symbol overlay (green boxes around symbol-bearing tokens)
    ocr_symbol_overlay_path = None
    if reconcile_result is not None and draw_symbol_overlay is not None:
        if cfg.ocr_reconcile_debug:
            ocr_symbol_overlay_path = (
                run_dir / "overlays" / f"{pdf_stem}_page_{page_num}_symbols.png"
            )
            draw_symbol_overlay(
                result=reconcile_result,
                page_width=page_w,
                page_height=page_h,
                out_path=ocr_symbol_overlay_path,
                scale=scale,
                background=bg_img,
            )

    # Return page results for manifest (don't write manifest here)
    page_result = {
        "page": page_num,
        "page_width": page_w,
        "page_height": page_h,
        "skew_degrees": skew,
        "stages": {name: sr.to_dict() for name, sr in stage_results.items()},
        "counts": {
            "boxes": len(boxes),
            "rows": sum(len(blk.rows) for blk in blocks),
            "lines": sum(len(blk.lines or []) for blk in blocks),
            "blocks": len(blocks),
            "tables": sum(1 for b in blocks if b.is_table),
            "notes_columns": len(notes_columns),
            "graphics": len(graphics),
            "filtered_graphics": len(filtered_graphics),
            "abbreviation_regions": len(abbreviation_regions),
            "abbreviation_entries": sum(len(ab.entries) for ab in abbreviation_regions),
            "legend_regions": len(legend_regions),
            "legend_entries": sum(len(leg.entries) for leg in legend_regions),
            "revision_regions": len(revision_regions),
            "revision_entries": sum(len(r.entries) for r in revision_regions),
            "misc_title_regions": len(misc_title_regions),
            "standard_detail_regions": len(standard_detail_regions),
            "standard_detail_entries": sum(
                len(sd.entries) for sd in standard_detail_regions
            ),
            "ocr_reconcile_accepted": (
                len(reconcile_result.added_tokens) if reconcile_result else 0
            ),
            "ocr_reconcile_total": (
                reconcile_result.stats.get("ocr_total", 0) if reconcile_result else 0
            ),
            "ocr_reconcile_candidates": (
                reconcile_result.stats.get("candidates_generated", 0)
                if reconcile_result
                else 0
            ),
            "ocr_reconcile_candidates_accepted": (
                reconcile_result.stats.get("candidates_accepted", 0)
                if reconcile_result
                else 0
            ),
            "ocr_reconcile_candidates_rejected": (
                reconcile_result.stats.get("candidates_rejected", 0)
                if reconcile_result
                else 0
            ),
            "ocr_reconcile_filtered_non_numeric": (
                reconcile_result.stats.get("filtered_non_numeric", 0)
                if reconcile_result
                else 0
            ),
        },
        "artifacts": {
            "boxes_json": str(boxes_path),
            "overlay_png": str(overlay_path),
            "legends_json": str(legends_path),
            "abbreviations_json": str(abbrev_path),
            "revisions_json": str(revisions_path),
            "standard_details_json": str(std_details_path),
            "zones_json": str(zones_path),
        },
        "zones": zone_summary(page_zones),
    }
    if ocr_reconcile_debug_path:
        page_result["artifacts"]["ocr_reconcile_png"] = str(ocr_reconcile_debug_path)

    # Include injection_log in manifest when ocr_debug is enabled (truncated)
    if (
        reconcile_result is not None
        and cfg.ocr_reconcile_debug
        and reconcile_result.stats
    ):
        inj_log = reconcile_result.stats.get("injection_log", [])
        page_result["ocr_injection_log"] = inj_log[:50]

    # ── Export structured CSVs ────────────────────────────────────────
    try:
        page_exports = export_page_results(page_result, run_dir, pdf_stem)
        page_result["exports"] = page_exports
    except Exception as exc:  # pragma: no cover
        print(f"  page {page_num}: export warning: {exc}", flush=True)

    print(f"  page {page_num}: done", flush=True)
    print(summarize(blocks), flush=True)
    return page_result


def run_pdf(
    pdf: Path,
    start: int,
    end: int | None,
    resolution: int,
    run_root: Path,
    run_prefix: str,
    color_overrides: dict | None = None,
    cfg: GroupingConfig | None = None,
) -> Path:
    """Process pages of a single PDF and create one run folder with all results."""
    if cfg is None:
        cfg = GroupingConfig()

    with pdfplumber.open(pdf) as pdf_doc:
        total_pages = len(pdf_doc.pages)
    end_page = end if end is not None else total_pages

    # Create single run folder for this PDF
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{stamp}_{run_prefix}"
    run_dir = make_run_dir(run_root, run_name)

    print(f"Processing {pdf.name} -> {run_dir}", flush=True)

    page_results = []
    for page_num in range(start, end_page):
        try:
            result = process_page(
                pdf,
                page_num,
                run_dir,
                resolution,
                color_overrides,
                cfg=cfg,
            )
            page_results.append(result)
        except Exception as exc:  # pragma: no cover
            print(f"  page {page_num}: ERROR {exc}", flush=True)
            page_results.append({"page": page_num, "error": str(exc)})

    # Write single manifest for entire run
    from plancheck.pipeline import input_fingerprint

    pages_list = list(range(start, end_page))
    manifest = {
        "run_id": run_dir.name,
        "created_at": datetime.now().isoformat(),
        "source_pdf": str(pdf.resolve()),
        "pdf_name": pdf.name,
        "input_fingerprint": input_fingerprint(pdf, pages_list, cfg),
        "render_resolution_dpi": resolution,
        "overlay_scale": resolution / 72.0,
        "config_snapshot": vars(cfg),
        # Keep legacy key for backward compat
        "settings": vars(cfg),
        "pages_processed": pages_list,
        "pages": page_results,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Run complete: {run_dir}", flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch run pages of a PDF")
    parser.add_argument("pdf", type=Path, help="Path to PDF")
    parser.add_argument("--start", type=int, default=0, help="Start page (inclusive)")
    parser.add_argument(
        "--end", type=int, default=None, help="End page (exclusive); default = all"
    )
    parser.add_argument(
        "--resolution", type=int, default=200, help="Overlay render DPI"
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default=None,
        help="Prefix for run folder name (default: derived from PDF name)",
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("runs"), help="Root directory for runs"
    )
    parser.add_argument(
        "--keep-runs",
        type=int,
        default=50,
        help="Number of runs to keep (oldest deleted)",
    )
    parser.add_argument(
        "--ocr-full-reconcile",
        action="store_true",
        default=False,
        help="Enable full-page OCR reconciliation (inject missing %% / ° ± symbols)",
    )
    parser.add_argument(
        "--ocr-debug",
        action="store_true",
        default=False,
        help="Force OCR reconcile debug overlay even when no tokens are injected",
    )
    parser.add_argument(
        "--ocr-resolution",
        type=int,
        default=300,
        help="DPI for OCR page render (default 300; use 120 to avoid Paddle resize)",
    )
    parser.add_argument(
        "--ocr-preprocess",
        action="store_true",
        default=False,
        help="Preprocess OCR image (grayscale + CLAHE contrast) before PaddleOCR",
    )
    args = parser.parse_args()

    cfg = GroupingConfig(
        enable_ocr_reconcile=args.ocr_full_reconcile,
        ocr_reconcile_debug=args.ocr_debug,
        ocr_reconcile_resolution=args.ocr_resolution,
        enable_ocr_preprocess=args.ocr_preprocess,
    )

    # Derive run prefix from PDF name if not provided
    run_prefix = args.run_prefix or args.pdf.stem.replace(" ", "_")[:20]

    run_pdf(
        pdf=args.pdf,
        start=args.start,
        end=args.end,
        resolution=args.resolution,
        run_root=args.run_root,
        run_prefix=run_prefix,
        cfg=cfg,
    )

    # Cleanup old runs
    cleanup_old_runs(args.run_root, args.keep_runs)


if __name__ == "__main__":
    main()
