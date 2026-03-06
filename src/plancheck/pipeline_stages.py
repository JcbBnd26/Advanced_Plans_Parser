"""Individual pipeline stage helper functions.

Extracted from :mod:`plancheck.pipeline` for maintainability.
Each ``_run_*_stage`` function executes a single pipeline stage,
records its outcome in the :class:`StageResult`, and mutates the
:class:`PageResult` in-place.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .config import GroupingConfig
from .page_result import PageResult, StageResult
from .pipeline import run_stage

if TYPE_CHECKING:
    from .ingest.ingest import PageContext

log = logging.getLogger(__name__)


# ── Stage 1: Ingest ────────────────────────────────────────────────────


def _run_ingest_stage(
    pr: PageResult,
    ctx: "PageContext",
    cfg: GroupingConfig,
    resolution: int,
) -> None:
    """Stage 1: render background image (from PageContext)."""

    with run_stage("ingest", cfg) as sr:
        pr.background_image = ctx.background_image
        sr.counts = {"render_dpi": resolution}
        sr.status = "success"
    pr.stages["ingest"] = sr


# ── Stages 2 + 3: TOCR and VOCRPP ─────────────────────────────────────


def _run_tocr_vocrpp_stages(
    pr: PageResult,
    ctx: "PageContext",
    cfg: GroupingConfig,
) -> tuple:
    """Stages 2+3: TOCR then VOCRPP (sequential).  Returns (boxes, page_w, page_h, preprocess_img)."""
    from .tocr.extract import extract_tocr_from_words

    def _do_tocr():
        """Execute the text-layer OCR extraction stage.

        Returns (sr, page_w, page_h, boxes) — all outputs explicit for clarity.
        """
        with run_stage("tocr", cfg) as sr_t:
            result = extract_tocr_from_words(
                ctx.words,
                ctx.page_num,
                ctx.page_width,
                ctx.page_height,
                cfg,
                mode="full",
            )
            b, pw, ph, diag = result.to_legacy_tuple()
            sr_t.counts = {"tokens_total": diag.get("tokens_total", 0)}
            sr_t.status = "success" if not diag.get("error") else "failed"
            return sr_t, pw, ph, b

    def _do_vocrpp(raw_img):
        """Execute the visual-OCR preprocessing stage.

        Returns (sr, preprocess_img) — all outputs explicit for clarity.
        """
        with run_stage("vocrpp", cfg) as sr_v:
            pp_img = None
            if sr_v.ran:
                from .vocrpp.preprocess import OcrPreprocessConfig
                from .vocrpp.preprocess import preprocess_image_for_ocr as _pp

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
                pp_result = _pp(raw_img, cfg=pp_cfg)
                pp_img = pp_result.image
                sr_v.counts = {"applied_steps": pp_result.applied_steps}
                sr_v.status = "success"
            return sr_v, pp_img

    # Sequential execution — no thread pool overhead on CPU
    sr_tocr, page_w, page_h, boxes = _do_tocr()
    raw_img = ctx.ocr_image if ctx.ocr_image is not None else ctx.background_image
    sr_vocrpp, preprocess_img = _do_vocrpp(raw_img)

    pr.stages["tocr"] = sr_tocr
    pr.stages["vocrpp"] = sr_vocrpp
    pr.page_width = page_w
    pr.page_height = page_h

    return boxes, page_w, page_h, preprocess_img


# ── Prune + deskew ─────────────────────────────────────────────────────


def _run_prune_deskew(
    boxes: list,
    cfg: GroupingConfig,
    page_w: float,
    page_h: float,
) -> tuple:
    """Prune overlapping boxes and optionally deskew.  Returns (boxes, skew)."""
    from .tocr.preprocess import estimate_skew_degrees, nms_prune, rotate_boxes

    boxes = nms_prune(boxes, cfg.iou_prune)
    if cfg.enable_skew:
        skew = estimate_skew_degrees(boxes, cfg.max_skew_degrees)
        boxes = rotate_boxes(
            boxes, -skew, page_w, page_h, min_rotation=cfg.preprocess_min_rotation
        )
    else:
        skew = 0.0
    return boxes, skew


# ── Stage 3.5: VOCR candidate detection ───────────────────────────────


def _run_vocr_candidates_stage(
    pr: PageResult,
    ctx: "PageContext",
    cfg: GroupingConfig,
    boxes: list,
    page_w: float,
    page_h: float,
) -> list:
    """Stage 3.5: VOCR candidate detection.  Returns candidate list."""
    from .vocr.candidates import detect_vocr_candidates
    from .vocr.method_stats import load_method_stats
    from .vocr.producer_stats import load_producer_stats

    candidates: list = []
    with run_stage("vocr_candidates", cfg) as sr_cand:
        if sr_cand.ran:
            # Load persistent method stats for adaptive confidence (Level 1)
            method_stats = load_method_stats(cfg.vocr_cand_stats_path)

            # Load per-producer stats (Level 3)
            producer_stats = load_producer_stats(cfg.vocr_cand_producer_stats_path)
            producer_id = getattr(ctx, "producer_id", "")

            candidates = detect_vocr_candidates(
                tokens=boxes,
                page_chars=getattr(ctx, "chars", []),
                page_lines=ctx.lines,
                page_curves=ctx.curves,
                page_rects=ctx.rects,
                page_width=page_w,
                page_height=page_h,
                page_num=ctx.page_num,
                cfg=cfg,
                method_stats=method_stats,
                producer_stats=producer_stats,
                producer_id=producer_id,
            )
            # Per-method breakdown for stage counts
            by_method: dict = {}
            for c in candidates:
                for m in c.trigger_methods:
                    by_method[m] = by_method.get(m, 0) + 1

            # Level 2: ML classifier filtering
            ml_pruned = 0
            if cfg.vocr_cand_ml_enabled and candidates:
                from .corrections.candidate_classifier import CandidateClassifier

                clf = CandidateClassifier(Path(cfg.vocr_cand_classifier_path))
                if clf.load():
                    before = len(candidates)
                    candidates = clf.filter_candidates(
                        candidates,
                        page_w,
                        page_h,
                        threshold=cfg.vocr_cand_ml_threshold,
                    )
                    ml_pruned = before - len(candidates)

            sr_cand.counts = {
                "candidates_total": len(candidates),
                "ml_pruned": ml_pruned,
                "by_method": by_method,
            }
            sr_cand.status = "success"
    pr.stages["vocr_candidates"] = sr_cand
    pr.vocr_candidates = candidates
    return candidates


# ── Stage 4: Visual OCR ───────────────────────────────────────────────


def _run_vocr_stage(
    pr: PageResult,
    ctx: "PageContext",
    cfg: GroupingConfig,
    page_w: float,
    page_h: float,
    preprocess_img,
) -> tuple:
    """Stage 4: Visual OCR.  Returns (ocr_tokens, ocr_confs).

    When VOCR candidates are available, runs **targeted** VOCR on those
    patches only.  Otherwise falls back to full-page OCR.
    """
    ocr_tokens = None
    ocr_confs = None
    with run_stage("vocr", cfg) as sr_vocr:
        if sr_vocr.ran:
            ocr_img = (
                preprocess_img
                if preprocess_img is not None
                else (
                    ctx.ocr_image if ctx.ocr_image is not None else ctx.background_image
                )
            )

            if pr.vocr_candidates:
                # Targeted mode: scan only candidate patches
                from .vocr.targeted import extract_vocr_targeted

                ocr_tokens, ocr_confs, pr.vocr_candidates = extract_vocr_targeted(
                    page_image=ocr_img,
                    candidates=pr.vocr_candidates,
                    page_num=ctx.page_num,
                    page_width=page_w,
                    page_height=page_h,
                    cfg=cfg,
                )
                sr_vocr.counts = {
                    "tokens_total": len(ocr_tokens),
                    "mode": "targeted",
                    "patches_scanned": len(pr.vocr_candidates),
                }
            else:
                # Full-page fallback
                from .vocr import extract_vocr_tokens

                ocr_tokens, ocr_confs = extract_vocr_tokens(
                    page_image=ocr_img,
                    page_num=ctx.page_num,
                    page_width=page_w,
                    page_height=page_h,
                    cfg=cfg,
                )
                sr_vocr.counts = {
                    "tokens_total": len(ocr_tokens),
                    "mode": "full_page",
                }
            sr_vocr.status = "success"
    pr.stages["vocr"] = sr_vocr
    pr.ocr_tokens = ocr_tokens
    pr.ocr_confs = ocr_confs
    return ocr_tokens, ocr_confs


# ── Stage 5: Reconcile ────────────────────────────────────────────────


def _run_reconcile_stage(
    pr: PageResult,
    ctx: "PageContext",
    cfg: GroupingConfig,
    boxes: list,
    page_w: float,
    page_h: float,
    preprocess_img,
    ocr_tokens,
    ocr_confs,
    sr_vocr: StageResult,
) -> list:
    """Stage 5: OCR reconciliation.  Returns updated boxes list."""
    from .tocr.preprocess import nms_prune

    reconcile_result = None
    recon_inputs = {
        "vocr_failed": sr_vocr.status == "failed",
        "tocr_tokens": len(boxes),
        "vocr_tokens": len(ocr_tokens) if ocr_tokens else 0,
    }
    with run_stage("reconcile", cfg, inputs=recon_inputs) as sr_recon:
        if sr_recon.ran:
            from .reconcile import reconcile_ocr as _reconcile

            ocr_img_ = (
                preprocess_img
                if preprocess_img is not None
                else (
                    ctx.ocr_image if ctx.ocr_image is not None else ctx.background_image
                )
            )
            reconcile_result = _reconcile(
                page_image=ocr_img_,
                tokens=boxes,
                page_num=ctx.page_num,
                page_width=page_w,
                page_height=page_h,
                cfg=cfg,
                ocr_tokens=ocr_tokens,
                ocr_confs=ocr_confs,
            )
            if reconcile_result.added_tokens:
                boxes.extend(reconcile_result.added_tokens)
                boxes = nms_prune(boxes, cfg.iou_prune)
            sr_recon.counts = {"accepted": len(reconcile_result.added_tokens)}
            # Compute candidate stats if targeted VOCR was used
            if pr.vocr_candidates:
                from .vocr.candidates import compute_candidate_stats
                from .vocr.method_stats import update_method_stats

                cand_stats = compute_candidate_stats(
                    pr.vocr_candidates,
                    page_w,
                    page_h,
                )
                sr_recon.counts["candidate_stats"] = cand_stats
                # Persist per-method hit/miss stats (Level 1 adaptive)
                update_method_stats(cfg.vocr_cand_stats_path, cand_stats)

                # Level 3: persist per-producer stats
                _producer = getattr(ctx, "producer_id", "")
                if _producer:
                    from .vocr.producer_stats import update_producer_stats

                    update_producer_stats(
                        cfg.vocr_cand_producer_stats_path,
                        _producer,
                        cand_stats,
                    )

                # Level 2: persist outcomes for classifier training
                if cfg.vocr_cand_ml_enabled:
                    try:
                        from .corrections.store import CorrectionStore

                        store = CorrectionStore()
                        n_saved = store.save_candidate_outcomes_batch(
                            pr.vocr_candidates,
                            page_width=page_w,
                            page_height=page_h,
                        )
                        sr_recon.counts["outcomes_saved"] = n_saved
                        store.close()
                    except Exception as exc:  # noqa: BLE001 — non-critical outcome save
                        log.warning("Failed to save candidate outcomes: %s", exc)
            sr_recon.status = "success"
    pr.stages["reconcile"] = sr_recon
    pr.reconcile_result = reconcile_result
    return boxes


# ── Stage 6: Grouping ─────────────────────────────────────────────────


def _run_grouping_stage(
    pr: PageResult,
    cfg: GroupingConfig,
    boxes: list,
    page_h: float,
) -> tuple:
    """Stage 6: clustering and notes-column grouping.  Returns (blocks, notes_columns)."""
    from .grouping import (
        build_clusters_v2,
        group_notes_columns,
        link_continued_columns,
        mark_headers,
        mark_notes,
    )

    with run_stage("grouping", cfg) as sr_grp:
        blocks = build_clusters_v2(boxes, page_h, cfg)
        mark_headers(blocks)
        mark_notes(blocks)
        notes_columns = group_notes_columns(blocks, cfg=cfg)
        link_continued_columns(notes_columns, blocks=blocks, cfg=cfg)
        sr_grp.counts = {
            "blocks": len(blocks),
            "notes_columns": len(notes_columns),
        }
        sr_grp.status = "success"
    pr.stages["grouping"] = sr_grp
    pr.tokens = boxes
    pr.blocks = blocks
    pr.notes_columns = notes_columns
    return blocks, notes_columns


# ── Stage 7: Analysis ─────────────────────────────────────────────────


def _run_analysis_stage(
    pr: PageResult,
    ctx: "PageContext",
    cfg: GroupingConfig,
    blocks: list,
    boxes: list,
    notes_columns: list,
    page_w: float,
    page_h: float,
) -> None:
    """Stage 7: graphics, structural boxes, regions, title blocks, zones."""
    from .analysis import (
        detect_abbreviation_regions,
        detect_legend_regions,
        detect_misc_title_regions,
        detect_revision_regions,
        detect_standard_detail_regions,
        extract_graphics_from_data,
        filter_graphics_outside_regions,
    )
    from .analysis.structural_boxes import detect_semantic_regions
    from .analysis.title_block import extract_title_blocks
    from .analysis.zoning import detect_zones

    with run_stage("analysis", cfg) as sr_ana:
        graphics = extract_graphics_from_data(
            ctx.page_num,
            ctx.lines,
            ctx.rects,
            ctx.curves,
        )
        structural_boxes, semantic_regions = detect_semantic_regions(
            blocks=blocks,
            graphics=graphics,
            page_width=page_w,
            page_height=page_h,
            merge_overlapping=cfg.merge_overlapping_boxes,
        )
        abbreviation_regions = detect_abbreviation_regions(
            blocks=blocks,
            graphics=graphics,
            page_width=page_w,
            page_height=page_h,
            cfg=cfg,
        )
        exclusion_zones = [ab.bbox() for ab in abbreviation_regions]

        misc_title_regions = detect_misc_title_regions(
            blocks=blocks,
            graphics=graphics,
            page_width=page_w,
            page_height=page_h,
            exclusion_zones=exclusion_zones,
            cfg=cfg,
        )
        exclusion_zones.extend(mt.bbox() for mt in misc_title_regions)

        revision_regions = detect_revision_regions(
            blocks=blocks,
            graphics=graphics,
            page_width=page_w,
            page_height=page_h,
            exclusion_zones=exclusion_zones,
            cfg=cfg,
        )
        exclusion_zones.extend(rv.bbox() for rv in revision_regions)

        filtered_gfx = filter_graphics_outside_regions(graphics, exclusion_zones)
        legend_regions = detect_legend_regions(
            blocks=blocks,
            graphics=filtered_gfx,
            page_width=page_w,
            page_height=page_h,
            exclusion_zones=exclusion_zones,
            cfg=cfg,
        )
        standard_detail_regions = detect_standard_detail_regions(
            blocks=blocks,
            graphics=graphics,
            page_width=page_w,
            page_height=page_h,
            exclusion_zones=exclusion_zones,
            cfg=cfg,
        )
        title_block_infos = extract_title_blocks(
            structural_boxes=structural_boxes,
            blocks=blocks,
            tokens=boxes,
            page=ctx.page_num,
        )
        page_zones = detect_zones(
            page_width=page_w,
            page_height=page_h,
            blocks=blocks,
            notes_columns=notes_columns,
            legend_bboxes=[lg.bbox() for lg in legend_regions],
            abbreviation_bboxes=[ab.bbox() for ab in abbreviation_regions],
            revision_bboxes=[rv.bbox() for rv in revision_regions],
            detail_bboxes=[sd.bbox() for sd in standard_detail_regions],
            cfg=cfg,
        )
        sr_ana.counts = {
            "graphics": len(graphics),
            "structural_boxes": len(structural_boxes),
            "abbreviation_regions": len(abbreviation_regions),
            "legend_regions": len(legend_regions),
            "revision_regions": len(revision_regions),
            "standard_detail_regions": len(standard_detail_regions),
            "title_blocks": len(title_block_infos),
        }
        sr_ana.status = "success"

    pr.stages["analysis"] = sr_ana
    pr.graphics = graphics
    pr.structural_boxes = structural_boxes
    pr.semantic_regions = semantic_regions
    pr.abbreviation_regions = abbreviation_regions
    pr.legend_regions = legend_regions
    pr.revision_regions = revision_regions
    pr.standard_detail_regions = standard_detail_regions
    pr.misc_title_regions = misc_title_regions
    pr.title_blocks = title_block_infos
    pr.page_zones = page_zones

    # Optional: LayoutLMv3-based layout detection (Phase 2.2)
    if cfg.ml_layout_enabled and pr.background_image is not None:
        try:
            from .analysis.layout_model import predict_layout

            layout_preds = predict_layout(
                pr.background_image,
                boxes,
                page_w,
                page_h,
                model_name_or_path=cfg.ml_layout_model_path,
            )
            pr.layout_predictions = layout_preds
            log.info("Layout model: %d predictions", len(layout_preds))
        except Exception:  # noqa: BLE001 — optional layout model
            log.debug("Layout model prediction failed", exc_info=True)


# ── Stage 8: Checks ───────────────────────────────────────────────────


def _run_checks_stage(
    pr: PageResult,
    cfg: GroupingConfig,
    page_num: int,
) -> list:
    """Stage 8: semantic checks.  Returns findings list."""
    from .checks.semantic_checks import run_all_checks

    with run_stage("checks", cfg) as sr_chk:
        # Compute mean OCR confidence for severity attenuation
        _mean_conf = 1.0
        if pr.ocr_confs:
            _mean_conf = sum(pr.ocr_confs) / len(pr.ocr_confs)
        findings = run_all_checks(
            notes_columns=pr.notes_columns,
            abbreviation_regions=pr.abbreviation_regions,
            revision_regions=pr.revision_regions,
            standard_detail_regions=pr.standard_detail_regions,
            legend_regions=pr.legend_regions,
            misc_title_regions=pr.misc_title_regions,
            structural_boxes=pr.structural_boxes,
            title_blocks=pr.title_blocks,
            blocks=pr.blocks,
            page=page_num,
            mean_ocr_confidence=_mean_conf,
        )
        # ── Optional LLM-assisted checks ─────────────────────────────
        if cfg.enable_llm_checks:
            try:
                from .checks.llm_checks import run_llm_checks

                llm_findings = run_llm_checks(
                    notes_columns=pr.notes_columns,
                    page=page_num,
                    provider=cfg.llm_provider,
                    model=cfg.llm_model,
                    api_key=cfg.llm_api_key,
                    api_base=cfg.llm_api_base,
                    temperature=cfg.llm_temperature,
                    policy=cfg.llm_policy,
                )
                findings.extend(llm_findings)
            except Exception as exc:  # noqa: BLE001 — optional LLM checks
                log.warning("LLM checks failed: %s", exc)

        sr_chk.counts = {"findings": len(findings)}
        sr_chk.status = "success"
    pr.stages["checks"] = sr_chk
    pr.semantic_findings = findings
    return findings
