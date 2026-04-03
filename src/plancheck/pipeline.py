"""Pipeline stage infrastructure: gating, timing, and stage-result recording.

Provides a canonical pipeline contract for the 9-stage flow:

    ingest → tocr → vocrpp → vocr → reconcile → grouping → analysis → checks → export

Every stage produces a :class:`StageResult` that is serialised into the
run manifest. Gating logic is centralised in :func:`gate` so that
GUI / CLI / benchmark runners all behave identically.

The :func:`run_pipeline` function orchestrates a single-page pipeline run
and returns structured results without performing any file I/O, making it
suitable for embedding in scripts, GUIs, or tests.
"""

from __future__ import annotations

import hashlib
import logging
import time
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional

from .config import GroupingConfig
from .document_checks import _run_document_checks  # noqa: F401 — re-export
from .ml_feedback import _apply_ml_feedback  # noqa: F401 — re-export
from .ml_feedback import _bbox_iou

# ── Backward-compatible re-exports (moved to dedicated modules) ────────
from .page_result import PageResult  # noqa: F401 — re-export
from .page_result import DocumentResult, SkipReason, StageResult

if TYPE_CHECKING:
    from .corrections.store import CorrectionStore

log = logging.getLogger(__name__)

# Optional stage status hook (used by the GUI to update a progress bar).
# Stored in a ContextVar so callers can scope the callback via a contextmanager.
_ON_STAGE: ContextVar[Callable[[str, str], None] | None] = ContextVar(
    "_ON_STAGE", default=None
)


@contextmanager
def stage_callback_hook(
    callback: Callable[[str, str], None] | None,
) -> Generator[None, None, None]:
    """Temporarily install a stage status callback for this context.

    The callback is invoked as ``callback(stage_name, status)`` where status is
    one of: ``pending`` (caller-driven), ``running``, ``done``, ``skipped``, ``error``.
    """

    token = _ON_STAGE.set(callback)
    try:
        yield
    finally:
        _ON_STAGE.reset(token)


# ── Dependency probes (cached at module level) ─────────────────────────


def _has_cv2() -> bool:
    """Return True if OpenCV (cv2) is importable."""
    try:
        import cv2  # noqa: F401

        return True
    except ImportError:
        return False


# ── Canonical gating function ──────────────────────────────────────────

# Ordered stage names — the canonical pipeline sequence.
STAGE_ORDER: List[str] = [
    "ingest",
    "tocr",
    "vocrpp",
    "vocr_candidates",
    "vocr",
    "reconcile",
    "grouping",
    "analysis",
    "checks",
    "export",
]


def gate(
    stage: str,
    cfg: GroupingConfig,
    inputs: Dict[str, Any] | None = None,
) -> tuple[bool, Optional[str]]:
    """Decide whether *stage* should run.

    Parameters
    ----------
    stage : str
        One of :data:`STAGE_ORDER`.
    cfg : GroupingConfig
        Effective configuration for the run.
    inputs : dict, optional
        Lightweight metadata about upstream outputs (e.g.
        ``{"tocr_tokens": 1500, "vocrpp_image": True}``).

    Returns
    -------
    (should_run, skip_reason)
        *should_run* is ``True`` when the stage should execute.
        When ``False``, *skip_reason* explains why.
    """
    if inputs is None:
        inputs = {}

    should_run = True
    skip_reason: Optional[str] = None

    # Stages that always run unconditionally.
    if stage in ("ingest", "grouping", "analysis", "checks", "export"):
        return True, None

    if stage == "tocr":
        if not cfg.enable_tocr:
            should_run, skip_reason = False, SkipReason.disabled_by_config.value
        elif not inputs.get("has_pdf", True):
            should_run, skip_reason = False, SkipReason.missing_inputs.value
        if not should_run:
            log.info("gate: skipping '%s' — %s", stage, skip_reason)
            return False, skip_reason
        return True, None

    if stage == "vocrpp":
        if not cfg.enable_ocr_preprocess:
            should_run, skip_reason = False, SkipReason.disabled_by_config.value
        elif not cfg.enable_vocr:
            should_run, skip_reason = False, SkipReason.disabled_by_config.value
        elif not _has_cv2():
            should_run, skip_reason = False, SkipReason.missing_dependency.value
        if not should_run:
            log.info("gate: skipping '%s' — %s", stage, skip_reason)
            return False, skip_reason
        return True, None

    if stage == "vocr_candidates":
        if not cfg.enable_vocr_candidates:
            should_run, skip_reason = False, SkipReason.disabled_by_config.value
        elif not cfg.enable_vocr:
            should_run, skip_reason = False, SkipReason.disabled_by_config.value
        if not should_run:
            log.info("gate: skipping '%s' — %s", stage, skip_reason)
            return False, skip_reason
        return True, None

    if stage == "vocr":
        if not cfg.enable_vocr:
            should_run, skip_reason = False, SkipReason.disabled_by_config.value
            log.info("gate: skipping '%s' — %s", stage, skip_reason)
            return False, skip_reason
        # Backend availability is checked lazily by get_ocr_backend()
        # which raises ImportError with a clear message if Surya is missing.
        return True, None

    if stage == "reconcile":
        if not cfg.enable_ocr_reconcile:
            should_run, skip_reason = False, SkipReason.disabled_by_config.value
        elif not cfg.enable_vocr:
            # Reconcile requires VOCR tokens to merge.
            should_run, skip_reason = False, SkipReason.disabled_by_config.value
        elif inputs.get("vocr_failed") and not inputs.get("tocr_tokens"):
            should_run, skip_reason = False, SkipReason.upstream_failed.value
        if not should_run:
            log.info("gate: skipping '%s' — %s", stage, skip_reason)
            return False, skip_reason
        return True, None

    # Unknown stage — treat as not applicable.
    log.info("gate: skipping unknown stage '%s'", stage)
    return False, SkipReason.not_applicable.value


# ── Stage context manager ──────────────────────────────────────────────


@contextmanager
def run_stage(
    stage: str,
    cfg: GroupingConfig,
    inputs: Dict[str, Any] | None = None,
) -> Generator[StageResult, None, None]:
    """Context manager that wraps a pipeline stage with gating + timing.

    Usage::

        with run_stage("tocr", cfg) as sr:
            if sr.ran:
                # … do the work …
                sr.counts["tokens_total"] = 1234
                sr.status = "success"

    The yielded :class:`StageResult` has ``ran=True`` only when
    :func:`gate` approves the stage. The caller should check ``sr.ran``
    before doing expensive work. Timing is handled automatically.
    """
    should_run, skip_reason = gate(stage, cfg, inputs)
    on_stage = _ON_STAGE.get()

    def _notify(status: str) -> None:
        if on_stage is None:
            return
        try:
            on_stage(stage, status)
        except Exception:  # noqa: BLE001 — callbacks must not break pipeline
            # Progress reporting must never break the pipeline.
            log.warning("Stage callback failed: %s %s", stage, status, exc_info=True)

    sr = StageResult(stage=stage)
    # A stage is "enabled" in config even if a runtime dependency is
    # missing.  It only truly *runs* when gate approves.
    if stage == "vocrpp":
        sr.enabled = cfg.enable_ocr_preprocess and cfg.enable_vocr
    elif stage == "vocr_candidates":
        sr.enabled = cfg.enable_vocr_candidates and cfg.enable_vocr
    elif stage == "vocr":
        sr.enabled = cfg.enable_vocr
    elif stage == "reconcile":
        sr.enabled = cfg.enable_ocr_reconcile
    else:
        # ingest, tocr, grouping, analysis, checks, export — always enabled.
        sr.enabled = True

    if inputs:
        sr.inputs = inputs

    if not should_run:
        sr.ran = False
        sr.status = "skipped"
        sr.skip_reason = skip_reason
        _notify("skipped")
        yield sr
        return

    sr.ran = True
    _notify("running")
    t0 = time.perf_counter()
    try:
        yield sr
        # If the caller didn't explicitly set status, mark success if no error.
        if sr.status not in ("success", "failed"):
            sr.status = "success"
    except Exception as exc:
        sr.status = "failed"
        sr.error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "stack": traceback.format_exc(),
        }
        _notify("error")
        # Re-raise so the outer handler can decide fallback policy.
        raise
    finally:
        elapsed = time.perf_counter() - t0
        sr.duration_ms = int(elapsed * 1000)
        if sr.status == "success":
            _notify("done")
        elif sr.status == "failed":
            _notify("error")
        else:
            _notify("skipped")


# ── Input fingerprint (determinism aid) ────────────────────────────────


def input_fingerprint(
    pdf_path: Path,
    pages: List[int],
    cfg: GroupingConfig,
) -> str:
    """Compute a reproducibility fingerprint for a pipeline run.

    Uses file size + mtime (fast) rather than full content hash.
    """
    stat = pdf_path.stat()
    parts = [
        str(pdf_path.resolve()),
        str(stat.st_size),
        str(stat.st_mtime_ns),
        ",".join(str(p) for p in sorted(pages)),
        str(sorted(vars(cfg).items())),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


# ── Composable pipeline phase functions ─────────────────────────────────
#
# The pipeline is split into three phases so that ``run_document()`` can
# batch all pages through TOCR first, then only flagged pages through
# VOCRPP/VOCR/reconcile, then all pages through grouping/analysis/checks.
#
# ``run_pipeline()`` still calls all three phases sequentially for
# backward compatibility (single-page callers).
# ────────────────────────────────────────────────────────────────────────


def _build_page_context(
    pdf_path: Path,
    page_num: int,
    cfg: GroupingConfig,
    resolution: int,
) -> "PageContext":
    """Open the PDF once and build a PageContext for one page."""
    log.debug("_build_page_context: page %d", page_num)
    from .ingest import build_page_context
    from .tocr.extract import build_extract_words_kwargs

    ocr_res = 0
    if cfg.enable_vocr or cfg.enable_ocr_preprocess or cfg.enable_ocr_reconcile:
        ocr_res = (
            cfg.vocr_resolution
            if cfg.vocr_resolution > 0
            else cfg.ocr_reconcile_resolution
        )

    return build_page_context(
        pdf_path,
        page_num,
        overlay_resolution=resolution,
        ocr_resolution=ocr_res,
        extract_words_kwargs=build_extract_words_kwargs(cfg, mode="full"),
    )


def _run_early_stages(
    pr: "PageResult",
    ctx: "PageContext",
    cfg: GroupingConfig,
    resolution: int,
) -> tuple:
    """Phase 1: ingest → tocr → prune/deskew → vocr_candidates.

    Returns (boxes, page_w, page_h) — lightweight data safe to stash.
    """
    from .pipeline_stages import (
        _run_ingest_stage,
        _run_prune_deskew,
        _run_tocr_stage,
        _run_vector_symbol_recovery,
        _run_vocr_candidates_stage,
    )

    _run_ingest_stage(pr, ctx, cfg, resolution)

    boxes, page_w, page_h = _run_tocr_stage(pr, ctx, cfg)
    log.info(
        "_run_early_stages: page %d — tocr produced %d tokens (%.0f×%.0f)",
        ctx.page_num,
        len(boxes),
        page_w,
        page_h,
    )

    boxes = _run_vector_symbol_recovery(pr, ctx, cfg, boxes)

    boxes, skew = _run_prune_deskew(boxes, cfg, page_w, page_h)
    pr.skew_degrees = skew

    if cfg.enable_vocr:
        _run_vocr_candidates_stage(pr, ctx, cfg, boxes, page_w, page_h)
        log.info(
            "_run_early_stages: page %d — %d vocr candidates detected",
            ctx.page_num,
            len(getattr(pr, "vocr_candidates", []) or []),
        )
    else:
        pr.vocr_candidates = []

    return boxes, page_w, page_h


def _run_vocr_phases(
    pr: "PageResult",
    ctx: "PageContext",
    cfg: GroupingConfig,
    boxes: list,
    page_w: float,
    page_h: float,
) -> list:
    """Phase 2: vocrpp → vocr → reconcile.  Returns updated boxes list."""
    if not cfg.enable_vocr:
        log.info("_run_vocr_phases: skipped (VOCR quarantined)")
        return boxes
    log.info("_run_vocr_phases: entering with %d boxes", len(boxes))
    from .pipeline_stages import (
        _run_reconcile_stage,
        _run_vocr_stage,
        _run_vocrpp_stage,
    )

    preprocess_img = _run_vocrpp_stage(pr, ctx, cfg)

    ocr_tokens, ocr_confs = _run_vocr_stage(
        pr,
        ctx,
        cfg,
        page_w,
        page_h,
        preprocess_img,
    )

    boxes = _run_reconcile_stage(
        pr,
        ctx,
        cfg,
        boxes,
        page_w,
        page_h,
        preprocess_img,
        ocr_tokens,
        ocr_confs,
        pr.stages["vocr"],
    )
    log.info("_run_vocr_phases: exiting with %d boxes", len(boxes))
    return boxes


def _skip_vocr_phases(pr: "PageResult", cfg: GroupingConfig) -> None:
    """Record skipped VOCRPP/VOCR/reconcile stages for non-flagged pages."""
    for stage_name in ("vocrpp", "vocr", "reconcile"):
        sr = StageResult(stage=stage_name)
        sr.enabled = True
        sr.ran = False
        sr.status = "skipped"
        sr.skip_reason = "no_candidates"
        pr.stages[stage_name] = sr
    pr.ocr_tokens = []
    pr.ocr_confs = []


def _run_late_stages(
    pr: "PageResult",
    ctx: "PageContext",
    cfg: GroupingConfig,
    boxes: list,
    page_w: float,
    page_h: float,
    page_num: int,
    correction_store: "CorrectionStore | None" = None,
    run_id: str | None = None,
    pdf_path: Path | None = None,
) -> list:
    """Phase 3: grouping → analysis → checks → correction store → export.

    Returns the semantic findings list.
    """
    log.info("_run_late_stages: entering page %d with %d boxes", page_num, len(boxes))
    from .pipeline_stages import (
        _run_analysis_stage,
        _run_checks_stage,
        _run_grouping_stage,
    )

    blocks, notes_columns = _run_grouping_stage(pr, cfg, boxes, page_h)

    _run_analysis_stage(
        pr,
        ctx,
        cfg,
        blocks,
        boxes,
        notes_columns,
        page_w,
        page_h,
    )

    findings = _run_checks_stage(pr, cfg, page_num)
    log.info(
        "_run_late_stages: page %d — %d blocks, %d findings",
        page_num,
        len(blocks),
        len(findings),
    )

    # Optional: persist detections to correction store
    if correction_store is not None and run_id is not None and pdf_path is not None:
        _persist_corrections(
            pr,
            correction_store,
            pdf_path,
            page_num,
            run_id,
            cfg,
        )

    # Stage 9: export (no-op in library mode)
    with run_stage("export", cfg) as sr_exp:
        sr_exp.status = "success"
        sr_exp.counts = {"note": "library mode — no file I/O"}
    pr.stages["export"] = sr_exp

    return findings


def _persist_corrections(
    pr: "PageResult",
    correction_store: "CorrectionStore",
    pdf_path: Path,
    page_num: int,
    run_id: str,
    cfg: GroupingConfig,
) -> None:
    """Persist block/region detections and run ML feedback."""
    log.info("_persist_corrections: page %d", page_num)
    from .analysis.zoning import classify_blocks
    from .corrections.features import featurize, featurize_region

    doc_id = correction_store.register_document(pdf_path)
    correction_store.purge_old_detections_for_doc(doc_id, keep_run_id=run_id)

    block_zone_map: dict[int, str] = {}
    if hasattr(pr, "page_zones") and pr.page_zones:
        _zone_assignments = classify_blocks(pr.blocks, pr.page_zones)
        block_zone_map = {idx: tag.value for idx, tag in _zone_assignments.items()}

    for block_idx, block in enumerate(pr.blocks):
        lbl = getattr(block, "label", None)
        if lbl in ("note_column_header", "note_column_subheader"):
            etype = "header"
        elif lbl == "notes_block" or getattr(block, "is_notes", False):
            etype = "notes_column"
        elif getattr(block, "is_header", False):
            etype = "header"
        else:
            continue
        bb = block.bbox()
        if bb == (0, 0, 0, 0):
            continue
        zone = block_zone_map.get(block_idx, "unknown")
        features = featurize(block, pr.page_width, pr.page_height, zone=zone)
        text = " ".join(b.text for b in block.get_all_boxes())[:500]
        correction_store.save_detection(
            doc_id=doc_id,
            page=page_num,
            run_id=run_id,
            element_type=etype,
            bbox=bb,
            text_content=text,
            features=features,
            confidence=None,
        )

    for region_list, etype in [
        (pr.abbreviation_regions, "abbreviations"),
        (pr.legend_regions, "legend"),
        (pr.revision_regions, "revision"),
        (pr.standard_detail_regions, "standard_detail"),
        (pr.misc_title_regions, "misc_title"),
    ]:
        for region in region_list:
            bbox = region.bbox()
            if bbox == (0, 0, 0, 0):
                continue
            header_block = getattr(region, "header", None)
            entry_count = len(getattr(region, "entries", []))
            features = featurize_region(
                etype,
                bbox,
                header_block,
                pr.page_width,
                pr.page_height,
                entry_count=entry_count,
            )
            try:
                text_content = region.header_text()
            except Exception:  # noqa: BLE001 — fallback for missing/broken header_text
                log.warning("header_text() failed for %s region", etype, exc_info=True)
                text_content = getattr(region, "text", "")
            correction_store.save_detection(
                doc_id=doc_id,
                page=page_num,
                run_id=run_id,
                element_type=etype,
                bbox=bbox,
                text_content=text_content or "",
                features=features,
                confidence=getattr(region, "confidence", None),
            )

    for tb in pr.title_blocks:
        bbox = tb.bbox
        if bbox == (0, 0, 0, 0):
            continue
        features = featurize_region(
            "title_block",
            bbox,
            None,
            pr.page_width,
            pr.page_height,
        )
        correction_store.save_detection(
            doc_id=doc_id,
            page=page_num,
            run_id=run_id,
            element_type="title_block",
            bbox=bbox,
            text_content=tb.raw_text[:500] if tb.raw_text else "",
            features=features,
            confidence=tb.confidence,
        )

    _drift_warnings = _apply_ml_feedback(
        correction_store,
        doc_id,
        page_num,
        cfg,
        page_image=pr.background_image,
    )
    pr.drift_warnings = _drift_warnings


# ── Single-page pipeline runner ────────────────────────────────────────


def run_pipeline(
    pdf_path: Path,
    page_num: int,
    cfg: GroupingConfig | None = None,
    resolution: int = 200,
    correction_store: "CorrectionStore | None" = None,
    run_id: str | None = None,
) -> PageResult:
    """Run the full 9-stage pipeline on a single page and return results.

    This is the **library-grade** entry point.  It performs no file I/O
    (no JSON writes, no overlay renders) — it only returns structured
    Python objects.  Callers (scripts, GUI, tests) are responsible for
    serialisation and overlay production.

    The PDF is opened **exactly once** via :func:`_build_page_context`;
    all stages consume the pre-extracted :class:`PageContext`.

    Internally delegates to three composable phase functions so that
    :func:`run_document` can batch pages through each phase independently.

    Parameters
    ----------
    pdf_path : Path
        Path to the source PDF.
    page_num : int
        0-based page index.
    cfg : GroupingConfig, optional
        Pipeline configuration.  Defaults to ``GroupingConfig()``.
    resolution : int
        Render resolution in DPI for the background image.

    Returns
    -------
    PageResult
        All artefacts produced by the pipeline.
    """
    if cfg is None:
        cfg = GroupingConfig()

    ctx = _build_page_context(pdf_path, page_num, cfg, resolution)
    pr = PageResult(page=page_num)

    # Phase 1: ingest → tocr → prune/deskew → vocr_candidates
    boxes, page_w, page_h = _run_early_stages(pr, ctx, cfg, resolution)

    # Phase 2: vocrpp → vocr → reconcile
    boxes = _run_vocr_phases(pr, ctx, cfg, boxes, page_w, page_h)

    # Phase 3: grouping → analysis → checks → export
    _run_late_stages(
        pr,
        ctx,
        cfg,
        boxes,
        page_w,
        page_h,
        page_num,
        correction_store=correction_store,
        run_id=run_id,
        pdf_path=pdf_path,
    )

    log.info("run_pipeline page %d complete", page_num)
    return pr


# ── Document-level runner ──────────────────────────────────────────────


def run_document(
    pdf_path: Path,
    pages: List[int] | None = None,
    cfg: GroupingConfig | None = None,
    resolution: int = 200,
) -> DocumentResult:
    """Process multiple pages in batch-by-stage order.

    Phase 1 — all pages run through ingest → tocr → prune/deskew →
    vocr_candidates.  Phase 2 — only *flagged* pages (non-empty
    ``vocr_candidates``) run through vocrpp → vocr → reconcile;
    non-flagged pages record skipped stages.  Phase 3 — all pages run
    through grouping → analysis → checks → export.

    This avoids loading the expensive VOCR backend for documents where
    most pages have no OCR-reconciliation candidates.

    Parameters
    ----------
    pdf_path : Path
        Path to the source PDF.
    pages : list[int], optional
        0-based page indices.  ``None`` = all pages.
    cfg : GroupingConfig, optional
        Pipeline configuration.
    resolution : int
        Render DPI.

    Returns
    -------
    DocumentResult
        Per-page + document-level results.
    """
    from .ingest import ingest_pdf

    if cfg is None:
        cfg = GroupingConfig()

    meta = ingest_pdf(pdf_path)
    if pages is None:
        pages = list(range(meta.num_pages))

    dr = DocumentResult(pdf_path=pdf_path, config=cfg)

    # Per-page intermediate state needed across phases.
    # Keys: page index in the *pages* list (not the PDF page number).
    page_states: dict[int, dict] = {}

    # ── Phase 1: early stages (all pages) ────────────────────────
    for idx, pg in enumerate(pages):
        try:
            ctx = _build_page_context(pdf_path, pg, cfg, resolution)
            pr = PageResult(page=pg)
            boxes, page_w, page_h = _run_early_stages(pr, ctx, cfg, resolution)
            page_states[idx] = {
                "pr": pr,
                "ctx": ctx,
                "boxes": boxes,
                "page_w": page_w,
                "page_h": page_h,
            }
            dr.pages.append(pr)
        except Exception as exc:  # noqa: BLE001 — page failure must not abort doc
            log.error("run_document page %d Phase 1 failed: %s", pg, exc)
            failed = PageResult(page=pg)
            failed.stages["error"] = StageResult(
                stage="pipeline",
                status="failed",
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            dr.pages.append(failed)

    log.info(
        "run_document Phase 1 complete: %d/%d pages succeeded",
        len(page_states),
        len(pages),
    )

    # ── Phase 2: VOCR stages (flagged pages only) ────────────────
    # QUARANTINED: VOCR is disabled — skip entire Phase 2 when not enabled.
    vocr_enabled = cfg.enable_vocr
    if not vocr_enabled:
        log.info("run_document Phase 2: skipped (VOCR quarantined)")
        # Still free chars/words since Phase 1 is done with them
        for state in page_states.values():
            ctx = state["ctx"]
            ctx.chars = []
            ctx.words = []
            ctx.ocr_image = None
        # Record skipped VOCR stages so downstream knows
        for state in page_states.values():
            _skip_vocr_phases(state["pr"], cfg)
    else:
        # A page is "flagged" when it has vocr_candidates OR when
        # candidate detection is disabled but VOCR itself is enabled
        # (full-page fallback mode).
        candidates_disabled = not cfg.enable_vocr_candidates

        # Free heavy data no longer needed after Phase 1 to make room
        # for the VOCR model (~1–1.5 GB for Surya transformers).  The
        # chars list is only used during TOCR extraction; lines/rects/
        # curves are still needed in Phase 3 analysis so we keep them.
        for state in page_states.values():
            ctx = state["ctx"]
            ctx.chars = []
            ctx.words = []

        # Pre-release OCR images for pages that won't run VOCR.
        for idx, state in page_states.items():
            pr = state["pr"]
            has_candidates = bool(getattr(pr, "vocr_candidates", None))
            needs_vocr = vocr_enabled and (has_candidates or candidates_disabled)
            if not needs_vocr:
                state["ctx"].ocr_image = None

        import gc

        # Force garbage collection between Phase 1 (TOCR) and Phase 2 (VOCR)
        # to reclaim ~300-500 MB of char/word data freed above.  Without
        # this, Python's generational GC may defer collection until after
        # Surya's 1-1.5 GB model loads, causing peak memory to exceed
        # available RAM on 8 GB machines.
        gc.collect()

        for idx, state in page_states.items():
            pr = state["pr"]
            has_candidates = bool(getattr(pr, "vocr_candidates", None))
            needs_vocr = vocr_enabled and (has_candidates or candidates_disabled)

            if needs_vocr:
                try:
                    state["boxes"] = _run_vocr_phases(
                        pr,
                        state["ctx"],
                        cfg,
                        state["boxes"],
                        state["page_w"],
                        state["page_h"],
                    )
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "run_document page %d Phase 2 failed: %s",
                        pages[idx],
                        exc,
                    )
                finally:
                    # Release OCR image after VOCR is done with this page
                    state["ctx"].ocr_image = None
            else:
                # Record skipped VOCR stages so downstream knows
                _skip_vocr_phases(pr, cfg)

        flagged_count = sum(
            1
            for idx in page_states
            if bool(getattr(page_states[idx]["pr"], "vocr_candidates", None))
            or candidates_disabled
        )
        log.info(
            "run_document Phase 2 complete: %d/%d pages ran VOCR",
            flagged_count,
            len(page_states),
        )

    # ── Phase 3: late stages (all pages) ─────────────────────────
    for idx, state in page_states.items():
        pr = state["pr"]
        try:
            _run_late_stages(
                pr,
                state["ctx"],
                cfg,
                state["boxes"],
                state["page_w"],
                state["page_h"],
                pages[idx],
                pdf_path=pdf_path,
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "run_document page %d Phase 3 failed: %s",
                pages[idx],
                exc,
            )
        finally:
            # Release per-page context now that all stages are done
            state["ctx"] = None

    log.info("run_document Phase 3 complete")

    # Cross-page checks
    dr.document_findings = _run_document_checks(dr.pages)

    # ── Optional GNN refinement ──────────────────────────────────
    graph = None
    if cfg.ml_gnn_enabled:
        try:
            from .analysis.gnn import (
                build_document_graph,
                is_gnn_available,
                predict_with_gnn,
            )

            if is_gnn_available():
                # Only request embeddings when a text embedder is available.
                # Without a real embedder, include_embeddings would produce
                # zero-padded 398-d features instead of the expected 14-d.
                _text_embedder = None
                _include_emb = cfg.ml_embeddings_enabled
                if _include_emb:
                    try:
                        from .corrections.text_embeddings import (
                            TextEmbedder,
                            is_embeddings_available,
                        )

                        if is_embeddings_available():
                            _text_embedder = TextEmbedder(
                                model_name=cfg.ml_embeddings_model
                            )
                        else:
                            _include_emb = False
                            log.info(
                                "GNN embeddings requested but text embedder "
                                "not available — using 14-d features."
                            )
                    except Exception:  # noqa: BLE001
                        _include_emb = False
                        log.warning("Text embedder init failed for GNN", exc_info=True)

                graph = build_document_graph(
                    dr.pages,
                    include_embeddings=_include_emb,
                    text_embedder=_text_embedder,
                )
                gnn_labels = predict_with_gnn(
                    graph,
                    model_path=cfg.ml_gnn_model_path,
                )
                if gnn_labels is not None and len(gnn_labels) > 0:
                    log.info("GNN refined %d node labels", len(gnn_labels))
                    # Store GNN predictions on the DocumentResult for
                    # downstream consumers (export, GUI, etc.)
                    dr.gnn_predictions = gnn_labels
                    dr.gnn_graph_nodes = graph.get("nodes", [])
        except Exception as exc:  # noqa: BLE001 — optional GNN stage
            log.warning("GNN refinement failed: %s", exc)

    # ── Level 4: GNN candidate prior confidence adjustment ───────
    # QUARANTINED: guard on enable_vocr prevents vocr imports
    if cfg.enable_vocr and cfg.vocr_cand_gnn_prior_enabled and graph is not None:
        try:
            from .analysis.gnn.model import is_gnn_available, load_gnn
            from .vocr.gnn_candidate_prior import (
                apply_gnn_prior,
                load_gnn_candidate_prior,
            )

            if is_gnn_available():
                gnn_model = load_gnn(cfg.ml_gnn_model_path)
                prior_head = load_gnn_candidate_prior(cfg.vocr_cand_gnn_prior_path)
                if prior_head is not None:
                    adjusted = apply_gnn_prior(
                        dr.pages,
                        graph,
                        gnn_model,
                        prior_head,
                        page_width=(
                            getattr(dr.pages[0], "page_width", 612.0)
                            if dr.pages
                            else 612.0
                        ),
                        page_height=(
                            getattr(dr.pages[0], "page_height", 792.0)
                            if dr.pages
                            else 792.0
                        ),
                        blend_weight=cfg.vocr_cand_gnn_prior_blend,
                    )
                    log.info("GNN candidate prior adjusted %d candidates", adjusted)
        except Exception as exc:  # noqa: BLE001 — optional GNN prior stage
            log.warning("GNN candidate prior failed: %s", exc)

    log.info(
        "run_document: %d pages, %d doc findings",
        len(dr.pages),
        len(dr.document_findings),
    )
    return dr
