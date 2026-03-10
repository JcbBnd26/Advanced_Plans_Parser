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
from typing import (TYPE_CHECKING, Any, Callable, Dict, Generator, List,
                    Optional)

from .config import GroupingConfig
from .document_checks import _run_document_checks  # noqa: F401 — re-export
from .ml_feedback import (_apply_ml_feedback,  # noqa: F401 — re-export
                          _bbox_iou)
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

    # Stages that always run unconditionally.
    if stage in ("ingest", "grouping", "analysis", "checks", "export"):
        return True, None

    if stage == "tocr":
        if not cfg.enable_tocr:
            return False, SkipReason.disabled_by_config.value
        if not inputs.get("has_pdf", True):
            return False, SkipReason.missing_inputs.value
        return True, None

    if stage == "vocrpp":
        if not cfg.enable_ocr_preprocess:
            return False, SkipReason.disabled_by_config.value
        if not cfg.enable_vocr:
            return False, SkipReason.disabled_by_config.value
        if not _has_cv2():
            return False, SkipReason.missing_dependency.value
        return True, None

    if stage == "vocr_candidates":
        if not cfg.enable_vocr_candidates:
            return False, SkipReason.disabled_by_config.value
        if not cfg.enable_vocr:
            return False, SkipReason.disabled_by_config.value
        return True, None

    if stage == "vocr":
        if not cfg.enable_vocr:
            return False, SkipReason.disabled_by_config.value
        # Backend availability is checked lazily by get_ocr_backend()
        # which raises ImportError with a clear message if Surya is missing.
        return True, None

    if stage == "reconcile":
        if not cfg.enable_ocr_reconcile:
            return False, SkipReason.disabled_by_config.value
        if not cfg.enable_vocr:
            # Reconcile requires VOCR tokens to merge.
            return False, SkipReason.disabled_by_config.value
        if inputs.get("vocr_failed") and not inputs.get("tocr_tokens"):
            return False, SkipReason.upstream_failed.value
        return True, None

    # Unknown stage — treat as not applicable.
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
            log.debug("Stage callback failed: %s %s", stage, status, exc_info=True)

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

    The PDF is opened **exactly once** via :func:`build_page_context`;
    all stages consume the pre-extracted :class:`PageContext`.

    Each stage is delegated to a ``_run_*_stage`` helper for
    maintainability; this function handles only orchestration.

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
    from .pipeline_stages import (_run_analysis_stage, _run_checks_stage,
                                  _run_grouping_stage, _run_ingest_stage,
                                  _run_prune_deskew, _run_reconcile_stage,
                                  _run_tocr_vocrpp_stages,
                                  _run_vocr_candidates_stage, _run_vocr_stage)

    if cfg is None:
        cfg = GroupingConfig()

    # ── Single PDF open: build PageContext ──────────────────────────
    from .ingest import build_page_context
    from .tocr.extract import build_extract_words_kwargs

    # Determine OCR resolution (0 = skip OCR image render)
    ocr_res = 0
    if cfg.enable_vocr or cfg.enable_ocr_preprocess or cfg.enable_ocr_reconcile:
        ocr_res = (
            cfg.vocr_resolution
            if cfg.vocr_resolution > 0
            else cfg.ocr_reconcile_resolution
        )

    ctx = build_page_context(
        pdf_path,
        page_num,
        overlay_resolution=resolution,
        ocr_resolution=ocr_res,
        extract_words_kwargs=build_extract_words_kwargs(cfg, mode="full"),
    )

    pr = PageResult(page=page_num)

    # Stage 1: ingest
    _run_ingest_stage(pr, ctx, cfg, resolution)

    # Stages 2+3: tocr → vocrpp (sequential)
    boxes, page_w, page_h, preprocess_img = _run_tocr_vocrpp_stages(
        pr,
        ctx,
        cfg,
    )

    # Prune + optional deskew
    boxes, skew = _run_prune_deskew(boxes, cfg, page_w, page_h)
    pr.skew_degrees = skew

    # Stage 3.5: vocr candidate detection (targeted patch selection)
    _run_vocr_candidates_stage(pr, ctx, cfg, boxes, page_w, page_h)

    # Stage 4: vocr (targeted or full-page depending on candidates)
    ocr_tokens, ocr_confs = _run_vocr_stage(
        pr,
        ctx,
        cfg,
        page_w,
        page_h,
        preprocess_img,
    )

    # Stage 5: reconcile
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

    # Stage 6: grouping
    blocks, notes_columns = _run_grouping_stage(pr, cfg, boxes, page_h)

    # Stage 7: analysis
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

    # Stage 8: checks
    findings = _run_checks_stage(pr, cfg, page_num)

    # Optional: persist detections to correction store
    if correction_store is not None and run_id is not None:
        from .analysis.zoning import classify_blocks
        from .corrections.features import featurize, featurize_region

        doc_id = correction_store.register_document(pdf_path)

        # Purge old pipeline detections for this document so the DB
        # only keeps the freshest run (manual annotations preserved).
        correction_store.purge_old_detections_for_doc(doc_id, keep_run_id=run_id)

        # Build zone map: block index → ZoneTag.value
        block_zone_map: dict[int, str] = {}
        if hasattr(pr, "page_zones") and pr.page_zones:
            _zone_assignments = classify_blocks(pr.blocks, pr.page_zones)
            block_zone_map = {idx: tag.value for idx, tag in _zone_assignments.items()}

        # Block-level detections
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

        # Region-level detections
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
                except (
                    Exception
                ):  # noqa: BLE001 — fallback for missing/broken header_text
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

        # Title blocks (bbox is a field, not a method)
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

        # ── ML feedback loop ────────────────────────────────────────
        #
        # 1.  Apply prior corrections: if the user already corrected a
        #     detection on this page (from a previous run), carry that
        #     label/bbox forward using spatial matching.
        # 2.  ML re-labelling: if a trained classifier exists and it
        #     disagrees with the rule-based label at high confidence,
        #     override the label so the user sees improved results.
        # 3.  Confidence scoring: write model confidence to each
        #     detection row for display (coloured dots in the GUI).
        #
        _drift_warnings = _apply_ml_feedback(
            correction_store,
            doc_id,
            page_num,
            cfg,
            page_image=pr.background_image,
        )
        pr.drift_warnings = _drift_warnings

    # Stage 9: export (no-op in library mode)
    with run_stage("export", cfg) as sr_exp:
        sr_exp.status = "success"
        sr_exp.counts = {"note": "library mode — no file I/O"}
    pr.stages["export"] = sr_exp

    log.info("run_pipeline page %d: %d findings", page_num, len(findings))
    return pr


# ── Document-level runner ──────────────────────────────────────────────


def run_document(
    pdf_path: Path,
    pages: List[int] | None = None,
    cfg: GroupingConfig | None = None,
    resolution: int = 200,
) -> DocumentResult:
    """Process multiple pages and run cross-page checks.

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

    for pg in pages:
        try:
            pr = run_pipeline(pdf_path, pg, cfg=cfg, resolution=resolution)
            dr.pages.append(pr)
        except Exception as exc:
            log.error("run_document page %d failed: %s", pg, exc)
            # Create a minimal failed PageResult
            failed = PageResult(page=pg)
            failed.stages["error"] = StageResult(
                stage="pipeline",
                status="failed",
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            dr.pages.append(failed)

    # Cross-page checks
    dr.document_findings = _run_document_checks(dr.pages)

    # ── Optional GNN refinement ──────────────────────────────────
    graph = None
    if cfg.ml_gnn_enabled:
        try:
            from .analysis.gnn import (build_document_graph, is_gnn_available,
                                       predict_with_gnn)

            if is_gnn_available():
                graph = build_document_graph(
                    dr.pages,
                    include_embeddings=cfg.ml_embeddings_enabled,
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
    if cfg.vocr_cand_gnn_prior_enabled and graph is not None:
        try:
            from .analysis.gnn.model import is_gnn_available, load_gnn
            from .vocr.gnn_candidate_prior import (apply_gnn_prior,
                                                   load_gnn_candidate_prior)

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
