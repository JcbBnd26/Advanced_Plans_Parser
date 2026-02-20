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
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional

from .config import GroupingConfig

if TYPE_CHECKING:
    from PIL import Image

    from .analysis.structural_boxes import SemanticRegion, StructuralBox
    from .analysis.title_block import TitleBlockInfo
    from .analysis.zoning import PageZone
    from .checks.semantic_checks import CheckResult
    from .models import (
        AbbreviationRegion,
        BlockCluster,
        GlyphBox,
        GraphicElement,
        LegendRegion,
        MiscTitleRegion,
        NotesColumn,
        RevisionRegion,
        StandardDetailRegion,
    )
    from .reconcile.reconcile import ReconcileResult

logger = logging.getLogger("plancheck.pipeline")

# ── Skip reasons (exhaustive enumeration) ──────────────────────────────


class SkipReason(str, Enum):
    """Why a pipeline stage was skipped."""

    disabled_by_config = "disabled_by_config"
    missing_dependency = "missing_dependency"
    missing_inputs = "missing_inputs"
    no_pages = "no_pages"
    no_images = "no_images"
    no_tokens = "no_tokens"
    upstream_failed = "upstream_failed"
    cache_hit = "cache_hit"
    not_applicable = "not_applicable"


# ── Stage result ───────────────────────────────────────────────────────


@dataclass
class StageResult:
    """Outcome record for a single pipeline stage."""

    stage: str
    enabled: bool = False
    ran: bool = False
    status: str = "skipped"  # "success" | "skipped" | "failed"
    skip_reason: Optional[str] = None
    duration_ms: int = 0
    counts: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize stage result to a JSON-compatible dict."""
        d: Dict[str, Any] = {
            "stage": self.stage,
            "enabled": self.enabled,
            "ran": self.ran,
            "status": self.status,
        }
        if self.skip_reason is not None:
            d["skip_reason"] = self.skip_reason
        d["duration_ms"] = self.duration_ms
        if self.counts:
            d["counts"] = self.counts
        if self.inputs:
            d["inputs"] = self.inputs
        if self.outputs:
            d["outputs"] = self.outputs
        if self.error is not None:
            d["error"] = self.error
        return d


# ── Dependency probes (cached at module level) ─────────────────────────


def _has_paddleocr() -> bool:
    """Return True if PaddleOCR is importable."""
    try:
        import os

        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        import paddleocr  # noqa: F401

        return True
    except ImportError:
        return False


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

    if stage == "vocr":
        if not cfg.enable_vocr:
            return False, SkipReason.disabled_by_config.value
        if not _has_paddleocr():
            return False, SkipReason.missing_dependency.value
        return True, None

    if stage == "reconcile":
        if not cfg.enable_ocr_reconcile:
            return False, SkipReason.disabled_by_config.value
        if not cfg.enable_vocr:
            # Reconcile requires VOCR tokens to merge.
            return False, SkipReason.disabled_by_config.value
        if not _has_paddleocr():
            return False, SkipReason.missing_dependency.value
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

    sr = StageResult(stage=stage)
    # A stage is "enabled" in config even if a runtime dependency is
    # missing.  It only truly *runs* when gate approves.
    if stage == "vocrpp":
        sr.enabled = cfg.enable_ocr_preprocess and cfg.enable_vocr
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
        yield sr
        return

    sr.ran = True
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
        # Re-raise so the outer handler can decide fallback policy.
        raise
    finally:
        elapsed = time.perf_counter() - t0
        sr.duration_ms = int(elapsed * 1000)


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


# ── Page-level result container ────────────────────────────────────────


@dataclass
class PageResult:
    """Structured result from :func:`run_pipeline` for a single page.

    Contains every artefact produced by the 9-stage pipeline so that the
    caller can serialise, render overlays, or feed into cross-page checks
    without repeating any computation.
    """

    page: int = 0
    page_width: float = 0.0
    page_height: float = 0.0
    skew_degrees: float = 0.0

    # Stage results
    stages: Dict[str, StageResult] = field(default_factory=dict)

    # Core artefacts
    tokens: List[GlyphBox] = field(default_factory=list)
    blocks: List[BlockCluster] = field(default_factory=list)
    notes_columns: List[NotesColumn] = field(default_factory=list)

    # Analysis artefacts
    graphics: List[GraphicElement] = field(default_factory=list)
    structural_boxes: List[StructuralBox] = field(default_factory=list)
    semantic_regions: List[SemanticRegion] = field(default_factory=list)
    abbreviation_regions: List[AbbreviationRegion] = field(default_factory=list)
    legend_regions: List[LegendRegion] = field(default_factory=list)
    revision_regions: List[RevisionRegion] = field(default_factory=list)
    standard_detail_regions: List[StandardDetailRegion] = field(default_factory=list)
    misc_title_regions: List[MiscTitleRegion] = field(default_factory=list)
    title_blocks: List[TitleBlockInfo] = field(default_factory=list)
    page_zones: List[PageZone] = field(default_factory=list)

    # Checks
    semantic_findings: List[CheckResult] = field(default_factory=list)

    # Quality
    page_quality: float = 0.0

    # Optional OCR artefacts
    ocr_tokens: Optional[List[GlyphBox]] = None
    reconcile_result: Optional[ReconcileResult] = None
    background_image: Optional[Image.Image] = None

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return a lightweight summary suitable for JSON serialisation."""
        return {
            "page": self.page,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "skew_degrees": self.skew_degrees,
            "page_quality": self.page_quality,
            "stages": {n: sr.to_dict() for n, sr in self.stages.items()},
            "counts": {
                "tokens": len(self.tokens),
                "blocks": len(self.blocks),
                "notes_columns": len(self.notes_columns),
                "abbreviation_regions": len(self.abbreviation_regions),
                "legend_regions": len(self.legend_regions),
                "revision_regions": len(self.revision_regions),
                "standard_detail_regions": len(self.standard_detail_regions),
                "misc_title_regions": len(self.misc_title_regions),
                "title_blocks": len(self.title_blocks),
                "structural_boxes": len(self.structural_boxes),
                "semantic_findings": len(self.semantic_findings),
            },
            "semantic_findings": [
                f.to_dict() if hasattr(f, "to_dict") else str(f)
                for f in self.semantic_findings
            ],
        }


# ── Stage helpers (keep run_pipeline focused on orchestration) ─────────


def _run_ingest_stage(
    pr: PageResult,
    pdf_path: Path,
    page_num: int,
    cfg: GroupingConfig,
    resolution: int,
) -> None:
    """Stage 1: render background image."""
    from .ingest import render_page_image

    with run_stage("ingest", cfg) as sr:
        pr.background_image = render_page_image(
            pdf_path, page_num, resolution=resolution
        )
        sr.counts = {"render_dpi": resolution}
        sr.status = "success"
    pr.stages["ingest"] = sr


def _run_tocr_vocrpp_stages(
    pr: PageResult,
    pdf_path: Path,
    page_num: int,
    cfg: GroupingConfig,
) -> tuple:
    """Stages 2+3: TOCR and VOCRPP in parallel.  Returns (boxes, page_w, page_h, preprocess_img)."""
    from concurrent.futures import ThreadPoolExecutor

    from .ingest import render_page_image
    from .tocr.extract import extract_tocr_page

    boxes: list = []
    page_w = page_h = 0.0
    preprocess_img = None

    def _do_tocr():
        """Execute the text-layer OCR extraction stage."""
        nonlocal boxes, page_w, page_h
        with run_stage("tocr", cfg) as sr_t:
            result = extract_tocr_page(pdf_path, page_num, cfg, mode="full")
            b, pw, ph, diag = result.to_legacy_tuple()
            boxes[:] = b
            page_w_h = (pw, ph)
            sr_t.counts = {"tokens_total": diag.get("tokens_total", 0)}
            sr_t.status = "success" if not diag.get("error") else "failed"
            return sr_t, page_w_h

    def _do_vocrpp():
        """Execute the visual-OCR preprocessing stage."""
        nonlocal preprocess_img
        with run_stage("vocrpp", cfg) as sr_v:
            if sr_v.ran:
                from .vocrpp.preprocess import OcrPreprocessConfig
                from .vocrpp.preprocess import preprocess_image_for_ocr as _pp

                ocr_res = (
                    cfg.vocr_resolution
                    if cfg.vocr_resolution > 0
                    else cfg.ocr_reconcile_resolution
                )
                raw_img = render_page_image(pdf_path, page_num, resolution=ocr_res)
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
                preprocess_img = pp_result.image
                sr_v.counts = {"applied_steps": pp_result.applied_steps}
                sr_v.status = "success"
            return sr_v

    with ThreadPoolExecutor(max_workers=2) as pool:
        tocr_fut = pool.submit(_do_tocr)
        vocrpp_fut = pool.submit(_do_vocrpp)
        sr_tocr, (page_w, page_h) = tocr_fut.result()
        sr_vocrpp = vocrpp_fut.result()

    pr.stages["tocr"] = sr_tocr
    pr.stages["vocrpp"] = sr_vocrpp
    pr.page_width = page_w
    pr.page_height = page_h

    return boxes, page_w, page_h, preprocess_img


def _run_prune_deskew(
    boxes: list,
    cfg: GroupingConfig,
    page_w: float,
    page_h: float,
) -> tuple:
    """Prune overlapping boxes and optionally deskew.  Returns (boxes, skew)."""
    from .grouping import nms_prune
    from .tocr.preprocess import estimate_skew_degrees, rotate_boxes

    boxes = nms_prune(boxes, cfg.iou_prune)
    if cfg.enable_skew:
        skew = estimate_skew_degrees(boxes, cfg.max_skew_degrees)
        boxes = rotate_boxes(
            boxes, -skew, page_w, page_h, min_rotation=cfg.preprocess_min_rotation
        )
    else:
        skew = 0.0
    return boxes, skew


def _run_vocr_stage(
    pr: PageResult,
    pdf_path: Path,
    page_num: int,
    cfg: GroupingConfig,
    page_w: float,
    page_h: float,
    preprocess_img,
) -> tuple:
    """Stage 4: Visual OCR.  Returns (ocr_tokens, ocr_confs)."""
    from .ingest import render_page_image

    ocr_tokens = None
    ocr_confs = None
    with run_stage("vocr", cfg) as sr_vocr:
        if sr_vocr.ran:
            from .vocr import extract_vocr_tokens

            ocr_img = (
                preprocess_img
                if preprocess_img is not None
                else render_page_image(
                    pdf_path,
                    page_num,
                    resolution=(
                        cfg.vocr_resolution
                        if cfg.vocr_resolution > 0
                        else cfg.ocr_reconcile_resolution
                    ),
                )
            )
            ocr_tokens, ocr_confs = extract_vocr_tokens(
                page_image=ocr_img,
                page_num=page_num,
                page_width=page_w,
                page_height=page_h,
                cfg=cfg,
            )
            sr_vocr.counts = {"tokens_total": len(ocr_tokens)}
            sr_vocr.status = "success"
    pr.stages["vocr"] = sr_vocr
    pr.ocr_tokens = ocr_tokens
    return ocr_tokens, ocr_confs


def _run_reconcile_stage(
    pr: PageResult,
    pdf_path: Path,
    page_num: int,
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
    from .grouping import nms_prune
    from .ingest import render_page_image

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
                else render_page_image(
                    pdf_path,
                    page_num,
                    resolution=cfg.ocr_reconcile_resolution,
                )
            )
            reconcile_result = _reconcile(
                page_image=ocr_img_,
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
            sr_recon.counts = {"accepted": len(reconcile_result.added_tokens)}
            sr_recon.status = "success"
    pr.stages["reconcile"] = sr_recon
    pr.reconcile_result = reconcile_result
    return boxes


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


def _run_analysis_stage(
    pr: PageResult,
    pdf_path: Path,
    page_num: int,
    cfg: GroupingConfig,
    blocks: list,
    boxes: list,
    notes_columns: list,
    page_w: float,
    page_h: float,
) -> None:
    """Stage 7: graphics, structural boxes, regions, title blocks, zones."""
    from .analysis.legends import (
        detect_abbreviation_regions,
        detect_legend_regions,
        detect_misc_title_regions,
        detect_revision_regions,
        detect_standard_detail_regions,
        extract_graphics,
        filter_graphics_outside_regions,
    )
    from .analysis.structural_boxes import detect_semantic_regions
    from .analysis.title_block import extract_title_blocks
    from .analysis.zoning import detect_zones

    with run_stage("analysis", cfg) as sr_ana:
        graphics = extract_graphics(str(pdf_path), page_num)
        structural_boxes, semantic_regions = detect_semantic_regions(
            blocks=blocks,
            graphics=graphics,
            page_width=page_w,
            page_height=page_h,
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
            page=page_num,
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


def _run_checks_stage(
    pr: PageResult,
    cfg: GroupingConfig,
    page_num: int,
) -> list:
    """Stage 8: semantic checks.  Returns findings list."""
    from .checks.semantic_checks import run_all_checks

    with run_stage("checks", cfg) as sr_chk:
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
        )
        sr_chk.counts = {"findings": len(findings)}
        sr_chk.status = "success"
    pr.stages["checks"] = sr_chk
    pr.semantic_findings = findings
    return findings


# ── Single-page pipeline runner ────────────────────────────────────────


def run_pipeline(
    pdf_path: Path,
    page_num: int,
    cfg: GroupingConfig | None = None,
    resolution: int = 200,
) -> PageResult:
    """Run the full 9-stage pipeline on a single page and return results.

    This is the **library-grade** entry point.  It performs no file I/O
    (no JSON writes, no overlay renders) — it only returns structured
    Python objects.  Callers (scripts, GUI, tests) are responsible for
    serialisation and overlay production.

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
    if cfg is None:
        cfg = GroupingConfig()

    pr = PageResult(page=page_num)

    # Stage 1: ingest
    _run_ingest_stage(pr, pdf_path, page_num, cfg, resolution)

    # Stages 2+3: tocr ‖ vocrpp (parallel)
    boxes, page_w, page_h, preprocess_img = _run_tocr_vocrpp_stages(
        pr,
        pdf_path,
        page_num,
        cfg,
    )

    # Prune + optional deskew
    boxes, skew = _run_prune_deskew(boxes, cfg, page_w, page_h)
    pr.skew_degrees = skew

    # Stage 4: vocr
    ocr_tokens, ocr_confs = _run_vocr_stage(
        pr,
        pdf_path,
        page_num,
        cfg,
        page_w,
        page_h,
        preprocess_img,
    )

    # Stage 5: reconcile
    boxes = _run_reconcile_stage(
        pr,
        pdf_path,
        page_num,
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
        pdf_path,
        page_num,
        cfg,
        blocks,
        boxes,
        notes_columns,
        page_w,
        page_h,
    )

    # Stage 8: checks
    findings = _run_checks_stage(pr, cfg, page_num)

    # Stage 9: export (no-op in library mode)
    with run_stage("export", cfg) as sr_exp:
        sr_exp.status = "success"
        sr_exp.counts = {"note": "library mode — no file I/O"}
    pr.stages["export"] = sr_exp

    logger.info("run_pipeline page %d: %d findings", page_num, len(findings))
    return pr


# ── Document-level result ──────────────────────────────────────────────


@dataclass
class DocumentResult:
    """Aggregated result for a multi-page document run."""

    pdf_path: Optional[Path] = None
    pages: List[PageResult] = field(default_factory=list)
    document_findings: List[CheckResult] = field(default_factory=list)
    config: Optional[GroupingConfig] = None

    def total_findings(self) -> int:
        """Total semantic findings across all pages + document-level."""
        return sum(len(pr.semantic_findings) for pr in self.pages) + len(
            self.document_findings
        )

    def to_summary_dict(self) -> Dict[str, Any]:
        """Serialize document result to a summary dict."""
        return {
            "pdf": str(self.pdf_path) if self.pdf_path else None,
            "pages_processed": len(self.pages),
            "total_page_findings": sum(len(pr.semantic_findings) for pr in self.pages),
            "document_findings": len(self.document_findings),
            "pages": [pr.to_summary_dict() for pr in self.pages],
            "document_level_findings": [
                f.to_dict() if hasattr(f, "to_dict") else str(f)
                for f in self.document_findings
            ],
        }


# ── Cross-page checks ─────────────────────────────────────────────────


def _run_document_checks(pages: List[PageResult]) -> List[Any]:
    """Run document-level (cross-page) checks.

    These checks look at data across pages and flag inconsistencies that
    single-page checks cannot detect.
    """
    from .checks.semantic_checks import CheckResult

    findings: List[CheckResult] = []

    if not pages:
        return findings

    # 1. Duplicate sheet numbers across pages
    sheet_map: Dict[str, List[int]] = {}
    for pr in pages:
        for tb in pr.title_blocks:
            sn = tb.sheet_number.strip().upper()
            if sn:
                sheet_map.setdefault(sn, []).append(pr.page)
    for sn, pg_list in sheet_map.items():
        if len(pg_list) > 1:
            findings.append(
                CheckResult(
                    check_id="DOC_DUP_SHEET",
                    severity="error",
                    message=(f"Sheet number '{sn}' appears on pages " f"{pg_list}"),
                    details={"sheet_number": sn, "pages": pg_list},
                )
            )

    # 2. Abbreviation consistency: same code should mean the same thing
    global_abbrevs: Dict[str, Dict[str, List[int]]] = {}
    for pr in pages:
        for region in pr.abbreviation_regions:
            for entry in region.entries:
                code = entry.code.strip().upper()
                meaning = entry.meaning.strip().upper()
                if code:
                    global_abbrevs.setdefault(code, {}).setdefault(meaning, []).append(
                        pr.page
                    )
    for code, meanings in global_abbrevs.items():
        if len(meanings) > 1:
            findings.append(
                CheckResult(
                    check_id="DOC_ABBREV_CONFLICT",
                    severity="error",
                    message=(
                        f"Abbreviation '{code}' has different meanings "
                        f"across pages: {list(meanings.keys())}"
                    ),
                    details={"code": code, "meanings": dict(meanings)},
                )
            )

    # 3. Pages missing title blocks
    pages_without_tb = [pr.page for pr in pages if not pr.title_blocks]
    if pages_without_tb and len(pages_without_tb) < len(pages):
        # Only flag if *some* pages have title blocks (mixed document)
        findings.append(
            CheckResult(
                check_id="DOC_TITLE_INCONSISTENT",
                severity="info",
                message=(
                    f"Pages {pages_without_tb} have no title block "
                    f"while other pages do"
                ),
                details={"pages_without": pages_without_tb},
            )
        )

    # 4. Legend consistency: same symbol description shouldn't change
    global_legends: Dict[str, Dict[str, List[int]]] = {}
    for pr in pages:
        for region in pr.legend_regions:
            for entry in region.entries:
                desc = entry.description.strip().upper()
                sym_key = ""
                if entry.symbol and entry.symbol.element_type:
                    sb = entry.symbol_bbox or (
                        entry.symbol.x0,
                        entry.symbol.y0,
                        entry.symbol.x1,
                        entry.symbol.y1,
                    )
                    sym_key = f"{entry.symbol.element_type}_{sb}"
                if sym_key and desc:
                    global_legends.setdefault(sym_key, {}).setdefault(desc, []).append(
                        pr.page
                    )
    for sym_key, descs in global_legends.items():
        if len(descs) > 1:
            findings.append(
                CheckResult(
                    check_id="DOC_LEGEND_CONFLICT",
                    severity="warning",
                    message=(
                        f"Legend symbol '{sym_key}' has different "
                        f"descriptions across pages: {list(descs.keys())}"
                    ),
                    details={"symbol": sym_key, "descriptions": dict(descs)},
                )
            )

    # 5. Revision sequence gaps: look for missing revision numbers
    all_rev_nums: List[int] = []
    for pr in pages:
        for region in pr.revision_regions:
            for entry in region.entries:
                num_str = entry.number.strip()
                if num_str.isdigit():
                    all_rev_nums.append(int(num_str))
    if all_rev_nums:
        rev_set = set(all_rev_nums)
        max_rev = max(rev_set)
        missing = [n for n in range(1, max_rev + 1) if n not in rev_set]
        if missing:
            findings.append(
                CheckResult(
                    check_id="DOC_REVISION_GAP",
                    severity="warning",
                    message=(
                        f"Revision sequence has gaps: missing "
                        f"revision(s) {missing} (found {sorted(rev_set)})"
                    ),
                    details={"missing": missing, "found": sorted(rev_set)},
                )
            )

    # 6. Page quality outliers: flag pages significantly below average
    quality_scores = [pr.page_quality for pr in pages if pr.page_quality > 0]
    if len(quality_scores) >= 3:
        avg_q = sum(quality_scores) / len(quality_scores)
        threshold = avg_q * 0.6  # 60% of average is suspicious
        low_quality = [
            (pr.page, pr.page_quality)
            for pr in pages
            if 0 < pr.page_quality < threshold
        ]
        if low_quality:
            findings.append(
                CheckResult(
                    check_id="DOC_LOW_QUALITY",
                    severity="info",
                    message=(
                        f"Pages with unusually low quality: "
                        f"{[(p, f'{q:.0%}') for p, q in low_quality]} "
                        f"(avg {avg_q:.0%})"
                    ),
                    details={
                        "low_pages": low_quality,
                        "average_quality": avg_q,
                    },
                )
            )

    return findings


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
            logger.error("run_document page %d failed: %s", pg, exc)
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

    logger.info(
        "run_document: %d pages, %d doc findings",
        len(dr.pages),
        len(dr.document_findings),
    )
    return dr
