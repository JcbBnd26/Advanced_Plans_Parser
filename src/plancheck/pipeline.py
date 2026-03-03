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
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Callable, Dict, Generator, List,
                    Optional)

from .config import GroupingConfig

if TYPE_CHECKING:
    from PIL import Image

    from .analysis.structural_boxes import SemanticRegion, StructuralBox
    from .analysis.title_block import TitleBlockInfo
    from .analysis.zoning import PageZone
    from .checks.semantic_checks import CheckResult
    from .ingest.ingest import PageContext
    from .models import (AbbreviationRegion, BlockCluster, GlyphBox,
                         GraphicElement, LegendRegion, MiscTitleRegion,
                         NotesColumn, RevisionRegion, StandardDetailRegion)
    from .reconcile.reconcile import ReconcileResult

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

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StageResult":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        return cls(
            stage=d["stage"],
            enabled=d.get("enabled", False),
            ran=d.get("ran", False),
            status=d.get("status", "skipped"),
            skip_reason=d.get("skip_reason"),
            duration_ms=d.get("duration_ms", 0),
            counts=d.get("counts", {}),
            inputs=d.get("inputs", {}),
            outputs=d.get("outputs", {}),
            error=d.get("error"),
        )


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
    on_stage = _ON_STAGE.get()

    def _notify(status: str) -> None:
        if on_stage is None:
            return
        try:
            on_stage(stage, status)
        except Exception:
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

    # Layout model predictions (Phase 2.2)
    layout_predictions: list = field(default_factory=list)

    # Drift warnings (Phase 4.1)
    drift_warnings: list = field(default_factory=list)

    # Checks
    semantic_findings: List[CheckResult] = field(default_factory=list)

    # Quality
    page_quality: float = 0.0

    # Optional OCR artefacts
    vocr_candidates: list = field(default_factory=list)
    ocr_tokens: Optional[List[GlyphBox]] = None
    ocr_confs: Optional[List[float]] = None
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
                "drift_warnings": len(self.drift_warnings),
                "vocr_candidates": len(self.vocr_candidates),
            },
            "semantic_findings": [
                f.to_dict() if hasattr(f, "to_dict") else str(f)
                for f in self.semantic_findings
            ],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Full serialization of PageResult to a JSON-compatible dict.

        Unlike :meth:`to_summary_dict`, this preserves all nested data
        structures so that :meth:`from_dict` can reconstruct the full
        ``PageResult``.  The ``background_image`` field is excluded
        (not JSON-serializable; regenerate from the PDF if needed).
        """
        from .analysis.structural_boxes import SemanticRegion, StructuralBox
        from .analysis.title_block import TitleBlockInfo
        from .analysis.zoning import PageZone
        from .checks.semantic_checks import CheckResult
        from .reconcile.reconcile import ReconcileResult

        blocks = self.blocks  # used for index-based references

        d: Dict[str, Any] = {
            "_version": 2,
            "page": self.page,
            "page_width": round(self.page_width, 3),
            "page_height": round(self.page_height, 3),
            "skew_degrees": round(self.skew_degrees, 4),
            "page_quality": round(self.page_quality, 4),
            "stages": {n: sr.to_dict() for n, sr in self.stages.items()},
            # Core artefacts
            "tokens": [t.to_dict() for t in self.tokens],
            "blocks": [b.to_dict() for b in self.blocks],
            "notes_columns": [nc.to_dict(blocks) for nc in self.notes_columns],
            # Graphics
            "graphics": [g.to_dict() for g in self.graphics],
            # Analysis artefacts (index-based references into blocks)
            "structural_boxes": [sb.to_dict() for sb in self.structural_boxes],
            "semantic_regions": [sr.to_dict(blocks) for sr in self.semantic_regions],
            "abbreviation_regions": [
                r.to_dict(blocks) for r in self.abbreviation_regions
            ],
            "legend_regions": [r.to_dict(blocks) for r in self.legend_regions],
            "revision_regions": [r.to_dict(blocks) for r in self.revision_regions],
            "standard_detail_regions": [
                r.to_dict(blocks) for r in self.standard_detail_regions
            ],
            "misc_title_regions": [r.to_dict(blocks) for r in self.misc_title_regions],
            "title_blocks": [tb.to_dict() for tb in self.title_blocks],
            "page_zones": [z.to_dict() for z in self.page_zones],
            # Layout / drift
            "layout_predictions": list(self.layout_predictions),
            "drift_warnings": list(self.drift_warnings),
            # Checks
            "semantic_findings": [
                f.to_dict() if hasattr(f, "to_dict") else {"raw": str(f)}
                for f in self.semantic_findings
            ],
            # OCR artefacts
            "vocr_candidates": [c.to_dict() for c in self.vocr_candidates],
            "ocr_tokens": (
                [t.to_dict() for t in self.ocr_tokens] if self.ocr_tokens else None
            ),
            "ocr_confs": list(self.ocr_confs) if self.ocr_confs else None,
            "reconcile_result": (
                self.reconcile_result.to_dict() if self.reconcile_result else None
            ),
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PageResult":
        """Reconstruct a full PageResult from a dict produced by :meth:`to_dict`."""
        from .analysis.structural_boxes import SemanticRegion, StructuralBox
        from .analysis.title_block import TitleBlockInfo
        from .analysis.zoning import PageZone
        from .checks.semantic_checks import CheckResult
        from .models import (AbbreviationRegion, BlockCluster, GlyphBox,
                             GraphicElement, LegendRegion, MiscTitleRegion,
                             NotesColumn, RevisionRegion, StandardDetailRegion,
                             VocrCandidate)
        from .reconcile.reconcile import ReconcileResult

        # 1. Tokens
        tokens = [GlyphBox.from_dict(t) for t in d.get("tokens", [])]

        # 2. Blocks (need tokens for line-based access)
        blocks = [BlockCluster.from_dict(b, tokens) for b in d.get("blocks", [])]

        # 3. Notes columns (need blocks for index-based references)
        notes_columns = [
            NotesColumn.from_dict(nc, blocks) for nc in d.get("notes_columns", [])
        ]

        # 4. Graphics
        graphics = [GraphicElement.from_dict(g) for g in d.get("graphics", [])]

        # 5. Analysis artefacts
        structural_boxes = [
            StructuralBox.from_dict(sb) for sb in d.get("structural_boxes", [])
        ]
        semantic_regions = [
            SemanticRegion.from_dict(sr, blocks) for sr in d.get("semantic_regions", [])
        ]
        abbreviation_regions = [
            AbbreviationRegion.from_dict(r, blocks)
            for r in d.get("abbreviation_regions", [])
        ]
        legend_regions = [
            LegendRegion.from_dict(r, blocks) for r in d.get("legend_regions", [])
        ]
        revision_regions = [
            RevisionRegion.from_dict(r, blocks) for r in d.get("revision_regions", [])
        ]
        standard_detail_regions = [
            StandardDetailRegion.from_dict(r, blocks)
            for r in d.get("standard_detail_regions", [])
        ]
        misc_title_regions = [
            MiscTitleRegion.from_dict(r, blocks)
            for r in d.get("misc_title_regions", [])
        ]
        title_blocks_list = [
            TitleBlockInfo.from_dict(tb) for tb in d.get("title_blocks", [])
        ]
        page_zones = [PageZone.from_dict(z) for z in d.get("page_zones", [])]

        # 6. Stages
        stages = {n: StageResult.from_dict(sr) for n, sr in d.get("stages", {}).items()}

        # 7. Checks
        findings_raw = d.get("semantic_findings", [])
        semantic_findings = []
        for f in findings_raw:
            if isinstance(f, dict) and "check_id" in f:
                semantic_findings.append(CheckResult.from_dict(f))

        # 8. OCR artefacts
        vocr_candidates = [
            VocrCandidate.from_dict(c) for c in d.get("vocr_candidates", [])
        ]
        ocr_tokens = (
            [GlyphBox.from_dict(t) for t in d["ocr_tokens"]]
            if d.get("ocr_tokens")
            else None
        )
        ocr_confs = list(d["ocr_confs"]) if d.get("ocr_confs") else None
        reconcile_result = (
            ReconcileResult.from_dict(d["reconcile_result"])
            if d.get("reconcile_result")
            else None
        )

        return cls(
            page=d.get("page", 0),
            page_width=d.get("page_width", 0.0),
            page_height=d.get("page_height", 0.0),
            skew_degrees=d.get("skew_degrees", 0.0),
            page_quality=d.get("page_quality", 0.0),
            stages=stages,
            tokens=tokens,
            blocks=blocks,
            notes_columns=notes_columns,
            graphics=graphics,
            structural_boxes=structural_boxes,
            semantic_regions=semantic_regions,
            abbreviation_regions=abbreviation_regions,
            legend_regions=legend_regions,
            revision_regions=revision_regions,
            standard_detail_regions=standard_detail_regions,
            misc_title_regions=misc_title_regions,
            title_blocks=title_blocks_list,
            page_zones=page_zones,
            layout_predictions=d.get("layout_predictions", []),
            drift_warnings=d.get("drift_warnings", []),
            semantic_findings=semantic_findings,
            vocr_candidates=vocr_candidates,
            ocr_tokens=ocr_tokens,
            ocr_confs=ocr_confs,
            reconcile_result=reconcile_result,
        )


# ── Stage helpers (keep run_pipeline focused on orchestration) ─────────


def _run_ingest_stage(
    pr: PageResult,
    ctx: "PageContext",
    cfg: GroupingConfig,
    resolution: int,
) -> None:
    """Stage 1: render background image (from PageContext)."""
    from .ingest import PageContext  # noqa: F811 (TYPE_CHECKING guard)

    with run_stage("ingest", cfg) as sr:
        pr.background_image = ctx.background_image
        sr.counts = {"render_dpi": resolution}
        sr.status = "success"
    pr.stages["ingest"] = sr


def _run_tocr_vocrpp_stages(
    pr: PageResult,
    ctx: "PageContext",
    cfg: GroupingConfig,
) -> tuple:
    """Stages 2+3: TOCR then VOCRPP (sequential).  Returns (boxes, page_w, page_h, preprocess_img)."""
    from .tocr.extract import extract_tocr_from_words

    boxes: list = []
    page_w = page_h = 0.0
    preprocess_img = None

    def _do_tocr():
        """Execute the text-layer OCR extraction stage."""
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

                raw_img = (
                    ctx.ocr_image if ctx.ocr_image is not None else ctx.background_image
                )
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

    # Sequential execution — no thread pool overhead on CPU
    sr_tocr, (page_w, page_h) = _do_tocr()
    sr_vocrpp = _do_vocrpp()

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
                from .corrections.candidate_classifier import \
                    CandidateClassifier

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
                    except Exception as exc:
                        log.warning("Failed to save candidate outcomes: %s", exc)
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
    from .grouping import (build_clusters_v2, group_notes_columns,
                           link_continued_columns, mark_headers, mark_notes)

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
    ctx: "PageContext",
    cfg: GroupingConfig,
    blocks: list,
    boxes: list,
    notes_columns: list,
    page_w: float,
    page_h: float,
) -> None:
    """Stage 7: graphics, structural boxes, regions, title blocks, zones."""
    from .analysis import (detect_abbreviation_regions, detect_legend_regions,
                           detect_misc_title_regions, detect_revision_regions,
                           detect_standard_detail_regions,
                           extract_graphics_from_data,
                           filter_graphics_outside_regions)
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
        except Exception:
            log.debug("Layout model prediction failed", exc_info=True)


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
            except Exception as exc:  # pragma: no cover
                log.warning("LLM checks failed: %s", exc)

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
                except Exception:
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


# ── ML feedback loop ───────────────────────────────────────────────────


def _bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Compute Intersection-over-Union between two bboxes."""
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _apply_ml_feedback(
    store: Any,
    doc_id: str,
    page_num: int,
    cfg: GroupingConfig,
    page_image: Any = None,
) -> list:
    """Apply prior corrections + ML predictions to this page's detections.

    Three passes (all best-effort — failures are logged and swallowed):

    1. **Prior corrections**: look up every correction the user previously
       made for this (doc_id, page).  Match each correction's
       ``original_bbox`` against new detections by IoU ≥ 0.5.  When
       matched, update the detection's ``element_type`` (and bbox if
       reshaped).  Delete-corrections hide the detection.

    2. **ML relabelling**: if ``cfg.ml_enabled`` is *True* and a trained
       :class:`ElementClassifier` exists, run it over every *uncorrected*
       detection.  When the model disagrees with the rule-based label
       **and** model confidence ≥ ``cfg.ml_relabel_confidence``, update
       the detection's label.

    3. **Confidence scoring**: write model confidence to every detection
       regardless of whether the label was changed.

    4. **Drift detection** (Phase 4.1): when ``cfg.ml_drift_enabled``
       is *True* and drift stats exist, check each feature vector
       against the reference distribution.

    When *page_image* is provided and ``cfg.ml_vision_enabled`` is True,
    CNN image embeddings are extracted for each detection and appended
    to the feature vector.

    Returns
    -------
    list[dict]
        Drift warning dicts (empty when drift detection is disabled).
    """
    drift_warnings: list = []
    try:
        dets = store.get_detections_for_page(doc_id, page_num)
        if not dets:
            return drift_warnings

        # ── Pass 1: prior corrections ──────────────────────────────
        prior = store.get_prior_corrections_by_bbox(doc_id, page_num)
        corrected_det_ids: set[str] = set()

        for corr in prior:
            orig_bbox = corr["original_bbox"]
            if orig_bbox is None:
                continue

            # Find best matching new detection by IoU.
            # Prefer a *different* detection than the one the correction
            # was originally linked to (so we match against new re-run
            # detections).  Fall back to the original if nothing else
            # matches (single-detection or no-re-run scenario).
            orig_det_id = corr["detection_id"]
            best_iou = 0.0
            best_det: dict | None = None
            fallback_det: dict | None = None
            for det in dets:
                iou = _bbox_iou(orig_bbox, det["bbox"])
                if det["detection_id"] == orig_det_id:
                    if iou >= 0.5:
                        fallback_det = det
                    continue
                if iou > best_iou:
                    best_iou = iou
                    best_det = det

            if best_iou < 0.5 or best_det is None:
                # No new detection matched — fall back to original
                if fallback_det is not None:
                    best_det = fallback_det
                else:
                    continue

            did = best_det["detection_id"]
            corrected_det_ids.add(did)

            if corr["correction_type"] == "delete":
                # Mark as very-low confidence so it visually fades
                store._conn.execute(
                    "UPDATE detections SET confidence = 0.0 " "WHERE detection_id = ?",
                    (did,),
                )
            elif corr["correction_type"] in ("relabel", "accept"):
                new_label = corr["corrected_label"]
                store._conn.execute(
                    "UPDATE detections SET element_type = ? " "WHERE detection_id = ?",
                    (new_label, did),
                )
            elif corr["correction_type"] == "reshape":
                new_bbox = corr["corrected_bbox"]
                new_label = corr["corrected_label"]
                store._conn.execute(
                    "UPDATE detections SET element_type = ?, "
                    "  bbox_x0 = ?, bbox_y0 = ?, bbox_x1 = ?, bbox_y1 = ? "
                    "WHERE detection_id = ?",
                    (new_label, *new_bbox, did),
                )

        store._conn.commit()

        # ── Pass 2 + 3: ML relabelling + confidence scoring ───────
        if not cfg.ml_enabled:
            return drift_warnings

        from .corrections.classifier import ElementClassifier

        clf = ElementClassifier(model_path=Path(cfg.ml_model_path))
        if not clf.model_exists():
            return

        # Optionally set up CNN image feature extractor
        img_extractor = None
        if cfg.ml_vision_enabled and page_image is not None:
            try:
                from .corrections.image_features import (ImageFeatureExtractor,
                                                         is_vision_available)

                if is_vision_available():
                    img_extractor = ImageFeatureExtractor(
                        backbone=cfg.ml_vision_backbone
                    )
            except Exception:
                log.debug("Vision feature extractor init failed", exc_info=True)

        # Optionally set up text embedder
        text_embedder = None
        if cfg.ml_embeddings_enabled:
            try:
                from .corrections.text_embeddings import (
                    TextEmbedder, is_embeddings_available)

                if is_embeddings_available():
                    text_embedder = TextEmbedder(model_name=cfg.ml_embeddings_model)
            except Exception:
                log.debug("Text embedder init failed", exc_info=True)

        # Refresh detections after pass 1 modifications
        dets = store.get_detections_for_page(doc_id, page_num)

        # Feature cache setup (Phase 4.3)
        use_cache = getattr(cfg, "ml_feature_cache_enabled", False)
        feat_version = 0
        if use_cache:
            try:
                from .corrections.classifier import FEATURE_VERSION

                feat_version = FEATURE_VERSION
            except ImportError:
                use_cache = False

        for det in dets:
            if not det["features"]:
                continue

            did = det["detection_id"]

            # Check feature cache first (Phase 4.3)
            cached_vec = None
            if use_cache:
                try:
                    raw = store.get_cached_features(did, feat_version)
                    if raw is not None:
                        cached_vec = raw
                except Exception:
                    pass

            if cached_vec is not None:
                # Use cached vector for prediction directly
                import numpy as _np

                vec = _np.array(cached_vec, dtype=_np.float64)
                pred_label, pred_conf = clf.predict_from_vector(vec)
            else:
                # Extract image features when vision is enabled
                img_feat = None
                if img_extractor is not None:
                    bbox = det["bbox"]
                    if bbox:
                        img_feat = img_extractor.extract(page_image, tuple(bbox))

                # Extract text embedding when embeddings are enabled
                text_emb = None
                if text_embedder is not None:
                    det_features = det["features"]
                    # Reconstruct text from features if available, or use
                    # a representative text snippet from the detection
                    det_text = det.get("text", "")
                    if not det_text and det_features:
                        det_text = str(det_features.get("_text", ""))
                    if det_text:
                        text_emb = text_embedder.embed(det_text)

                pred_label, pred_conf = clf.predict(
                    det["features"],
                    image_features=img_feat,
                    text_embedding=text_emb,
                )

                # Store in cache for next time (Phase 4.3)
                if use_cache:
                    try:
                        from .corrections.classifier import \
                            encode_features as _ef

                        vec = _ef(
                            det["features"],
                            image_features=img_feat,
                            text_embedding=text_emb,
                        )
                        store.cache_features(did, vec.tolist(), feat_version)
                    except Exception:
                        pass

            # Always write confidence — unless already set by prior
            # delete correction (pass 1 sets it to 0.0).
            if did not in corrected_det_ids:
                store._conn.execute(
                    "UPDATE detections SET confidence = ? " "WHERE detection_id = ?",
                    (pred_conf, did),
                )

            # Only relabel if: not already corrected by user AND
            # model disagrees AND model is confident enough
            if (
                did not in corrected_det_ids
                and pred_label != det["element_type"]
                and pred_conf >= cfg.ml_relabel_confidence
            ):
                log.info(
                    "ML relabel: %s %s → %s (conf=%.2f)",
                    did[:12],
                    det["element_type"],
                    pred_label,
                    pred_conf,
                )
                store._conn.execute(
                    "UPDATE detections SET element_type = ? " "WHERE detection_id = ?",
                    (pred_label, did),
                )

        store._conn.commit()

        # ── Pass 4: drift detection (Phase 4.1) ───────────────────
        if getattr(cfg, "ml_drift_enabled", False):
            try:
                from .corrections.classifier import encode_features
                from .corrections.drift_detection import DriftDetector

                drift_stats_path = Path(
                    getattr(cfg, "ml_drift_stats_path", "data/drift_stats.json")
                )
                if drift_stats_path.exists():
                    drift_threshold = getattr(cfg, "ml_drift_threshold", 0.3)
                    detector = DriftDetector.load(
                        drift_stats_path, threshold=drift_threshold
                    )
                    dets = store.get_detections_for_page(doc_id, page_num)
                    for det in dets:
                        if not det["features"]:
                            continue
                        vec = encode_features(det["features"])
                        dr = detector.check(vec)
                        if dr.is_drifted:
                            drift_warnings.append(
                                {
                                    "detection_id": det["detection_id"],
                                    "drift_fraction": dr.drift_fraction,
                                    "flagged_features": dr.flagged_features,
                                }
                            )
                    if drift_warnings:
                        log.info(
                            "Drift: %d/%d detections flagged on page %d",
                            len(drift_warnings),
                            len(dets),
                            page_num,
                        )
            except Exception:
                log.debug("Drift detection failed", exc_info=True)

    except Exception:
        log.debug("ML feedback failed", exc_info=True)

    return drift_warnings


# ── Document-level result ──────────────────────────────────────────────


@dataclass
class DocumentResult:
    """Aggregated result for a multi-page document run."""

    pdf_path: Optional[Path] = None
    pages: List[PageResult] = field(default_factory=list)
    document_findings: List[CheckResult] = field(default_factory=list)
    config: Optional[GroupingConfig] = None

    # GNN cross-page predictions (populated when ml_gnn_enabled)
    gnn_predictions: Optional[Any] = None
    gnn_graph_nodes: list = field(default_factory=list)

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

    def to_dict(self) -> Dict[str, Any]:
        """Full serialization of DocumentResult to a JSON-compatible dict."""
        return {
            "_version": 2,
            "pdf_path": str(self.pdf_path) if self.pdf_path else None,
            "pages": [pr.to_dict() for pr in self.pages],
            "document_findings": [
                f.to_dict() if hasattr(f, "to_dict") else {"raw": str(f)}
                for f in self.document_findings
            ],
            "config": self.config.to_dict() if self.config else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DocumentResult":
        """Reconstruct a DocumentResult from a dict produced by :meth:`to_dict`."""
        from .checks.semantic_checks import CheckResult

        pages = [PageResult.from_dict(pd) for pd in d.get("pages", [])]
        findings_raw = d.get("document_findings", [])
        document_findings = []
        for f in findings_raw:
            if isinstance(f, dict) and "check_id" in f:
                document_findings.append(CheckResult.from_dict(f))

        cfg = None
        if d.get("config"):
            cfg = GroupingConfig.from_dict(d["config"])

        pdf_path = Path(d["pdf_path"]) if d.get("pdf_path") else None

        return cls(
            pdf_path=pdf_path,
            pages=pages,
            document_findings=document_findings,
            config=cfg,
        )


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
            from .analysis.document_graph import build_document_graph
            from .analysis.gnn_model import is_gnn_available, predict_with_gnn

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
        except Exception as exc:
            log.warning("GNN refinement failed: %s", exc)

    # ── Level 4: GNN candidate prior confidence adjustment ───────
    if cfg.vocr_cand_gnn_prior_enabled and graph is not None:
        try:
            from .analysis.gnn_model import is_gnn_available, load_gnn
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
        except Exception as exc:
            log.warning("GNN candidate prior failed: %s", exc)

    log.info(
        "run_document: %d pages, %d doc findings",
        len(dr.pages),
        len(dr.document_findings),
    )
    return dr
