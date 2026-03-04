"""Data containers for pipeline stage and page-level results.

Extracted from :mod:`plancheck.pipeline` for maintainability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

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

from .config import GroupingConfig

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
            VocrCandidate,
        )
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
