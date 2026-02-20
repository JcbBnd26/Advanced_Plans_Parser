"""Core geometry-first grouping package for plan checking.

Frequently-used symbols are re-exported here for convenience.
For specialised imports (analysis detectors, CSV exporters, font
metrics, reconciliation internals, etc.) import directly from
the relevant submodule — e.g.::

    from plancheck.analysis.structural_boxes import detect_structural_boxes
    from plancheck.export.csv_export import export_notes_csv
    from plancheck.reconcile import reconcile_ocr
"""

# ── Core models & config ──────────────────────────────────────────────

from .analysis.zoning import zone_summary
from .config import GroupingConfig
from .export.overlay import draw_overlay
from .grouping import build_clusters_v2, group_blocks, group_rows, mark_tables
from .ingest import IngestError, PdfMeta, ingest_pdf, render_page_image
from .models import BlockCluster, GlyphBox, Line, RowBand, Span, SuspectRegion
from .pipeline import (
    DocumentResult,
    PageResult,
    StageResult,
    run_document,
    run_pipeline,
)
from .tocr.extract import extract_tocr_page
from .tocr.preprocess import estimate_skew_degrees, nms_prune, rotate_boxes

# ── Grouping ──────────────────────────────────────────────────────────


# ── Pipeline ──────────────────────────────────────────────────────────


# ── Ingest ────────────────────────────────────────────────────────────


# ── Text OCR preprocessing ───────────────────────────────────────────


# ── Overlay & export ──────────────────────────────────────────────────


# ── Zoning ────────────────────────────────────────────────────────────


# ── Optional (VOCR / reconcile) ───────────────────────────────────────

try:
    from .export.reconcile_overlay import draw_reconcile_debug, draw_symbol_overlay
except ImportError:
    draw_reconcile_debug = None  # type: ignore[assignment]
    draw_symbol_overlay = None  # type: ignore[assignment]


__all__ = [
    # Models & config
    "GroupingConfig",
    "GlyphBox",
    "RowBand",
    "Line",
    "Span",
    "BlockCluster",
    "SuspectRegion",
    # Grouping
    "build_clusters_v2",
    "group_blocks",
    "group_rows",
    "mark_tables",
    # Pipeline
    "DocumentResult",
    "PageResult",
    "StageResult",
    "run_document",
    "run_pipeline",
    # Ingest
    "IngestError",
    "PdfMeta",
    "ingest_pdf",
    "render_page_image",
    # Text OCR
    "extract_tocr_page",
    "estimate_skew_degrees",
    "nms_prune",
    "rotate_boxes",
    # Overlay
    "draw_overlay",
    "draw_reconcile_debug",
    "draw_symbol_overlay",
    # Zoning
    "zone_summary",
]
