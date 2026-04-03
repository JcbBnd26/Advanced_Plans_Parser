"""Core geometry-first grouping package for plan checking.

Frequently-used symbols are re-exported here for convenience.
For specialised imports (analysis detectors, CSV exporters, font
metrics, reconciliation internals, etc.) import directly from
the relevant submodule — e.g.::

    from plancheck.analysis.structural_boxes import detect_structural_boxes
    from plancheck.export.csv_export import export_notes_csv
    from plancheck.reconcile import reconcile_ocr

The exports are resolved lazily so importing ``plancheck`` does not
eagerly import the whole pipeline stack.
"""

from __future__ import annotations

_EXPORTS: dict[str, tuple[str, str]] = {
    "GroupingConfig": ("plancheck.config", "GroupingConfig"),
    "GlyphBox": ("plancheck.models", "GlyphBox"),
    "RowBand": ("plancheck.models", "RowBand"),
    "Line": ("plancheck.models", "Line"),
    "Span": ("plancheck.models", "Span"),
    "BlockCluster": ("plancheck.models", "BlockCluster"),
    "SuspectRegion": ("plancheck.models", "SuspectRegion"),
    "build_clusters_v2": ("plancheck.grouping", "build_clusters_v2"),
    "group_blocks": ("plancheck.grouping", "group_blocks"),
    "group_rows": ("plancheck.grouping", "group_rows"),
    "mark_tables": ("plancheck.grouping", "mark_tables"),
    "DocumentResult": ("plancheck.pipeline", "DocumentResult"),
    "PageResult": ("plancheck.pipeline", "PageResult"),
    "StageResult": ("plancheck.pipeline", "StageResult"),
    "run_document": ("plancheck.pipeline", "run_document"),
    "run_pipeline": ("plancheck.pipeline", "run_pipeline"),
    "IngestError": ("plancheck.ingest", "IngestError"),
    "PageContext": ("plancheck.ingest", "PageContext"),
    "PdfMeta": ("plancheck.ingest", "PdfMeta"),
    "build_page_context": ("plancheck.ingest", "build_page_context"),
    "ingest_pdf": ("plancheck.ingest", "ingest_pdf"),
    "render_page_image": ("plancheck.ingest", "render_page_image"),
    "extract_tocr_page": ("plancheck.tocr.extract", "extract_tocr_page"),
    "estimate_skew_degrees": (
        "plancheck.tocr.preprocess",
        "estimate_skew_degrees",
    ),
    "nms_prune": ("plancheck.tocr.preprocess", "nms_prune"),
    "rotate_boxes": ("plancheck.tocr.preprocess", "rotate_boxes"),
    "draw_overlay": ("plancheck.export.overlay", "draw_overlay"),
    "zone_summary": ("plancheck.analysis.zoning", "zone_summary"),
}

_OPTIONAL_EXPORTS: dict[str, tuple[str, str]] = {
    "draw_reconcile_debug": (
        "plancheck.export.reconcile_overlay",
        "draw_reconcile_debug",
    ),
    "draw_symbol_overlay": (
        "plancheck.export.reconcile_overlay",
        "draw_symbol_overlay",
    ),
}

__all__ = list(_EXPORTS) + list(_OPTIONAL_EXPORTS)


def __getattr__(name: str):
    """Resolve public exports lazily on first access."""
    from importlib import import_module

    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value

    if name in _OPTIONAL_EXPORTS:
        module_name, attr_name = _OPTIONAL_EXPORTS[name]
        try:
            value = getattr(import_module(module_name), attr_name)
        except ImportError:
            value = None
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return module attributes including lazy public exports."""
    return sorted(set(globals()) | set(__all__))
