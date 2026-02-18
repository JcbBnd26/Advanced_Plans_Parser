"""Core geometry-first grouping package for plan checking."""

from .config import GroupingConfig
from .grouping import (
    build_clusters_v2,
    build_lines,
    group_blocks,
    group_blocks_from_lines,
    group_rows,
    mark_notes,
    mark_tables,
)
from .models import BlockCluster, GlyphBox, Line, RowBand, Span, SuspectRegion

try:
    from .ocr_reconcile import (
        draw_reconcile_debug,
        draw_symbol_overlay,
        extract_vocr_tokens,
        reconcile_ocr,
    )
except ImportError:
    # PaddleOCR not installed — OCR reconciliation unavailable
    reconcile_ocr = None  # type: ignore[assignment]
    draw_reconcile_debug = None  # type: ignore[assignment]
    draw_symbol_overlay = None  # type: ignore[assignment]
    extract_vocr_tokens = None  # type: ignore[assignment]

from ._structural_boxes import (
    BoxType,
    SemanticRegion,
    StructuralBox,
    classify_structural_boxes,
    create_synthetic_regions,
    detect_semantic_regions,
    detect_structural_boxes,
    mask_blocks_by_structural_boxes,
)
from .export import (
    export_abbreviations_csv,
    export_blocks_csv,
    export_from_manifest,
    export_legends_csv,
    export_notes_csv,
    export_page_results,
    export_page_summary_csv,
    export_revisions_csv,
    export_standard_details_csv,
)
from .ocr_preprocess_pipeline import (
    OcrPreprocessConfig,
    OcrPreprocessResult,
    preprocess_image_for_ocr,
)
from .overlay import draw_overlay
from .pipeline import (
    STAGE_ORDER,
    SkipReason,
    StageResult,
    gate,
    input_fingerprint,
    run_stage,
)
from .preprocess import estimate_skew_degrees, nms_prune, rotate_boxes
from .zoning import PageZone, ZoneTag, classify_blocks, detect_zones, zone_summary

__all__ = [
    "GroupingConfig",
    "GlyphBox",
    "RowBand",
    "Line",
    "Span",
    "BlockCluster",
    "SuspectRegion",
    "nms_prune",
    "estimate_skew_degrees",
    "rotate_boxes",
    "draw_overlay",
    "group_rows",
    "group_blocks",
    "group_blocks_from_lines",
    "build_lines",
    "mark_tables",
    "build_clusters_v2",
    "reconcile_ocr",
    "extract_vocr_tokens",
    "draw_reconcile_debug",
    "draw_symbol_overlay",
    "OcrPreprocessConfig",
    "OcrPreprocessResult",
    "preprocess_image_for_ocr",
    "STAGE_ORDER",
    "SkipReason",
    "StageResult",
    "gate",
    "run_stage",
    "input_fingerprint",
    "PageZone",
    "ZoneTag",
    "detect_zones",
    "classify_blocks",
    "zone_summary",
    "export_page_results",
    "export_page_summary_csv",
    "export_notes_csv",
    "export_abbreviations_csv",
    "export_legends_csv",
    "export_standard_details_csv",
    "export_revisions_csv",
    "export_blocks_csv",
    "export_from_manifest",
    "BoxType",
    "StructuralBox",
    "SemanticRegion",
    "detect_structural_boxes",
    "classify_structural_boxes",
    "create_synthetic_regions",
    "detect_semantic_regions",
    "mask_blocks_by_structural_boxes",
]
