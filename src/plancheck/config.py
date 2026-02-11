from dataclasses import dataclass


@dataclass
class GroupingConfig:
    """Tunables for geometry-first grouping."""

    # IoU threshold for pruning overlapping boxes.
    iou_prune: float = 0.5
    # Skew handling.
    enable_skew: bool = False
    max_skew_degrees: float = 3.0
    # Relative tolerances derived from median text size.
    horizontal_tol_mult: float = 1.2
    vertical_tol_mult: float = 0.45
    row_gap_mult: float = 1.0
    block_gap_mult: float = 0.9
    # Maximum block height relative to median row height; beyond this, start a new block.
    max_block_height_mult: float = 60.0
    # Horizontal gap multiplier to split a row when boxes are too far apart (prevents spanning whole page).
    row_split_gap_mult: float = 6.0
    # Gap between columns (in median word widths) to split page into columns before row grouping.
    column_gap_mult: float = 1.2
    # Enable histogram-based gutter detection before gap-based splitting.
    use_hist_gutter: bool = False
    # Minimum gutter width (in median word widths) for histogram-based column detection.
    gutter_width_mult: float = 2.0
    # Maximum allowed column width (in median word widths) before forcing an internal split.
    max_column_width_mult: float = 15.0
    # Max row width relative to its column span; rows wider than this are split on largest gaps.
    max_row_width_mult: float = 1.1
    # Table regularity tolerance (coefficient of variation threshold).
    table_regular_tol: float = 0.22
    # Span gap multiplier: split spans when gap > median_space_gap * span_gap_mult.
    span_gap_mult: float = 12.0
    # Content band for column detection (exclude headers/footers from gutter analysis).
    # Values are fractions of page height.
    content_band_top: float = 0.15  # Exclude top 15%
    content_band_bottom: float = 0.85  # Exclude bottom 15%

    # ── OCR full-page reconciliation settings ──────────────────────────
    # Enable dual-source OCR reconciliation (full-page PaddleOCR + spatial
    # matching against PDF text layer, injecting only missing symbols).
    enable_ocr_reconcile: bool = False
    # Characters OCR is allowed to inject (symbol whitelist).
    ocr_reconcile_allowed_symbols: str = "%/°±"
    # Render resolution (DPI) for the full-page OCR image.
    ocr_reconcile_resolution: int = 300
    # Minimum PaddleOCR confidence (0-1) to consider an OCR token.
    ocr_reconcile_confidence: float = 0.6
    # IoU threshold above which an OCR token is considered "matched" to a PDF token.
    ocr_reconcile_iou_threshold: float = 0.5
    # Center-to-center proximity tolerance (pts) for a "likely match" fallback.
    ocr_reconcile_center_tol_x: float = 3.0
    ocr_reconcile_center_tol_y: float = 2.0
    # Max horizontal distance (pts) to look for a neighbouring digit token
    # when accepting an unmatched OCR symbol (Case B contextual check).
    ocr_reconcile_proximity_pts: float = 10.0
    # Force OCR reconcile debug overlay even when no tokens are injected.
    ocr_reconcile_debug: bool = False
