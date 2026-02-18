from dataclasses import dataclass


class ConfigValidationError(ValueError):
    """Raised when a GroupingConfig field has an invalid value."""


def _check_range(
    name: str, value: float, lo: float, hi: float, *, inclusive: bool = True
) -> None:
    if inclusive:
        if not (lo <= value <= hi):
            raise ConfigValidationError(f"{name}={value} out of range [{lo}, {hi}]")
    else:
        if not (lo < value < hi):
            raise ConfigValidationError(f"{name}={value} out of range ({lo}, {hi})")


def _check_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ConfigValidationError(f"{name}={value} must be > 0")


def _check_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ConfigValidationError(f"{name}={value} must be >= 0")


def _check_odd(name: str, value: int, floor: int = 3) -> None:
    if value < floor:
        raise ConfigValidationError(f"{name}={value} must be >= {floor}")
    if value % 2 == 0:
        raise ConfigValidationError(f"{name}={value} must be odd")


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
    use_hist_gutter: bool = True
    # Minimum gutter width (in median word widths) for histogram-based column detection.
    gutter_width_mult: float = 1.0
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

    # ── Text OCR (pdfplumber text extraction) ─────────────────────────
    # When True (default), extract word boxes from the PDF text layer.
    # Disable to run VOCR-only or skip text extraction entirely.
    enable_tocr: bool = True
    # pdfplumber extract_words() horizontal tolerance (pts).
    # Characters within this distance are grouped into the same word.
    tocr_x_tolerance: float = 3.0
    # pdfplumber extract_words() vertical tolerance (pts).
    # Characters within this distance are considered same-line.
    tocr_y_tolerance: float = 3.0
    # When True, include extra attributes (fontname, size) per word.
    tocr_extra_attrs: bool = True
    # Filter control characters (U+0000–U+001F except \t\n\r, plus U+FEFF BOM).
    tocr_filter_control_chars: bool = True
    # Deduplicate overlapping text boxes with identical text (IoU > threshold).
    tocr_dedup_iou: float = 0.8
    # Drop tokens shorter than N characters (0 = keep all).
    tocr_min_word_length: int = 0
    # Ignore text below this point size (0.0 = keep all).
    tocr_min_font_size: float = 0.0
    # Ignore text above this point size (0.0 = no upper limit).
    tocr_max_font_size: float = 0.0
    # Drop tokens that are entirely whitespace after cleanup.
    tocr_strip_whitespace_tokens: bool = True
    # Clip word coordinates to page bounds (False keeps raw coords).
    tocr_clip_to_page: bool = True
    # Inset margin (pts): ignore words whose centre falls within this
    # distance of a page edge (title blocks, borders, revision stamps).
    tocr_margin_pts: float = 0.0
    # Keep non-upright (rotated/vertical) text.  False drops them.
    tocr_keep_rotated: bool = True
    # Apply NFKC unicode normalisation (collapses ligatures, fullwidth chars).
    tocr_normalize_unicode: bool = False
    # Lowercase all tokens.
    tocr_case_fold: bool = False
    # Replace runs of internal whitespace with a single space.
    tocr_collapse_whitespace: bool = True
    # Flag/skip page if token density (tokens/sq-in) falls below this (0 = off).
    tocr_min_token_density: float = 0.0
    # Warn if mojibake fraction of tokens exceeds this threshold.
    tocr_mojibake_threshold: float = 0.1
    # Preserve PDF's internal text flow ordering instead of spatial sort.
    tocr_use_text_flow: bool = False
    # Keep blank chars as individual character objects (for fixed-width forms).
    tocr_keep_blank_chars: bool = False

    # ── Visual OCR (PaddleOCR full-page extraction) ────────────────────
    # When True, run PaddleOCR on the rendered page image to extract
    # visual tokens.  Required as input for the reconcile stage.
    enable_vocr: bool = False
    # Model tier: "mobile" for speed (~1x), "server" for accuracy (~10–20x slower).
    vocr_model_tier: str = "mobile"
    # Enable PaddleOCR document-orientation classification (auto-rotate).
    vocr_use_orientation_classify: bool = False
    # Enable PaddleOCR document unwarping (perspective correction).
    vocr_use_doc_unwarping: bool = False
    # Enable PaddleOCR textline orientation detection.
    vocr_use_textline_orientation: bool = False
    # Render resolution (DPI) for the VOCR page image.
    # 0 = fall back to ocr_reconcile_resolution for backward compat.
    vocr_resolution: int = 0
    # Minimum PaddleOCR confidence (0–1) to keep a detected token.
    vocr_min_confidence: float = 0.6
    # Maximum pixel dimension for a single OCR tile before splitting.
    # PaddleOCR silently downscales images above ~4000 px — tiling
    # avoids that.  Lower values tile more aggressively (slower but
    # preserves small text); higher values let Paddle downscale.
    vocr_max_tile_px: int = 3800
    # Fractional overlap between adjacent tiles (0.0–0.25).
    # More overlap catches text on tile boundaries but is slower.
    vocr_tile_overlap: float = 0.05
    # IoU threshold for deduplicating tokens in tile-overlap zones.
    vocr_tile_dedup_iou: float = 0.5
    # Minimum OCR token text length to keep (0 = keep all).
    # Useful to filter single-character OCR noise.
    vocr_min_text_length: int = 0
    # Drop OCR tokens that are entirely whitespace.
    vocr_strip_whitespace: bool = True
    # Maximum horizontal skew (degrees) for PaddleOCR text detection.
    # 0 = use PaddleOCR default.
    vocr_max_det_skew: float = 0.0
    # Heartbeat interval (seconds) printed to stdout during long OCR
    # calls so terminals / CI don't consider the process idle.
    vocr_heartbeat_interval: float = 15.0

    # ── OCR reconciliation (merge VOCR tokens into TOCR) ───────────────
    # Enable dual-source OCR reconciliation (spatial matching of VOCR
    # tokens against PDF text layer, injecting only missing symbols).
    # Requires enable_vocr to also be True.
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
    # Horizontal search margin (pts) around OCR bbox for finding digit
    # anchors during composite matching (Case C).
    ocr_reconcile_anchor_margin: float = 25.0
    # Horizontal padding (pts) when placing after-digit symbols.
    ocr_reconcile_symbol_pad: float = 0.5
    # Force OCR reconcile debug overlay even when no tokens are injected.
    ocr_reconcile_debug: bool = False

    # ── OCR image preprocessing (runs BEFORE OCR reconciliation) ────────
    # When True and OCR reconciliation is also enabled, the rendered OCR
    # image is preprocessed (grayscale, CLAHE contrast, optional denoise)
    # before being passed to PaddleOCR.
    enable_ocr_preprocess: bool = False
    # Convert rendered image to grayscale before further processing.
    vocrpp_grayscale: bool = True
    # Apply PIL autocontrast (stretch histogram to full range).
    vocrpp_autocontrast: bool = False
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation).
    vocrpp_clahe: bool = True
    # CLAHE clip limit (higher = more contrast enhancement).
    vocrpp_clahe_clip_limit: float = 2.0
    # CLAHE tile grid size (pixels per tile side).
    vocrpp_clahe_grid_size: int = 8
    # Apply median filter for denoising.
    vocrpp_median_denoise: bool = False
    # Median filter kernel size (must be odd, ≥ 3).
    vocrpp_median_kernel: int = 3
    # Apply adaptive binarisation (Gaussian threshold).
    vocrpp_adaptive_binarize: bool = False
    # Adaptive threshold block size (must be odd, ≥ 3).
    vocrpp_binarize_block_size: int = 11
    # Adaptive threshold constant subtracted from mean.
    vocrpp_binarize_constant: float = 2.0
    # Apply UnsharpMask sharpening after other steps.
    vocrpp_sharpen: bool = False
    # UnsharpMask radius (pixels).
    vocrpp_sharpen_radius: int = 2
    # UnsharpMask percent (strength, 0–500).
    vocrpp_sharpen_percent: int = 140

    # ── Reconcile stage: symbol generation & candidate acceptance ──────
    # Multiplier on glyph height for digit-neighbour vertical band tolerance.
    ocr_reconcile_digit_band_tol_mult: float = 0.5
    # Horizontal overshoot (pts) when searching for digit neighbours.
    ocr_reconcile_digit_overshoot: float = -2.0
    # Fallback estimated character width (pts) when no neighbours found.
    ocr_reconcile_char_width_fallback: float = 5.0
    # Multiplier on glyph height for line-neighbour vertical band tolerance.
    ocr_reconcile_line_neighbour_tol_mult: float = 0.6
    # Minimum vertical band tolerance (pts) for line-neighbour search.
    ocr_reconcile_line_neighbour_min_tol: float = 3.0
    # Fraction of text characters that must be digits for _is_digit_group.
    ocr_reconcile_digit_ratio: float = 0.5
    # Slash symbol width as fraction of glyph height.
    ocr_reconcile_slash_width_mult: float = 0.35
    # Percent symbol width as fraction of glyph height.
    ocr_reconcile_pct_width_mult: float = 0.95
    # Degree / plus-minus symbol width as fraction of glyph height.
    ocr_reconcile_degree_width_mult: float = 0.5
    # Max proximity (pts) for accepting a generated candidate.
    ocr_reconcile_accept_proximity: float = 4.0
    # IoU threshold for candidate overlap rejection.
    ocr_reconcile_accept_iou: float = 0.15
    # Coverage threshold for candidate overlap rejection.
    ocr_reconcile_accept_coverage: float = 0.30
    # Maximum debug candidates logged per page.
    ocr_reconcile_max_debug: int = 200

    # ── Grouping stage: histogram, lines & blocks ──────────────────────
    # Density threshold for histogram-based gutter detection.
    grouping_histogram_density: float = 0.08
    # Minimum number of histogram bins for gutter detection (actual count
    # is adaptive: max(this, content_width / (median_w * 0.5))).
    grouping_histogram_bins: int = 80
    # Minimum vertical overlap ratio to merge two boxes into one line.
    grouping_line_overlap_ratio: float = 0.3
    # Fallback median space-gap width when no spaces detected.
    grouping_space_gap_fallback: float = 5.0
    # Percentile cutoff for space-gap estimation (filters outliers).
    grouping_space_gap_percentile: float = 0.9
    # Column partition width guard (× median width).
    grouping_partition_width_guard_mult: float = 30.0
    # Column partition density decay factor.
    grouping_partition_decay: float = 0.7
    # Column partition minimum density floor.
    grouping_partition_floor: float = 1.0
    # Majority fraction of rows that must match note-number regex.
    grouping_note_majority: float = 0.5
    # Maximum rows in a block to consider note-number column.
    grouping_note_max_rows: int = 50
    # Column-gap fallback multiplier (× median row/line width).
    grouping_col_gap_fallback_mult: float = 0.6
    # Block merge gap multiplier (× block_gap).
    grouping_block_merge_mult: float = 1.5
    # Notes column X-alignment tolerance (pts).
    grouping_notes_x_tolerance: float = 30.0
    # Notes column max Y-gap between blocks (pts).
    grouping_notes_y_gap_max: float = 50.0
    # First-gap multiplier for notes column detection.
    grouping_notes_first_gap_mult: float = 2.0
    # Continued-column X-alignment tolerance (pts).
    grouping_link_x_tolerance: float = 50.0

    # ── Legend / abbreviation / revision detection ─────────────────────
    # Tolerance (pts) for matching a rect enclosing a header.
    legend_enclosure_tolerance: float = 20.0
    # Maximum side length (pts) for a graphic to qualify as legend symbol.
    legend_max_symbol_size: float = 50.0
    # Minimum area (pts²) to keep a symbol (filters noise).
    legend_symbol_min_area: float = 10.0
    # Maximum area (pts²) to keep a symbol (filters large graphics).
    legend_symbol_max_area: float = 2500.0
    # X-tolerance (pts) for aligning symbols into columns.
    legend_column_x_tolerance: float = 30.0
    # Y-tolerance (pts) for pairing a symbol with its text description.
    legend_text_y_tolerance: float = 20.0
    # Maximum X-gap (pts) allowed between symbol and text description.
    legend_text_x_gap_max: float = 300.0
    # Left margin (pts) to extend search region for unboxed legends.
    legend_unboxed_x_margin: float = 100.0
    # Right extent (pts) from header for unboxed legend search.
    legend_unboxed_x_extent: float = 600.0
    # Downward extent (pts) from header for unboxed legend search.
    legend_unboxed_y_extent: float = 500.0

    # ── Font metrics anomaly detection ─────────────────────────────────
    # Inflation factor threshold above which a font is flagged anomalous.
    font_metrics_inflation_threshold: float = 1.3
    # Minimum character samples required to assess a font.
    font_metrics_min_samples: int = 5
    # Minimum confidence (0–1) for an anomaly to be considered valid.
    font_metrics_confidence_min: float = 0.7
    # Render DPI for visual font metrics analysis.
    font_metrics_visual_dpi: int = 300
    # Grayscale threshold (0–255) below which pixels count as "dark".
    font_metrics_dark_threshold: int = 200

    # ── Overlay / debug visualisation ──────────────────────────────────
    # Base font size (px) for overlay labels.
    overlay_label_font_base: int = 10
    # Minimum font size (px) for overlay labels.
    overlay_label_font_floor: int = 8
    # Background alpha (0–255) for overlay label backgrounds.
    overlay_label_bg_alpha: int = 200
    # Fill alpha (0–255) for table block overlays.
    overlay_table_fill_alpha: int = 60
    # Outline width (px) for individual glyph-box overlays.
    overlay_glyph_outline_width: int = 1
    # Outline width (px) for block-level overlays (rows, blocks, headers).
    overlay_block_outline_width: int = 3
    # Outline width (px) for region-level overlays (legend, notes-column, revision, etc.).
    overlay_region_outline_width: int = 4
    # Outline width (px) for span / entry sub-element overlays.
    overlay_span_outline_width: int = 2
    # Y-overlap ratio to count two elements on the same line.
    overlay_same_line_overlap: float = 0.5
    # Proximity (pts) tolerance for misc-title same-line detection.
    overlay_proximity_pts: float = 50.0

    # ── Preprocessing (deskew) ─────────────────────────────────────────
    # Minimum detected skew angle (degrees) below which no rotation is applied.
    preprocess_min_rotation: float = 0.01

    def __post_init__(self) -> None:
        """Validate field ranges to catch misconfiguration early."""
        # -- Thresholds that must be in [0, 1] --
        _unit = [
            "iou_prune",
            "tocr_dedup_iou",
            "vocr_min_confidence",
            "vocr_tile_overlap",
            "vocr_tile_dedup_iou",
            "ocr_reconcile_confidence",
            "ocr_reconcile_iou_threshold",
            "ocr_reconcile_digit_ratio",
            "ocr_reconcile_accept_iou",
            "ocr_reconcile_accept_coverage",
            "grouping_line_overlap_ratio",
            "grouping_space_gap_percentile",
            "grouping_note_majority",
            "table_regular_tol",
            "content_band_top",
            "content_band_bottom",
            "overlay_same_line_overlap",
            "tocr_mojibake_threshold",
            "font_metrics_confidence_min",
        ]
        for name in _unit:
            _check_range(name, getattr(self, name), 0.0, 1.0)

        # -- Strictly positive floats --
        _pos_floats = [
            "horizontal_tol_mult",
            "vertical_tol_mult",
            "row_gap_mult",
            "block_gap_mult",
            "max_block_height_mult",
            "row_split_gap_mult",
            "column_gap_mult",
            "span_gap_mult",
            "max_row_width_mult",
            "tocr_x_tolerance",
            "tocr_y_tolerance",
            "vocrpp_clahe_clip_limit",
            "font_metrics_inflation_threshold",
        ]
        for name in _pos_floats:
            _check_positive(name, getattr(self, name))

        # -- Non-negative floats --
        _nn_floats = [
            "max_skew_degrees",
            "ocr_reconcile_proximity_pts",
            "ocr_reconcile_anchor_margin",
            "ocr_reconcile_accept_proximity",
            "ocr_reconcile_char_width_fallback",
            "grouping_space_gap_fallback",
            "preprocess_min_rotation",
        ]
        for name in _nn_floats:
            _check_non_negative(name, getattr(self, name))

        # -- Positive ints --
        _pos_ints = [
            "grouping_histogram_bins",
            "grouping_note_max_rows",
            "ocr_reconcile_max_debug",
            "font_metrics_min_samples",
            "overlay_glyph_outline_width",
            "overlay_block_outline_width",
            "overlay_region_outline_width",
            "overlay_span_outline_width",
        ]
        for name in _pos_ints:
            val = getattr(self, name)
            if val < 1:
                raise ConfigValidationError(f"{name}={val} must be >= 1")

        # -- DPI / resolution must be positive when set --
        if self.ocr_reconcile_resolution < 1:
            raise ConfigValidationError(
                f"ocr_reconcile_resolution={self.ocr_reconcile_resolution} must be >= 1"
            )

        # -- Odd kernel sizes --
        if self.vocrpp_median_denoise:
            _check_odd("vocrpp_median_kernel", self.vocrpp_median_kernel, floor=3)
        if self.vocrpp_adaptive_binarize:
            _check_odd(
                "vocrpp_binarize_block_size", self.vocrpp_binarize_block_size, floor=3
            )

        # -- VOCR model tier must be known --
        if self.vocr_model_tier not in ("mobile", "server"):
            raise ConfigValidationError(
                f"vocr_model_tier={self.vocr_model_tier!r} must be 'mobile' or 'server'"
            )

        # -- content_band ordering --
        if self.content_band_top >= self.content_band_bottom:
            raise ConfigValidationError(
                f"content_band_top ({self.content_band_top}) must be < "
                f"content_band_bottom ({self.content_band_bottom})"
            )

        # -- Alpha ranges 0-255 --
        for name in ("overlay_label_bg_alpha", "overlay_table_fill_alpha"):
            val = getattr(self, name)
            if not (0 <= val <= 255):
                raise ConfigValidationError(f"{name}={val} out of range [0, 255]")

        # -- Dark threshold 0-255 --
        if not (0 <= self.font_metrics_dark_threshold <= 255):
            raise ConfigValidationError(
                f"font_metrics_dark_threshold={self.font_metrics_dark_threshold} "
                f"out of range [0, 255]"
            )
