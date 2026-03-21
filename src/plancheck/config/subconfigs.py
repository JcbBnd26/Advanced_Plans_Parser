"""Sub-configuration dataclasses for focused domain settings."""

from __future__ import annotations

from dataclasses import dataclass

from .constants import (
    DEFAULT_DRIFT_STATS,
    DEFAULT_GNN_MODEL,
    DEFAULT_ML_MODEL,
    DEFAULT_SUBTYPE_MODEL,
)


@dataclass
class TOCRConfig:
    """Text OCR (pdfplumber) settings."""

    enable_tocr: bool = True
    tocr_x_tolerance: float = 3.0
    tocr_y_tolerance: float = 3.0
    tocr_extra_attrs: bool = True
    tocr_filter_control_chars: bool = True
    tocr_dedup_iou: float = 0.8
    tocr_min_word_length: int = 0
    tocr_min_font_size: float = 0.0
    tocr_max_font_size: float = 0.0
    tocr_strip_whitespace_tokens: bool = True
    tocr_clip_to_page: bool = True
    tocr_margin_pts: float = 0.0
    tocr_keep_rotated: bool = True
    tocr_normalize_unicode: bool = False
    tocr_case_fold: bool = False
    tocr_collapse_whitespace: bool = True
    tocr_min_token_density: float = 0.0
    tocr_mojibake_threshold: float = 0.1
    tocr_use_text_flow: bool = False
    tocr_keep_blank_chars: bool = False


@dataclass
class VOCRConfig:
    """Visual OCR (Surya) and image-preprocessing settings."""

    enable_vocr: bool = True
    vocr_backend: str = "surya"
    vocr_device: str = "cpu"
    surya_init_timeout_sec: int = 45
    surya_languages: str = "en"  # Comma-separated language codes
    vocr_resolution: int = 0
    vocr_min_confidence: float = 0.6
    vocr_max_tile_px: int = 3800
    vocr_tile_overlap: float = 0.05
    vocr_tile_dedup_iou: float = 0.5
    vocr_min_text_length: int = 0
    vocr_strip_whitespace: bool = True
    # VOCR candidate detection
    enable_vocr_candidates: bool = True
    vocr_cand_gap_multiplier: float = 2.0
    vocr_cand_patch_margin: float = 4.0
    vocr_cand_min_confidence: float = 0.3
    vocr_cand_max_candidates: int = 200
    vocr_cand_density_grid_size: float = 20.0
    vocr_cand_char_width_ratio: float = 0.7
    vocr_cand_vector_circle_max_diameter: float = 8.0
    vocr_cand_stats_path: str = "data/candidate_method_stats.json"
    vocr_cand_ml_enabled: bool = False
    vocr_cand_classifier_path: str = "data/candidate_classifier.pkl"
    vocr_cand_ml_threshold: float = 0.3
    vocr_cand_retrain_min_rows: int = 100
    vocr_cand_producer_stats_path: str = "data/producer_method_stats.json"
    vocr_cand_gnn_prior_enabled: bool = False
    vocr_cand_gnn_prior_path: str = "data/gnn_candidate_prior.pt"
    vocr_cand_gnn_prior_blend: float = 0.25
    # VOCR image preprocessing
    enable_ocr_preprocess: bool = True
    vocrpp_grayscale: bool = True
    vocrpp_autocontrast: bool = False
    vocrpp_clahe: bool = True
    vocrpp_clahe_clip_limit: float = 2.0
    vocrpp_clahe_grid_size: int = 8
    vocrpp_median_denoise: bool = False
    vocrpp_median_kernel: int = 3
    vocrpp_adaptive_binarize: bool = False
    vocrpp_binarize_block_size: int = 11
    vocrpp_binarize_constant: float = 2.0
    vocrpp_sharpen: bool = False
    vocrpp_sharpen_radius: int = 2
    vocrpp_sharpen_percent: int = 140


@dataclass
class ReconcileConfig:
    """OCR reconciliation (merge VOCR tokens into TOCR) settings."""

    enable_ocr_reconcile: bool = True
    ocr_reconcile_allowed_symbols: str = "%/°±Ø×'\"#@"
    ocr_reconcile_resolution: int = 300
    ocr_reconcile_confidence: float = 0.6
    ocr_reconcile_iou_threshold: float = 0.5
    ocr_reconcile_center_tol_x: float = 3.0
    ocr_reconcile_center_tol_y: float = 2.0
    ocr_reconcile_proximity_pts: float = 10.0
    ocr_reconcile_anchor_margin: float = 25.0
    ocr_reconcile_symbol_pad: float = 0.5
    ocr_reconcile_debug: bool = False
    ocr_reconcile_digit_band_tol_mult: float = 0.5
    ocr_reconcile_digit_overshoot: float = -2.0
    ocr_reconcile_char_width_fallback: float = 5.0
    ocr_reconcile_line_neighbour_tol_mult: float = 0.6
    ocr_reconcile_line_neighbour_min_tol: float = 3.0
    ocr_reconcile_digit_ratio: float = 0.5
    ocr_reconcile_slash_width_mult: float = 0.35
    ocr_reconcile_pct_width_mult: float = 0.95
    ocr_reconcile_degree_width_mult: float = 0.5
    ocr_reconcile_accept_proximity: float = 4.0
    ocr_reconcile_accept_iou: float = 0.15
    ocr_reconcile_accept_coverage: float = 0.30
    ocr_reconcile_max_debug: int = 200


@dataclass
class GroupingStageConfig:
    """Core geometry, histogram, and clustering settings."""

    iou_prune: float = 0.5
    enable_skew: bool = False
    max_skew_degrees: float = 3.0
    horizontal_tol_mult: float = 1.2
    vertical_tol_mult: float = 0.45
    row_gap_mult: float = 1.0
    block_gap_mult: float = 0.9
    max_block_height_mult: float = 60.0
    row_split_gap_mult: float = 6.0
    column_gap_mult: float = 1.2
    use_hist_gutter: bool = True
    gutter_width_mult: float = 1.0
    max_column_width_mult: float = 15.0
    max_row_width_mult: float = 1.1
    table_regular_tol: float = 0.22
    span_gap_mult: float = 12.0
    content_band_top: float = 0.15
    content_band_bottom: float = 0.85
    merge_overlapping_boxes: bool = False
    grouping_histogram_density: float = 0.08
    grouping_histogram_bins: int = 80
    grouping_line_overlap_ratio: float = 0.3
    grouping_space_gap_fallback: float = 5.0
    grouping_space_gap_percentile: float = 0.9
    grouping_partition_width_guard_mult: float = 30.0
    grouping_partition_decay: float = 0.7
    grouping_partition_floor: float = 1.0
    grouping_note_majority: float = 0.5
    grouping_note_max_rows: int = 50
    grouping_col_gap_fallback_mult: float = 0.6
    grouping_block_merge_mult: float = 1.5
    grouping_notes_x_tolerance: float = 30.0
    grouping_notes_y_gap_max: float = 50.0
    grouping_notes_first_gap_mult: float = 2.0
    grouping_link_x_tolerance: float = 50.0
    preprocess_min_rotation: float = 0.01


@dataclass
class AnalysisConfig:
    """Semantic region growth, legend, header, and font-metrics settings."""

    region_growth_max_gap: float = 0.0
    region_gap_adaptive_mult: float = 3.0
    region_growth_x_tolerance: float = 80.0
    region_font_size_ratio: float = 1.8
    header_large_font_mult: float = 1.25
    header_max_rows: int = 3
    notes_column_running_mean: bool = True
    legend_enclosure_tolerance: float = 20.0
    legend_max_symbol_size: float = 50.0
    legend_symbol_min_area: float = 10.0
    legend_symbol_max_area: float = 2500.0
    legend_column_x_tolerance: float = 30.0
    legend_text_y_tolerance: float = 20.0
    legend_text_x_gap_max: float = 300.0
    legend_unboxed_x_margin: float = 100.0
    legend_unboxed_x_extent: float = 600.0
    legend_unboxed_y_extent: float = 500.0
    font_metrics_inflation_threshold: float = 1.3
    font_metrics_min_samples: int = 5
    font_metrics_confidence_min: float = 0.7
    font_metrics_visual_dpi: int = 300
    font_metrics_dark_threshold: int = 200
    box_containment_pad: float = 4.0
    box_promote_gap_tolerance: float = 50.0
    box_promote_overlap_frac: float = 0.60
    box_callout_area_frac: float = 0.01
    box_callout_max_parts: int = 6


@dataclass
class ExportConfig:
    """Overlay / debug visualisation settings."""

    overlay_label_font_base: int = 10
    overlay_label_font_floor: int = 8
    overlay_label_bg_alpha: int = 200
    overlay_table_fill_alpha: int = 60
    overlay_glyph_outline_width: int = 1
    overlay_block_outline_width: int = 3
    overlay_region_outline_width: int = 4
    overlay_span_outline_width: int = 2
    overlay_same_line_overlap: float = 0.5
    overlay_proximity_pts: float = 50.0


@dataclass
class MLConfig:
    """ML classifier, LLM, GNN, drift, and retraining settings."""

    ml_model_path: str = str(DEFAULT_ML_MODEL)
    ml_stage2_model_path: str = str(DEFAULT_SUBTYPE_MODEL)
    ml_hierarchical_enabled: bool = False
    ml_relabel_confidence: float = 0.8
    ml_enabled: bool = True
    ml_min_training_examples: int = 10
    ml_ensemble_enabled: bool = False
    ml_vision_enabled: bool = False
    ml_vision_backbone: str = "resnet18"
    ml_layout_enabled: bool = False
    ml_layout_model_path: str = "microsoft/layoutlmv3-base"
    ml_embeddings_enabled: bool = False
    ml_embeddings_model: str = "all-MiniLM-L6-v2"
    enable_llm_checks: bool = False
    llm_provider: str = "ollama"
    llm_model: str = "llama3.1:8b"
    llm_api_key: str = ""
    llm_api_base: str = "http://localhost:11434"
    llm_temperature: float = 0.1
    llm_policy: str = "local_only"
    ml_gnn_enabled: bool = False
    ml_gnn_model_path: str = str(DEFAULT_GNN_MODEL)
    ml_gnn_hidden_dim: int = 64
    ml_gnn_patience: int = 20
    ml_drift_enabled: bool = False
    ml_drift_stats_path: str = str(DEFAULT_DRIFT_STATS)
    ml_drift_threshold: float = 0.3
    ml_retrain_threshold: int = 50
    ml_retrain_on_startup: bool = False
    ml_feature_cache_enabled: bool = True
    ml_comparison_threshold: float = 0.005
