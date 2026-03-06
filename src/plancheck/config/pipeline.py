"""Main PipelineConfig dataclass with all tunables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .constants import DEFAULT_DRIFT_STATS, DEFAULT_GNN_MODEL, DEFAULT_ML_MODEL
from .exceptions import ConfigLoadError, ConfigValidationError
from .subconfigs import (
    AnalysisConfig,
    ExportConfig,
    GroupingStageConfig,
    MLConfig,
    ReconcileConfig,
    TOCRConfig,
    VOCRConfig,
)
from .validation import _check_non_negative, _check_odd, _check_positive, _check_range


@dataclass
class PipelineConfig:
    """Tunables for geometry-first grouping.

    Fields are organised into logical sections:

    * **Core geometry** – IoU pruning, skew, tolerance multipliers,
      row/block/column splitting, and table detection.
    * **Text OCR (TOCR)** – pdfplumber text-layer extraction options.
    * **Visual OCR (VOCR)** – PaddleOCR full-page extraction options.
    * **OCR reconciliation** – merging VOCR tokens into the TOCR layer.
    * **OCR image preprocessing (VOCRPP)** – grayscale, CLAHE, denoise,
      binarise, and sharpen the rendered page before OCR.
    * **Reconcile-stage tuning** – symbol sizing and candidate acceptance.
    * **Grouping-stage tuning** – histogram gutters, line overlap, gap
      fallbacks, column detection, and notes-column linking.
    * **Semantic region growth** – anchor-based bbox expansion for
      legends, abbreviations, revisions, and standard details.
    * **Legend / abbreviation / revision detection** – enclosure
      tolerance, symbol sizing, and unboxed-region search extents.
    * **Font metrics** – inflation-factor anomaly detection thresholds.
    * **Overlay / debug visualisation** – label sizing, outline widths,
      fill alpha, and same-line overlap tolerance.
    * **Preprocessing (deskew)** – minimum rotation angle.
    """

    # ── Core geometry ──────────────────────────────────────────────────
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

    # ── Text OCR (pdfplumber text extraction) ─────────────────────────
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

    # ── Visual OCR (PaddleOCR full-page extraction) ────────────────────
    enable_vocr: bool = False
    vocr_model_tier: str = "mobile"
    vocr_use_orientation_classify: bool = False
    vocr_use_doc_unwarping: bool = False
    vocr_use_textline_orientation: bool = False
    vocr_resolution: int = 0
    vocr_min_confidence: float = 0.6
    vocr_max_tile_px: int = 3800
    vocr_tile_overlap: float = 0.05
    vocr_tile_dedup_iou: float = 0.5
    vocr_min_text_length: int = 0
    vocr_strip_whitespace: bool = True
    vocr_max_det_skew: float = 0.0
    vocr_heartbeat_interval: float = 15.0

    # ── VOCR candidate detection (targeted patch selection) ────────────
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

    # ── OCR image preprocessing ────────────────────────────────────────
    enable_ocr_preprocess: bool = False
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

    # ── OCR reconciliation (merge VOCR tokens into TOCR) ───────────────
    enable_ocr_reconcile: bool = False
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

    # ── Grouping stage: histogram, lines & blocks ──────────────────────
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

    # ── Semantic region growth ─────────────────────────────────────────
    region_growth_max_gap: float = 0.0
    region_gap_adaptive_mult: float = 3.0
    region_growth_x_tolerance: float = 80.0
    region_font_size_ratio: float = 1.8
    header_large_font_mult: float = 1.25
    header_max_rows: int = 3
    notes_column_running_mean: bool = True

    # ── Legend / abbreviation / revision detection ─────────────────────
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

    # ── Font metrics anomaly detection ─────────────────────────────────
    font_metrics_inflation_threshold: float = 1.3
    font_metrics_min_samples: int = 5
    font_metrics_confidence_min: float = 0.7
    font_metrics_visual_dpi: int = 300
    font_metrics_dark_threshold: int = 200

    # ── Overlay / debug visualisation ──────────────────────────────────
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

    # ── Preprocessing (deskew) ─────────────────────────────────────────
    preprocess_min_rotation: float = 0.01

    # ── ML classifier ─────────────────────────────────────────────────
    ml_model_path: str = str(DEFAULT_ML_MODEL)
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

    # ── ML drift & retraining ─────────────────────────────────────────
    ml_drift_enabled: bool = False
    ml_drift_stats_path: str = str(DEFAULT_DRIFT_STATS)
    ml_drift_threshold: float = 0.3
    ml_retrain_threshold: int = 50
    ml_retrain_on_startup: bool = False
    ml_feature_cache_enabled: bool = True

    # ── Pipeline metadata ──────────────────────────────────────────────
    version: int = 1

    def __post_init__(self) -> None:
        """Validate field ranges to catch misconfiguration early."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration values.

        This is called automatically on construction (via ``__post_init__``),
        but is also safe to call explicitly in case a caller mutates a config
        instance after creation.
        """
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
            "vocr_cand_min_confidence",
            "vocr_cand_char_width_ratio",
            "grouping_line_overlap_ratio",
            "grouping_space_gap_percentile",
            "grouping_note_majority",
            "table_regular_tol",
            "content_band_top",
            "content_band_bottom",
            "overlay_same_line_overlap",
            "tocr_mojibake_threshold",
            "font_metrics_confidence_min",
            "ml_relabel_confidence",
            "ml_drift_threshold",
        ]
        for name in _unit:
            _check_range(name, getattr(self, name), 0.0, 1.0)

        # -- Strictly positive floats --
        _pos_floats = [
            "vocr_cand_gap_multiplier",
            "vocr_cand_patch_margin",
            "vocr_cand_density_grid_size",
            "vocr_cand_vector_circle_max_diameter",
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
            "region_growth_max_gap",
            "region_gap_adaptive_mult",
            "region_growth_x_tolerance",
            "region_font_size_ratio",
        ]
        for name in _nn_floats:
            _check_non_negative(name, getattr(self, name))

        # -- Positive ints --
        _pos_ints = [
            "vocr_cand_max_candidates",
            "grouping_histogram_bins",
            "grouping_note_max_rows",
            "ocr_reconcile_max_debug",
            "font_metrics_min_samples",
            "overlay_glyph_outline_width",
            "overlay_block_outline_width",
            "overlay_region_outline_width",
            "overlay_span_outline_width",
            "ml_min_training_examples",
            "ml_retrain_threshold",
        ]
        for name in _pos_ints:
            val = getattr(self, name)
            if val < 1:
                raise ConfigValidationError(f"{name}={val} must be >= 1")

        # -- LLM temperature [0, 2] --
        _check_range("llm_temperature", self.llm_temperature, 0.0, 2.0)

        # -- LLM policy must be a recognised value --
        _allowed_policies = {"local_only", "cloud_allowed", "cloud_with_consent"}
        if self.llm_policy not in _allowed_policies:
            raise ConfigValidationError(
                f"llm_policy={self.llm_policy!r} must be one of {sorted(_allowed_policies)}"
            )

        # -- GNN hidden dim must be positive --
        if self.ml_gnn_hidden_dim < 1:
            raise ConfigValidationError(
                f"ml_gnn_hidden_dim={self.ml_gnn_hidden_dim} must be >= 1"
            )

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

    # ── Sub-config view properties ──────────────────────────────────────

    @property
    def tocr(self) -> TOCRConfig:
        """View of TOCR-related settings."""
        import dataclasses

        names = {f.name for f in dataclasses.fields(TOCRConfig)}
        return TOCRConfig(**{n: getattr(self, n) for n in names})

    @property
    def vocr(self) -> VOCRConfig:
        """View of VOCR and image-preprocessing settings."""
        import dataclasses

        names = {f.name for f in dataclasses.fields(VOCRConfig)}
        return VOCRConfig(**{n: getattr(self, n) for n in names})

    @property
    def reconcile(self) -> ReconcileConfig:
        """View of OCR reconciliation settings."""
        import dataclasses

        names = {f.name for f in dataclasses.fields(ReconcileConfig)}
        return ReconcileConfig(**{n: getattr(self, n) for n in names})

    @property
    def grouping(self) -> GroupingStageConfig:
        """View of core geometry and clustering settings."""
        import dataclasses

        names = {f.name for f in dataclasses.fields(GroupingStageConfig)}
        return GroupingStageConfig(**{n: getattr(self, n) for n in names})

    @property
    def analysis(self) -> AnalysisConfig:
        """View of semantic analysis settings."""
        import dataclasses

        names = {f.name for f in dataclasses.fields(AnalysisConfig)}
        return AnalysisConfig(**{n: getattr(self, n) for n in names})

    @property
    def export(self) -> ExportConfig:
        """View of overlay / export settings."""
        import dataclasses

        names = {f.name for f in dataclasses.fields(ExportConfig)}
        return ExportConfig(**{n: getattr(self, n) for n in names})

    @property
    def ml(self) -> MLConfig:
        """View of ML / LLM / drift settings."""
        import dataclasses

        names = {f.name for f in dataclasses.fields(MLConfig)}
        return MLConfig(**{n: getattr(self, n) for n in names})

    # ── Serialisation / deserialisation ────────────────────────────────

    def to_dict(self) -> dict:
        """Return all config fields as a plain dict."""
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        """Create a config from a plain dict, ignoring unknown keys."""
        import dataclasses

        valid = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load configuration from a YAML file.

        Requires the ``pyyaml`` package.  Only keys matching
        ``PipelineConfig`` field names are used; unknown keys are
        silently ignored.

        Raises
        ------
        ConfigLoadError
            If the file cannot be read or parsed.
        """
        from pathlib import Path as _Path

        path = _Path(path)
        try:
            import yaml
        except ImportError as exc:
            raise ConfigLoadError(
                "pyyaml is required for YAML config loading. "
                "Install it with: pip install pyyaml"
            ) from exc

        try:
            text = path.read_text(encoding="utf-8")
            data = yaml.safe_load(text)
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigLoadError(
                f"Failed to load YAML config from {path}: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ConfigLoadError(
                f"YAML config must be a mapping, got {type(data).__name__}"
            )

        return cls.from_dict(data)

    @classmethod
    def from_toml(cls, path: str | Path) -> "PipelineConfig":
        """Load configuration from a TOML file.

        Looks for config values at the top level, or under ``[plancheck]``
        or ``[grouping]`` subtables.

        Raises
        ------
        ConfigLoadError
            If the file cannot be read or parsed.
        """
        from pathlib import Path as _Path

        path = _Path(path)

        if not path.exists():
            raise ConfigLoadError(f"TOML config file not found: {path}")

        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ConfigLoadError(
                    "tomllib (Python 3.11+) or tomli is required for TOML "
                    "config loading. Install tomli with: pip install tomli"
                ) from exc

        try:
            text = path.read_text(encoding="utf-8")
            data = tomllib.loads(text)
        except (OSError, Exception) as exc:
            raise ConfigLoadError(
                f"Failed to load TOML config from {path}: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ConfigLoadError(
                f"TOML config must be a mapping, got {type(data).__name__}"
            )

        # Check for subtables: [plancheck] or [grouping]
        if "plancheck" in data and isinstance(data["plancheck"], dict):
            data = data["plancheck"]
        elif "grouping" in data and isinstance(data["grouping"], dict):
            data = data["grouping"]

        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: str | Path) -> "PipelineConfig":
        """Load configuration from a file, auto-detecting format by extension.

        Supported formats:
        - ``.yaml``, ``.yml``: YAML format
        - ``.toml``: TOML format

        Raises
        ------
        ConfigLoadError
            If the file cannot be read, parsed, or has an unsupported extension.
        """
        from pathlib import Path as _Path

        path = _Path(path)
        suffix = path.suffix.lower()

        if suffix in (".yaml", ".yml"):
            return cls.from_yaml(path)
        elif suffix == ".toml":
            return cls.from_toml(path)
        else:
            raise ConfigLoadError(
                f"Unsupported config file extension '{suffix}'. "
                "Use .yaml, .yml, or .toml"
            )


def migrate_config(old_dict: dict) -> dict:
    """Migrate a flat config dict from an older version to the current schema.

    Unknown or renamed keys are silently dropped; missing keys retain their
    default values when ``PipelineConfig.from_dict`` is called on the result.

    Parameters
    ----------
    old_dict:
        A plain dict loaded from an older config file or pickled config.

    Returns
    -------
    dict
        A cleaned dict ready to pass to ``PipelineConfig.from_dict``.
    """
    import dataclasses

    known = {f.name for f in dataclasses.fields(PipelineConfig)}
    return {k: v for k, v in old_dict.items() if k in known}
