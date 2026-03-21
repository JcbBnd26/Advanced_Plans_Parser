"""Tests for plancheck.config — GroupingConfig defaults, mutation, and validation."""

import pytest

from plancheck.config import ConfigValidationError, GroupingConfig


class TestGroupingConfig:
    def test_defaults(self):
        cfg = GroupingConfig()
        assert cfg.iou_prune == 0.5
        assert cfg.enable_skew is False
        assert cfg.enable_tocr is True
        assert cfg.enable_vocr is True
        assert cfg.enable_ocr_reconcile is True
        assert cfg.enable_ocr_preprocess is True
        assert cfg.ml_enabled is True
        assert cfg.ml_hierarchical_enabled is False

    def test_override(self):
        cfg = GroupingConfig(iou_prune=0.8, enable_skew=True)
        assert cfg.iou_prune == 0.8
        assert cfg.enable_skew is True

    def test_vars_round_trip(self):
        """vars(cfg) should produce a dict that can reconstruct the config."""
        cfg = GroupingConfig(enable_vocr=True, vocr_backend="surya")
        d = vars(cfg)
        cfg2 = GroupingConfig(**d)
        assert cfg2.enable_vocr is True
        assert cfg2.vocr_backend == "surya"
        assert vars(cfg) == vars(cfg2)

    def test_ocr_reconcile_allowed_symbols(self):
        cfg = GroupingConfig()
        assert "%" in cfg.ocr_reconcile_allowed_symbols
        assert "/" in cfg.ocr_reconcile_allowed_symbols
        assert "°" in cfg.ocr_reconcile_allowed_symbols
        assert "±" in cfg.ocr_reconcile_allowed_symbols


class TestConfigValidation:
    """Validate __post_init__ range guards."""

    # ── Unit-range fields [0, 1] ──────────────────────────────────────

    def test_iou_prune_negative_rejected(self):
        with pytest.raises(ConfigValidationError, match="iou_prune"):
            GroupingConfig(iou_prune=-0.1)

    def test_iou_prune_above_one_rejected(self):
        with pytest.raises(ConfigValidationError, match="iou_prune"):
            GroupingConfig(iou_prune=1.5)

    def test_iou_prune_boundary_zero(self):
        cfg = GroupingConfig(iou_prune=0.0)
        assert cfg.iou_prune == 0.0

    def test_iou_prune_boundary_one(self):
        cfg = GroupingConfig(iou_prune=1.0)
        assert cfg.iou_prune == 1.0

    def test_vocr_min_confidence_out_of_range(self):
        with pytest.raises(ConfigValidationError, match="vocr_min_confidence"):
            GroupingConfig(vocr_min_confidence=200)

    def test_ocr_reconcile_confidence_negative(self):
        with pytest.raises(ConfigValidationError, match="ocr_reconcile_confidence"):
            GroupingConfig(ocr_reconcile_confidence=-1)

    def test_content_band_top_out_of_range(self):
        with pytest.raises(ConfigValidationError, match="content_band_top"):
            GroupingConfig(content_band_top=1.5)

    # ── Strictly positive floats ──────────────────────────────────────

    def test_horizontal_tol_mult_zero_rejected(self):
        with pytest.raises(ConfigValidationError, match="horizontal_tol_mult"):
            GroupingConfig(horizontal_tol_mult=0.0)

    def test_horizontal_tol_mult_negative_rejected(self):
        with pytest.raises(ConfigValidationError, match="horizontal_tol_mult"):
            GroupingConfig(horizontal_tol_mult=-1.0)

    def test_clahe_clip_limit_zero_rejected(self):
        with pytest.raises(ConfigValidationError, match="vocrpp_clahe_clip_limit"):
            GroupingConfig(vocrpp_clahe_clip_limit=0)

    # ── Non-negative floats ───────────────────────────────────────────

    def test_proximity_pts_negative_rejected(self):
        with pytest.raises(ConfigValidationError, match="ocr_reconcile_proximity_pts"):
            GroupingConfig(ocr_reconcile_proximity_pts=-5)

    def test_proximity_pts_zero_accepted(self):
        cfg = GroupingConfig(ocr_reconcile_proximity_pts=0.0)
        assert cfg.ocr_reconcile_proximity_pts == 0.0

    # ── Positive ints ─────────────────────────────────────────────────

    def test_histogram_bins_zero_rejected(self):
        with pytest.raises(ConfigValidationError, match="grouping_histogram_bins"):
            GroupingConfig(grouping_histogram_bins=0)

    def test_resolution_zero_rejected(self):
        with pytest.raises(ConfigValidationError, match="ocr_reconcile_resolution"):
            GroupingConfig(ocr_reconcile_resolution=0)

    # ── Odd kernel checks ─────────────────────────────────────────────

    def test_median_kernel_even_rejected(self):
        with pytest.raises(ConfigValidationError, match="vocrpp_median_kernel"):
            GroupingConfig(vocrpp_median_denoise=True, vocrpp_median_kernel=4)

    def test_median_kernel_too_small_rejected(self):
        with pytest.raises(ConfigValidationError, match="vocrpp_median_kernel"):
            GroupingConfig(vocrpp_median_denoise=True, vocrpp_median_kernel=1)

    def test_median_kernel_odd_accepted(self):
        cfg = GroupingConfig(vocrpp_median_denoise=True, vocrpp_median_kernel=5)
        assert cfg.vocrpp_median_kernel == 5

    def test_binarize_block_even_rejected(self):
        with pytest.raises(ConfigValidationError, match="vocrpp_binarize_block_size"):
            GroupingConfig(vocrpp_adaptive_binarize=True, vocrpp_binarize_block_size=10)

    # ── OCR backend ────────────────────────────────────────────────────

    def test_invalid_vocr_backend(self):
        with pytest.raises(ConfigValidationError, match="vocr_backend"):
            GroupingConfig(vocr_backend="unsupported_backend")

    def test_valid_vocr_backend(self):
        cfg = GroupingConfig(vocr_backend="surya")
        assert cfg.vocr_backend == "surya"

    def test_surya_init_timeout_zero_rejected(self):
        with pytest.raises(ConfigValidationError, match="surya_init_timeout_sec"):
            GroupingConfig(surya_init_timeout_sec=0)

    def test_surya_init_timeout_positive_accepted(self):
        cfg = GroupingConfig(surya_init_timeout_sec=90)
        assert cfg.surya_init_timeout_sec == 90

    # ── Content band ordering ─────────────────────────────────────────

    def test_content_band_inverted_rejected(self):
        with pytest.raises(ConfigValidationError, match="content_band_top"):
            GroupingConfig(content_band_top=0.9, content_band_bottom=0.1)

    def test_content_band_equal_rejected(self):
        with pytest.raises(ConfigValidationError, match="content_band_top"):
            GroupingConfig(content_band_top=0.5, content_band_bottom=0.5)

    # ── Alpha ranges ──────────────────────────────────────────────────

    def test_alpha_above_255_rejected(self):
        with pytest.raises(ConfigValidationError, match="overlay_label_bg_alpha"):
            GroupingConfig(overlay_label_bg_alpha=300)

    def test_alpha_negative_rejected(self):
        with pytest.raises(ConfigValidationError, match="overlay_table_fill_alpha"):
            GroupingConfig(overlay_table_fill_alpha=-1)

    def test_dark_threshold_above_255(self):
        with pytest.raises(ConfigValidationError, match="font_metrics_dark_threshold"):
            GroupingConfig(font_metrics_dark_threshold=256)


# ══════════════════════════════════════════════════════════════════════
# Config file loading tests
# ══════════════════════════════════════════════════════════════════════


class TestConfigFromDict:
    def test_from_dict_round_trip(self):
        cfg = GroupingConfig(
            iou_prune=0.7,
            enable_vocr=True,
            ml_hierarchical_enabled=True,
            ml_stage2_model_path="data/custom_stage2.pkl",
        )
        d = cfg.to_dict()
        cfg2 = GroupingConfig.from_dict(d)
        assert cfg2.iou_prune == 0.7
        assert cfg2.enable_vocr is True
        assert cfg2.ml_hierarchical_enabled is True
        assert cfg2.ml_stage2_model_path == "data/custom_stage2.pkl"

    def test_from_dict_ignores_unknown_keys(self):
        d = {"iou_prune": 0.6, "unknown_field": 42}
        cfg = GroupingConfig.from_dict(d)
        assert cfg.iou_prune == 0.6

    def test_to_dict_no_private(self):
        cfg = GroupingConfig()
        d = cfg.to_dict()
        assert all(not k.startswith("_") for k in d)


class TestConfigFromYaml:
    def test_load_yaml(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "iou_prune: 0.75\n"
            "enable_vocr: true\n"
            "ml_hierarchical_enabled: true\n"
            'ml_stage2_model_path: "data/title_stage2.pkl"\n'
        )
        cfg = GroupingConfig.from_yaml(yaml_file)
        assert cfg.iou_prune == 0.75
        assert cfg.enable_vocr is True
        assert cfg.ml_hierarchical_enabled is True
        assert cfg.ml_stage2_model_path == "data/title_stage2.pkl"

    def test_load_yaml_ignores_unknown(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("iou_prune: 0.6\nunknown_key: hello\n")
        cfg = GroupingConfig.from_yaml(yaml_file)
        assert cfg.iou_prune == 0.6

    def test_yaml_invalid_content(self, tmp_path):
        from plancheck.config import ConfigLoadError

        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("- list\n- not mapping\n")
        with pytest.raises(ConfigLoadError, match="mapping"):
            GroupingConfig.from_yaml(yaml_file)

    def test_yaml_missing_file(self, tmp_path):
        from plancheck.config import ConfigLoadError

        with pytest.raises(ConfigLoadError):
            GroupingConfig.from_yaml(tmp_path / "nonexistent.yaml")


class TestConfigFromToml:
    def test_load_toml(self, tmp_path):
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(
            'iou_prune = 0.8\nenable_vocr = true\nml_hierarchical_enabled = true\nml_stage2_model_path = "data/title_stage2.pkl"\n'
        )
        cfg = GroupingConfig.from_toml(toml_file)
        assert cfg.iou_prune == 0.8
        assert cfg.enable_vocr is True
        assert cfg.ml_hierarchical_enabled is True
        assert cfg.ml_stage2_model_path == "data/title_stage2.pkl"

    def test_load_toml_subtable(self, tmp_path):
        toml_file = tmp_path / "config.toml"
        toml_file.write_text("[plancheck]\niou_prune = 0.65\n")
        cfg = GroupingConfig.from_toml(toml_file)
        assert cfg.iou_prune == 0.65

    def test_load_toml_grouping_subtable(self, tmp_path):
        toml_file = tmp_path / "config.toml"
        toml_file.write_text("[grouping]\niou_prune = 0.55\n")
        cfg = GroupingConfig.from_toml(toml_file)
        assert cfg.iou_prune == 0.55

    def test_toml_missing_file(self, tmp_path):
        from plancheck.config import ConfigLoadError

        with pytest.raises(ConfigLoadError):
            GroupingConfig.from_toml(tmp_path / "nonexistent.toml")


class TestConfigFromFile:
    def test_auto_detect_yaml(self, tmp_path):
        f = tmp_path / "config.yml"
        f.write_text("iou_prune: 0.9\n")
        cfg = GroupingConfig.from_file(f)
        assert cfg.iou_prune == 0.9

    def test_auto_detect_toml(self, tmp_path):
        f = tmp_path / "config.toml"
        f.write_text("iou_prune = 0.85\n")
        cfg = GroupingConfig.from_file(f)
        assert cfg.iou_prune == 0.85

    def test_unsupported_extension(self, tmp_path):
        from plancheck.config import ConfigLoadError

        f = tmp_path / "config.json"
        f.write_text("{}")
        with pytest.raises(ConfigLoadError, match="Unsupported"):
            GroupingConfig.from_file(f)


# ══════════════════════════════════════════════════════════════════════
# Sub-config field synchronization tests
# ══════════════════════════════════════════════════════════════════════


class TestSubconfigSync:
    """Ensure PipelineConfig has every field from every sub-config."""

    def test_subconfig_fields_match_pipeline(self):
        """Fail immediately if a subconfig field is missing from PipelineConfig."""
        import dataclasses

        from plancheck.config import PipelineConfig
        from plancheck.config.subconfigs import (
            AnalysisConfig,
            ExportConfig,
            GroupingStageConfig,
            MLConfig,
            ReconcileConfig,
            TOCRConfig,
            VOCRConfig,
        )

        pipeline_fields = {f.name for f in dataclasses.fields(PipelineConfig)}
        subconfigs = [
            TOCRConfig,
            VOCRConfig,
            ReconcileConfig,
            GroupingStageConfig,
            AnalysisConfig,
            ExportConfig,
            MLConfig,
        ]
        for sub_cls in subconfigs:
            for f in dataclasses.fields(sub_cls):
                assert (
                    f.name in pipeline_fields
                ), f"{sub_cls.__name__}.{f.name} missing from PipelineConfig"
