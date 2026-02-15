"""Tests for plancheck.config — GroupingConfig defaults, mutation, and validation."""

import pytest

from plancheck.config import ConfigValidationError, GroupingConfig


class TestGroupingConfig:
    def test_defaults(self):
        cfg = GroupingConfig()
        assert cfg.iou_prune == 0.5
        assert cfg.enable_skew is False
        assert cfg.enable_tocr is True
        assert cfg.enable_vocr is False
        assert cfg.enable_ocr_reconcile is False
        assert cfg.enable_ocr_preprocess is False

    def test_override(self):
        cfg = GroupingConfig(iou_prune=0.8, enable_skew=True)
        assert cfg.iou_prune == 0.8
        assert cfg.enable_skew is True

    def test_vars_round_trip(self):
        """vars(cfg) should produce a dict that can reconstruct the config."""
        cfg = GroupingConfig(enable_vocr=True, vocr_model_tier="server")
        d = vars(cfg)
        cfg2 = GroupingConfig(**d)
        assert cfg2.enable_vocr is True
        assert cfg2.vocr_model_tier == "server"
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

    # ── Model tier ────────────────────────────────────────────────────

    def test_invalid_model_tier(self):
        with pytest.raises(ConfigValidationError, match="vocr_model_tier"):
            GroupingConfig(vocr_model_tier="turbo")

    def test_valid_model_tiers(self):
        GroupingConfig(vocr_model_tier="mobile")
        GroupingConfig(vocr_model_tier="server")

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
