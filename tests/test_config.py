"""Tests for plancheck.config — GroupingConfig defaults and mutation."""

from plancheck.config import GroupingConfig


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
