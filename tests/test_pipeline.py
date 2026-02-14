"""Tests for plancheck.pipeline — gating, stage results, fingerprinting."""

import pytest

from plancheck.config import GroupingConfig
from plancheck.pipeline import (
    STAGE_ORDER,
    SkipReason,
    StageResult,
    gate,
    input_fingerprint,
)


class TestStageOrder:
    def test_all_stages_present(self):
        assert "ingest" in STAGE_ORDER
        assert "tocr" in STAGE_ORDER
        assert "vocrpp" in STAGE_ORDER
        assert "vocr" in STAGE_ORDER
        assert "reconcile" in STAGE_ORDER

    def test_correct_order(self):
        assert STAGE_ORDER.index("ingest") < STAGE_ORDER.index("tocr")
        assert STAGE_ORDER.index("tocr") < STAGE_ORDER.index("vocr")
        assert STAGE_ORDER.index("vocr") < STAGE_ORDER.index("reconcile")


class TestGate:
    def test_ingest_always_runs(self):
        cfg = GroupingConfig()
        should_run, reason = gate("ingest", cfg)
        assert should_run is True
        assert reason is None

    def test_tocr_disabled(self):
        cfg = GroupingConfig(enable_tocr=False)
        should_run, reason = gate("tocr", cfg)
        assert should_run is False
        assert reason == SkipReason.disabled_by_config.value

    def test_tocr_enabled(self):
        cfg = GroupingConfig(enable_tocr=True)
        should_run, _ = gate("tocr", cfg)
        assert should_run is True

    def test_vocr_disabled(self):
        cfg = GroupingConfig(enable_vocr=False)
        should_run, reason = gate("vocr", cfg)
        assert should_run is False
        assert reason == SkipReason.disabled_by_config.value

    def test_vocrpp_requires_vocr(self):
        cfg = GroupingConfig(enable_ocr_preprocess=True, enable_vocr=False)
        should_run, reason = gate("vocrpp", cfg)
        assert should_run is False

    def test_reconcile_requires_vocr(self):
        cfg = GroupingConfig(enable_ocr_reconcile=True, enable_vocr=False)
        should_run, reason = gate("reconcile", cfg)
        assert should_run is False

    def test_unknown_stage(self):
        cfg = GroupingConfig()
        should_run, reason = gate("nonexistent", cfg)
        assert should_run is False
        assert reason == SkipReason.not_applicable.value


class TestStageResult:
    def test_to_dict_minimal(self):
        sr = StageResult(stage="test")
        d = sr.to_dict()
        assert d["stage"] == "test"
        assert d["status"] == "skipped"
        assert d["ran"] is False

    def test_to_dict_full(self):
        sr = StageResult(
            stage="tocr",
            enabled=True,
            ran=True,
            status="success",
            duration_ms=42,
            counts={"tokens": 100},
            error=None,
        )
        d = sr.to_dict()
        assert d["status"] == "success"
        assert d["duration_ms"] == 42
        assert d["counts"]["tokens"] == 100

    def test_to_dict_with_error(self):
        sr = StageResult(stage="vocr", error={"message": "boom"})
        d = sr.to_dict()
        assert d["error"]["message"] == "boom"


class TestInputFingerprint:
    def test_same_inputs_same_hash(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake pdf content")
        cfg = GroupingConfig()
        fp1 = input_fingerprint(pdf, [0, 1], cfg)
        fp2 = input_fingerprint(pdf, [0, 1], cfg)
        assert fp1 == fp2

    def test_different_pages_different_hash(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake pdf content")
        cfg = GroupingConfig()
        fp1 = input_fingerprint(pdf, [0], cfg)
        fp2 = input_fingerprint(pdf, [0, 1], cfg)
        assert fp1 != fp2

    def test_different_config_different_hash(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake pdf content")
        cfg1 = GroupingConfig(iou_prune=0.5)
        cfg2 = GroupingConfig(iou_prune=0.9)
        fp1 = input_fingerprint(pdf, [0], cfg1)
        fp2 = input_fingerprint(pdf, [0], cfg2)
        assert fp1 != fp2
