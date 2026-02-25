"""Tests for retrain_trigger module (Phase 4.2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plancheck.corrections.retrain_trigger import RetrainResult, check_retrain_needed
from plancheck.corrections.store import CorrectionStore

# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def store_with_corrections(tmp_path: Path, sample_features) -> CorrectionStore:
    """Store with some documents, detections, and corrections."""
    store = CorrectionStore(db_path=tmp_path / "test.db")

    # Insert a fake document row directly (bypasses pdfplumber)
    store._conn.execute(
        "INSERT INTO documents (doc_id, filename, page_count, ingested_at) "
        "VALUES (?, ?, ?, ?)",
        ("doc_test", "test.pdf", 1, "2025-01-01T00:00:00Z"),
    )
    store._conn.commit()

    doc_id = "doc_test"
    page = 1

    for i in range(60):
        det_id = store.save_detection(
            doc_id=doc_id,
            page=page,
            run_id="run_test",
            element_type="note",
            bbox=(10.0 + i, 20.0, 100.0, 50.0),
            text_content=f"text {i}",
            confidence=0.9,
            features=sample_features,
        )
        store.save_correction(
            doc_id=doc_id,
            page=page,
            correction_type="relabel",
            corrected_label="header",
            corrected_bbox=(10.0 + i, 20.0, 100.0, 50.0),
            detection_id=det_id,
            original_label="note",
            original_bbox=(10.0 + i, 20.0, 100.0, 50.0),
        )

    return store


# ── RetrainResult ──────────────────────────────────────────────────────


class TestRetrainResult:
    def test_default_state(self):
        r = RetrainResult()
        assert r.retrained is False
        assert r.accepted is False
        assert r.error == ""

    def test_to_dict(self):
        r = RetrainResult(retrained=True, accepted=True, metrics={"f1_weighted": 0.85})
        d = r.to_dict()
        assert d["retrained"] is True
        assert d["f1_weighted"] == 0.85


# ── check_retrain_needed ──────────────────────────────────────────────


class TestCheckRetrainNeeded:
    def test_below_threshold(self, tmp_store):
        assert check_retrain_needed(tmp_store, threshold=50) is False

    def test_above_threshold(self, store_with_corrections):
        assert check_retrain_needed(store_with_corrections, threshold=50) is True

    def test_custom_threshold(self, store_with_corrections):
        # 60 corrections, threshold=100 → not needed
        assert check_retrain_needed(store_with_corrections, threshold=100) is False


# ── Store retrain helpers ─────────────────────────────────────────────


class TestStoreRetrainHelpers:
    def test_last_train_date_none(self, tmp_store):
        assert tmp_store.last_train_date() is None

    def test_last_train_date_after_run(self, tmp_store):
        metrics = {
            "accuracy": 0.9,
            "f1_macro": 0.85,
            "f1_weighted": 0.88,
            "labels": ["note", "header"],
            "per_class": {},
            "n_train": 50,
            "n_val": 10,
        }
        tmp_store.save_training_run(metrics, notes="test")
        date = tmp_store.last_train_date()
        assert date is not None
        assert "T" in date  # ISO format

    def test_count_corrections_since_none(self, store_with_corrections):
        n = store_with_corrections.count_corrections_since(None)
        assert n == 60

    def test_count_corrections_since_far_future(self, store_with_corrections):
        n = store_with_corrections.count_corrections_since("2099-01-01T00:00:00Z")
        assert n == 0

    def test_count_corrections_since_last_train_no_runs(self, store_with_corrections):
        # No training runs → all corrections count
        n = store_with_corrections.count_corrections_since_last_train()
        assert n == 60

    def test_should_retrain(self, store_with_corrections):
        assert store_with_corrections.should_retrain(threshold=50) is True
        assert store_with_corrections.should_retrain(threshold=100) is False
