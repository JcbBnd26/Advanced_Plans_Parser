"""Tests for retrain_trigger module (Phase 4.2)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from plancheck.corrections.retrain_trigger import (
    RetrainResult,
    auto_retrain,
    check_retrain_needed,
)
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


class _FakeStore:
    def __init__(self, db_path: Path, export_rows: list[dict]) -> None:
        self._db_path = db_path
        self._export_rows = export_rows
        self.saved_runs: list[tuple[dict, str, str]] = []

    def count_corrections_since_last_train(self) -> int:
        return 20

    def build_training_set(self) -> int:
        return len(self._export_rows)

    def export_training_jsonl(self, output_path: Path) -> int:
        output_path.write_text(
            "\n".join(json.dumps(row) for row in self._export_rows) + "\n",
            encoding="utf-8",
        )
        return len(self._export_rows)

    def save_training_run(self, metrics, model_path="", notes="", **kwargs):
        self.saved_runs.append((metrics, model_path, notes))
        return f"run_{len(self.saved_runs)}"

    def get_training_history(self) -> list[dict]:
        return []


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


class TestAutoRetrainStage2:
    def test_trains_stage2_when_subtype_rows_are_present(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        sample_features: dict,
    ) -> None:
        rows = []
        for idx, label in enumerate(
            [
                "page_title",
                "plan_title",
                "page_title",
                "plan_title",
                "page_title",
                "plan_title",
                "page_title",
                "plan_title",
                "page_title",
                "plan_title",
            ]
        ):
            rows.append(
                {
                    "example_id": f"ex_{idx}",
                    "label": label,
                    "features": sample_features,
                    "split": "train" if idx < 8 else "val",
                }
            )
        rows.append(
            {
                "example_id": "ex_non_subtype",
                "label": "header",
                "features": sample_features,
                "split": "train",
            }
        )
        store = _FakeStore(tmp_path / "test.db", rows)

        monkeypatch.setattr(
            "plancheck.corrections.classifier.ElementClassifier.train",
            lambda self, path, calibrate=True, ensemble=False: {
                "accuracy": 0.9,
                "f1_weighted": 0.88,
                "n_train": 8,
                "n_val": 2,
                "holdout_predictions": [],
            },
        )

        captured_stage2: dict[str, object] = {}

        def _fake_stage2_train(self, path, calibrate=True, ensemble=False):
            jsonl_path = Path(path)
            captured_stage2["path"] = jsonl_path
            captured_stage2["lines"] = jsonl_path.read_text(
                encoding="utf-8"
            ).splitlines()
            return {
                "accuracy": 0.8,
                "f1_weighted": 0.77,
                "n_train": 8,
                "n_val": 2,
                "holdout_predictions": [],
            }

        monkeypatch.setattr(
            "plancheck.corrections.subtype_classifier.TitleSubtypeClassifier.train",
            _fake_stage2_train,
        )
        monkeypatch.setattr(
            "plancheck.corrections.drift_detection.DriftDetector",
            lambda threshold=0.3: SimpleNamespace(
                fit=lambda path: None,
                save=lambda path: None,
            ),
        )

        result = auto_retrain(
            store,
            model_path=tmp_path / "stage1.pkl",
            stage2_model_path=tmp_path / "stage2.pkl",
            threshold=0,
        )

        assert result.accepted is True
        assert result.stage2_trained is True
        assert result.stage2_metrics["f1_weighted"] == 0.77
        assert result.stage2_skipped_reason == ""
        assert len(store.saved_runs) == 2
        assert store.saved_runs[1][1] == str(tmp_path / "stage2.pkl")
        assert store.saved_runs[1][2] == "auto-retrain-stage2"
        assert captured_stage2["path"] == tmp_path / "training_data_stage2.jsonl"
        assert all('"label": "header"' not in line for line in captured_stage2["lines"])

    def test_skips_stage2_when_subtype_data_is_insufficient(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        sample_features: dict,
    ) -> None:
        rows = [
            {
                "example_id": f"ex_{idx}",
                "label": "page_title",
                "features": sample_features,
                "split": "train",
            }
            for idx in range(4)
        ]
        rows.extend(
            {
                "example_id": f"ex_header_{idx}",
                "label": "header",
                "features": sample_features,
                "split": "train",
            }
            for idx in range(6)
        )
        store = _FakeStore(tmp_path / "test.db", rows)

        monkeypatch.setattr(
            "plancheck.corrections.classifier.ElementClassifier.train",
            lambda self, path, calibrate=True, ensemble=False: {
                "accuracy": 0.9,
                "f1_weighted": 0.88,
                "n_train": 8,
                "n_val": 2,
                "holdout_predictions": [],
            },
        )
        monkeypatch.setattr(
            "plancheck.corrections.drift_detection.DriftDetector",
            lambda threshold=0.3: SimpleNamespace(
                fit=lambda path: None,
                save=lambda path: None,
            ),
        )

        result = auto_retrain(
            store,
            model_path=tmp_path / "stage1.pkl",
            stage2_model_path=tmp_path / "stage2.pkl",
            threshold=0,
        )

        assert result.accepted is True
        assert result.stage2_trained is False
        assert result.stage2_error == ""
        assert "only 4 subtype examples" in result.stage2_skipped_reason
        assert len(store.saved_runs) == 1


# ── Auto-rollback ─────────────────────────────────────────────────────


class TestAutoRetrainRollback:
    """Verify the F1-regression rollback logic in auto_retrain()."""

    def _make_rows(self, sample_features: dict, n: int = 20) -> list[dict]:
        labels = ["header", "notes_column", "legend", "abbreviation_table"]
        return [
            {
                "example_id": f"ex_{i}",
                "label": labels[i % len(labels)],
                "features": sample_features,
                "split": "train" if i < int(n * 0.8) else "val",
            }
            for i in range(n)
        ]

    def test_rolled_back_when_f1_regresses(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        sample_features: dict,
    ) -> None:
        """When the new F1 is lower than the previous run, the old model is restored."""
        rows = self._make_rows(sample_features)
        store = _FakeStore(tmp_path / "test.db", rows)

        # Simulate a prior training run with high F1
        monkeypatch.setattr(
            store,
            "get_training_history",
            lambda: [{"run_id": "prior_run_001", "f1_weighted": 0.90}],
        )

        # Mock train to produce a regressed F1 (no model file written)
        monkeypatch.setattr(
            "plancheck.corrections.classifier.ElementClassifier.train",
            lambda self, path, calibrate=True, ensemble=False: {
                "accuracy": 0.40,
                "f1_macro": 0.35,
                "f1_weighted": 0.35,
                "n_train": 16,
                "n_val": 4,
                "holdout_predictions": [],
            },
        )

        # Write a sentinel model so a backup can be created & restored
        model_path = tmp_path / "model.pkl"
        model_path.write_bytes(b"original-model-sentinel")

        result = auto_retrain(store, model_path=model_path, threshold=0)

        assert result.rolled_back is True
        assert result.accepted is False
        assert model_path.read_bytes() == b"original-model-sentinel"
        assert not model_path.with_suffix(".pkl.bak").exists()

    def test_not_rolled_back_when_f1_improves(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        sample_features: dict,
    ) -> None:
        """When F1 improves, the new model is accepted and backup is removed."""
        rows = self._make_rows(sample_features)
        store = _FakeStore(tmp_path / "test.db", rows)

        monkeypatch.setattr(
            store,
            "get_training_history",
            lambda: [{"run_id": "prior_run_001", "f1_weighted": 0.60}],
        )

        monkeypatch.setattr(
            "plancheck.corrections.classifier.ElementClassifier.train",
            lambda self, path, calibrate=True, ensemble=False: {
                "accuracy": 0.90,
                "f1_macro": 0.85,
                "f1_weighted": 0.88,
                "n_train": 16,
                "n_val": 4,
                "holdout_predictions": [],
            },
        )

        model_path = tmp_path / "model.pkl"
        model_path.write_bytes(b"original")

        result = auto_retrain(store, model_path=model_path, threshold=0)

        assert result.rolled_back is False
        assert result.accepted is True
        # Backup file should have been removed after successful accept
        assert not model_path.with_suffix(".pkl.bak").exists()

    def test_no_rollback_without_prior_history(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        sample_features: dict,
    ) -> None:
        """When there are no prior runs, rollback is never triggered."""
        rows = self._make_rows(sample_features)
        store = _FakeStore(tmp_path / "test.db", rows)
        # get_training_history already returns [] in _FakeStore, so no patch needed

        monkeypatch.setattr(
            "plancheck.corrections.classifier.ElementClassifier.train",
            lambda self, path, calibrate=True, ensemble=False: {
                "accuracy": 0.70,
                "f1_macro": 0.68,
                "f1_weighted": 0.69,
                "n_train": 16,
                "n_val": 4,
                "holdout_predictions": [],
            },
        )

        model_path = tmp_path / "model.pkl"
        model_path.write_bytes(b"original")

        result = auto_retrain(store, model_path=model_path, threshold=0)

        assert result.rolled_back is False
        assert result.accepted is True
