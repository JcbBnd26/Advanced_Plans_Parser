"""Tests for experiment_tracker module (Phase 4.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from plancheck.corrections.experiment_tracker import (
    ExperimentComparison,
    ExperimentSummary,
    ExperimentTracker,
)
from plancheck.corrections.store import CorrectionStore

# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def store_with_runs(tmp_store: CorrectionStore) -> CorrectionStore:
    """Store with three training runs of increasing quality."""
    for i, (acc, f1w) in enumerate([(0.70, 0.68), (0.80, 0.78), (0.90, 0.88)]):
        tmp_store.save_training_run(
            {
                "accuracy": acc,
                "f1_macro": f1w - 0.02,
                "f1_weighted": f1w,
                "labels": ["note", "header", "legend"],
                "per_class": {
                    "note": {"f1": f1w},
                    "header": {"f1": f1w - 0.1},
                    "legend": {"f1": f1w + 0.05},
                },
                "n_train": 50 + i * 10,
                "n_val": 10 + i * 2,
            },
            model_path=f"data/model_{i}.pkl",
            notes=f"run {i}",
            hyperparams={"n_estimators": 200, "run": i},
            feature_version=5,
        )
    return tmp_store


# ── ExperimentSummary ──────────────────────────────────────────────────


class TestExperimentSummary:
    def test_to_dict(self):
        es = ExperimentSummary(
            run_id="run_001",
            accuracy=0.92,
            f1_weighted=0.90,
        )
        d = es.to_dict()
        assert d["run_id"] == "run_001"
        assert d["accuracy"] == 0.92


# ── ExperimentTracker ──────────────────────────────────────────────────


class TestExperimentTracker:
    def test_list_experiments(self, store_with_runs):
        tracker = ExperimentTracker(store_with_runs)
        exps = tracker.list_experiments()
        assert len(exps) == 3
        notes = {e.notes for e in exps}
        assert notes == {"run 0", "run 1", "run 2"}

    def test_list_experiments_limit(self, store_with_runs):
        tracker = ExperimentTracker(store_with_runs)
        exps = tracker.list_experiments(limit=2)
        assert len(exps) == 2

    def test_list_experiments_sort_by_f1(self, store_with_runs):
        tracker = ExperimentTracker(store_with_runs)
        exps = tracker.list_experiments(sort_by="f1_weighted")
        # Best first
        assert exps[0].f1_weighted >= exps[1].f1_weighted

    def test_list_experiments_ascending(self, store_with_runs):
        tracker = ExperimentTracker(store_with_runs)
        exps = tracker.list_experiments(sort_by="accuracy", ascending=True)
        assert exps[0].accuracy <= exps[-1].accuracy

    def test_best_experiment(self, store_with_runs):
        tracker = ExperimentTracker(store_with_runs)
        best = tracker.best_experiment(metric="f1_weighted")
        assert best is not None
        assert best.f1_weighted == 0.88

    def test_best_experiment_empty(self, tmp_store):
        tracker = ExperimentTracker(tmp_store)
        assert tracker.best_experiment() is None

    def test_compare_experiments(self, store_with_runs):
        tracker = ExperimentTracker(store_with_runs)
        exps = tracker.list_experiments()
        # Compare oldest vs newest
        comp = tracker.compare_experiments(exps[-1].run_id, exps[0].run_id)
        assert isinstance(comp, ExperimentComparison)
        assert comp.f1_weighted_delta > 0  # newer is better

    def test_compare_experiments_to_dict(self, store_with_runs):
        tracker = ExperimentTracker(store_with_runs)
        exps = tracker.list_experiments()
        comp = tracker.compare_experiments(exps[-1].run_id, exps[0].run_id)
        d = comp.to_dict()
        assert "f1_weighted_delta" in d
        assert "run_a" in d
        assert "run_b" in d

    def test_export_csv(self, store_with_runs, tmp_path: Path):
        tracker = ExperimentTracker(store_with_runs)
        csv_path = tmp_path / "experiments.csv"
        n = tracker.export_csv(csv_path)
        assert n == 3
        assert csv_path.exists()
        lines = csv_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 4  # header + 3 rows
        assert "run_id" in lines[0]

    def test_export_csv_empty(self, tmp_store, tmp_path: Path):
        tracker = ExperimentTracker(tmp_store)
        n = tracker.export_csv(tmp_path / "empty.csv")
        assert n == 0


# ── Extended save_training_run ────────────────────────────────────────


class TestExtendedTrainingRun:
    def test_hyperparams_persisted(self, tmp_store):
        run_id = tmp_store.save_training_run(
            {
                "accuracy": 0.9,
                "f1_macro": 0.85,
                "f1_weighted": 0.88,
                "labels": [],
                "per_class": {},
                "n_train": 50,
                "n_val": 10,
            },
            hyperparams={"n_estimators": 200, "lr": 0.1},
            feature_version=5,
        )
        history = tmp_store.get_training_history()
        assert len(history) == 1
        assert history[0]["hyperparams"]["n_estimators"] == 200
        assert history[0]["feature_version"] == 5

    def test_feature_set_persisted(self, tmp_store):
        run_id = tmp_store.save_training_run(
            {
                "accuracy": 0.8,
                "f1_macro": 0.75,
                "f1_weighted": 0.78,
                "labels": ["note"],
                "per_class": {},
                "n_train": 30,
                "n_val": 5,
            },
            feature_set={"base_dim": 51, "total_dim": 51},
        )
        history = tmp_store.get_training_history()
        assert history[0]["feature_set"]["base_dim"] == 51

    def test_training_curves_persisted(self, tmp_store):
        curves = {"epoch": [1, 2, 3], "train_loss": [0.5, 0.3, 0.2]}
        run_id = tmp_store.save_training_run(
            {
                "accuracy": 0.8,
                "f1_macro": 0.75,
                "f1_weighted": 0.78,
                "labels": [],
                "per_class": {},
                "n_train": 30,
                "n_val": 5,
            },
            training_curves=curves,
        )
        history = tmp_store.get_training_history()
        assert history[0]["training_curves"]["epoch"] == [1, 2, 3]
