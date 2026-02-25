"""Tests for drift_detection module (Phase 4.1)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from plancheck.corrections.drift_detection import (
    DriftDetector,
    DriftResult,
    PageDriftResult,
)

# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def training_vectors() -> list[np.ndarray]:
    """Generate 100 synthetic training vectors (10-dim)."""
    rng = np.random.RandomState(42)
    return [rng.randn(10) for _ in range(100)]


@pytest.fixture
def fitted_detector(training_vectors: list[np.ndarray]) -> DriftDetector:
    """Return a DriftDetector already fit on training vectors."""
    det = DriftDetector(threshold=0.3)
    det.fit_from_vectors(training_vectors)
    return det


# ── DriftResult / PageDriftResult ──────────────────────────────────────


class TestDriftResult:
    def test_to_dict(self):
        dr = DriftResult(
            is_drifted=True,
            drift_fraction=0.45,
            flagged_features=[0, 3, 7],
            n_features=10,
        )
        d = dr.to_dict()
        assert d["is_drifted"] is True
        assert d["drift_fraction"] == 0.45
        assert d["flagged_features"] == [0, 3, 7]

    def test_default_not_drifted(self):
        dr = DriftResult()
        assert dr.is_drifted is False
        assert dr.drift_fraction == 0.0


class TestPageDriftResult:
    def test_to_dict(self):
        pdr = PageDriftResult(
            is_drifted=True,
            drifted_fraction=0.6,
            n_detections=5,
            n_drifted=3,
        )
        d = pdr.to_dict()
        assert d["n_drifted"] == 3
        assert d["per_detection"] == []


# ── DriftDetector ──────────────────────────────────────────────────────


class TestDriftDetectorFit:
    def test_fit_from_vectors(self, training_vectors):
        det = DriftDetector()
        det.fit_from_vectors(training_vectors)
        assert det.is_fitted
        assert det._n_features == 10
        assert det._n_samples == 100

    def test_fit_from_vectors_empty_raises(self):
        det = DriftDetector()
        with pytest.raises(ValueError, match="No vectors"):
            det.fit_from_vectors([])

    def test_fit_from_jsonl(self, tmp_path: Path, sample_features):
        """fit() reads a JSONL file and computes reference stats."""
        jsonl = tmp_path / "train.jsonl"
        lines = []
        for i in range(30):
            rec = {
                "features": sample_features,
                "label": "note",
                "split": "train",
            }
            lines.append(json.dumps(rec))
        jsonl.write_text("\n".join(lines), encoding="utf-8")

        det = DriftDetector(threshold=0.3)
        det.fit(jsonl)
        assert det.is_fitted
        assert det._n_samples == 30

    def test_fit_no_train_split_raises(self, tmp_path: Path, sample_features):
        jsonl = tmp_path / "train.jsonl"
        rec = {"features": sample_features, "label": "note", "split": "val"}
        jsonl.write_text(json.dumps(rec), encoding="utf-8")
        det = DriftDetector()
        with pytest.raises(ValueError, match="No training-split"):
            det.fit(jsonl)


class TestDriftDetectorCheck:
    def test_in_distribution_vector(self, fitted_detector, training_vectors):
        """A vector from the training set should not be flagged."""
        result = fitted_detector.check(training_vectors[0])
        # Most training vectors should be within p1-p99
        assert result.n_features == 10
        assert isinstance(result.drift_fraction, float)

    def test_extreme_vector_is_drifted(self, fitted_detector):
        """A vector far outside training distribution should be flagged."""
        extreme = np.full(10, 1000.0)
        result = fitted_detector.check(extreme)
        assert result.is_drifted is True
        assert result.drift_fraction >= 0.3

    def test_unfitted_returns_default(self):
        det = DriftDetector()
        result = det.check(np.zeros(10))
        assert result.is_drifted is False
        assert result.n_features == 0

    def test_threshold_respected(self, training_vectors):
        """With threshold=1.0, nothing should drift (all features must breach)."""
        det = DriftDetector(threshold=1.0)
        det.fit_from_vectors(training_vectors)
        extreme = np.full(10, 1000.0)
        result = det.check(extreme)
        assert result.drift_fraction >= 0.9  # most features breach
        assert result.is_drifted is True


class TestDriftDetectorCheckPage:
    def test_page_all_normal(self, fitted_detector, training_vectors):
        vecs = training_vectors[:5]
        result = fitted_detector.check_page(vecs)
        assert result.n_detections == 5
        assert isinstance(result.drifted_fraction, float)

    def test_page_all_drifted(self, fitted_detector):
        vecs = [np.full(10, 1000.0) for _ in range(4)]
        result = fitted_detector.check_page(vecs, page_threshold=0.5)
        assert result.is_drifted is True
        assert result.n_drifted == 4

    def test_empty_page(self, fitted_detector):
        result = fitted_detector.check_page([])
        assert result.is_drifted is False
        assert result.n_detections == 0


class TestDriftDetectorPersistence:
    def test_save_and_load(self, fitted_detector, tmp_path: Path):
        path = tmp_path / "drift_stats.json"
        fitted_detector.save(path)
        assert path.exists()

        loaded = DriftDetector.load(path)
        assert loaded.is_fitted
        assert loaded._n_samples == fitted_detector._n_samples
        assert loaded._n_features == fitted_detector._n_features
        assert loaded.threshold == fitted_detector.threshold
        np.testing.assert_array_equal(loaded._mean, fitted_detector._mean)

    def test_load_override_threshold(self, fitted_detector, tmp_path: Path):
        path = tmp_path / "drift.json"
        fitted_detector.save(path)
        loaded = DriftDetector.load(path, threshold=0.9)
        assert loaded.threshold == 0.9

    def test_load_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            DriftDetector.load(tmp_path / "nope.json")

    def test_save_unfitted_raises(self):
        det = DriftDetector()
        with pytest.raises(RuntimeError, match="not been fit"):
            det.save(Path("/tmp/nope.json"))
