"""Tests for Level 2 — candidate features, classifier, and outcome store."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from plancheck.corrections.candidate_features import (
    CANDIDATE_FEATURE_DIM,
    CANDIDATE_FEATURE_VERSION,
    featurize_candidate,
    featurize_candidates_batch,
    featurize_outcome_row,
)
from plancheck.corrections.store import CorrectionStore
from plancheck.models import VocrCandidate

# ── Feature extraction ──────────────────────────────────────────────────


class TestFeaturizeCandidate:
    """Tests for featurize_candidate."""

    def _make_candidate(self, **overrides):
        defaults = dict(
            page=0,
            x0=100,
            y0=200,
            x1=120,
            y1=210,
            trigger_methods=["placeholder_token"],
            predicted_symbol="%",
            confidence=0.8,
            context={"gap_pts": 5.0, "neighbor_text": "hello"},
        )
        defaults.update(overrides)
        return VocrCandidate(**defaults)

    def test_output_shape(self):
        c = self._make_candidate()
        vec = featurize_candidate(c, 612, 792)
        assert vec.shape == (CANDIDATE_FEATURE_DIM,)
        assert vec.dtype == np.float32

    def test_trigger_method_one_hot(self):
        c = self._make_candidate(trigger_methods=["placeholder_token"])
        vec = featurize_candidate(c)
        # placeholder_token is index 2 in VOCR_TRIGGER_METHODS
        assert vec[2] == 1.0
        # Other method slots should be 0
        assert vec[0] == 0.0

    def test_confidence_in_vector(self):
        c = self._make_candidate(confidence=0.75)
        vec = featurize_candidate(c)
        assert vec[18] == pytest.approx(0.75)  # offset 18 = confidence

    def test_bbox_normalised(self):
        c = self._make_candidate(x0=306, y0=396, x1=306, y1=396)
        vec = featurize_candidate(c, 612, 792)
        assert vec[19] == pytest.approx(0.5)  # x0/pw
        assert vec[20] == pytest.approx(0.5)  # y0/ph

    def test_predicted_symbol_one_hot(self):
        c = self._make_candidate(predicted_symbol="°")
        vec = featurize_candidate(c)
        assert vec[25 + 1] == 1.0  # ° is index 1

    def test_unknown_symbol_uses_other(self):
        c = self._make_candidate(predicted_symbol="★")
        vec = featurize_candidate(c)
        assert vec[25 + 3] == 1.0  # "other" slot

    def test_context_features(self):
        c = self._make_candidate(context={"gap_pts": 12.5, "neighbor_text": "abc"})
        vec = featurize_candidate(c)
        assert vec[29] == pytest.approx(12.5)  # gap_pts
        assert vec[30] == pytest.approx(3.0)  # len("abc")

    def test_empty_context(self):
        c = self._make_candidate(context={})
        vec = featurize_candidate(c)
        assert vec[29] == 0.0
        assert vec[30] == 0.0


class TestFeaturizeBatch:
    def test_batch_shape(self):
        cands = [
            VocrCandidate(page=0, x0=10, y0=20, x1=30, y1=40),
            VocrCandidate(page=0, x0=50, y0=60, x1=70, y1=80),
        ]
        X = featurize_candidates_batch(cands)
        assert X.shape == (2, CANDIDATE_FEATURE_DIM)

    def test_empty_batch(self):
        X = featurize_candidates_batch([])
        assert X.shape == (0, CANDIDATE_FEATURE_DIM)


class TestFeaturizeOutcomeRow:
    def test_from_db_row(self):
        row = {
            "trigger_methods": "placeholder_token,intraline_gap",
            "confidence": 0.65,
            "bbox_x0": 100,
            "bbox_y0": 200,
            "bbox_x1": 120,
            "bbox_y1": 210,
            "page_width": 612,
            "page_height": 792,
            "predicted_symbol": "%",
            "features_json": "{}",
        }
        vec = featurize_outcome_row(row)
        assert vec.shape == (CANDIDATE_FEATURE_DIM,)
        # Both methods should be flagged
        assert vec[2] == 1.0  # placeholder_token
        assert vec[3] == 1.0  # intraline_gap


# ── Candidate outcomes store ────────────────────────────────────────────


class TestCandidateOutcomeStore:
    """Tests for CorrectionStore candidate outcome methods."""

    @pytest.fixture()
    def store(self, tmp_path: Path):
        s = CorrectionStore(tmp_path / "test.db")
        yield s
        s.close()

    def test_save_and_retrieve(self, store: CorrectionStore):
        oid = store.save_candidate_outcome(
            page=0,
            trigger_methods=["placeholder_token"],
            outcome="hit",
            confidence=0.8,
            bbox=(10, 20, 30, 40),
        )
        assert oid.startswith("co_")
        rows = store.get_candidate_outcomes()
        assert len(rows) == 1
        assert rows[0]["outcome"] == "hit"

    def test_count_outcomes(self, store: CorrectionStore):
        store.save_candidate_outcome(
            page=0,
            trigger_methods=["a"],
            outcome="hit",
            confidence=0.5,
            bbox=(0, 0, 1, 1),
        )
        store.save_candidate_outcome(
            page=0,
            trigger_methods=["b"],
            outcome="miss",
            confidence=0.3,
            bbox=(0, 0, 1, 1),
        )
        counts = store.count_candidate_outcomes()
        assert counts["total"] == 2
        assert counts["hits"] == 1
        assert counts["misses"] == 1

    def test_min_rows_filter(self, store: CorrectionStore):
        store.save_candidate_outcome(
            page=0,
            trigger_methods=["a"],
            outcome="hit",
            confidence=0.5,
            bbox=(0, 0, 1, 1),
        )
        # min_rows=10 → should return empty
        rows = store.get_candidate_outcomes(min_rows=10)
        assert rows == []

    def test_outcome_filter(self, store: CorrectionStore):
        store.save_candidate_outcome(
            page=0,
            trigger_methods=["a"],
            outcome="hit",
            confidence=0.5,
            bbox=(0, 0, 1, 1),
        )
        store.save_candidate_outcome(
            page=0,
            trigger_methods=["b"],
            outcome="miss",
            confidence=0.3,
            bbox=(0, 0, 1, 1),
        )
        hits = store.get_candidate_outcomes(outcome="hit")
        assert len(hits) == 1
        assert hits[0]["outcome"] == "hit"

    def test_batch_save(self, store: CorrectionStore):
        cands = [
            VocrCandidate(
                page=0,
                x0=10,
                y0=20,
                x1=30,
                y1=40,
                trigger_methods=["a"],
                outcome="hit",
                confidence=0.8,
            ),
            VocrCandidate(
                page=0,
                x0=50,
                y0=60,
                x1=70,
                y1=80,
                trigger_methods=["b"],
                outcome="miss",
                confidence=0.3,
            ),
            VocrCandidate(
                page=0,
                x0=90,
                y0=100,
                x1=110,
                y1=120,
                trigger_methods=["c"],
                outcome="pending",
                confidence=0.5,
            ),
        ]
        n = store.save_candidate_outcomes_batch(cands)
        assert n == 2  # "pending" is skipped

    def test_features_roundtrip(self, store: CorrectionStore):
        store.save_candidate_outcome(
            page=0,
            trigger_methods=["a"],
            outcome="hit",
            confidence=0.5,
            bbox=(0, 0, 1, 1),
            features={"gap_pts": 3.5},
        )
        rows = store.get_candidate_outcomes()
        feat = json.loads(rows[0]["features_json"])
        assert feat["gap_pts"] == 3.5


# ── Candidate classifier ───────────────────────────────────────────────


class TestCandidateClassifier:
    """Tests for CandidateClassifier."""

    def test_load_nonexistent_returns_false(self, tmp_path: Path):
        from plancheck.corrections.candidate_classifier import CandidateClassifier

        clf = CandidateClassifier(tmp_path / "nope.pkl")
        assert clf.load() is False
        assert clf.is_ready is False

    def test_predict_without_model_returns_ones(self):
        from plancheck.corrections.candidate_classifier import CandidateClassifier

        clf = CandidateClassifier()
        cands = [
            VocrCandidate(page=0, x0=10, y0=20, x1=30, y1=40),
        ]
        probs = clf.predict(cands)
        assert len(probs) == 1
        assert probs[0] == pytest.approx(1.0)

    def test_filter_without_model_passes_all(self):
        from plancheck.corrections.candidate_classifier import CandidateClassifier

        clf = CandidateClassifier()
        cands = [
            VocrCandidate(page=0, x0=10, y0=20, x1=30, y1=40),
        ]
        result = clf.filter_candidates(cands, threshold=0.9)
        assert len(result) == 1


class TestTrainCandidateClassifier:
    """Test the full train → predict cycle with synthetic data."""

    @pytest.fixture()
    def synthetic_outcomes(self):
        """Generate 200 synthetic outcome rows for training."""
        rng = np.random.RandomState(42)
        rows = []
        for i in range(200):
            is_hit = i % 3 != 0  # ~67% hit rate
            rows.append(
                {
                    "trigger_methods": (
                        "placeholder_token" if is_hit else "dense_cluster_hole"
                    ),
                    "confidence": (
                        rng.uniform(0.5, 0.9) if is_hit else rng.uniform(0.2, 0.5)
                    ),
                    "bbox_x0": rng.uniform(0, 500),
                    "bbox_y0": rng.uniform(0, 700),
                    "bbox_x1": rng.uniform(0, 500) + 20,
                    "bbox_y1": rng.uniform(0, 700) + 10,
                    "page_width": 612,
                    "page_height": 792,
                    "predicted_symbol": "%" if is_hit else "",
                    "features_json": "{}",
                    "outcome": "hit" if is_hit else "miss",
                }
            )
        return rows

    def test_train_and_predict(self, tmp_path: Path, synthetic_outcomes):
        from plancheck.corrections.candidate_classifier import (
            CandidateClassifier,
            train_candidate_classifier,
        )

        model_path = tmp_path / "clf.pkl"
        metrics = train_candidate_classifier(synthetic_outcomes, model_path=model_path)

        assert model_path.exists()
        assert metrics["accuracy"] > 0.5
        assert "f1" in metrics
        assert "auc_roc" in metrics

        # Load and predict
        clf = CandidateClassifier(model_path)
        assert clf.load() is True
        assert clf.is_ready

        cands = [
            VocrCandidate(
                page=0,
                x0=100,
                y0=200,
                x1=120,
                y1=210,
                trigger_methods=["placeholder_token"],
                confidence=0.8,
                predicted_symbol="%",
            ),
        ]
        probs = clf.predict(cands, 612, 792)
        assert len(probs) == 1
        assert 0.0 <= probs[0] <= 1.0

    def test_single_class_returns_error(self, tmp_path: Path):
        from plancheck.corrections.candidate_classifier import (
            train_candidate_classifier,
        )

        rows = [
            {
                "trigger_methods": "a",
                "confidence": 0.5,
                "bbox_x0": 0,
                "bbox_y0": 0,
                "bbox_x1": 10,
                "bbox_y1": 10,
                "page_width": 612,
                "page_height": 792,
                "predicted_symbol": "",
                "features_json": "{}",
                "outcome": "hit",
            }
        ] * 20
        result = train_candidate_classifier(rows, model_path=tmp_path / "c.pkl")
        assert result.get("error") == "single_class"
