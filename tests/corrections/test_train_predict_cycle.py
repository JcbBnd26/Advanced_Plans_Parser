"""End-to-end integration test: train → serialize → load → predict.

Covers the correction → train → predict round-trip that the GUI training
workflow exercises.  Marked ``slow`` and ``integration`` so they can be
skipped in fast unit-test runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plancheck.corrections.classifier import ElementClassifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, sample_features: dict, n: int = 100) -> None:
    """Write a JSONL training file with 4 classes, evenly distributed."""
    labels = ["header", "notes_column", "legend", "abbreviation_table"]
    records: list[str] = []
    for i in range(n):
        feats = dict(sample_features)
        feats["zone"] = "unknown"  # must not leak label via zone
        feats["font_size_pt"] = 6.0 + (i % 20) * 0.5
        feats["x_frac"] = round((i % 10) * 0.1, 2)
        records.append(
            json.dumps(
                {
                    "example_id": f"ex_{i}",
                    "label": labels[i % len(labels)],
                    "features": feats,
                    "split": "train" if i < int(n * 0.8) else "val",
                }
            )
        )
    path.write_text("\n".join(records) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
class TestTrainPredictCycle:
    """Full train → serialise → load → predict round-trip."""

    def test_train_writes_model_and_returns_metrics(
        self, tmp_path: Path, sample_features: dict
    ) -> None:
        model_path = tmp_path / "model.pkl"
        jsonl_path = tmp_path / "train.jsonl"
        _write_jsonl(jsonl_path, sample_features)

        clf = ElementClassifier(model_path=model_path)
        metrics = clf.train(str(jsonl_path))

        assert model_path.exists(), "Model file was not written after training"
        assert isinstance(metrics, dict)
        assert 0.0 <= metrics.get("f1_weighted", -1.0) <= 1.0
        assert metrics.get("n_train", 0) >= 10

    def test_fresh_load_predicts_trained_label(
        self, tmp_path: Path, sample_features: dict
    ) -> None:
        """A fresh ElementClassifier loaded from disk produces a valid prediction."""
        model_path = tmp_path / "model.pkl"
        jsonl_path = tmp_path / "train.jsonl"
        _write_jsonl(jsonl_path, sample_features)

        # Train with first instance
        clf = ElementClassifier(model_path=model_path)
        clf.train(str(jsonl_path))

        # Load with a fresh instance (simulates GUI reload after training)
        fresh = ElementClassifier(model_path=model_path)
        label, confidence = fresh.predict(sample_features)

        known = {"header", "notes_column", "legend", "abbreviation_table"}
        assert label in known, f"Predicted unknown label: {label!r}"
        assert 0.0 <= confidence <= 1.0, f"Confidence out of range: {confidence}"

    def test_predict_with_sparse_features_does_not_raise(
        self, tmp_path: Path, sample_features: dict
    ) -> None:
        """predict() must not raise when optional v2/v3/v4 feature keys are absent."""
        model_path = tmp_path / "model.pkl"
        jsonl_path = tmp_path / "train.jsonl"
        _write_jsonl(jsonl_path, sample_features)

        clf = ElementClassifier(model_path=model_path)
        clf.train(str(jsonl_path))

        # Only core v1 keys — strip out v2 keyword patterns and v3/v4 features
        core_keys = {
            "font_size_pt",
            "font_size_max_pt",
            "font_size_min_pt",
            "is_all_caps",
            "is_bold",
            "token_count",
            "row_count",
            "x_frac",
            "y_frac",
            "x_center_frac",
            "y_center_frac",
            "width_frac",
            "height_frac",
            "aspect_ratio",
            "contains_digit",
            "starts_with_digit",
            "has_colon",
            "has_period_after_num",
            "text_length",
            "avg_chars_per_token",
            "zone",
            "neighbor_count",
        }
        sparse = {k: v for k, v in sample_features.items() if k in core_keys}

        label, confidence = clf.predict(sparse)
        assert isinstance(label, str)
        assert 0.0 <= confidence <= 1.0
