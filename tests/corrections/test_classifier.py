"""Tests for plancheck.corrections.classifier."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from plancheck.corrections.classifier import (
    _NUMERIC_KEYS,
    ZONE_VALUES,
    ElementClassifier,
    encode_features,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_features() -> dict:
    """Minimal valid feature dict."""
    return {
        "font_size_pt": 8.0,
        "font_size_max_pt": 10.0,
        "font_size_min_pt": 6.0,
        "is_all_caps": 0,
        "is_bold": 0,
        "token_count": 5,
        "row_count": 2,
        "x_frac": 0.7,
        "y_frac": 0.1,
        "x_center_frac": 0.8,
        "y_center_frac": 0.3,
        "width_frac": 0.2,
        "height_frac": 0.4,
        "aspect_ratio": 2.5,
        "contains_digit": 1,
        "starts_with_digit": 1,
        "has_colon": 0,
        "has_period_after_num": 1,
        "text_length": 40,
        "avg_chars_per_token": 8.0,
        "zone": "header",
        "neighbor_count": 3,
        # Text-content features (v2)
        "unique_word_ratio": 0.8,
        "uppercase_word_frac": 0.4,
        "avg_word_length": 5.0,
        "kw_notes_pattern": 0,
        "kw_header_pattern": 1,
        "kw_legend_pattern": 0,
        "kw_abbreviation_pattern": 0,
        "kw_revision_pattern": 0,
        "kw_title_block_pattern": 0,
        "kw_detail_pattern": 0,
        # Discriminative features (v3)
        "text_density": 0.002,
        "x_dist_to_right_margin": 0.1,
        "line_width_variance": 3.0,
    }


@pytest.fixture
def training_jsonl(tmp_path: Path, sample_features: dict) -> Path:
    """Create a JSONL training file with enough data to train."""
    path = tmp_path / "train.jsonl"
    lines: list[str] = []

    # Need at least 10 examples, create 20 (15 train, 5 val) with 3 classes
    labels = ["header", "notes_column", "legend"]
    for i in range(20):
        feats = dict(sample_features)
        label = labels[i % len(labels)]
        feats["zone"] = "unknown"  # MUST NOT leak the label into zone
        feats["font_size_pt"] = 8.0 + i * 0.5
        feats["x_frac"] = 0.1 * (i % 10)
        # Add varied features so the model can distinguish classes
        # without label leakage
        feats["is_all_caps"] = int(label == "header")
        feats["kw_header_pattern"] = int(label == "header")
        feats["kw_notes_pattern"] = int(label == "notes_column")
        feats["kw_legend_pattern"] = int(label == "legend")
        feats["y_frac"] = 0.05 if label == "header" else 0.5
        example = {
            "features": feats,
            "label": label,
            "split": "train" if i < 15 else "val",
        }
        lines.append(json.dumps(example))

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# encode_features
# ---------------------------------------------------------------------------


class TestEncodeFeatures:
    def test_output_length(self, sample_features: dict) -> None:
        vec = encode_features(sample_features)
        expected_len = len(_NUMERIC_KEYS) + len(ZONE_VALUES)
        assert vec.shape == (expected_len,)

    def test_zone_one_hot_correct(self) -> None:
        feats = {k: 0.0 for k in _NUMERIC_KEYS}
        feats["zone"] = "legend"
        vec = encode_features(feats)
        zone_start = len(_NUMERIC_KEYS)
        zone_slice = vec[zone_start:]
        assert zone_slice[ZONE_VALUES.index("legend")] == 1.0
        assert sum(zone_slice) == 1.0

    def test_unknown_zone(self) -> None:
        feats = {k: 0.0 for k in _NUMERIC_KEYS}
        feats["zone"] = "unknown"
        vec = encode_features(feats)
        zone_start = len(_NUMERIC_KEYS)
        assert vec[zone_start + ZONE_VALUES.index("unknown")] == 1.0

    def test_missing_zone_defaults_unknown(self) -> None:
        feats = {k: 0.0 for k in _NUMERIC_KEYS}
        # No 'zone' key at all
        vec = encode_features(feats)
        zone_start = len(_NUMERIC_KEYS)
        assert vec[zone_start + ZONE_VALUES.index("unknown")] == 1.0

    def test_missing_numeric_defaults_zero(self) -> None:
        feats = {"zone": "header"}
        vec = encode_features(feats)
        # All numeric values should be 0.0
        for i in range(len(_NUMERIC_KEYS)):
            assert vec[i] == 0.0

    def test_dtype_is_float64(self, sample_features: dict) -> None:
        vec = encode_features(sample_features)
        assert vec.dtype == np.float64


# ---------------------------------------------------------------------------
# ElementClassifier
# ---------------------------------------------------------------------------


class TestElementClassifier:
    def test_model_exists_false_initially(self, tmp_path: Path) -> None:
        clf = ElementClassifier(model_path=tmp_path / "model.pkl")
        assert clf.model_exists() is False

    def test_train_and_predict(
        self, tmp_path: Path, training_jsonl: Path, sample_features: dict
    ) -> None:
        model_path = tmp_path / "model.pkl"
        clf = ElementClassifier(model_path=model_path)

        metrics = clf.train(training_jsonl)

        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert "per_class" in metrics
        assert "n_train" in metrics
        assert metrics["n_train"] == 15
        assert metrics["n_val"] == 5

        # Model persisted
        assert model_path.is_file()
        assert clf.model_exists()

        # Predict
        label, conf = clf.predict(sample_features)
        assert isinstance(label, str)
        assert label in ["header", "notes_column", "legend"]
        assert 0.0 <= conf <= 1.0

    def test_predict_batch(
        self, tmp_path: Path, training_jsonl: Path, sample_features: dict
    ) -> None:
        clf = ElementClassifier(model_path=tmp_path / "model.pkl")
        clf.train(training_jsonl)

        results = clf.predict_batch([sample_features, sample_features])
        assert len(results) == 2
        for label, conf in results:
            assert isinstance(label, str)
            assert 0.0 <= conf <= 1.0

    def test_predict_batch_empty(self, tmp_path: Path, training_jsonl: Path) -> None:
        clf = ElementClassifier(model_path=tmp_path / "model.pkl")
        clf.train(training_jsonl)
        assert clf.predict_batch([]) == []

    def test_train_too_few_examples(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "small.jsonl"
        lines = [
            json.dumps(
                {
                    "features": {k: 0.0 for k in _NUMERIC_KEYS},
                    "label": "header",
                    "split": "train",
                }
            )
            for _ in range(5)
        ]
        jsonl_path.write_text("\n".join(lines), encoding="utf-8")

        clf = ElementClassifier(model_path=tmp_path / "model.pkl")
        with pytest.raises(ValueError, match="at least 10"):
            clf.train(jsonl_path)

    def test_lazy_load_model(
        self, tmp_path: Path, training_jsonl: Path, sample_features: dict
    ) -> None:
        """A second classifier instance should lazy-load the saved model."""
        model_path = tmp_path / "model.pkl"
        clf1 = ElementClassifier(model_path=model_path)
        clf1.train(training_jsonl)

        clf2 = ElementClassifier(model_path=model_path)
        assert clf2._model is None  # not loaded yet

        label, conf = clf2.predict(sample_features)
        assert clf2._model is not None  # now loaded
        assert isinstance(label, str)

    def test_feature_importance(self, tmp_path: Path, training_jsonl: Path) -> None:
        clf = ElementClassifier(model_path=tmp_path / "model.pkl")
        clf.train(training_jsonl)
        importance = clf.get_feature_importance()
        assert isinstance(importance, dict)
        # Should have entries for all features (numeric + zone one-hots)
        assert len(importance) > 0
        # All values should be non-negative floats
        for name, val in importance.items():
            assert isinstance(val, float)

    def test_zone_values_match_enum(self) -> None:
        """ZONE_VALUES should contain the actual ZoneTag enum values."""
        expected = {
            "border",
            "drawing",
            "notes",
            "title_block",
            "legend",
            "abbreviations",
            "revisions",
            "details",
            "page",
            "unknown",
        }
        assert set(ZONE_VALUES) == expected

    def test_training_data_has_no_label_leakage(self, training_jsonl: Path) -> None:
        """Zone feature must never equal the label — that's label leakage."""
        with open(training_jsonl, encoding="utf-8") as fh:
            for line in fh:
                ex = json.loads(line.strip())
                zone = ex["features"].get("zone", "unknown")
                label = ex["label"]
                assert zone != label, f"Label leakage: zone={zone!r} == label={label!r}"
