"""Element-type classifier using scikit-learn GradientBoosting.

Trains on JSONL exported by :meth:`CorrectionStore.export_training_jsonl`
and predicts ``(label, confidence)`` for new feature dicts.

Uses :class:`~sklearn.ensemble.GradientBoostingClassifier` with balanced
class weighting via ``compute_sample_weight``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Zone values for one-hot encoding — must match ZoneTag enum in analysis.zoning
ZONE_VALUES: list[str] = [
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
]

# Numeric feature keys (order matters — must be consistent between train & predict)
_NUMERIC_KEYS: list[str] = [
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
    "neighbor_count",
    # Text-content features (added in v2)
    "unique_word_ratio",
    "uppercase_word_frac",
    "avg_word_length",
    "kw_notes_pattern",
    "kw_header_pattern",
    "kw_legend_pattern",
    "kw_abbreviation_pattern",
    "kw_revision_pattern",
    "kw_title_block_pattern",
    "kw_detail_pattern",
]

_DEFAULT_MODEL_PATH = Path("data/element_classifier.pkl")


def encode_features(feature_dict: dict) -> np.ndarray:
    """Convert a feature dict to a flat float array.

    The ``zone`` string is one-hot encoded into binary columns
    (one per :data:`ZONE_VALUES`).  All other keys are cast to float.

    Parameters
    ----------
    feature_dict : dict
        Output of :func:`~plancheck.corrections.features.featurize`.

    Returns
    -------
    numpy.ndarray
        1-D float64 array of length ``len(_NUMERIC_KEYS) + len(ZONE_VALUES)``.
    """
    numeric = [float(feature_dict.get(k, 0.0)) for k in _NUMERIC_KEYS]
    zone = feature_dict.get("zone", "unknown")
    one_hot = [1.0 if zone == z else 0.0 for z in ZONE_VALUES]
    return np.array(numeric + one_hot, dtype=np.float64)


class ElementClassifier:
    """Gradient-boosted element-type classifier.

    Wraps ``sklearn.ensemble.GradientBoostingClassifier`` with helpers
    for encoding features, training from JSONL, and predicting labels.
    Uses balanced sample weights to handle class imbalance.

    Parameters
    ----------
    model_path : Path
        Location of the persisted ``.pkl`` model file.
    """

    def __init__(self, model_path: Path = _DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self._model = None
        self._classes: list[str] = []

    # ── Training ───────────────────────────────────────────────────

    def train(self, jsonl_path: Path) -> dict:
        """Train a new model from exported JSONL.

        Parameters
        ----------
        jsonl_path : Path
            Path to the JSONL file written by
            :meth:`CorrectionStore.export_training_jsonl`.

        Returns
        -------
        dict
            Training metrics: ``accuracy``, ``per_class``,
            ``confusion_matrix``, ``labels``, ``n_train``, ``n_val``.

        Raises
        ------
        ValueError
            If fewer than 10 training examples are available.
        """
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.utils.class_weight import compute_sample_weight

        from .metrics import compute_metrics

        # Load data
        examples: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))

        if len(examples) < 10:
            raise ValueError(f"Need at least 10 training examples, got {len(examples)}")

        # Split
        train_ex = [e for e in examples if e.get("split") == "train"]
        val_ex = [e for e in examples if e.get("split") == "val"]

        if not train_ex:
            raise ValueError("No training-split examples found")

        X_train = np.array([encode_features(e["features"]) for e in train_ex])
        y_train = [e["label"] for e in train_ex]

        # Balanced class weighting
        sample_weights = compute_sample_weight("balanced", y_train)

        # Fit
        clf = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
        clf.fit(X_train, y_train, sample_weight=sample_weights)

        # Evaluate on validation set (or training set if no val)
        eval_ex = val_ex if val_ex else train_ex
        X_eval = np.array([encode_features(e["features"]) for e in eval_ex])
        y_eval = [e["label"] for e in eval_ex]
        y_pred = clf.predict(X_eval).tolist()

        labels = sorted(set(y_train) | set(y_eval))
        metrics = compute_metrics(y_eval, y_pred, labels=labels)
        metrics["n_train"] = len(train_ex)
        metrics["n_val"] = len(val_ex)

        # Persist
        import joblib

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": clf, "classes": labels}, self.model_path)

        self._model = clf
        self._classes = labels

        return metrics

    # ── Prediction ─────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Lazy-load the model from disk."""
        if self._model is not None:
            return
        import joblib

        data = joblib.load(self.model_path)
        self._model = data["model"]
        self._classes = data["classes"]

    def predict(self, feature_dict: dict) -> Tuple[str, float]:
        """Predict element type and confidence for a single feature dict.

        Returns
        -------
        tuple[str, float]
            ``(predicted_label, confidence)`` where confidence is the
            maximum class probability.
        """
        self._load_model()
        x = encode_features(feature_dict).reshape(1, -1)
        proba = self._model.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        label = self._model.classes_[idx]
        return str(label), float(proba[idx])

    def predict_batch(self, feature_dicts: List[dict]) -> List[Tuple[str, float]]:
        """Predict element types for a batch of feature dicts.

        Returns
        -------
        list[tuple[str, float]]
            List of ``(predicted_label, confidence)`` pairs.
        """
        if not feature_dicts:
            return []
        self._load_model()
        X = np.array([encode_features(f) for f in feature_dicts])
        probas = self._model.predict_proba(X)
        results: list[tuple[str, float]] = []
        for row in probas:
            idx = int(np.argmax(row))
            label = self._model.classes_[idx]
            results.append((str(label), float(row[idx])))
        return results

    def model_exists(self) -> bool:
        """Return *True* if a trained model file exists on disk."""
        return self.model_path.is_file()

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature-name → importance mapping from the trained model.

        Uses the model's ``feature_importances_`` property.
        Returns an empty dict if no model is loaded or importances
        cannot be computed.
        """
        self._load_model()
        if self._model is None:
            return {}

        feature_names = list(_NUMERIC_KEYS) + [f"zone_{z}" for z in ZONE_VALUES]

        try:
            importances = np.asarray(self._model.feature_importances_)
        except Exception:
            return {}

        if importances.shape[0] != len(feature_names):
            return {}

        return {
            name: round(float(imp), 6)
            for name, imp in sorted(
                zip(feature_names, importances), key=lambda x: -x[1]
            )
        }
