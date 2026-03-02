"""Element-type classifier using scikit-learn GradientBoosting.

Trains on JSONL exported by :meth:`CorrectionStore.export_training_jsonl`
and predicts ``(label, confidence)`` for new feature dicts.

Uses :class:`~sklearn.ensemble.GradientBoostingClassifier` with balanced
class weighting via ``compute_sample_weight``.  When ensemble mode is
enabled, combines GBM + HistGradientBoosting (+ optional LightGBM/XGBoost)
via soft-voting for more robust predictions.  Post-training, the model
is wrapped with isotonic calibration via
:class:`~sklearn.calibration.CalibratedClassifierCV` so that
``predict_proba`` returns well-calibrated probabilities.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Feature schema version — increment when _NUMERIC_KEYS or ZONE_VALUES change.
# Used by the feature cache to invalidate stale entries.
FEATURE_VERSION: int = 5

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
    # Discriminative features (added in v3)
    "text_density",
    "x_dist_to_right_margin",
    "line_width_variance",
    # OCR confidence features (added in v4)
    "mean_token_confidence",
    "min_token_confidence",
]

_DEFAULT_MODEL_PATH = Path("data/element_classifier.pkl")


def encode_features(
    feature_dict: dict,
    image_features: np.ndarray | None = None,
    text_embedding: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a feature dict to a flat float array.

    The ``zone`` string is one-hot encoded into binary columns
    (one per :data:`ZONE_VALUES`).  All other keys are cast to float.

    When *image_features* is provided (from
    :func:`~plancheck.corrections.image_features.extract_image_features`),
    the CNN embedding is appended after the base features.

    When *text_embedding* is provided (from
    :func:`~plancheck.corrections.text_embeddings.embed`),
    the dense text vector is appended after image features (if any).

    Parameters
    ----------
    feature_dict : dict
        Output of :func:`~plancheck.corrections.features.featurize`.
    image_features : numpy.ndarray, optional
        1-D float array of CNN image embeddings (512-d for ResNet-18).
    text_embedding : numpy.ndarray, optional
        1-D float array of text embeddings (384-d for MiniLM).

    Returns
    -------
    numpy.ndarray
        1-D float64 array of length ``len(_NUMERIC_KEYS) + len(ZONE_VALUES)``
        (+ image dims when vision active, + embedding dims when embeddings active).
    """
    numeric = [float(feature_dict.get(k, 0.0)) for k in _NUMERIC_KEYS]
    zone = feature_dict.get("zone", "unknown")
    one_hot = [1.0 if zone == z else 0.0 for z in ZONE_VALUES]
    base = np.array(numeric + one_hot, dtype=np.float64)
    parts = [base]
    if image_features is not None and len(image_features) > 0:
        parts.append(image_features.astype(np.float64))
    if text_embedding is not None and len(text_embedding) > 0:
        parts.append(text_embedding.astype(np.float64))
    if len(parts) == 1:
        return base
    return np.concatenate(parts)


class ElementClassifier:
    """Gradient-boosted element-type classifier.

    Wraps ``sklearn.ensemble.GradientBoostingClassifier`` with helpers
    for encoding features, training from JSONL, and predicting labels.
    Uses balanced sample weights to handle class imbalance.  After
    training, the model is calibrated using isotonic regression so
    that confidence scores are well-calibrated.

    When *ensemble=True* is passed to :meth:`train`, a soft-voting
    ensemble of GBM + HistGBM (+ optional LightGBM / XGBoost) is
    used instead.  Falls back gracefully if optional packages are
    not installed.

    Parameters
    ----------
    model_path : Path
        Location of the persisted ``.pkl`` model file.
    """

    def __init__(self, model_path: Path = _DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self._model = None
        self._raw_model = None

    # ── Private estimator builders ────────────────────────────────

    @staticmethod
    def _build_gbm(sample_weights=None):
        """Build a GradientBoostingClassifier."""
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
        )

    @staticmethod
    def _build_hist_gbm():
        """Build a HistGradientBoostingClassifier (sklearn built-in)."""
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_leaf=10,
            random_state=42,
        )

    @staticmethod
    def _build_lgbm():
        """Build a LGBMClassifier if *lightgbm* is installed, else None."""
        try:
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                min_child_samples=5,
                subsample=0.8,
                random_state=42,
                verbose=-1,
            )
        except ImportError:
            log.info("lightgbm not installed — skipping LGBMClassifier.")
            return None

    @staticmethod
    def _build_xgb():
        """Build an XGBClassifier if *xgboost* is installed, else None."""
        try:
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric="mlogloss",
                verbosity=0,
            )
        except ImportError:
            log.info("xgboost not installed — skipping XGBClassifier.")
            return None

    def _build_ensemble(self):
        """Build a VotingClassifier with available estimators.

        Always includes GBM + HistGBM.  Adds LightGBM and XGBoost
        when the packages are installed.

        Returns
        -------
        VotingClassifier
            Soft-voting ensemble.
        list[str]
            Names of the constituent estimators.
        """
        from sklearn.ensemble import VotingClassifier

        estimators: list[tuple] = [
            ("gbm", self._build_gbm()),
            ("hist_gbm", self._build_hist_gbm()),
        ]

        lgbm = self._build_lgbm()
        if lgbm is not None:
            estimators.append(("lgbm", lgbm))

        xgb = self._build_xgb()
        if xgb is not None:
            estimators.append(("xgb", xgb))

        names = [name for name, _ in estimators]
        log.info(
            "Building ensemble with %d estimators: %s",
            len(estimators),
            ", ".join(names),
        )

        return (
            VotingClassifier(
                estimators=estimators,
                voting="soft",
            ),
            names,
        )

    # ── Training ───────────────────────────────────────────────────

    def train(
        self,
        jsonl_path: Path,
        *,
        calibrate: bool = True,
        ensemble: bool = False,
    ) -> dict:
        """Train a new model from exported JSONL.

        Parameters
        ----------
        jsonl_path : Path
            Path to the JSONL file written by
            :meth:`CorrectionStore.export_training_jsonl`.
        calibrate : bool
            If *True* (default), apply isotonic calibration via
            :class:`~sklearn.calibration.CalibratedClassifierCV`
            so that ``predict_proba`` values are well-calibrated.
        ensemble : bool
            If *True*, use a soft-voting ensemble of GBM + HistGBM
            (+ optional LightGBM / XGBoost).  Defaults to *False*
            (single GBM).

        Returns
        -------
        dict
            Training metrics: ``accuracy``, ``per_class``,
            ``confusion_matrix``, ``labels``, ``n_train``, ``n_val``,
            ``calibrated``, ``ensemble``, ``ensemble_members``,
            ``holdout_predictions``.

        Raises
        ------
        ValueError
            If fewer than 10 training examples are available.
        """
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

        # Build and fit the model (single GBM or ensemble)
        ensemble_members: list[str] = []
        if ensemble:
            clf, ensemble_members = self._build_ensemble()
            # VotingClassifier doesn't natively forward sample_weight
            # to all estimators.  Fit manually to pass weights to GBM.
            clf.fit(X_train, y_train, sample_weight=sample_weights)
            raw_model = clf  # keep ref for feature importance
        else:
            from sklearn.ensemble import GradientBoostingClassifier

            clf = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.1,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=42,
            )
            clf.fit(X_train, y_train, sample_weight=sample_weights)
            raw_model = clf

        # ── Confidence calibration (isotonic regression) ──────────
        calibrated_clf = self._calibrate(
            clf,
            X_train,
            y_train,
            calibrate=calibrate,
        )

        # Evaluate on validation set (or training set if no val)
        eval_on_train = not bool(val_ex)
        if eval_on_train:
            log.warning(
                "No validation data available — metrics computed on "
                "TRAINING data and are unreliable"
            )
        eval_ex = val_ex if val_ex else train_ex
        X_eval = np.array([encode_features(e["features"]) for e in eval_ex])
        y_eval = [e["label"] for e in eval_ex]
        # Use calibrated model for evaluation when available
        eval_model = calibrated_clf if calibrated_clf is not None else clf
        y_pred = eval_model.predict(X_eval).tolist()

        # Capture per-example holdout predictions for model comparison
        holdout_predictions: list[dict] = []
        if val_ex:
            proba = eval_model.predict_proba(X_eval)
            proba_max = proba.max(axis=1).tolist()
            for i, ex in enumerate(val_ex):
                holdout_predictions.append(
                    {
                        "label_true": y_eval[i],
                        "label_pred": y_pred[i],
                        "confidence": round(proba_max[i], 6),
                    }
                )

        labels = sorted(set(y_train) | set(y_eval))
        metrics = compute_metrics(y_eval, y_pred, labels=labels)
        metrics["n_train"] = len(train_ex)
        metrics["n_val"] = len(val_ex)
        metrics["eval_on_train"] = eval_on_train
        metrics["calibrated"] = calibrated_clf is not None
        metrics["ensemble"] = ensemble
        metrics["ensemble_members"] = ensemble_members
        metrics["holdout_predictions"] = holdout_predictions

        # Hyperparameters & feature set metadata (Phase 4.4)
        hyperparams: dict = {}
        if not ensemble:
            hyperparams = {
                "n_estimators": 200,
                "max_depth": 3,
                "learning_rate": 0.1,
                "min_samples_leaf": 5,
                "subsample": 0.8,
                "random_state": 42,
                "calibrated": calibrated_clf is not None,
            }
        else:
            hyperparams = {
                "ensemble": True,
                "members": ensemble_members,
                "calibrated": calibrated_clf is not None,
            }
        metrics["hyperparams"] = hyperparams
        metrics["feature_set"] = {
            "numeric_keys": len(_NUMERIC_KEYS),
            "zone_values": len(ZONE_VALUES),
            "base_dim": len(_NUMERIC_KEYS) + len(ZONE_VALUES),
            "total_dim": X_train.shape[1],
            "feature_version": FEATURE_VERSION,
        }
        metrics["feature_version"] = FEATURE_VERSION

        # Persist
        import joblib

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {"model": raw_model, "classes": labels}
        if calibrated_clf is not None:
            payload["calibrated_model"] = calibrated_clf
        if ensemble:
            payload["ensemble_members"] = ensemble_members
        # Record the expected feature vector length so that predict()
        # can detect whether the model was trained with image features.
        payload["n_features_in"] = X_train.shape[1]
        joblib.dump(payload, self.model_path)

        self._raw_model = raw_model
        self._model = calibrated_clf if calibrated_clf is not None else raw_model
        self._n_features_in: int | None = X_train.shape[1]

        return metrics

    # ── Negative-class handling ─────────────────────────────────

    #: Label used for false-positive (deleted) training examples.
    NEGATIVE_LABEL: str = "__negative__"

    @staticmethod
    def _resolve_negative(
        classes: np.ndarray, proba: np.ndarray
    ) -> Tuple[str, float, float]:
        """Pick the best *real* class, returning negative probability too.

        When the model includes a ``__negative__`` class trained from
        deleted detections, this helper finds:

        1. ``p_negative`` — the probability this is a false positive.
        2. The best non-negative class label and its probability.

        If ``__negative__`` is not among the model classes, the
        overall argmax is returned with ``p_negative = 0``.

        Returns
        -------
        tuple[str, float, float]
            ``(best_label, best_confidence, p_negative)``
        """
        neg_label = ElementClassifier.NEGATIVE_LABEL
        classes_list = list(classes)
        if neg_label in classes_list:
            neg_idx = classes_list.index(neg_label)
            p_neg = float(proba[neg_idx])
            # Mask negative class and pick best real class
            masked = proba.copy()
            masked[neg_idx] = -1.0
            best_idx = int(np.argmax(masked))
            best_label = str(classes[best_idx])
            # Scale real-class confidence by (1 - p_negative) so that
            # deleted-like regions get visibly lower confidence.
            best_conf = float(proba[best_idx]) * (1.0 - p_neg)
            return best_label, best_conf, p_neg
        else:
            idx = int(np.argmax(proba))
            return str(classes[idx]), float(proba[idx]), 0.0

    # ── Prediction ─────────────────────────────────────────────────

    @staticmethod
    def _calibrate(
        clf,
        X_train: np.ndarray,
        y_train: list[str],
        *,
        calibrate: bool = True,
    ):
        """Wrap *clf* with isotonic calibration.

        Returns the calibrated estimator, or *None* if calibration is
        skipped or fails (e.g. too few examples per class for CV).
        """
        if not calibrate:
            return None
        try:
            from sklearn.calibration import CalibratedClassifierCV

            n_unique = len(set(y_train))
            n_samples = len(y_train)
            # Need enough samples for cross-validation folds
            cv_folds = min(5, max(2, n_samples // max(n_unique, 1)))
            if n_samples < cv_folds * n_unique:
                log.warning(
                    "Too few samples (%d) for calibration CV — skipping.",
                    n_samples,
                )
                return None

            cal = CalibratedClassifierCV(
                estimator=clf,
                method="isotonic",
                cv=cv_folds,
            )
            cal.fit(X_train, y_train)
            log.info(
                "Model calibrated with isotonic regression (cv=%d).",
                cv_folds,
            )
            return cal
        except Exception:
            log.warning("Calibration failed — using raw model.", exc_info=True)
            return None

    def _load_model(self) -> None:
        """Lazy-load the model from disk.

        Prefers ``calibrated_model`` when present in the pickle for
        backward compatibility with pre-calibration model files.
        """
        if self._model is not None:
            return
        import joblib

        data = joblib.load(self.model_path)
        self._raw_model = data.get("model")
        self._model = data.get("calibrated_model", data["model"])
        self._n_features_in: int | None = data.get("n_features_in")

    def predict(
        self,
        feature_dict: dict,
        image_features: np.ndarray | None = None,
        text_embedding: np.ndarray | None = None,
    ) -> Tuple[str, float]:
        """Predict element type and confidence for a single feature dict.

        Parameters
        ----------
        feature_dict : dict
            Hand-crafted features from :func:`featurize`.
        image_features : numpy.ndarray, optional
            CNN image embedding for the element's visual crop.
        text_embedding : numpy.ndarray, optional
            Dense text embedding from sentence-transformer.

        Returns
        -------
        tuple[str, float]
            ``(predicted_label, confidence)`` where confidence is the
            maximum class probability.
        """
        self._load_model()
        # Only append optional features if the model was trained with them.
        # Base vector length = len(_NUMERIC_KEYS) + len(ZONE_VALUES).
        base_dim = len(_NUMERIC_KEYS) + len(ZONE_VALUES)
        if self._n_features_in is not None and self._n_features_in <= base_dim:
            image_features = None  # model doesn't expect vision dims
            text_embedding = None  # model doesn't expect embedding dims
        x = encode_features(feature_dict, image_features, text_embedding).reshape(1, -1)
        # If model was trained with fewer features, trim to match
        if self._n_features_in is not None and x.shape[1] > self._n_features_in:
            x = x[:, : self._n_features_in]
        proba = self._model.predict_proba(x)[0]
        label, conf, _ = self._resolve_negative(self._model.classes_, proba)
        return label, conf

    def predict_from_vector(self, vector: np.ndarray) -> Tuple[str, float]:
        """Predict from a pre-encoded feature vector (e.g. from cache).

        Parameters
        ----------
        vector : numpy.ndarray
            1-D float array already encoded by :func:`encode_features`.

        Returns
        -------
        tuple[str, float]
            ``(predicted_label, confidence)``.
        """
        self._load_model()
        x = np.asarray(vector, dtype=np.float64).reshape(1, -1)
        if self._n_features_in is not None and x.shape[1] > self._n_features_in:
            x = x[:, : self._n_features_in]
        proba = self._model.predict_proba(x)[0]
        label, conf, _ = self._resolve_negative(self._model.classes_, proba)
        return label, conf

    def predict_batch(
        self,
        feature_dicts: List[dict],
        image_features_list: List[np.ndarray] | None = None,
        text_embeddings_list: List[np.ndarray] | None = None,
    ) -> List[Tuple[str, float]]:
        """Predict element types for a batch of feature dicts.

        Parameters
        ----------
        feature_dicts : list[dict]
            Hand-crafted features from :func:`featurize`.
        image_features_list : list[numpy.ndarray], optional
            Parallel list of CNN image embeddings (one per element).
        text_embeddings_list : list[numpy.ndarray], optional
            Parallel list of text embeddings (one per element).

        Returns
        -------
        list[tuple[str, float]]
            List of ``(predicted_label, confidence)`` pairs.
        """
        if not feature_dicts:
            return []
        self._load_model()
        # Only use optional features if the model was trained with them
        base_dim = len(_NUMERIC_KEYS) + len(ZONE_VALUES)
        if self._n_features_in is not None and self._n_features_in <= base_dim:
            image_features_list = None  # model doesn't expect vision dims
            text_embeddings_list = None  # model doesn't expect embedding dims

        rows = []
        for i, f in enumerate(feature_dicts):
            img = image_features_list[i] if image_features_list else None
            emb = text_embeddings_list[i] if text_embeddings_list else None
            rows.append(encode_features(f, img, emb))
        X = np.array(rows)

        # Trim if model expects fewer features
        if self._n_features_in is not None and X.shape[1] > self._n_features_in:
            X = X[:, : self._n_features_in]

        probas = self._model.predict_proba(X)
        results: list[tuple[str, float]] = []
        for row in probas:
            label, conf, _ = self._resolve_negative(self._model.classes_, row)
            results.append((label, conf))
        return results

    def predict_negative_probability(
        self,
        feature_dict: dict,
        image_features: np.ndarray | None = None,
        text_embedding: np.ndarray | None = None,
    ) -> float:
        """Return the probability that a region is a false positive.

        If the model was not trained with ``__negative__`` examples
        (i.e. no deletions in the training data), returns ``0.0``.

        Parameters
        ----------
        feature_dict : dict
            Hand-crafted features from :func:`featurize`.

        Returns
        -------
        float
            P(negative) in [0, 1].
        """
        self._load_model()
        base_dim = len(_NUMERIC_KEYS) + len(ZONE_VALUES)
        if self._n_features_in is not None and self._n_features_in <= base_dim:
            image_features = None
            text_embedding = None
        x = encode_features(feature_dict, image_features, text_embedding).reshape(1, -1)
        if self._n_features_in is not None and x.shape[1] > self._n_features_in:
            x = x[:, : self._n_features_in]
        proba = self._model.predict_proba(x)[0]
        _, _, p_neg = self._resolve_negative(self._model.classes_, proba)
        return p_neg

    def model_exists(self) -> bool:
        """Return *True* if a trained model file exists on disk."""
        return self.model_path.is_file()

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature-name -> importance mapping from the trained model.

        Uses the raw (uncalibrated) model's ``feature_importances_``
        property.  For ensemble models, extracts importances from the
        first GBM estimator.  Returns an empty dict if no model is
        loaded or importances cannot be computed.
        """
        self._load_model()
        # Prefer the raw model — calibrated wrappers don't expose
        # feature_importances_ directly.
        model = self._raw_model if self._raw_model is not None else self._model
        if model is None:
            return {}

        # For VotingClassifier, extract the first named estimator
        from sklearn.ensemble import VotingClassifier

        if isinstance(model, VotingClassifier):
            try:
                model = model.estimators_[0]
            except (IndexError, AttributeError):
                pass

        feature_names = list(_NUMERIC_KEYS) + [f"zone_{z}" for z in ZONE_VALUES]

        try:
            importances = np.asarray(model.feature_importances_)
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

    def calibration_curve(self, jsonl_path: Path) -> dict:
        """Compute calibration curves for each class.

        Parameters
        ----------
        jsonl_path : Path
            Path to the JSONL file with ``features``, ``label``, and
            ``split`` keys (typically the training export).

        Returns
        -------
        dict
            ``{"curves": {label: {"mean_predicted": [...], "fraction_positive": [...]}},
              "ece": float}``
            where ``ece`` is the Expected Calibration Error (weighted).
        """
        self._load_model()
        if self._model is None:
            return {"curves": {}, "ece": 0.0}

        examples: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))

        # Use validation examples when available, else all
        val_ex = [e for e in examples if e.get("split") == "val"]
        if not val_ex:
            val_ex = examples
        if not val_ex:
            return {"curves": {}, "ece": 0.0}

        X = np.array([encode_features(e["features"]) for e in val_ex])
        y_true = [e["label"] for e in val_ex]
        probas = self._model.predict_proba(X)
        classes = list(self._model.classes_)

        from sklearn.calibration import calibration_curve as _sk_cal_curve

        curves: dict[str, dict] = {}
        total_ece = 0.0
        total_weight = 0

        for i, cls in enumerate(classes):
            y_binary = np.array([1 if yt == cls else 0 for yt in y_true])
            p_cls = probas[:, i]
            n_pos = int(y_binary.sum())
            if n_pos == 0 or n_pos == len(y_binary):
                continue  # skip classes with no positive/negative examples
            try:
                frac_pos, mean_pred = _sk_cal_curve(
                    y_binary,
                    p_cls,
                    n_bins=min(10, max(3, n_pos)),
                    strategy="uniform",
                )
                curves[cls] = {
                    "mean_predicted": [round(float(v), 4) for v in mean_pred],
                    "fraction_positive": [round(float(v), 4) for v in frac_pos],
                }
                # Per-class ECE contribution
                bin_counts = np.histogram(
                    p_cls,
                    bins=len(frac_pos),
                    range=(0, 1),
                )[0]
                ece = float(
                    np.sum(np.abs(frac_pos - mean_pred) * bin_counts)
                    / max(len(p_cls), 1)
                )
                total_ece += ece * n_pos
                total_weight += n_pos
            except Exception:
                continue

        weighted_ece = total_ece / total_weight if total_weight > 0 else 0.0
        return {"curves": curves, "ece": round(weighted_ece, 4)}
