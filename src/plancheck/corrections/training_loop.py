"""Training loop for the element-type classifier.

Contains the core training orchestration extracted from
:class:`~plancheck.corrections.classifier.ElementClassifier`.
Called by :meth:`ElementClassifier.train` so that model construction,
prediction, and evaluation remain in ``classifier.py``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)


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


def train_classifier(
    jsonl_path: Path,
    model_path: Path,
    build_single_fn: Callable[[], Any],
    build_ensemble_fn: Callable[[], tuple[Any, list[str]]],
    *,
    calibrate: bool = True,
    ensemble: bool = False,
) -> tuple[dict, Any, Any, int]:
    """Execute the full training loop and persist the model.

    Parameters
    ----------
    jsonl_path : Path
        Path to the JSONL file written by
        :meth:`CorrectionStore.export_training_jsonl`.
    model_path : Path
        Destination path for the serialised model pickle.
    build_single_fn : callable
        Zero-argument callable returning a fresh
        :class:`~sklearn.ensemble.GradientBoostingClassifier`.
    build_ensemble_fn : callable
        Zero-argument callable returning
        ``(VotingClassifier, member_names)``.
    calibrate : bool
        If *True* (default), apply isotonic calibration via
        :class:`~sklearn.calibration.CalibratedClassifierCV`.
    ensemble : bool
        If *True*, use the ensemble builder instead of the single GBM.

    Returns
    -------
    tuple[dict, object, object, int]
        ``(metrics, raw_model, fitted_model, n_features_in)``

    Raises
    ------
    ValueError
        If fewer than 10 training examples are available or no
        training-split examples are found.
    """
    # Lazy imports avoid circular dependencies at module load time
    from sklearn.utils.class_weight import compute_sample_weight

    from .classifier import (
        FEATURE_VERSION,
        _NUMERIC_KEYS,
        ZONE_VALUES,
        encode_features,
    )
    from .metrics import compute_metrics

    # ── Load data ────────────────────────────────────────────────
    examples: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if len(examples) < 10:
        raise ValueError(f"Need at least 10 training examples, got {len(examples)}")

    # ── Split ────────────────────────────────────────────────────
    train_ex = [e for e in examples if e.get("split") == "train"]
    val_ex = [e for e in examples if e.get("split") == "val"]

    if not train_ex:
        raise ValueError("No training-split examples found")

    X_train = np.array([encode_features(e["features"]) for e in train_ex])
    y_train = [e["label"] for e in train_ex]

    # ── Balanced class weighting ─────────────────────────────────
    sample_weights = compute_sample_weight("balanced", y_train)

    # ── Build and fit ─────────────────────────────────────────────
    ensemble_members: list[str] = []
    if ensemble:
        clf, ensemble_members = build_ensemble_fn()
        clf.fit(X_train, y_train, sample_weight=sample_weights)
        raw_model = clf
    else:
        clf = build_single_fn()
        clf.fit(X_train, y_train, sample_weight=sample_weights)
        raw_model = clf

    # ── Confidence calibration (isotonic regression) ──────────────
    calibrated_clf = _calibrate(clf, X_train, y_train, calibrate=calibrate)

    # ── Evaluate on validation set (or training set if no val) ────
    eval_on_train = not bool(val_ex)
    if eval_on_train:
        log.warning(
            "No validation data available — metrics computed on "
            "TRAINING data and are unreliable"
        )
    eval_ex = val_ex if val_ex else train_ex
    X_eval = np.array([encode_features(e["features"]) for e in eval_ex])
    y_eval = [e["label"] for e in eval_ex]
    eval_model = calibrated_clf if calibrated_clf is not None else clf
    y_pred = eval_model.predict(X_eval).tolist()

    # ── Per-example holdout predictions ──────────────────────────
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

    # ── Hyperparameters & feature-set metadata (Phase 4.4) ───────
    if not ensemble:
        hyperparams: dict = {
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

    # ── Persist ───────────────────────────────────────────────────
    import joblib

    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {"model": raw_model, "classes": labels}
    if calibrated_clf is not None:
        payload["calibrated_model"] = calibrated_clf
    if ensemble:
        payload["ensemble_members"] = ensemble_members
    payload["n_features_in"] = X_train.shape[1]
    joblib.dump(payload, model_path)

    fitted_model = calibrated_clf if calibrated_clf is not None else raw_model
    return metrics, raw_model, fitted_model, X_train.shape[1]
