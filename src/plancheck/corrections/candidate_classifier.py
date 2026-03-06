"""Binary hit/miss classifier for VOCR candidates (Level 2).

Trains a lightweight ``HistGradientBoostingClassifier`` on outcome data
stored in the ``candidate_outcomes`` table.  At inference time,
:func:`predict_hit_probability` returns P(hit) for each candidate so
the pipeline can skip unlikely candidates.

Workflow
--------
1. :func:`train_candidate_classifier` — train from DB rows, save model.
2. :func:`load_candidate_classifier` — load saved model for inference.
3. :func:`predict_hit_probability` — score a batch of candidates.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from .candidate_features import (
    CANDIDATE_FEATURE_DIM,
    CANDIDATE_FEATURE_VERSION,
    featurize_candidate,
    featurize_candidates_batch,
    featurize_outcome_row,
)

if TYPE_CHECKING:
    from ..models import VocrCandidate

log = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = Path("data/candidate_classifier.pkl")


# ── Training ────────────────────────────────────────────────────────────


def train_candidate_classifier(
    outcome_rows: list[dict],
    *,
    model_path: Path = _DEFAULT_MODEL_PATH,
    calibrate: bool = True,
) -> dict:
    """Train a binary classifier on candidate outcome rows.

    Parameters
    ----------
    outcome_rows : list[dict]
        Rows from ``CorrectionStore.get_candidate_outcomes()``.
    model_path : Path
        Where to save the trained model (``.pkl``).
    calibrate : bool
        Apply isotonic calibration for well-calibrated P(hit).

    Returns
    -------
    dict
        Training metrics: ``n_train``, ``n_val``, ``accuracy``, ``f1``,
        ``auc_roc``, ``model_path``.
    """
    import joblib
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    # Build feature matrix + labels
    X_list = []
    y_list = []
    for row in outcome_rows:
        X_list.append(featurize_outcome_row(row))
        y_list.append(1 if row["outcome"] == "hit" else 0)

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int32)

    # Stratified train/val split
    if len(set(y)) < 2:
        log.warning(
            "Only one class present in %d outcome rows — cannot train.",
            len(y),
        )
        return {"error": "single_class", "n_rows": len(y)}

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train HistGBM (fast, handles small datasets well)
    base_model = HistGradientBoostingClassifier(
        max_iter=150,
        max_depth=4,
        learning_rate=0.1,
        min_samples_leaf=5,
        random_state=42,
    )

    # Apply cross-validated calibration on training data (avoid data leakage)
    if calibrate and len(X_train) >= 10:
        # Use CV-based calibration on training data, not prefit on val
        cal_model = CalibratedClassifierCV(
            estimator=base_model, method="isotonic", cv=3
        )
        cal_model.fit(X_train, y_train)
        final_model = cal_model
    else:
        base_model.fit(X_train, y_train)
        final_model = base_model

    # Evaluate
    y_pred = final_model.predict(X_val)
    y_prob = final_model.predict_proba(X_val)[:, 1]
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, zero_division=0.0)
    try:
        auc = roc_auc_score(y_val, y_prob)
    except ValueError:
        auc = 0.0

    # Save
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "feature_version": CANDIDATE_FEATURE_VERSION,
            "n_features": CANDIDATE_FEATURE_DIM,
        },
        model_path,
    )

    metrics = {
        "n_train": len(X_train),
        "n_val": len(X_val),
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "auc_roc": round(auc, 4),
        "model_path": str(model_path),
    }
    log.info(
        "Candidate classifier trained: acc=%.3f  f1=%.3f  auc=%.3f  "
        "(train=%d, val=%d) → %s",
        acc,
        f1,
        auc,
        len(X_train),
        len(X_val),
        model_path,
    )
    return metrics


# ── Inference ───────────────────────────────────────────────────────────


class CandidateClassifier:
    """Lightweight wrapper for the saved candidate hit/miss model.

    Usage::

        clf = CandidateClassifier()
        clf.load()
        probs = clf.predict(candidates, page_width, page_height)
    """

    def __init__(self, model_path: Path = _DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self._model = None

    def load(self) -> bool:
        """Load the persisted model.  Returns *True* on success."""
        if not self.model_path.exists():
            log.debug("No candidate classifier at %s", self.model_path)
            return False
        try:
            import joblib

            bundle = joblib.load(self.model_path)
            if bundle.get("feature_version") != CANDIDATE_FEATURE_VERSION:
                log.warning(
                    "Candidate classifier feature version mismatch "
                    "(model=%s, current=%s) — skipping",
                    bundle.get("feature_version"),
                    CANDIDATE_FEATURE_VERSION,
                )
                return False
            self._model = bundle["model"]
            log.info("Loaded candidate classifier from %s", self.model_path)
            return True
        except Exception as exc:
            log.warning("Failed to load candidate classifier: %s", exc)
            return False

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def predict(
        self,
        candidates: list["VocrCandidate"],
        page_width: float = 612.0,
        page_height: float = 792.0,
    ) -> np.ndarray:
        """Return P(hit) for each candidate.

        Parameters
        ----------
        candidates : list[VocrCandidate]
            Candidates to score.
        page_width, page_height : float
            Page dimensions.

        Returns
        -------
        np.ndarray
            Shape ``(N,)`` with P(hit) in [0, 1].
            All ones if model is not loaded (fail-open).
        """
        if not self.is_ready or not candidates:
            return np.ones(len(candidates), dtype=np.float32)

        X = featurize_candidates_batch(candidates, page_width, page_height)
        return self._model.predict_proba(X)[:, 1].astype(np.float32)

    def filter_candidates(
        self,
        candidates: list["VocrCandidate"],
        page_width: float = 612.0,
        page_height: float = 792.0,
        threshold: float = 0.3,
    ) -> list["VocrCandidate"]:
        """Return only candidates with P(hit) ≥ *threshold*.

        If the model isn't loaded, all candidates pass through (fail-open).
        """
        if not self.is_ready:
            return candidates
        probs = self.predict(candidates, page_width, page_height)
        return [c for c, p in zip(candidates, probs) if p >= threshold]
