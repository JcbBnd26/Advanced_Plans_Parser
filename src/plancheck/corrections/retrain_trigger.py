"""Automated retraining trigger for the element-type classifier.

Provides functions to check whether a retrain is needed, perform the
retrain, and run a startup check that can be called when the GUI
opens or the pipeline initialises.

The trigger logic is simple:

1. Count corrections since the last training run.
2. If the count exceeds the configured threshold → retrain.
3. After retraining, auto-rollback if F1 regresses (same logic as
   ``train_model.py``).

Usage
-----
::

    from plancheck.corrections.retrain_trigger import (
        check_retrain_needed,
        auto_retrain,
        startup_check,
    )

    # Quick predicate
    if check_retrain_needed(store, threshold=50):
        result = auto_retrain(store, model_path, db_path)

    # Or call at startup for hands-off use
    startup_check(cfg)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class RetrainResult:
    """Outcome of an automatic retrain attempt."""

    retrained: bool = False
    accepted: bool = False
    rolled_back: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    new_corrections: int = 0
    threshold: int = 50

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrained": self.retrained,
            "accepted": self.accepted,
            "rolled_back": self.rolled_back,
            "f1_weighted": self.metrics.get("f1_weighted", 0.0),
            "accuracy": self.metrics.get("accuracy", 0.0),
            "error": self.error,
            "new_corrections": self.new_corrections,
            "threshold": self.threshold,
        }


def check_retrain_needed(
    store: Any,
    threshold: int = 50,
) -> bool:
    """Return *True* when enough new corrections have accumulated.

    Parameters
    ----------
    store : CorrectionStore
        Open correction database.
    threshold : int
        Minimum number of new corrections required.
    """
    return store.should_retrain(threshold=threshold)


def auto_retrain(
    store: Any,
    model_path: Path | str = "data/element_classifier.pkl",
    *,
    calibrate: bool = True,
    ensemble: bool = False,
    threshold: int = 50,
) -> RetrainResult:
    """Retrain the classifier if enough corrections have accumulated.

    This function mirrors the logic of ``scripts/train_model.py`` but
    is callable programmatically from the GUI or pipeline.

    Parameters
    ----------
    store : CorrectionStore
        Open correction database.
    model_path : Path or str
        Where to save the trained model.
    calibrate : bool
        Enable isotonic calibration.
    ensemble : bool
        Enable soft-voting ensemble.
    threshold : int
        Minimum corrections required before retraining.

    Returns
    -------
    RetrainResult
        Outcome of the retrain attempt.
    """
    from .classifier import ElementClassifier
    from .metrics import format_metrics_table

    model_path = Path(model_path)
    result = RetrainResult(threshold=threshold)

    # Check if retrain is needed
    n_new = store.count_corrections_since_last_train()
    result.new_corrections = n_new
    if n_new < threshold:
        log.info(
            "Retrain not needed: %d corrections < threshold %d",
            n_new,
            threshold,
        )
        return result

    log.info("Auto-retrain triggered: %d new corrections", n_new)

    try:
        # Build training set
        n_examples = store.build_training_set()
        if n_examples < 10:
            result.error = f"Only {n_examples} training examples (need ≥10)"
            return result

        # Export JSONL
        db_parent = store._db_path.parent
        jsonl_path = db_parent / "training_data.jsonl"
        store.export_training_jsonl(jsonl_path)

        # Train
        clf = ElementClassifier(model_path=model_path)
        metrics = clf.train(
            jsonl_path,
            calibrate=calibrate,
            ensemble=ensemble,
        )
        result.retrained = True
        result.metrics = metrics

        log.info(
            "Auto-retrain complete: F1=%.4f accuracy=%.4f",
            metrics.get("f1_weighted", 0.0),
            metrics.get("accuracy", 0.0),
        )

        # Record in database
        holdout_preds = metrics.get("holdout_predictions")
        try:
            run_id = store.save_training_run(
                metrics,
                model_path=str(model_path),
                notes="auto-retrain",
                holdout_predictions=holdout_preds,
            )
        except Exception as exc:
            log.warning("Could not record training run: %s", exc)
            run_id = None

        # Auto-rollback check
        try:
            history = store.get_training_history()
            prior_runs = [r for r in history if r.get("run_id") != run_id]
            if prior_runs:
                prev_f1 = prior_runs[0]["f1_weighted"]
                curr_f1 = metrics.get("f1_weighted", 0.0)
                if curr_f1 < prev_f1:
                    log.warning(
                        "Auto-retrain: F1 regressed %.4f → %.4f, rolling back",
                        prev_f1,
                        curr_f1,
                    )
                    result.accepted = False
                    result.rolled_back = True
                    return result
        except Exception:
            log.debug("Auto-rollback check failed", exc_info=True)

        result.accepted = True

        # Fit drift detector on the new training data
        try:
            from .drift_detection import DriftDetector

            drift_stats_path = db_parent / "drift_stats.json"
            drift_det = DriftDetector(threshold=0.3)
            drift_det.fit(jsonl_path)
            drift_det.save(drift_stats_path)
            log.info("Drift stats updated: %s", drift_stats_path)
        except Exception:
            log.debug("Drift stats update failed", exc_info=True)

    except Exception as exc:
        log.error("Auto-retrain failed: %s", exc, exc_info=True)
        result.error = str(exc)

    return result


def startup_check(
    cfg: Any,
    *,
    db_path: Path | None = None,
) -> RetrainResult | None:
    """Run on application startup to check if retraining is advisable.

    Only acts when ``cfg.ml_retrain_on_startup`` is *True*.

    Parameters
    ----------
    cfg : GroupingConfig
        Pipeline configuration.
    db_path : Path, optional
        Override for the corrections database path.

    Returns
    -------
    RetrainResult or None
        Result of the retrain, or *None* if startup retrain is disabled
        or the database doesn't exist.
    """
    if not getattr(cfg, "ml_retrain_on_startup", False):
        return None

    db = db_path or Path("data/corrections.db")
    if not db.exists():
        return None

    from .store import CorrectionStore

    store = CorrectionStore(db)
    try:
        threshold = getattr(cfg, "ml_retrain_threshold", 50)
        model_path = getattr(cfg, "ml_model_path", "data/element_classifier.pkl")
        ensemble = getattr(cfg, "ml_ensemble_enabled", False)

        if not check_retrain_needed(store, threshold=threshold):
            return RetrainResult(
                new_corrections=store.count_corrections_since_last_train(),
                threshold=threshold,
            )

        return auto_retrain(
            store,
            model_path=model_path,
            ensemble=ensemble,
            threshold=threshold,
        )
    finally:
        store.close()


# ── Level 2: candidate classifier retrain ──────────────────────────────


def auto_retrain_candidate_classifier(
    *,
    db_path: Path | str = "data/corrections.db",
    model_path: Path | str = "data/candidate_classifier.pkl",
    min_rows: int = 100,
) -> dict[str, Any]:
    """Retrain the candidate hit/miss classifier if enough outcome data exists.

    Parameters
    ----------
    db_path : Path or str
        Corrections database path.
    model_path : Path or str
        Where to save the trained model.
    min_rows : int
        Minimum outcome rows before training.

    Returns
    -------
    dict
        Training metrics or ``{"skipped": True}`` if insufficient data.
    """
    from .candidate_classifier import train_candidate_classifier
    from .store import CorrectionStore

    db_path = Path(db_path)
    model_path = Path(model_path)

    if not db_path.exists():
        return {"skipped": True, "reason": "no_db"}

    store = CorrectionStore(db_path)
    try:
        rows = store.get_candidate_outcomes(min_rows=min_rows)
        if not rows:
            counts = store.count_candidate_outcomes()
            return {
                "skipped": True,
                "reason": "insufficient_data",
                "total_rows": counts.get("total", 0),
                "min_rows": min_rows,
            }
        return train_candidate_classifier(rows, model_path=model_path)
    finally:
        store.close()
