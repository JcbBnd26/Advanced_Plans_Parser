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

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from plancheck.config import DEFAULT_CORRECTIONS_DB, DEFAULT_ML_MODEL
from plancheck.config.constants import DEFAULT_SUBTYPE_MODEL

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
    stage2_trained: bool = False
    stage2_metrics: dict[str, Any] = field(default_factory=dict)
    stage2_error: str = ""
    stage2_skipped_reason: str = ""

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
            "stage2_trained": self.stage2_trained,
            "stage2_f1_weighted": self.stage2_metrics.get("f1_weighted", 0.0),
            "stage2_accuracy": self.stage2_metrics.get("accuracy", 0.0),
            "stage2_error": self.stage2_error,
            "stage2_skipped_reason": self.stage2_skipped_reason,
        }


_MIN_STAGE2_EXAMPLES = 10


def _export_stage2_training_jsonl(
    source_jsonl: Path,
    output_jsonl: Path,
) -> tuple[int, Counter[str]]:
    """Filter exported training data down to valid Stage-2 subtype labels."""
    from .subtype_classifier import TITLE_SUBTYPES

    allowed = set(TITLE_SUBTYPES)
    label_counts: Counter[str] = Counter()
    kept_lines: list[str] = []

    with source_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            label = str(record.get("label", ""))
            if label not in allowed:
                continue
            kept_lines.append(stripped)
            label_counts[label] += 1

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text(
        "\n".join(kept_lines) + ("\n" if kept_lines else ""),
        encoding="utf-8",
    )
    return len(kept_lines), label_counts


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
    stage2_model_path: Path | str | None = None,
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
    from .subtype_classifier import TitleSubtypeClassifier

    model_path = Path(model_path)
    stage2_model_path = Path(stage2_model_path or DEFAULT_SUBTYPE_MODEL)
    result = RetrainResult(threshold=threshold)

    # Backup existing model before training (for rollback)
    backup_path = model_path.with_suffix(".pkl.bak")
    model_existed = model_path.exists()
    if model_existed:
        import shutil

        shutil.copy2(model_path, backup_path)
        log.debug("Backed up existing model to %s", backup_path)

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
                    # Restore the backed-up model
                    if model_existed and backup_path.exists():
                        import shutil

                        shutil.copy2(backup_path, model_path)
                        backup_path.unlink()
                        log.info("Restored previous model from backup")
                        result.accepted = False
                        result.rolled_back = True
                    else:
                        log.error(
                            "Rollback failed: backup unavailable. "
                            "Regressed model persisted!"
                        )
                        result.error = "Rollback failed: backup unavailable"
                        result.accepted = False
                        result.rolled_back = False
                    return result
        except Exception:  # noqa: BLE001 — auto-rollback check should not break retrain
            log.debug("Auto-rollback check failed", exc_info=True)

        result.accepted = True

        # Clean up backup since model was accepted
        if backup_path.exists():
            backup_path.unlink()
            log.debug("Removed model backup after successful retrain")

        # Fit drift detector on the new training data
        try:
            from .drift_detection import DriftDetector

            drift_stats_path = db_parent / "drift_stats.json"
            drift_det = DriftDetector(threshold=0.3)
            drift_det.fit(jsonl_path)
            drift_det.save(drift_stats_path)
            log.info("Drift stats updated: %s", drift_stats_path)
        except Exception:  # noqa: BLE001 — drift stats update is best-effort
            log.debug("Drift stats update failed", exc_info=True)

        # Train Stage 2 when subtype labels are present and sufficiently diverse.
        try:
            stage2_jsonl_path = db_parent / "training_data_stage2.jsonl"
            stage2_count, stage2_label_counts = _export_stage2_training_jsonl(
                jsonl_path,
                stage2_jsonl_path,
            )
            stage2_labels = sorted(stage2_label_counts)

            if stage2_count < _MIN_STAGE2_EXAMPLES:
                result.stage2_skipped_reason = f"only {stage2_count} subtype examples (need >= {_MIN_STAGE2_EXAMPLES})"
            elif len(stage2_labels) < 2:
                result.stage2_skipped_reason = (
                    "need at least 2 subtype labels to train Stage 2"
                )
            else:
                subtype_clf = TitleSubtypeClassifier(model_path=stage2_model_path)
                stage2_metrics = subtype_clf.train(
                    stage2_jsonl_path,
                    calibrate=calibrate,
                    ensemble=ensemble,
                )
                result.stage2_trained = True
                result.stage2_metrics = stage2_metrics
                store.save_training_run(
                    stage2_metrics,
                    model_path=str(stage2_model_path),
                    notes="auto-retrain-stage2",
                    holdout_predictions=stage2_metrics.get("holdout_predictions"),
                )
                log.info(
                    "Stage-2 retrain complete: labels=%s F1=%.4f accuracy=%.4f",
                    ", ".join(stage2_labels),
                    stage2_metrics.get("f1_weighted", 0.0),
                    stage2_metrics.get("accuracy", 0.0),
                )
        except Exception as exc:
            log.warning("Stage-2 retrain failed: %s", exc, exc_info=True)
            result.stage2_error = str(exc)

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

    db = Path(db_path) if db_path else DEFAULT_CORRECTIONS_DB
    if not db.exists():
        return None

    from .store import CorrectionStore

    store = CorrectionStore(db)
    try:
        threshold = getattr(cfg, "ml_retrain_threshold", 50)
        model_path = getattr(cfg, "ml_model_path", str(DEFAULT_ML_MODEL))
        stage2_model_path = getattr(
            cfg,
            "ml_stage2_model_path",
            str(DEFAULT_SUBTYPE_MODEL),
        )
        ensemble = getattr(cfg, "ml_ensemble_enabled", False)

        if not check_retrain_needed(store, threshold=threshold):
            return RetrainResult(
                new_corrections=store.count_corrections_since_last_train(),
                threshold=threshold,
            )

        return auto_retrain(
            store,
            model_path=model_path,
            stage2_model_path=stage2_model_path,
            ensemble=ensemble,
            threshold=threshold,
        )
    finally:
        store.close()


# ── Level 2: candidate classifier retrain ──────────────────────────────


def auto_retrain_candidate_classifier(
    *,
    db_path: Path | str | None = None,
    model_path: Path | str = "data/candidate_classifier.pkl",
    min_rows: int = 100,
) -> dict[str, Any]:
    """Retrain the candidate hit/miss classifier if enough outcome data exists.

    Parameters
    ----------
    db_path : Path or str or None
        Corrections database path. Defaults to config.DEFAULT_CORRECTIONS_DB.
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

    db_path_resolved = Path(db_path) if db_path else DEFAULT_CORRECTIONS_DB
    model_path = Path(model_path)

    if not db_path_resolved.exists():
        return {"skipped": True, "reason": "no_db"}

    store = CorrectionStore(db_path_resolved)
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
