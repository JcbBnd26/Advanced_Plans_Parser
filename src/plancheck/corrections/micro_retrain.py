"""Lightweight micro-retrain for page-by-page training sessions.

Unlike :func:`auto_retrain`, this skips rollback checks, threshold
gates, and Stage-2 subtype training — designed for sub-2-second
inter-page retraining during an active annotation session.

Usage
-----
::

    from plancheck.corrections.micro_retrain import micro_retrain

    result = micro_retrain(store, model_path)
    if result.retrained:
        print(f"F1: {result.metrics.get('f1_weighted', 0):.1%}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_MIN_EXAMPLES = 10


@dataclass
class MicroRetrainResult:
    """Outcome of a micro-retrain attempt."""

    retrained: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    n_examples: int = 0
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for GUI consumption."""
        return {
            "retrained": self.retrained,
            "f1_weighted": self.metrics.get("f1_weighted", 0.0),
            "accuracy": self.metrics.get("accuracy", 0.0),
            "error": self.error,
            "n_examples": self.n_examples,
            "elapsed_s": self.elapsed_s,
        }


def micro_retrain(
    store: Any,
    model_path: Path | str = "data/element_classifier.pkl",
    *,
    calibrate: bool = True,
) -> MicroRetrainResult:
    """Retrain the classifier from ALL accumulated corrections.

    This is the fast-path retrain for the page-by-page training loop.
    It rebuilds the full training set every time to avoid catastrophic
    forgetting — with typical data sizes (~1000 rows) this takes < 2s.

    Parameters
    ----------
    store : CorrectionStore
        Open correction database.
    model_path : Path or str
        Where to save the trained model.
    calibrate : bool
        Enable isotonic calibration.

    Returns
    -------
    MicroRetrainResult
        Outcome of the retrain attempt.
    """
    from .classifier import ElementClassifier

    t0 = time.perf_counter()
    model_path = Path(model_path)
    result = MicroRetrainResult()

    try:
        # 1. Rebuild training set from all corrections
        n_examples = store.build_training_set()
        result.n_examples = n_examples
        if n_examples < _MIN_EXAMPLES:
            result.error = (
                f"Only {n_examples} training examples (need >= {_MIN_EXAMPLES})"
            )
            result.elapsed_s = time.perf_counter() - t0
            return result

        # 2. Export JSONL
        db_parent = store._db_path.parent
        jsonl_path = db_parent / "training_data.jsonl"
        store.export_training_jsonl(jsonl_path)

        # 3. Train (reuse ElementClassifier.train which wires the builders)
        clf = ElementClassifier(model_path=model_path)
        metrics = clf.train(
            jsonl_path,
            calibrate=calibrate,
            ensemble=False,
        )
        result.retrained = True
        result.metrics = metrics

        log.info(
            "Micro-retrain complete: F1=%.4f accuracy=%.4f (%d examples)",
            metrics.get("f1_weighted", 0.0),
            metrics.get("accuracy", 0.0),
            n_examples,
        )

        # 4. Record training run (best-effort)
        try:
            store.save_training_run(
                metrics,
                model_path=str(model_path),
                notes="micro-retrain",
                holdout_predictions=metrics.get("holdout_predictions"),
            )
        except Exception as exc:  # noqa: BLE001 — recording is optional
            log.warning("Could not record micro-retrain run: %s", exc)

    except Exception as exc:
        log.error("Micro-retrain failed: %s", exc, exc_info=True)
        result.error = str(exc)

    result.elapsed_s = time.perf_counter() - t0
    return result
