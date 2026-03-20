"""Training runs mixin for CorrectionStore.

Provides persistence and management of training experiment metadata,
including metrics, hyperparameters, and retrain heuristics.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from .store_utils import _gen_id, _utcnow_iso

if TYPE_CHECKING:
    pass


class TrainingRunsMixin:
    """Mixin providing training run tracking operations."""

    # These attributes are provided by CorrectionStore
    _conn: object
    _write_lock: object

    def save_training_run(
        self,
        metrics: dict,
        model_path: str = "",
        notes: str = "",
        holdout_predictions: list[dict] | None = None,
        hyperparams: dict | None = None,
        feature_set: dict | None = None,
        training_curves: dict | None = None,
        feature_version: int = 0,
    ) -> str:
        """Persist a training-run record and return its ``run_…`` ID.

        Parameters
        ----------
        metrics : dict
            Output of :func:`~plancheck.corrections.metrics.compute_metrics`,
            augmented with ``n_train`` and ``n_val``.
        model_path : str
            Path to the saved model file (informational).
        notes : str
            Free-text annotation for this run.
        holdout_predictions : list[dict] | None
            Optional list of ``{"label_true", "label_pred", "confidence"}``
            dicts from the validation set evaluation.
        hyperparams : dict | None
            Hyperparameters used for training (Phase 4.4).
        feature_set : dict | None
            Feature schema description (Phase 4.4).
        training_curves : dict | None
            Epoch-level training/validation loss curves (Phase 4.4).
        feature_version : int
            Feature schema version (Phase 4.3).
        """
        run_id = _gen_id("run_")
        hp_json = json.dumps(holdout_predictions) if holdout_predictions else ""
        hyper_json = json.dumps(hyperparams) if hyperparams else ""
        fs_json = json.dumps(feature_set) if feature_set else ""
        tc_json = json.dumps(training_curves) if training_curves else ""
        with self._write_lock():
            self._conn.execute(
                "INSERT INTO training_runs "
                "(run_id, trained_at, n_train, n_val, accuracy, f1_macro, f1_weighted, "
                " labels_json, per_class_json, model_path, notes, holdout_preds_json, "
                " hyperparams_json, feature_set_json, training_curves_json, "
                " feature_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    _utcnow_iso(),
                    metrics.get("n_train", 0),
                    metrics.get("n_val", 0),
                    metrics.get("accuracy", 0.0),
                    metrics.get("f1_macro", 0.0),
                    metrics.get("f1_weighted", 0.0),
                    json.dumps(metrics.get("labels", [])),
                    json.dumps(metrics.get("per_class", {})),
                    model_path,
                    notes,
                    hp_json,
                    hyper_json,
                    fs_json,
                    tc_json,
                    feature_version,
                ),
            )
            self._conn.commit()
        return run_id

    def get_training_history(self) -> list[dict[str, Any]]:
        """Return all training runs ordered newest-first.

        Returns
        -------
        list[dict]
            Each dict has: ``run_id``, ``trained_at``, ``n_train``,
            ``n_val``, ``accuracy``, ``f1_macro``, ``f1_weighted``,
            ``labels``, ``per_class``, ``model_path``, ``notes``,
            ``holdout_predictions``, ``hyperparams``, ``feature_set``,
            ``training_curves``, ``feature_version``.
        """
        rows = self._conn.execute(
            "SELECT * FROM training_runs ORDER BY trained_at DESC, rowid DESC"
        ).fetchall()
        results: list[dict[str, Any]] = []
        for r in rows:
            keys = r.keys()
            hp_raw = r["holdout_preds_json"] if "holdout_preds_json" in keys else ""
            hyper_raw = r["hyperparams_json"] if "hyperparams_json" in keys else ""
            fs_raw = r["feature_set_json"] if "feature_set_json" in keys else ""
            tc_raw = r["training_curves_json"] if "training_curves_json" in keys else ""
            fv = r["feature_version"] if "feature_version" in keys else 0
            results.append(
                {
                    "run_id": r["run_id"],
                    "trained_at": r["trained_at"],
                    "n_train": r["n_train"],
                    "n_val": r["n_val"],
                    "accuracy": r["accuracy"],
                    "f1_macro": r["f1_macro"],
                    "f1_weighted": r["f1_weighted"],
                    "labels": json.loads(r["labels_json"]),
                    "per_class": json.loads(r["per_class_json"]),
                    "model_path": r["model_path"],
                    "notes": r["notes"],
                    "holdout_predictions": json.loads(hp_raw) if hp_raw else [],
                    "hyperparams": json.loads(hyper_raw) if hyper_raw else {},
                    "feature_set": json.loads(fs_raw) if fs_raw else {},
                    "training_curves": json.loads(tc_raw) if tc_raw else {},
                    "feature_version": fv,
                }
            )
        return results

    def compare_runs(
        self, run_id_a: str, run_id_b: str, *, threshold: float = 0.005
    ) -> dict[str, Any]:
        """Compare two training runs and return metric deltas.

        Parameters
        ----------
        run_id_a, run_id_b : str
            Run IDs to compare.  Convention: *a* is the baseline (older),
            *b* is the candidate (newer).
        threshold : float
            Minimum absolute F1 delta to count as improved/regressed.

        Returns
        -------
        dict
            ``f1_weighted_delta``, ``accuracy_delta``,
            ``improved_classes``, ``regressed_classes``,
            ``per_class_deltas``.

        Raises
        ------
        ValueError
            If either *run_id* is not found.
        """
        history = {r["run_id"]: r for r in self.get_training_history()}
        if run_id_a not in history:
            raise ValueError(f"Run not found: {run_id_a}")
        if run_id_b not in history:
            raise ValueError(f"Run not found: {run_id_b}")

        a = history[run_id_a]
        b = history[run_id_b]

        f1_delta = b["f1_weighted"] - a["f1_weighted"]
        acc_delta = b["accuracy"] - a["accuracy"]

        # Per-class F1 deltas
        all_classes = sorted(set(a["per_class"]) | set(b["per_class"]))
        per_class_deltas: dict[str, float] = {}
        improved: list[str] = []
        regressed: list[str] = []
        for cls in all_classes:
            f1_a = a["per_class"].get(cls, {}).get("f1", 0.0)
            f1_b = b["per_class"].get(cls, {}).get("f1", 0.0)
            delta = f1_b - f1_a
            per_class_deltas[cls] = round(delta, 6)
            if delta > threshold:
                improved.append(cls)
            elif delta < -threshold:
                regressed.append(cls)

        return {
            "f1_weighted_delta": round(f1_delta, 6),
            "accuracy_delta": round(acc_delta, 6),
            "improved_classes": improved,
            "regressed_classes": regressed,
            "per_class_deltas": per_class_deltas,
        }

    # ── retrain helpers (Phase 4.2) ────────────────────────────────

    def last_train_date(self) -> str | None:
        """Return ISO-8601 timestamp of the most recent training run.

        Returns *None* when no runs have been recorded.
        """
        row = self._conn.execute(
            "SELECT trained_at FROM training_runs ORDER BY trained_at DESC LIMIT 1"
        ).fetchone()
        return row["trained_at"] if row else None

    def count_corrections_since(self, since_iso: str | None = None) -> int:
        """Count corrections added after *since_iso*.

        Parameters
        ----------
        since_iso : str | None
            ISO-8601 timestamp.  When *None*, returns the total count.
        """
        if since_iso is None:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM corrections").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM corrections WHERE corrected_at > ?",
                (since_iso,),
            ).fetchone()
        return row["n"] if row else 0

    def count_corrections_since_last_train(self) -> int:
        """Return number of new corrections since the last training run."""
        return self.count_corrections_since(self.last_train_date())

    def should_retrain(self, threshold: int = 50) -> bool:
        """Return *True* when enough new corrections have accumulated.

        Parameters
        ----------
        threshold : int
            Minimum number of new corrections needed (default 50).
        """
        return self.count_corrections_since_last_train() >= threshold


__all__ = ["TrainingRunsMixin"]
