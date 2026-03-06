"""Lightweight experiment tracking for element-type classifier training.

Extends the existing ``training_runs`` table with higher-level query
functions for listing, comparing, ranking, and exporting experiments.

This is intentionally *not* MLflow/W&B — it stores everything in the
same SQLite database that already holds corrections, keeping the
desktop application self-contained with zero extra dependencies.

Usage
-----
::

    tracker = ExperimentTracker(store)
    runs = tracker.list_experiments(limit=10)
    best = tracker.best_experiment(metric="f1_weighted")
    comparison = tracker.compare_experiments(run_a, run_b)
    tracker.export_csv(Path("experiments.csv"))
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ExperimentSummary:
    """Condensed view of a single experiment (training run)."""

    run_id: str = ""
    trained_at: str = ""
    n_train: int = 0
    n_val: int = 0
    accuracy: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    model_path: str = ""
    notes: str = ""
    hyperparams: dict = field(default_factory=dict)
    feature_version: int = 0
    n_classes: int = 0
    per_class: dict = field(default_factory=dict)
    holdout_predictions: list = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "trained_at": self.trained_at,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "model_path": self.model_path,
            "notes": self.notes,
            "hyperparams": self.hyperparams,
            "feature_version": self.feature_version,
            "n_classes": self.n_classes,
            "per_class": self.per_class,
            "holdout_predictions": self.holdout_predictions,
        }


@dataclass
class ExperimentComparison:
    """Side-by-side comparison of two experiments."""

    run_a: ExperimentSummary = field(default_factory=ExperimentSummary)
    run_b: ExperimentSummary = field(default_factory=ExperimentSummary)
    f1_weighted_delta: float = 0.0
    accuracy_delta: float = 0.0
    improved_classes: list[str] = field(default_factory=list)
    regressed_classes: list[str] = field(default_factory=list)
    per_class_deltas: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_a": self.run_a.to_dict(),
            "run_b": self.run_b.to_dict(),
            "f1_weighted_delta": self.f1_weighted_delta,
            "accuracy_delta": self.accuracy_delta,
            "improved_classes": self.improved_classes,
            "regressed_classes": self.regressed_classes,
            "per_class_deltas": self.per_class_deltas,
        }


class ExperimentTracker:
    """High-level experiment tracking backed by CorrectionStore.

    Parameters
    ----------
    store : CorrectionStore
        Open correction database with ``training_runs`` table.
    """

    def __init__(self, store: Any) -> None:
        self._store = store

    # ── Listing ────────────────────────────────────────────────────

    def list_experiments(
        self,
        limit: int = 50,
        *,
        sort_by: str = "trained_at",
        ascending: bool = False,
    ) -> list[ExperimentSummary]:
        """Return recent experiments as summary objects.

        Parameters
        ----------
        limit : int
            Maximum number of runs to return.
        sort_by : str
            Column to sort by: ``trained_at``, ``f1_weighted``,
            ``accuracy``, ``n_train``.
        ascending : bool
            Sort order (default newest/best first).
        """
        history = self._store.get_training_history()

        # Sort
        reverse = not ascending
        if sort_by in ("f1_weighted", "accuracy", "n_train", "n_val"):
            history.sort(key=lambda r: r.get(sort_by, 0), reverse=reverse)
        else:
            history.sort(key=lambda r: r.get("trained_at", ""), reverse=reverse)

        results: list[ExperimentSummary] = []
        for r in history[:limit]:
            results.append(
                ExperimentSummary(
                    run_id=r["run_id"],
                    trained_at=r["trained_at"],
                    n_train=r["n_train"],
                    n_val=r["n_val"],
                    accuracy=r["accuracy"],
                    f1_macro=r["f1_macro"],
                    f1_weighted=r["f1_weighted"],
                    model_path=r.get("model_path", ""),
                    notes=r.get("notes", ""),
                    hyperparams=r.get("hyperparams", {}),
                    feature_version=r.get("feature_version", 0),
                    n_classes=len(r.get("labels", [])),
                    per_class=r.get("per_class", {}),
                    holdout_predictions=r.get("holdout_predictions", []),
                )
            )
        return results

    # ── Comparison ─────────────────────────────────────────────────

    def compare_experiments(
        self,
        run_id_a: str,
        run_id_b: str,
    ) -> ExperimentComparison:
        """Compare two training runs side-by-side.

        Parameters
        ----------
        run_id_a : str
            Baseline (older) run ID.
        run_id_b : str
            Candidate (newer) run ID.

        Returns
        -------
        ExperimentComparison
        """
        raw = self._store.compare_runs(run_id_a, run_id_b)
        exps = {e.run_id: e for e in self.list_experiments(limit=10000)}

        run_a = exps.get(run_id_a, ExperimentSummary(run_id=run_id_a))
        run_b = exps.get(run_id_b, ExperimentSummary(run_id=run_id_b))

        return ExperimentComparison(
            run_a=run_a,
            run_b=run_b,
            f1_weighted_delta=raw["f1_weighted_delta"],
            accuracy_delta=raw["accuracy_delta"],
            improved_classes=raw["improved_classes"],
            regressed_classes=raw["regressed_classes"],
            per_class_deltas=raw["per_class_deltas"],
        )

    # ── Best run ───────────────────────────────────────────────────

    def best_experiment(
        self,
        metric: str = "f1_weighted",
    ) -> ExperimentSummary | None:
        """Return the experiment with the highest value for *metric*.

        Parameters
        ----------
        metric : str
            One of ``f1_weighted``, ``f1_macro``, ``accuracy``.

        Returns
        -------
        ExperimentSummary or None
            The best run, or *None* if no runs exist.
        """
        exps = self.list_experiments(
            limit=1,
            sort_by=metric,
            ascending=False,
        )
        return exps[0] if exps else None

    # ── Export ─────────────────────────────────────────────────────

    def export_csv(self, path: Path) -> int:
        """Export all experiments to a CSV file.

        Parameters
        ----------
        path : Path
            Output file (parent directories created automatically).

        Returns
        -------
        int
            Number of rows written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        exps = self.list_experiments(limit=100000, ascending=True)
        if not exps:
            return 0

        fieldnames = [
            "run_id",
            "trained_at",
            "n_train",
            "n_val",
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "model_path",
            "notes",
            "feature_version",
            "n_classes",
        ]

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for exp in exps:
                writer.writerow(exp.to_dict())

        log.info("Exported %d experiments to %s", len(exps), path)
        return len(exps)
