"""Data-drift detection for element-type classification features.

Computes reference statistics (mean, stddev, percentile bounds) from
training data and detects when production inputs drift outside the
expected distribution.

Two granularities:

* **Feature-level** — per-dimension Z-score or percentile-breach tests.
* **Page-level** — fraction of features flagged on a single page.

Usage
-----
::

    detector = DriftDetector()
    detector.fit(Path("data/training_data.jsonl"))
    detector.save(Path("data/drift_stats.json"))

    # Later, in the pipeline:
    detector = DriftDetector.load(Path("data/drift_stats.json"))
    result = detector.check(feature_vector)
    page_result = detector.check_page(feature_vectors)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ── Result containers ──────────────────────────────────────────────────


@dataclass
class DriftResult:
    """Drift check result for a single feature vector."""

    is_drifted: bool = False
    drift_fraction: float = 0.0
    flagged_features: list[int] = field(default_factory=list)
    n_features: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize drift result to a plain dict for JSON export."""
        return asdict(self)


@dataclass
class PageDriftResult:
    """Aggregated drift result across all detections on a page."""

    is_drifted: bool = False
    drifted_fraction: float = 0.0
    n_detections: int = 0
    n_drifted: int = 0
    per_detection: list[DriftResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize page-level drift result to a plain dict."""
        return {
            "is_drifted": self.is_drifted,
            "drifted_fraction": self.drifted_fraction,
            "n_detections": self.n_detections,
            "n_drifted": self.n_drifted,
            "per_detection": [d.to_dict() for d in self.per_detection],
        }


# ── Detector ───────────────────────────────────────────────────────────


class DriftDetector:
    """Statistical data-drift detector using percentile bounds.

    For each feature dimension, the detector records the 1st and 99th
    percentiles from training data.  At inference time, a feature is
    *flagged* when its value falls outside these bounds.  A vector is
    considered *drifted* when the fraction of flagged features exceeds
    ``threshold``.

    Parameters
    ----------
    threshold : float
        Fraction of features that must be flagged for a vector to be
        considered drifted (default 0.3 = 30%).
    """

    def __init__(self, threshold: float = 0.3) -> None:
        self.threshold = threshold
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._p1: np.ndarray | None = None
        self._p99: np.ndarray | None = None
        self._n_features: int = 0
        self._n_samples: int = 0

    # ── Fitting ────────────────────────────────────────────────────

    def fit(self, jsonl_path: Path) -> None:
        """Compute reference statistics from a training JSONL file.

        Parameters
        ----------
        jsonl_path : Path
            JSONL file exported by
            :meth:`CorrectionStore.export_training_jsonl`.
            Each line must have a ``features`` dict.
        """
        from .classifier import encode_features

        vectors: list[np.ndarray] = []
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("split") != "train":
                    continue
                vec = encode_features(rec["features"])
                vectors.append(vec)

        if not vectors:
            raise ValueError("No training-split examples found in JSONL")

        data = np.array(vectors, dtype=np.float64)
        self._n_samples = data.shape[0]
        self._n_features = data.shape[1]
        self._mean = np.mean(data, axis=0)
        self._std = np.std(data, axis=0)
        # Avoid division by zero for constant features
        self._std[self._std < 1e-12] = 1.0
        self._p1 = np.percentile(data, 1, axis=0)
        self._p99 = np.percentile(data, 99, axis=0)

        log.info(
            "DriftDetector fit: %d samples, %d features",
            self._n_samples,
            self._n_features,
        )

    def fit_from_vectors(self, vectors: list[np.ndarray]) -> None:
        """Fit from pre-computed feature vectors (for testing)."""
        if not vectors:
            raise ValueError("No vectors provided")
        data = np.array(vectors, dtype=np.float64)
        self._n_samples = data.shape[0]
        self._n_features = data.shape[1]
        self._mean = np.mean(data, axis=0)
        self._std = np.std(data, axis=0)
        self._std[self._std < 1e-12] = 1.0
        self._p1 = np.percentile(data, 1, axis=0)
        self._p99 = np.percentile(data, 99, axis=0)

    @property
    def is_fitted(self) -> bool:
        """Return *True* if reference statistics have been computed."""
        return self._mean is not None

    # ── Persistence ────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Serialize reference statistics to JSON.

        Parameters
        ----------
        path : Path
            Output file path (parent directories created automatically).
        """
        if not self.is_fitted:
            raise RuntimeError("DriftDetector has not been fit yet")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_samples": self._n_samples,
            "n_features": self._n_features,
            "threshold": self.threshold,
            "mean": self._mean.tolist(),  # type: ignore[union-attr]
            "std": self._std.tolist(),  # type: ignore[union-attr]
            "p1": self._p1.tolist(),  # type: ignore[union-attr]
            "p99": self._p99.tolist(),  # type: ignore[union-attr]
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("Drift stats saved to %s", path)

    @classmethod
    def load(cls, path: Path, *, threshold: float | None = None) -> "DriftDetector":
        """Load reference statistics from a previously saved JSON file.

        Parameters
        ----------
        path : Path
            JSON file written by :meth:`save`.
        threshold : float, optional
            Override the stored threshold.

        Returns
        -------
        DriftDetector
            Ready-to-use detector.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        det = cls(threshold=threshold if threshold is not None else data["threshold"])
        det._n_samples = data["n_samples"]
        det._n_features = data["n_features"]
        det._mean = np.array(data["mean"], dtype=np.float64)
        det._std = np.array(data["std"], dtype=np.float64)
        det._p1 = np.array(data["p1"], dtype=np.float64)
        det._p99 = np.array(data["p99"], dtype=np.float64)
        return det

    # ── Checking ───────────────────────────────────────────────────

    def check(self, feature_vector: np.ndarray) -> DriftResult:
        """Check a single feature vector for drift.

        Parameters
        ----------
        feature_vector : numpy.ndarray
            1-D float array of the same dimension as training data.

        Returns
        -------
        DriftResult
            Drift assessment for this vector.
        """
        if not self.is_fitted:
            return DriftResult()

        vec = np.asarray(feature_vector, dtype=np.float64)
        n = min(len(vec), self._n_features)
        flagged: list[int] = []
        for i in range(n):
            if vec[i] < self._p1[i] or vec[i] > self._p99[i]:  # type: ignore[index]
                flagged.append(i)

        fraction = len(flagged) / n if n > 0 else 0.0
        return DriftResult(
            is_drifted=fraction >= self.threshold,
            drift_fraction=round(fraction, 6),
            flagged_features=flagged,
            n_features=n,
        )

    def check_page(
        self,
        feature_vectors: list[np.ndarray],
        *,
        page_threshold: float = 0.5,
    ) -> PageDriftResult:
        """Check all detections on a page for aggregate drift.

        Parameters
        ----------
        feature_vectors : list[numpy.ndarray]
            Feature vectors for each detection on the page.
        page_threshold : float
            Fraction of drifted detections to flag the entire page
            (default 0.5 = 50%).

        Returns
        -------
        PageDriftResult
            Aggregated drift assessment.
        """
        if not feature_vectors or not self.is_fitted:
            return PageDriftResult()

        per_det: list[DriftResult] = [self.check(v) for v in feature_vectors]
        n_drifted = sum(1 for d in per_det if d.is_drifted)
        frac = n_drifted / len(per_det) if per_det else 0.0

        return PageDriftResult(
            is_drifted=frac >= page_threshold,
            drifted_fraction=round(frac, 6),
            n_detections=len(per_det),
            n_drifted=n_drifted,
            per_detection=per_det,
        )
