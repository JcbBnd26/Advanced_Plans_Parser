"""Training data mixin for CorrectionStore.

Provides training set generation from corrections and pseudo-labels.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from .store_utils import _deterministic_sort_key, _gen_id, _utcnow_iso

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


class TrainingDataMixin:
    """Mixin providing training data generation operations."""

    # These attributes are provided by CorrectionStore
    _conn: object
    _write_lock: object

    def generate_pseudo_labels(
        self,
        confidence_threshold: float = 0.95,
        max_per_label: int = 500,
    ) -> int:
        """Generate pseudo-labels from high-confidence rule-based detections.

        Bootstraps training data by treating detections with very high
        pipeline confidence as ground truth. This is useful for cold-start
        scenarios before human corrections are available.

        Parameters
        ----------
        confidence_threshold : float
            Minimum confidence score required for a detection to be
            considered as a pseudo-label. Default 0.95.
        max_per_label : int
            Maximum number of pseudo-labels per element type to avoid
            class imbalance. Default 500.

        Returns
        -------
        int
            Number of pseudo-label examples inserted into training_examples.
        """
        with self._write_lock():
            # Remove existing pseudo-labels before regenerating
            self._conn.execute("DELETE FROM training_examples WHERE source = 'pseudo'")

            # Select high-confidence detections, grouped by label
            rows = self._conn.execute(
                """
                SELECT d.detection_id, d.element_type, d.features_json, d.confidence
                FROM detections d
                WHERE d.confidence >= ?
                  AND d.element_type IS NOT NULL
                  AND d.features_json IS NOT NULL
                  AND d.detection_id NOT IN (
                      SELECT detection_id FROM corrections WHERE detection_id IS NOT NULL
                  )
                ORDER BY d.element_type, d.confidence DESC
                """,
                (confidence_threshold,),
            ).fetchall()

            # Group by label and cap at max_per_label
            by_label: dict[str, list] = defaultdict(list)
            for r in rows:
                label = r["element_type"]
                if len(by_label[label]) < max_per_label:
                    by_label[label].append(r)

            now = _utcnow_iso()
            count = 0
            for label, group in sorted(by_label.items()):
                # Sort deterministically within class for reproducible splits
                group.sort(key=lambda r: _deterministic_sort_key(r["detection_id"]))
                n = len(group)
                train_end = max(1, int(n * 0.7))
                val_end = max(train_end + 1, int(n * 0.9)) if n >= 2 else train_end
                for i, r in enumerate(group):
                    if i < train_end:
                        split = "train"
                    elif i < val_end:
                        split = "val"
                    else:
                        split = "test"
                    self._conn.execute(
                        "INSERT INTO training_examples "
                        "(example_id, source, correction_id, detection_id, "
                        " label, features_json, split, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            _gen_id("ex_"),
                            "pseudo",
                            None,
                            r["detection_id"],
                            label,
                            r["features_json"],
                            split,
                            now,
                        ),
                    )
                    count += 1

            self._conn.commit()
            return count

    def build_training_set(self, include_pseudo_labels: bool = False) -> int:
        """Rebuild the ``training_examples`` table from corrections.

        For each non-delete correction, creates a training example
        whose label is the *corrected* element type.  Delete
        corrections are included as negative examples with label
        ``__negative__`` so the classifier learns to recognise
        false-positive regions.

        The split is **truly stratified**: within each label group,
        examples are sorted by a deterministic MD5 hash of
        ``detection_id`` then the first 70 % are assigned to
        ``train``, the next 20 % to ``val``, and the remaining
        10 % to ``test``.

        This guarantees every class with ≥2 examples has
        representation in at least two splits.

        Parameters
        ----------
        include_pseudo_labels : bool
            If True, also generate pseudo-labels from high-confidence
            detections to bootstrap training data. Default False.

        Returns the number of examples inserted.
        """
        with self._write_lock():
            self._conn.execute("DELETE FROM training_examples")

            # ── Positive examples (relabel / accept / reshape) ──────────
            rows = self._conn.execute(
                "SELECT c.correction_id, c.detection_id, c.corrected_element_type, "
                "       c.corrected_features_json, d.features_json "
                "FROM corrections c "
                "JOIN detections d ON c.detection_id = d.detection_id "
                "WHERE c.correction_type != 'delete' "
                "  AND c.detection_id IS NOT NULL"
            ).fetchall()

            # ── Negative examples (deletions → false positives) ─────────
            delete_rows = self._conn.execute(
                "SELECT c.correction_id, c.detection_id, "
                "       c.corrected_features_json, d.features_json "
                "FROM corrections c "
                "JOIN detections d ON c.detection_id = d.detection_id "
                "WHERE c.correction_type = 'delete' "
                "  AND c.detection_id IS NOT NULL"
            ).fetchall()

            # ── Stratified split: group by label, force distribution ────
            by_label: dict[str, list] = defaultdict(list)
            for r in rows:
                by_label[r["corrected_element_type"]].append(r)
            for r in delete_rows:
                by_label["__negative__"].append(r)

            now = _utcnow_iso()
            count = 0
            for _label, group in sorted(by_label.items()):
                # Sort deterministically within this class using MD5
                group.sort(key=lambda r: _deterministic_sort_key(r["detection_id"]))
                n = len(group)
                if n < 5:
                    log.warning(
                        "Class %r has only %d examples — "
                        "stratification is unreliable at this size",
                        _label,
                        n,
                    )
                train_end = max(1, int(n * 0.7))  # at least 1 in train
                val_end = max(train_end + 1, int(n * 0.9)) if n >= 2 else train_end
                for i, r in enumerate(group):
                    features_json = r["corrected_features_json"] or r["features_json"]
                    if i < train_end:
                        split = "train"
                    elif i < val_end:
                        split = "val"
                    else:
                        split = "test"
                    self._conn.execute(
                        "INSERT INTO training_examples "
                        "(example_id, source, correction_id, detection_id, "
                        " label, features_json, split, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            _gen_id("ex_"),
                            "correction",
                            r["correction_id"],
                            r["detection_id"],
                            _label,
                            features_json,
                            split,
                            now,
                        ),
                    )
                    count += 1

            self._conn.commit()

        # Optionally add pseudo-labels from high-confidence detections
        if include_pseudo_labels:
            count += self.generate_pseudo_labels()

        return count

    def export_training_jsonl(self, output_path: Path) -> int:
        """Write training examples as JSON-Lines to *output_path*.

        Each line: ``{"example_id": …, "label": …, "features": {…}, "split": …}``

        Returns the number of lines written.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows = self._conn.execute(
            "SELECT example_id, label, features_json, split " "FROM training_examples"
        ).fetchall()

        count = 0
        with open(output_path, "w", encoding="utf-8") as fh:
            for r in rows:
                obj = {
                    "example_id": r["example_id"],
                    "label": r["label"],
                    "features": json.loads(r["features_json"]),
                    "split": r["split"],
                }
                fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count += 1
        return count


__all__ = ["TrainingDataMixin"]
