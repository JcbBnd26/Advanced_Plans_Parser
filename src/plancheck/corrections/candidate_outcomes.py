"""Candidate outcomes mixin for CorrectionStore.

Provides persistence of VOCR candidate outcomes for Level 2 learning,
enabling the pipeline to learn from hit/miss patterns over time.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from .store_utils import _gen_id, _utcnow_iso

if TYPE_CHECKING:
    pass


class CandidateOutcomesMixin:
    """Mixin providing VOCR candidate outcome tracking."""

    # These attributes are provided by CorrectionStore
    _conn: object
    _write_lock: object

    def save_candidate_outcome(
        self,
        page: int,
        trigger_methods: list[str],
        outcome: str,
        confidence: float,
        bbox: tuple[float, float, float, float],
        *,
        doc_id: str = "",
        run_id: str = "",
        predicted_symbol: str = "",
        found_symbol: str = "",
        page_width: float = 0.0,
        page_height: float = 0.0,
        features: dict | None = None,
    ) -> str:
        """Persist one VOCR candidate outcome and return its ``co_…`` ID.

        Parameters
        ----------
        page : int
            Zero-based page index.
        trigger_methods : list[str]
            Detection methods that flagged this candidate.
        outcome : str
            ``"hit"`` or ``"miss"``.
        confidence : float
            Candidate confidence at decision time.
        bbox : tuple
            ``(x0, y0, x1, y1)`` in PDF points.
        doc_id, run_id : str
            Optional document / run identifiers.
        predicted_symbol, found_symbol : str
            The symbol the candidate predicted and what VOCR actually found.
        page_width, page_height : float
            Page dimensions (for normalisation later).
        features : dict | None
            Pre-computed feature vector for the classifier.
        """
        outcome_id = _gen_id("co_")
        with self._write_lock():
            self._conn.execute(
                "INSERT INTO candidate_outcomes "
                "(outcome_id, doc_id, page, run_id, trigger_methods, "
                " predicted_symbol, found_symbol, outcome, confidence, "
                " bbox_x0, bbox_y0, bbox_x1, bbox_y1, "
                " page_width, page_height, features_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    outcome_id,
                    doc_id,
                    page,
                    run_id,
                    ",".join(trigger_methods),
                    predicted_symbol,
                    found_symbol,
                    outcome,
                    confidence,
                    *bbox,
                    page_width,
                    page_height,
                    json.dumps(features or {}),
                    _utcnow_iso(),
                ),
            )
            self._conn.commit()
        return outcome_id

    def save_candidate_outcomes_batch(
        self,
        candidates: list,
        *,
        doc_id: str = "",
        run_id: str = "",
        page_width: float = 0.0,
        page_height: float = 0.0,
    ) -> int:
        """Persist a batch of :class:`VocrCandidate` objects.

        Parameters
        ----------
        candidates : list[VocrCandidate]
            Candidates with ``outcome`` set to ``"hit"`` or ``"miss"``.
        doc_id, run_id : str
            Optional context identifiers.
        page_width, page_height : float
            Page dimensions.

        Returns
        -------
        int
            Number of rows inserted.
        """
        with self._write_lock():
            now = _utcnow_iso()
            count = 0
            for c in candidates:
                if c.outcome not in ("hit", "miss"):
                    continue
                outcome_id = _gen_id("co_")
                self._conn.execute(
                    "INSERT INTO candidate_outcomes "
                    "(outcome_id, doc_id, page, run_id, trigger_methods, "
                    " predicted_symbol, found_symbol, outcome, confidence, "
                    " bbox_x0, bbox_y0, bbox_x1, bbox_y1, "
                    " page_width, page_height, features_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        outcome_id,
                        doc_id,
                        c.page,
                        run_id,
                        ",".join(c.trigger_methods),
                        c.predicted_symbol or "",
                        c.found_symbol or "",
                        c.outcome,
                        c.confidence,
                        c.x0,
                        c.y0,
                        c.x1,
                        c.y1,
                        page_width,
                        page_height,
                        json.dumps({}),
                        now,
                    ),
                )
                count += 1
            if count:
                self._conn.commit()
            return count

    def get_candidate_outcomes(
        self,
        *,
        min_rows: int = 0,
        outcome: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve candidate outcome rows for classifier training.

        Parameters
        ----------
        min_rows : int
            If the table has fewer than this many rows, return an empty
            list (avoids training on tiny datasets).
        outcome : str | None
            Filter to ``"hit"`` or ``"miss"`` only.  ``None`` = all.

        Returns
        -------
        list[dict]
            Each dict has keys matching the ``candidate_outcomes`` schema.
        """
        total = self._conn.execute(
            "SELECT COUNT(*) AS n FROM candidate_outcomes"
        ).fetchone()["n"]
        if total < min_rows:
            return []

        query = "SELECT * FROM candidate_outcomes"
        params: list = []
        if outcome:
            query += " WHERE outcome = ?"
            params.append(outcome)
        query += " ORDER BY created_at"

        return [dict(r) for r in self._conn.execute(query, params).fetchall()]

    def count_candidate_outcomes(self) -> dict[str, int]:
        """Return ``{total, hits, misses}`` counts."""
        rows = self._conn.execute(
            "SELECT outcome, COUNT(*) AS n FROM candidate_outcomes GROUP BY outcome"
        ).fetchall()
        counts = {r["outcome"]: r["n"] for r in rows}
        return {
            "total": sum(counts.values()),
            "hits": counts.get("hit", 0),
            "misses": counts.get("miss", 0),
        }


__all__ = ["CandidateOutcomesMixin"]
