"""Active learning — rank pages by model uncertainty.

Uses the trained :class:`ElementClassifier` to score every detection in
the database, then ranks pages by average uncertainty so the annotator
focuses on the most informative examples first.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

from .classifier import _DEFAULT_MODEL_PATH, ElementClassifier
from .store import CorrectionStore


def rank_pages_by_uncertainty(
    store: CorrectionStore,
    model_path: Path = _DEFAULT_MODEL_PATH,
) -> List[Tuple[str, int, float]]:
    """Rank all (doc_id, page) pairs by mean model uncertainty.

    Uncertainty for a single detection is ``1 - max(predict_proba)``.
    Pages with higher mean uncertainty should be annotated first.

    Parameters
    ----------
    store : CorrectionStore
        Open database connection.
    model_path : Path
        Location of the trained classifier pickle.

    Returns
    -------
    list[tuple[str, int, float]]
        ``(doc_id, page, mean_uncertainty)`` sorted descending by
        uncertainty.  Pages with zero detections are excluded.
    """
    clf = ElementClassifier(model_path)
    if not clf.model_exists():
        return []

    # Fetch all detections grouped by (doc_id, page)
    rows = store._conn.execute(
        "SELECT doc_id, page, features_json FROM detections " "ORDER BY doc_id, page"
    ).fetchall()

    # Group by (doc_id, page)

    page_groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        key = (row["doc_id"], row["page"])
        features = json.loads(row["features_json"])
        page_groups.setdefault(key, []).append(features)

    # Score each page
    results: list[tuple[str, int, float]] = []
    for (doc_id, page), feature_list in page_groups.items():
        predictions = clf.predict_batch(feature_list)
        uncertainties = [1.0 - conf for _, conf in predictions]
        mean_unc = sum(uncertainties) / len(uncertainties) if uncertainties else 0.0
        results.append((doc_id, page, round(mean_unc, 4)))

    # Sort by uncertainty descending
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def suggest_next_page(
    store: CorrectionStore,
    model_path: Path = _DEFAULT_MODEL_PATH,
) -> Optional[Tuple[str, int]]:
    """Suggest the highest-uncertainty page that is not fully corrected.

    A page is considered "fully corrected" when every detection on it
    has at least one correction row.

    Returns
    -------
    tuple[str, int] | None
        ``(doc_id, page)`` or *None* if no uncorrected pages remain.
    """
    ranked = rank_pages_by_uncertainty(store, model_path)
    if not ranked:
        return None

    for doc_id, page, _unc in ranked:
        # Count detections vs corrections
        row = store._conn.execute(
            "SELECT COUNT(*) FROM detections WHERE doc_id = ? AND page = ?",
            (doc_id, page),
        ).fetchone()
        det_count = row[0] if row else 0

        row = store._conn.execute(
            "SELECT COUNT(DISTINCT d.detection_id) "
            "FROM detections d "
            "JOIN corrections c ON d.detection_id = c.detection_id "
            "WHERE d.doc_id = ? AND d.page = ?",
            (doc_id, page),
        ).fetchone()
        cor_count = row[0] if row else 0

        if cor_count < det_count:
            return (doc_id, page)

    return None
