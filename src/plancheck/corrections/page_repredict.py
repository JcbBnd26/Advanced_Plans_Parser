"""Single-page re-prediction using a freshly retrained classifier.

After a micro-retrain, call :func:`repredict_page` to re-score
detections on the *next* page using the updated model. Only the
ML label and confidence are updated — features, bounding boxes,
and text remain unchanged.

Usage
-----
::

    from plancheck.corrections.page_repredict import repredict_page

    result = repredict_page(store, doc_id, page=3, model_path=path)
    print(f"Updated {result.n_updated}, skipped {result.n_skipped}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Set

log = logging.getLogger(__name__)


@dataclass
class RepredictResult:
    """Outcome of a single-page re-prediction."""

    n_updated: int = 0
    n_skipped: int = 0
    elapsed_s: float = 0.0


def repredict_page(
    store: Any,
    doc_id: str,
    page: int,
    model_path: Path | str = "data/element_classifier.pkl",
    *,
    corrected_det_ids: Optional[Set[str]] = None,
) -> RepredictResult:
    """Re-predict labels for all detections on a single page.

    Loads cached features from the database, runs the freshly-trained
    classifier, and writes updated labels + confidence back. Detections
    whose IDs appear in *corrected_det_ids* are skipped to respect
    prior manual corrections.

    Parameters
    ----------
    store : CorrectionStore
        Open correction database.
    doc_id : str
        Document identifier.
    page : int
        Zero-based page number.
    model_path : Path or str
        Path to the trained model pickle.
    corrected_det_ids : set[str], optional
        Detection IDs to skip (already corrected by the user).

    Returns
    -------
    RepredictResult
        Counts of updated and skipped detections plus elapsed time.
    """
    from .classifier import ElementClassifier

    t0 = time.perf_counter()
    model_path = Path(model_path)
    result = RepredictResult()
    skip_ids = corrected_det_ids or set()

    # Also gather any historical corrections for this page so we
    # don't overwrite labels the user previously corrected.
    try:
        page_corrections = store.get_corrections_for_page(doc_id, page)
        for corr in page_corrections:
            det_id = corr.get("detection_id")
            if det_id:
                skip_ids.add(det_id)
    except Exception:  # noqa: BLE001 — corrections lookup is best-effort
        log.debug("Could not load corrections for skip list", exc_info=True)

    # Load detections for this page
    detections = store.get_latest_detections_for_page(doc_id, page)
    if not detections:
        log.debug(
            "No detections for doc=%s page=%d — nothing to re-predict", doc_id, page
        )
        result.elapsed_s = time.perf_counter() - t0
        return result

    # Instantiate a fresh classifier (picks up the just-saved model)
    clf = ElementClassifier(model_path=model_path)

    for det in detections:
        det_id = det["detection_id"]
        if det_id in skip_ids:
            result.n_skipped += 1
            continue

        features = det.get("features")
        if not features:
            result.n_skipped += 1
            continue

        label, confidence = clf.predict(features)
        store.update_detection_prediction(det_id, label, confidence)
        result.n_updated += 1

    log.info(
        "Re-predicted page %d: %d updated, %d skipped (%.3fs)",
        page,
        result.n_updated,
        result.n_skipped,
        time.perf_counter() - t0,
    )

    result.elapsed_s = time.perf_counter() - t0
    return result
