"""ML feedback loop: prior corrections, classifier relabelling, drift detection.

Extracted from :mod:`plancheck.pipeline` for maintainability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import GroupingConfig
from .models.geometry import bbox_iou as _bbox_iou

log = logging.getLogger(__name__)


def _apply_ml_feedback(
    store: Any,
    doc_id: str,
    page_num: int,
    cfg: GroupingConfig,
    page_image: Any = None,
) -> list:
    """Apply prior corrections + ML predictions to this page's detections.

    Three passes (all best-effort — failures are logged and swallowed):

    1. **Prior corrections**: look up every correction the user previously
       made for this (doc_id, page).  Match each correction's
       ``original_bbox`` against new detections by IoU ≥ 0.5.  When
       matched, update the detection's ``element_type`` (and bbox if
       reshaped).  Delete-corrections hide the detection.

    2. **ML relabelling**: if ``cfg.ml_enabled`` is *True* and a trained
       :class:`ElementClassifier` exists, run it over every *uncorrected*
       detection.  When the model disagrees with the rule-based label
       **and** model confidence ≥ ``cfg.ml_relabel_confidence``, update
       the detection's label.

    3. **Confidence scoring**: write model confidence to every detection
       regardless of whether the label was changed.

    4. **Drift detection** (Phase 4.1): when ``cfg.ml_drift_enabled``
       is *True* and drift stats exist, check each feature vector
       against the reference distribution.

    When *page_image* is provided and ``cfg.ml_vision_enabled`` is True,
    CNN image embeddings are extracted for each detection and appended
    to the feature vector.

    Returns
    -------
    list[dict]
        Drift warning dicts (empty when drift detection is disabled).
    """
    drift_warnings: list = []
    try:
        dets = store.get_detections_for_page(doc_id, page_num)
        if not dets:
            return drift_warnings

        # ── Pass 1: prior corrections ──────────────────────────────
        prior = store.get_prior_corrections_by_bbox(doc_id, page_num)
        corrected_det_ids: set[str] = set()

        # Wrap all Pass 1 updates in a write lock for transaction safety
        with store._write_lock():
            for corr in prior:
                orig_bbox = corr["original_bbox"]
                if orig_bbox is None:
                    continue

                # Find best matching new detection by IoU.
                # Prefer a *different* detection than the one the correction
                # was originally linked to (so we match against new re-run
                # detections).  Fall back to the original if nothing else
                # matches (single-detection or no-re-run scenario).
                orig_det_id = corr["detection_id"]
                best_iou = 0.0
                best_det: dict | None = None
                fallback_det: dict | None = None
                for det in dets:
                    iou = _bbox_iou(orig_bbox, det["bbox"])
                    if det["detection_id"] == orig_det_id:
                        if iou >= 0.5:
                            fallback_det = det
                        continue
                    if iou > best_iou:
                        best_iou = iou
                        best_det = det

                if best_iou < 0.5 or best_det is None:
                    # No new detection matched — fall back to original
                    if fallback_det is not None:
                        best_det = fallback_det
                    else:
                        continue

                did = best_det["detection_id"]
                corrected_det_ids.add(did)

                if corr["correction_type"] == "delete":
                    # Mark as very-low confidence so it visually fades
                    store._conn.execute(
                        "UPDATE detections SET confidence = 0.0 "
                        "WHERE detection_id = ?",
                        (did,),
                    )
                elif corr["correction_type"] in ("relabel", "accept"):
                    new_label = corr["corrected_label"]
                    store._conn.execute(
                        "UPDATE detections SET element_type = ? "
                        "WHERE detection_id = ?",
                        (new_label, did),
                    )
                elif corr["correction_type"] == "reshape":
                    new_bbox = corr["corrected_bbox"]
                    new_label = corr["corrected_label"]
                    store._conn.execute(
                        "UPDATE detections SET element_type = ?, "
                        "  bbox_x0 = ?, bbox_y0 = ?, bbox_x1 = ?, bbox_y1 = ? "
                        "WHERE detection_id = ?",
                        (new_label, *new_bbox, did),
                    )

            store._conn.commit()

        # ── Pass 2 + 3: ML relabelling + confidence scoring ───────
        if not cfg.ml_enabled:
            return drift_warnings

        # Choose between hierarchical router and direct Stage-1 classifier.
        use_hierarchical = getattr(cfg, "ml_hierarchical_enabled", False)

        from .corrections.classifier import ElementClassifier

        clf = ElementClassifier(model_path=Path(cfg.ml_model_path))
        if not clf.model_exists():
            return drift_warnings

        # Optionally set up CNN image feature extractor
        img_extractor = None
        if cfg.ml_vision_enabled and page_image is not None:
            try:
                from .corrections.image_features import (
                    ImageFeatureExtractor,
                    is_vision_available,
                )

                if is_vision_available():
                    img_extractor = ImageFeatureExtractor(
                        backbone=cfg.ml_vision_backbone
                    )
            except Exception:  # noqa: BLE001 — optional dep may fail to load
                log.warning("Vision feature extractor init failed", exc_info=True)

        # Optionally set up text embedder
        text_embedder = None
        if cfg.ml_embeddings_enabled:
            try:
                from .corrections.text_embeddings import (
                    TextEmbedder,
                    is_embeddings_available,
                )

                if is_embeddings_available():
                    text_embedder = TextEmbedder(model_name=cfg.ml_embeddings_model)
            except Exception:  # noqa: BLE001 — optional dep may fail to load
                log.warning("Text embedder init failed", exc_info=True)

        # Refresh detections after pass 1 modifications
        dets = store.get_detections_for_page(doc_id, page_num)

        # Feature cache setup (Phase 4.3)
        use_cache = getattr(cfg, "ml_feature_cache_enabled", False)
        feat_version = 0
        if use_cache:
            try:
                from .corrections.classifier import FEATURE_VERSION

                feat_version = FEATURE_VERSION
            except ImportError:
                use_cache = False

        # Collect updates for batch execution (transaction safety)
        pass2_conf_updates: list[tuple[float, str]] = []
        pass2_label_updates: list[tuple[str, str]] = []

        for det in dets:
            if not det["features"]:
                continue

            did = det["detection_id"]

            # Check feature cache first (Phase 4.3)
            cached_vec = None
            if use_cache:
                try:
                    raw = store.get_cached_features(did, feat_version)
                    if raw is not None:
                        cached_vec = raw
                except Exception:  # noqa: BLE001
                    log.warning("Feature cache read failed for %s", did, exc_info=True)

            if cached_vec is not None:
                # Use cached vector for prediction directly
                import numpy as _np

                vec = _np.array(cached_vec, dtype=_np.float64)
                pred_label, pred_conf = clf.predict_from_vector(vec)
            elif use_hierarchical:
                # Hierarchical: Stage 1 → family normalisation → Stage 2 → LLM
                from .corrections.hierarchical_classifier import classify_element

                det_text = det.get("text_content", "") or det.get("text", "")
                if not det_text and det["features"]:
                    det_text = str(det["features"].get("_text", ""))

                img_feat = None
                if img_extractor is not None:
                    bbox = det["bbox"]
                    if bbox:
                        img_feat = img_extractor.extract(page_image, tuple(bbox))

                text_emb = None
                if text_embedder is not None and det_text:
                    text_emb = text_embedder.embed(det_text)

                result = classify_element(
                    det["features"],
                    text=det_text,
                    stage1_model_path=Path(cfg.ml_model_path),
                    stage2_model_path=Path(cfg.ml_stage2_model_path),
                    enable_llm=getattr(cfg, "enable_llm_checks", False),
                    llm_provider=getattr(cfg, "llm_provider", "ollama"),
                    llm_model=getattr(cfg, "llm_model", "llama3.1:8b"),
                    llm_api_key=getattr(cfg, "llm_api_key", ""),
                    llm_api_base=getattr(cfg, "llm_api_base", "http://localhost:11434"),
                    llm_policy=getattr(cfg, "llm_policy", "local_only"),
                    image_features=img_feat,
                    text_embedding=text_emb,
                )
                pred_label = result.label
                pred_conf = result.confidence
            else:
                # Extract image features when vision is enabled
                img_feat = None
                if img_extractor is not None:
                    bbox = det["bbox"]
                    if bbox:
                        img_feat = img_extractor.extract(page_image, tuple(bbox))

                # Extract text embedding when embeddings are enabled
                text_emb = None
                if text_embedder is not None:
                    det_features = det["features"]
                    # Reconstruct text from features if available, or use
                    # a representative text snippet from the detection
                    det_text = det.get("text_content", "") or det.get("text", "")
                    if not det_text and det_features:
                        det_text = str(det_features.get("_text", ""))
                    if det_text:
                        text_emb = text_embedder.embed(det_text)

                pred_label, pred_conf = clf.predict(
                    det["features"],
                    image_features=img_feat,
                    text_embedding=text_emb,
                )

                # Store in cache for next time (Phase 4.3)
                if use_cache:
                    try:
                        from .corrections.classifier import encode_features as _ef

                        vec = _ef(
                            det["features"],
                            image_features=img_feat,
                            text_embedding=text_emb,
                        )
                        store.cache_features(did, vec.tolist(), feat_version)
                    except Exception:  # noqa: BLE001
                        log.warning(
                            "Feature cache write failed for %s", did, exc_info=True
                        )

            # Collect updates for batch execution within write lock
            if did not in corrected_det_ids:
                # Always write confidence — unless already set by prior
                # delete correction (pass 1 sets it to 0.0).
                pass2_conf_updates.append((pred_conf, did))

                # Only relabel if: not already corrected by user AND
                # model disagrees AND model is confident enough
                if (
                    pred_label != det["element_type"]
                    and pred_conf >= cfg.ml_relabel_confidence
                ):
                    log.info(
                        "ML relabel: %s %s → %s (conf=%.2f)",
                        did[:12],
                        det["element_type"],
                        pred_label,
                        pred_conf,
                    )
                    pass2_label_updates.append((pred_label, did))

        # Execute Pass 2 updates in a single transaction
        with store._write_lock():
            for conf, did in pass2_conf_updates:
                store._conn.execute(
                    "UPDATE detections SET confidence = ? WHERE detection_id = ?",
                    (conf, did),
                )
            for label, did in pass2_label_updates:
                store._conn.execute(
                    "UPDATE detections SET element_type = ? WHERE detection_id = ?",
                    (label, did),
                )
            store._conn.commit()

        # ── Pass 4: drift detection (Phase 4.1) ───────────────────
        if getattr(cfg, "ml_drift_enabled", False):
            try:
                from .corrections.classifier import encode_features
                from .corrections.drift_detection import DriftDetector

                drift_stats_path = Path(
                    getattr(cfg, "ml_drift_stats_path", "data/drift_stats.json")
                )
                if drift_stats_path.exists():
                    drift_threshold = getattr(cfg, "ml_drift_threshold", 0.3)
                    detector = DriftDetector.load(
                        drift_stats_path, threshold=drift_threshold
                    )
                    dets = store.get_detections_for_page(doc_id, page_num)
                    for det in dets:
                        if not det["features"]:
                            continue
                        vec = encode_features(det["features"])
                        dr = detector.check(vec)
                        if dr.is_drifted:
                            drift_warnings.append(
                                {
                                    "detection_id": det["detection_id"],
                                    "drift_fraction": dr.drift_fraction,
                                    "flagged_features": dr.flagged_features,
                                }
                            )
                    if drift_warnings:
                        log.info(
                            "Drift: %d/%d detections flagged on page %d",
                            len(drift_warnings),
                            len(dets),
                            page_num,
                        )
            except Exception:  # noqa: BLE001 — drift detection is best-effort
                log.warning("Drift detection failed", exc_info=True)

    except Exception:  # noqa: BLE001 — ML feedback must not break pipeline
        log.warning("ML feedback failed", exc_info=True)

    return drift_warnings
