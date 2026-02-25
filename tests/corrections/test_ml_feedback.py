"""Tests for the ML feedback loop in pipeline + store."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from plancheck.config import GroupingConfig
from plancheck.corrections.classifier import _NUMERIC_KEYS, ElementClassifier
from plancheck.corrections.store import CorrectionStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _register_doc(store: CorrectionStore, doc_id: str = "sha256:test123") -> str:
    store._conn.execute(
        "INSERT OR IGNORE INTO documents "
        "(doc_id, filename, page_count, ingested_at) VALUES (?, ?, ?, ?)",
        (doc_id, "test.pdf", 3, "2026-01-01T00:00:00Z"),
    )
    store._conn.commit()
    return doc_id


def _make_features(zone: str = "header", font_size: float = 8.0) -> dict:
    return {
        "font_size_pt": font_size,
        "font_size_max_pt": font_size + 2,
        "font_size_min_pt": font_size - 2,
        "is_all_caps": 0,
        "is_bold": 0,
        "token_count": 5,
        "row_count": 2,
        "x_frac": 0.5,
        "y_frac": 0.5,
        "x_center_frac": 0.6,
        "y_center_frac": 0.6,
        "width_frac": 0.2,
        "height_frac": 0.3,
        "aspect_ratio": 1.5,
        "contains_digit": 0,
        "starts_with_digit": 0,
        "has_colon": 0,
        "has_period_after_num": 0,
        "text_length": 20,
        "avg_chars_per_token": 4.0,
        "zone": zone,
        "neighbor_count": 2,
    }


def _train_model(tmp_path: Path) -> Path:
    """Train a small model and return its path."""
    model_path = tmp_path / "model.pkl"
    jsonl_path = tmp_path / "train.jsonl"

    lines: list[str] = []
    labels = ["header", "notes_column", "legend"]
    for i in range(15):
        feats = _make_features(zone=labels[i % 3], font_size=8.0 + i)
        example = {
            "features": feats,
            "label": labels[i % 3],
            "split": "train",
        }
        lines.append(json.dumps(example))
    jsonl_path.write_text("\n".join(lines), encoding="utf-8")

    clf = ElementClassifier(model_path=model_path)
    clf.train(str(jsonl_path))
    return model_path


# ---------------------------------------------------------------------------
# get_prior_corrections_by_bbox
# ---------------------------------------------------------------------------


class TestGetPriorCorrectionsByBbox:
    def test_returns_empty_no_corrections(self, tmp_path: Path) -> None:
        store = CorrectionStore(db_path=tmp_path / "test.db")
        doc_id = _register_doc(store)

        store.save_detection(
            doc_id=doc_id,
            page=0,
            run_id="run1",
            element_type="header",
            bbox=(100, 200, 400, 300),
            text_content="test",
            features=_make_features(),
        )

        result = store.get_prior_corrections_by_bbox(doc_id, 0)
        assert result == []

    def test_returns_relabel_correction(self, tmp_path: Path) -> None:
        store = CorrectionStore(db_path=tmp_path / "test.db")
        doc_id = _register_doc(store)

        det_id = store.save_detection(
            doc_id=doc_id,
            page=0,
            run_id="run1",
            element_type="header",
            bbox=(100, 200, 400, 300),
            text_content="test",
            features=_make_features(),
        )
        store.save_correction(
            doc_id=doc_id,
            page=0,
            correction_type="relabel",
            corrected_label="notes_column",
            corrected_bbox=(100, 200, 400, 300),
            detection_id=det_id,
            original_label="header",
            original_bbox=(100, 200, 400, 300),
            session_id="s1",
        )

        result = store.get_prior_corrections_by_bbox(doc_id, 0)
        assert len(result) == 1
        assert result[0]["corrected_label"] == "notes_column"
        assert result[0]["correction_type"] == "relabel"

    def test_returns_most_recent_correction(self, tmp_path: Path) -> None:
        store = CorrectionStore(db_path=tmp_path / "test.db")
        doc_id = _register_doc(store)

        det_id = store.save_detection(
            doc_id=doc_id,
            page=0,
            run_id="run1",
            element_type="header",
            bbox=(100, 200, 400, 300),
            text_content="test",
            features=_make_features(),
        )
        # First correction: relabel to notes_column
        store.save_correction(
            doc_id=doc_id,
            page=0,
            correction_type="relabel",
            corrected_label="notes_column",
            corrected_bbox=(100, 200, 400, 300),
            detection_id=det_id,
            original_label="header",
            original_bbox=(100, 200, 400, 300),
            session_id="s1",
        )
        # Second correction: relabel to legend (most recent wins)
        store.save_correction(
            doc_id=doc_id,
            page=0,
            correction_type="relabel",
            corrected_label="legend",
            corrected_bbox=(100, 200, 400, 300),
            detection_id=det_id,
            original_label="notes_column",
            original_bbox=(100, 200, 400, 300),
            session_id="s2",
        )

        result = store.get_prior_corrections_by_bbox(doc_id, 0)
        assert len(result) == 1
        assert result[0]["corrected_label"] == "legend"

    def test_returns_delete_correction(self, tmp_path: Path) -> None:
        store = CorrectionStore(db_path=tmp_path / "test.db")
        doc_id = _register_doc(store)

        det_id = store.save_detection(
            doc_id=doc_id,
            page=0,
            run_id="run1",
            element_type="header",
            bbox=(100, 200, 400, 300),
            text_content="test",
            features=_make_features(),
        )
        store.save_correction(
            doc_id=doc_id,
            page=0,
            correction_type="delete",
            corrected_label="header",
            corrected_bbox=(100, 200, 400, 300),
            detection_id=det_id,
            original_label="header",
            original_bbox=(100, 200, 400, 300),
            session_id="s1",
        )

        result = store.get_prior_corrections_by_bbox(doc_id, 0)
        assert len(result) == 1
        assert result[0]["correction_type"] == "delete"


# ---------------------------------------------------------------------------
# _bbox_iou
# ---------------------------------------------------------------------------


class TestBboxIou:
    def test_identical_boxes(self) -> None:
        from plancheck.pipeline import _bbox_iou

        assert _bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_no_overlap(self) -> None:
        from plancheck.pipeline import _bbox_iou

        assert _bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_partial_overlap(self) -> None:
        from plancheck.pipeline import _bbox_iou

        iou = _bbox_iou((0, 0, 10, 10), (5, 5, 15, 15))
        # Intersection = 5*5 = 25
        # Union = 100 + 100 - 25 = 175
        assert pytest.approx(iou, abs=0.01) == 25 / 175

    def test_zero_area_box(self) -> None:
        from plancheck.pipeline import _bbox_iou

        assert _bbox_iou((0, 0, 0, 0), (0, 0, 10, 10)) == 0.0


# ---------------------------------------------------------------------------
# _apply_ml_feedback
# ---------------------------------------------------------------------------


class TestApplyMlFeedback:
    def test_prior_relabel_updates_detection(self, tmp_path: Path) -> None:
        """A prior relabel correction should update the detection's label."""
        from plancheck.pipeline import _apply_ml_feedback

        store = CorrectionStore(db_path=tmp_path / "test.db")
        doc_id = _register_doc(store)
        cfg = GroupingConfig()  # real config — ml_enabled=True by default

        # Save a detection
        det_id = store.save_detection(
            doc_id=doc_id,
            page=0,
            run_id="run1",
            element_type="header",
            bbox=(100, 200, 400, 300),
            text_content="test",
            features=_make_features(),
        )

        # Save a correction
        store.save_correction(
            doc_id=doc_id,
            page=0,
            correction_type="relabel",
            corrected_label="notes_column",
            corrected_bbox=(100, 200, 400, 300),
            detection_id=det_id,
            original_label="header",
            original_bbox=(100, 200, 400, 300),
            session_id="s1",
        )

        # Now simulate re-run: new detection at same bbox
        det_id_2 = store.save_detection(
            doc_id=doc_id,
            page=0,
            run_id="run2",
            element_type="header",  # rule-based says header again
            bbox=(100, 200, 400, 300),
            text_content="test",
            features=_make_features(),
        )

        _apply_ml_feedback(store, doc_id, 0, cfg)

        # Check that the new detection was relabelled
        dets = store.get_detections_for_page(doc_id, 0)
        # Find the new detection
        new_det = next(d for d in dets if d["detection_id"] == det_id_2)
        assert new_det["element_type"] == "notes_column"

    def test_prior_delete_sets_low_confidence(self, tmp_path: Path) -> None:
        """A prior delete correction should set confidence to 0."""
        from plancheck.pipeline import _apply_ml_feedback

        store = CorrectionStore(db_path=tmp_path / "test.db")
        doc_id = _register_doc(store)
        cfg = GroupingConfig()

        det_id = store.save_detection(
            doc_id=doc_id,
            page=0,
            run_id="run1",
            element_type="header",
            bbox=(100, 200, 400, 300),
            text_content="test",
            features=_make_features(),
        )
        store.save_correction(
            doc_id=doc_id,
            page=0,
            correction_type="delete",
            corrected_label="header",
            corrected_bbox=(100, 200, 400, 300),
            detection_id=det_id,
            original_label="header",
            original_bbox=(100, 200, 400, 300),
            session_id="s1",
        )

        # Re-run: same detection reappears
        det_id_2 = store.save_detection(
            doc_id=doc_id,
            page=0,
            run_id="run2",
            element_type="header",
            bbox=(100, 200, 400, 300),
            text_content="test",
            features=_make_features(),
        )

        _apply_ml_feedback(store, doc_id, 0, cfg)

        dets = store.get_detections_for_page(doc_id, 0)
        new_det = next(d for d in dets if d["detection_id"] == det_id_2)
        assert new_det["confidence"] == 0.0

    def test_no_corrections_no_error(self, tmp_path: Path) -> None:
        """No corrections and no model should not raise."""
        from plancheck.pipeline import _apply_ml_feedback

        store = CorrectionStore(db_path=tmp_path / "test.db")
        doc_id = _register_doc(store)
        cfg = GroupingConfig()

        store.save_detection(
            doc_id=doc_id,
            page=0,
            run_id="run1",
            element_type="header",
            bbox=(100, 200, 400, 300),
            text_content="test",
            features=_make_features(),
        )

        # Should not raise
        _apply_ml_feedback(store, doc_id, 0, cfg)

    def test_ml_relabel_with_model(self, tmp_path: Path) -> None:
        """When a model exists and is confident, it should relabel uncorrected detections."""
        from plancheck.corrections.classifier import ElementClassifier
        from plancheck.pipeline import _apply_ml_feedback

        store = CorrectionStore(db_path=tmp_path / "test.db")
        doc_id = _register_doc(store)
        cfg = GroupingConfig()

        model_path = _train_model(tmp_path)

        det_id = store.save_detection(
            doc_id=doc_id,
            page=0,
            run_id="run1",
            element_type="header",
            bbox=(100, 200, 400, 300),
            text_content="test",
            features=_make_features(zone="header"),
        )

        # Patch ElementClassifier to use our trained model
        real_clf = ElementClassifier(model_path=model_path)
        with patch(
            "plancheck.corrections.classifier.ElementClassifier",
            return_value=real_clf,
        ) as mock_cls:
            mock_cls.return_value = real_clf
            _apply_ml_feedback(store, doc_id, 0, cfg)

        # Detection should have confidence set
        dets = store.get_detections_for_page(doc_id, 0)
        det = next(d for d in dets if d["detection_id"] == det_id)
        assert det["confidence"] is not None
        assert 0.0 <= det["confidence"] <= 1.0

    def test_corrected_detections_not_relabelled_by_ml(self, tmp_path: Path) -> None:
        """ML should not override a prior user correction."""
        from plancheck.corrections.classifier import ElementClassifier
        from plancheck.pipeline import _apply_ml_feedback

        store = CorrectionStore(db_path=tmp_path / "test.db")
        doc_id = _register_doc(store)
        cfg = GroupingConfig()

        model_path = _train_model(tmp_path)

        det_id = store.save_detection(
            doc_id=doc_id,
            page=0,
            run_id="run1",
            element_type="legend",
            bbox=(100, 200, 400, 300),
            text_content="test",
            features=_make_features(zone="legend"),
        )
        # User corrects to notes_column
        store.save_correction(
            doc_id=doc_id,
            page=0,
            correction_type="relabel",
            corrected_label="notes_column",
            corrected_bbox=(100, 200, 400, 300),
            detection_id=det_id,
            original_label="legend",
            original_bbox=(100, 200, 400, 300),
            session_id="s1",
        )

        real_clf = ElementClassifier(model_path=model_path)
        with patch(
            "plancheck.corrections.classifier.ElementClassifier",
            return_value=real_clf,
        ) as mock_cls:
            mock_cls.return_value = real_clf
            _apply_ml_feedback(store, doc_id, 0, cfg)

        dets = store.get_detections_for_page(doc_id, 0)
        det = next(d for d in dets if d["detection_id"] == det_id)
        # User's correction should be preserved (notes_column from pass 1)
        assert det["element_type"] == "notes_column"

    def test_ml_disabled_skips_relabelling(self, tmp_path: Path) -> None:
        """When cfg.ml_enabled is False, ML relabelling should be skipped entirely."""
        from plancheck.corrections.classifier import ElementClassifier
        from plancheck.pipeline import _apply_ml_feedback

        store = CorrectionStore(db_path=tmp_path / "test.db")
        doc_id = _register_doc(store)
        cfg = GroupingConfig(ml_enabled=False)

        model_path = _train_model(tmp_path)
        cfg.ml_model_path = str(model_path)

        det_id = store.save_detection(
            doc_id=doc_id,
            page=0,
            run_id="run1",
            element_type="header",
            bbox=(100, 200, 400, 300),
            text_content="test",
            features=_make_features(zone="legend"),
        )

        _apply_ml_feedback(store, doc_id, 0, cfg)

        # ML should not have touched the detection — label stays header
        dets = store.get_detections_for_page(doc_id, 0)
        det = next(d for d in dets if d["detection_id"] == det_id)
        assert det["element_type"] == "header"
        # Confidence should not have been set by ML either
        assert det["confidence"] is None

    def test_ml_relabel_confidence_from_config(self, tmp_path: Path) -> None:
        """The ML relabel threshold should come from cfg.ml_relabel_confidence."""
        from plancheck.corrections.classifier import ElementClassifier
        from plancheck.pipeline import _apply_ml_feedback

        store = CorrectionStore(db_path=tmp_path / "test.db")
        doc_id = _register_doc(store)

        model_path = _train_model(tmp_path)

        det_id = store.save_detection(
            doc_id=doc_id,
            page=0,
            run_id="run1",
            element_type="header",
            bbox=(100, 200, 400, 300),
            text_content="test",
            features=_make_features(zone="header"),
        )

        # Use a very high threshold so ML never relabels
        cfg = GroupingConfig(ml_relabel_confidence=0.9999)
        cfg.ml_model_path = str(model_path)

        real_clf = ElementClassifier(model_path=model_path)
        with patch(
            "plancheck.corrections.classifier.ElementClassifier",
            return_value=real_clf,
        ) as mock_cls:
            mock_cls.return_value = real_clf
            _apply_ml_feedback(store, doc_id, 0, cfg)

        dets = store.get_detections_for_page(doc_id, 0)
        det = next(d for d in dets if d["detection_id"] == det_id)
        # Even though model may disagree, the extreme threshold prevents relabelling
        # (confidence will be set but label is preserved unless confidence >= 0.9999)
        assert det["confidence"] is not None
        assert 0.0 <= det["confidence"] <= 1.0
        # With threshold at 0.9999, label should stay as "header" unless the
        # model has near-perfect confidence — which is unlikely on 15 examples
        assert det["element_type"] == "header"
