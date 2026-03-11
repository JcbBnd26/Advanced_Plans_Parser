from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from plancheck.config import GroupingConfig
from plancheck.corrections.hierarchical_classifier import ClassificationResult
from plancheck.corrections.retrain_trigger import auto_retrain
from plancheck.corrections.store import CorrectionStore


def _register_doc(store: CorrectionStore, doc_id: str = "sha256:hier-flow") -> str:
    store._conn.execute(
        "INSERT OR IGNORE INTO documents "
        "(doc_id, filename, page_count, ingested_at) VALUES (?, ?, ?, ?)",
        (doc_id, "hier-flow.pdf", 2, "2026-01-01T00:00:00Z"),
    )
    store._conn.commit()
    return doc_id


def _save_corrected_detection(
    store: CorrectionStore,
    *,
    doc_id: str,
    page: int,
    detection_index: int,
    corrected_label: str,
    original_label: str,
    features: dict,
) -> str:
    bbox = (
        10.0 + detection_index * 5.0,
        20.0 + page * 10.0,
        110.0 + detection_index * 5.0,
        50.0 + page * 10.0,
    )
    det_id = store.save_detection(
        doc_id=doc_id,
        page=page,
        run_id="run_train",
        element_type=original_label,
        bbox=bbox,
        text_content=f"title text {detection_index}",
        confidence=0.9,
        features=features,
    )
    store.save_correction(
        doc_id=doc_id,
        page=page,
        correction_type="relabel",
        corrected_label=corrected_label,
        corrected_bbox=bbox,
        detection_id=det_id,
        original_label=original_label,
        original_bbox=bbox,
        session_id="session_train",
    )
    return det_id


def test_subtype_annotations_flow_into_stage2_training_and_hierarchical_feedback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_features: dict,
) -> None:
    from plancheck.pipeline import _apply_ml_feedback

    store = CorrectionStore(db_path=tmp_path / "test.db")
    doc_id = _register_doc(store)

    title_features = dict(sample_features)
    title_features["zone"] = "title_block"
    title_features["kw_title_block_pattern"] = 1

    # Create enough subtype corrections for Stage 2, plus a non-subtype label
    # that should remain in the Stage-1 export but be filtered from Stage 2.
    subtype_labels = [
        "page_title",
        "plan_title",
        "page_title",
        "plan_title",
        "page_title",
        "plan_title",
        "page_title",
        "plan_title",
        "page_title",
        "plan_title",
    ]
    for idx, label in enumerate(subtype_labels):
        _save_corrected_detection(
            store,
            doc_id=doc_id,
            page=0,
            detection_index=idx,
            corrected_label=label,
            original_label="title_block",
            features=title_features,
        )

    _save_corrected_detection(
        store,
        doc_id=doc_id,
        page=0,
        detection_index=50,
        corrected_label="header",
        original_label="header",
        features=dict(sample_features),
    )

    captured: dict[str, object] = {}

    def _fake_stage1_train(self, path, calibrate=True, ensemble=False):
        jsonl_path = Path(path)
        captured["stage1_labels"] = [
            json.loads(line)["label"]
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path.write_text("stage1", encoding="utf-8")
        return {
            "accuracy": 0.91,
            "f1_weighted": 0.89,
            "n_train": 8,
            "n_val": 3,
            "labels": ["header", "page_title", "plan_title"],
            "per_class": {},
            "holdout_predictions": [],
        }

    def _fake_stage2_train(self, path, calibrate=True, ensemble=False):
        jsonl_path = Path(path)
        captured["stage2_labels"] = [
            json.loads(line)["label"]
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path.write_text("stage2", encoding="utf-8")
        return {
            "accuracy": 0.84,
            "f1_weighted": 0.81,
            "n_train": 8,
            "n_val": 2,
            "labels": ["page_title", "plan_title"],
            "per_class": {},
            "holdout_predictions": [],
        }

    monkeypatch.setattr(
        "plancheck.corrections.classifier.ElementClassifier.train",
        _fake_stage1_train,
    )
    monkeypatch.setattr(
        "plancheck.corrections.subtype_classifier.TitleSubtypeClassifier.train",
        _fake_stage2_train,
    )
    monkeypatch.setattr(
        "plancheck.corrections.drift_detection.DriftDetector",
        lambda threshold=0.3: SimpleNamespace(
            fit=lambda path: None,
            save=lambda path: None,
        ),
    )

    stage1_model_path = tmp_path / "stage1.pkl"
    stage2_model_path = tmp_path / "stage2.pkl"
    retrain_result = auto_retrain(
        store,
        model_path=stage1_model_path,
        stage2_model_path=stage2_model_path,
        threshold=0,
    )

    assert retrain_result.accepted is True
    assert retrain_result.stage2_trained is True

    training_data_path = tmp_path / "training_data.jsonl"
    stage2_training_data_path = tmp_path / "training_data_stage2.jsonl"
    assert training_data_path.exists()
    assert stage2_training_data_path.exists()
    assert set(captured["stage1_labels"]) == {"header", "page_title", "plan_title"}
    assert set(captured["stage2_labels"]) == {"page_title", "plan_title"}
    assert "header" not in captured["stage2_labels"]

    runtime_det_id = store.save_detection(
        doc_id=doc_id,
        page=1,
        run_id="run_feedback",
        element_type="title_block",
        bbox=(300.0, 320.0, 420.0, 360.0),
        text_content="SHEET A-101",
        confidence=None,
        features=title_features,
    )

    cfg = GroupingConfig(
        ml_hierarchical_enabled=True,
        ml_relabel_confidence=0.8,
    )
    cfg.ml_model_path = str(stage1_model_path)
    cfg.ml_stage2_model_path = str(stage2_model_path)

    router_calls: dict[str, object] = {}

    def _fake_classify_element(
        features,
        *,
        text="",
        stage1_model_path=None,
        stage2_model_path=None,
        **kwargs,
    ):
        router_calls["text"] = text
        router_calls["stage1_model_path"] = stage1_model_path
        router_calls["stage2_model_path"] = stage2_model_path
        router_calls["features"] = dict(features)
        return ClassificationResult(
            label="page_title",
            confidence=0.93,
            family="title",
            family_confidence=0.95,
            subtype="page_title",
            subtype_confidence=0.93,
        )

    monkeypatch.setattr(
        "plancheck.corrections.hierarchical_classifier.classify_element",
        _fake_classify_element,
    )

    _apply_ml_feedback(store, doc_id, 1, cfg)

    dets = store.get_detections_for_page(doc_id, 1)
    runtime_det = next(d for d in dets if d["detection_id"] == runtime_det_id)
    assert runtime_det["element_type"] == "page_title"
    assert runtime_det["confidence"] == pytest.approx(0.93)
    assert router_calls["text"] == "SHEET A-101"
    assert router_calls["stage1_model_path"] == stage1_model_path
    assert router_calls["stage2_model_path"] == stage2_model_path
    assert router_calls["features"]["zone"] == "title_block"
