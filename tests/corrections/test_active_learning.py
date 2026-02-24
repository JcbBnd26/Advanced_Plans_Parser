"""Tests for plancheck.corrections.active_learning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plancheck.corrections.active_learning import (
    rank_pages_by_uncertainty,
    suggest_next_page,
)
from plancheck.corrections.classifier import _NUMERIC_KEYS, ElementClassifier
from plancheck.corrections.store import CorrectionStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_features(zone: str = "header", font_size: float = 8.0) -> dict:
    """Build a minimal feature dict."""
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


def _register_doc(store: CorrectionStore, doc_id: str = "sha256:test123") -> str:
    store._conn.execute(
        "INSERT OR IGNORE INTO documents "
        "(doc_id, filename, page_count, ingested_at) VALUES (?, ?, ?, ?)",
        (doc_id, "test.pdf", 3, "2026-01-01T00:00:00Z"),
    )
    store._conn.commit()
    return doc_id


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
# Tests
# ---------------------------------------------------------------------------


class TestRankPagesByUncertainty:
    def test_no_model_returns_empty(self, tmp_path: Path) -> None:
        store = CorrectionStore(db_path=tmp_path / "test.db")
        result = rank_pages_by_uncertainty(store, model_path=tmp_path / "missing.pkl")
        assert result == []

    def test_no_detections_returns_empty(self, tmp_path: Path) -> None:
        model_path = _train_model(tmp_path)
        store = CorrectionStore(db_path=tmp_path / "test.db")
        result = rank_pages_by_uncertainty(store, model_path=model_path)
        assert result == []

    def test_returns_ranked_list(self, tmp_path: Path) -> None:
        model_path = _train_model(tmp_path)
        store = CorrectionStore(db_path=tmp_path / "test.db")
        doc_id = _register_doc(store)

        # Add detections on two pages
        for page in [0, 1]:
            for i in range(3):
                feats = _make_features(zone="header", font_size=8.0 + i + page * 10)
                store.save_detection(
                    doc_id=doc_id,
                    page=page,
                    run_id="run_test",
                    element_type="header",
                    bbox=(100, 100 + i * 50, 300, 150 + i * 50),
                    text_content="test",
                    features=feats,
                )

        result = rank_pages_by_uncertainty(store, model_path=model_path)
        assert len(result) == 2
        # Each entry is (doc_id, page, uncertainty)
        for doc, page, unc in result:
            assert doc == doc_id
            assert isinstance(page, int)
            assert 0.0 <= unc <= 1.0

        # Should be sorted by uncertainty descending
        assert result[0][2] >= result[1][2]


class TestSuggestNextPage:
    def test_no_model_returns_none(self, tmp_path: Path) -> None:
        store = CorrectionStore(db_path=tmp_path / "test.db")
        result = suggest_next_page(store, model_path=tmp_path / "missing.pkl")
        assert result is None

    def test_suggests_uncorrected_page(self, tmp_path: Path) -> None:
        model_path = _train_model(tmp_path)
        store = CorrectionStore(db_path=tmp_path / "test.db")
        doc_id = _register_doc(store)

        # Page 0: add detection + correction (fully corrected)
        feats = _make_features(zone="header")
        det_id_0 = store.save_detection(
            doc_id=doc_id,
            page=0,
            run_id="run_test",
            element_type="header",
            bbox=(100, 100, 300, 200),
            text_content="corrected",
            features=feats,
        )
        store.save_correction(
            doc_id=doc_id,
            page=0,
            correction_type="accept",
            corrected_label="header",
            corrected_bbox=(100, 100, 300, 200),
            detection_id=det_id_0,
            session_id="s1",
        )

        # Page 1: add detection, no correction (uncorrected)
        feats2 = _make_features(zone="notes_column")
        store.save_detection(
            doc_id=doc_id,
            page=1,
            run_id="run_test",
            element_type="notes_column",
            bbox=(400, 100, 600, 200),
            text_content="uncorrected",
            features=feats2,
        )

        result = suggest_next_page(store, model_path=model_path)
        assert result is not None
        assert result == (doc_id, 1)

    def test_all_corrected_returns_none(self, tmp_path: Path) -> None:
        model_path = _train_model(tmp_path)
        store = CorrectionStore(db_path=tmp_path / "test.db")
        doc_id = _register_doc(store)

        feats = _make_features(zone="header")
        det_id = store.save_detection(
            doc_id=doc_id,
            page=0,
            run_id="run_test",
            element_type="header",
            bbox=(100, 100, 300, 200),
            text_content="test",
            features=feats,
        )
        store.save_correction(
            doc_id=doc_id,
            page=0,
            correction_type="accept",
            corrected_label="header",
            corrected_bbox=(100, 100, 300, 200),
            detection_id=det_id,
            session_id="s1",
        )

        result = suggest_next_page(store, model_path=model_path)
        assert result is None
