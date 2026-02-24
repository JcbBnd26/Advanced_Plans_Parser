"""Tests for plancheck.corrections.store.CorrectionStore."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from plancheck.corrections.store import CorrectionStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_register(store: CorrectionStore, doc_id: str = "sha256:abc123") -> str:
    """Directly insert a document row to avoid needing a real PDF."""
    store._conn.execute(
        "INSERT OR IGNORE INTO documents "
        "(doc_id, filename, page_count, ingested_at) VALUES (?, ?, ?, ?)",
        (doc_id, "test.pdf", 3, "2026-01-01T00:00:00Z"),
    )
    store._conn.commit()
    return doc_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegisterDocument:
    """test_register_document_creates_record"""

    def test_register_same_pdf_twice_inserts_one_row(self, tmp_path: Path) -> None:
        """Register the same PDF twice — only one documents row should exist."""
        store = CorrectionStore(db_path=tmp_path / "test.db")

        # Create a small fake PDF-like file (constant content → same hash)
        fake_pdf = tmp_path / "plan.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake content for hashing")

        # Patch pdfplumber so we don't need a real PDF structure
        mock_pdf = MagicMock()
        mock_pdf.pages = [None, None]  # 2 pages
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = lambda s: mock_pdf
        mock_ctx.__exit__ = lambda s, *a: None
        with patch.dict("sys.modules", {"pdfplumber": MagicMock()}) as _:
            import pdfplumber as plumber_mod

            plumber_mod.open = MagicMock(return_value=mock_ctx)

            doc_id_1 = store.register_document(fake_pdf)
            doc_id_2 = store.register_document(fake_pdf)

        assert doc_id_1 == doc_id_2
        assert doc_id_1.startswith("sha256:")

        rows = store._conn.execute("SELECT * FROM documents").fetchall()
        assert len(rows) == 1


class TestSaveAndRetrieveDetection:
    """test_save_and_retrieve_detection"""

    def test_round_trip(
        self, tmp_store: CorrectionStore, sample_features: dict
    ) -> None:
        """Save a detection and retrieve it; all fields must round-trip."""
        doc_id = _mock_register(tmp_store)
        bbox = (100.0, 200.0, 300.0, 250.0)

        det_id = tmp_store.save_detection(
            doc_id=doc_id,
            page=0,
            run_id="run_20260101_000000",
            element_type="notes_column",
            bbox=bbox,
            text_content="1. SHALL COMPLY",
            features=sample_features,
            confidence=0.87,
        )
        assert det_id.startswith("det_")

        dets = tmp_store.get_detections_for_page(doc_id, page=0)
        assert len(dets) == 1

        d = dets[0]
        assert d["detection_id"] == det_id
        assert d["element_type"] == "notes_column"
        assert d["confidence"] == pytest.approx(0.87)
        assert d["bbox"] == bbox
        assert d["text_content"] == "1. SHALL COMPLY"
        assert d["features"]["font_size_pt"] == 8.0
        assert isinstance(d["created_at"], str)


class TestSaveCorrectionRelabel:
    """test_save_correction_relabel"""

    def test_relabel_correction(
        self, tmp_store: CorrectionStore, sample_features: dict
    ) -> None:
        doc_id = _mock_register(tmp_store)
        bbox = (100.0, 200.0, 300.0, 250.0)

        det_id = tmp_store.save_detection(
            doc_id=doc_id,
            page=1,
            run_id="run_1",
            element_type="notes_column",
            bbox=bbox,
            text_content="LEGEND",
            features=sample_features,
        )

        cor_id = tmp_store.save_correction(
            doc_id=doc_id,
            page=1,
            correction_type="relabel",
            corrected_label="legend",
            corrected_bbox=bbox,
            detection_id=det_id,
            original_label="notes_column",
            original_bbox=bbox,
        )
        assert cor_id.startswith("cor_")

        corrections = tmp_store.get_corrections_for_page(doc_id, page=1)
        assert len(corrections) == 1
        c = corrections[0]
        assert c["correction_type"] == "relabel"
        assert c["corrected_element_type"] == "legend"
        assert c["original_element_type"] == "notes_column"


class TestAcceptDetection:
    """test_accept_detection"""

    def test_accept_creates_accept_correction(
        self,
        tmp_store: CorrectionStore,
        sample_features: dict,
    ) -> None:
        doc_id = _mock_register(tmp_store)
        bbox = (10.0, 20.0, 110.0, 40.0)

        det_id = tmp_store.save_detection(
            doc_id=doc_id,
            page=2,
            run_id="run_2",
            element_type="header",
            bbox=bbox,
            text_content="GENERAL NOTES",
            features=sample_features,
        )

        cor_id = tmp_store.accept_detection(det_id, doc_id, page=2)
        assert cor_id.startswith("cor_")

        corrections = tmp_store.get_corrections_for_page(doc_id, page=2)
        assert len(corrections) == 1
        c = corrections[0]
        assert c["correction_type"] == "accept"
        assert c["corrected_element_type"] == "header"
        assert c["original_element_type"] == "header"
        assert c["corrected_bbox"] == bbox


class TestBuildTrainingSet:
    """test_build_training_set_count"""

    def test_count_matches_non_delete_corrections(
        self,
        tmp_store: CorrectionStore,
        sample_features: dict,
    ) -> None:
        doc_id = _mock_register(tmp_store)

        # Save 5 detections
        det_ids = []
        for i in range(5):
            did = tmp_store.save_detection(
                doc_id=doc_id,
                page=0,
                run_id="run_3",
                element_type="notes_column",
                bbox=(i * 10, 0, i * 10 + 50, 20),
                text_content=f"note {i}",
                features=sample_features,
            )
            det_ids.append(did)

        # 2 relabels + 1 accept = 3 non-delete corrections
        tmp_store.save_correction(
            doc_id=doc_id,
            page=0,
            correction_type="relabel",
            corrected_label="header",
            corrected_bbox=(0, 0, 50, 20),
            detection_id=det_ids[0],
            original_label="notes_column",
            original_bbox=(0, 0, 50, 20),
        )
        tmp_store.save_correction(
            doc_id=doc_id,
            page=0,
            correction_type="relabel",
            corrected_label="legend",
            corrected_bbox=(10, 0, 60, 20),
            detection_id=det_ids[1],
            original_label="notes_column",
            original_bbox=(10, 0, 60, 20),
        )
        tmp_store.accept_detection(det_ids[2], doc_id, page=0)

        count = tmp_store.build_training_set()
        assert count == 3

        # Verify splits are assigned
        rows = tmp_store._conn.execute("SELECT split FROM training_examples").fetchall()
        splits = [r["split"] for r in rows]
        assert all(s in ("train", "val", "test") for s in splits)


class TestExportJsonl:
    """test_export_jsonl"""

    def test_writes_valid_json_lines(
        self,
        tmp_store: CorrectionStore,
        sample_features: dict,
        tmp_path: Path,
    ) -> None:
        doc_id = _mock_register(tmp_store)

        det_id = tmp_store.save_detection(
            doc_id=doc_id,
            page=0,
            run_id="run_4",
            element_type="header",
            bbox=(0, 0, 100, 20),
            text_content="PLAN VIEW",
            features=sample_features,
        )
        tmp_store.accept_detection(det_id, doc_id, page=0)
        tmp_store.build_training_set()

        out = tmp_path / "train.jsonl"
        count = tmp_store.export_training_jsonl(out)
        assert count == 1

        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

        obj = json.loads(lines[0])
        assert "example_id" in obj
        assert obj["label"] == "header"
        assert "features" in obj
        assert obj["split"] in ("train", "val", "test")


# ── Polygon persistence tests ─────────────────────────────────────────


class TestPolygonPersistence:
    """Tests for polygon_json storage and retrieval."""

    def test_save_with_polygon_round_trips(
        self, tmp_store: CorrectionStore, sample_features: dict
    ) -> None:
        doc_id = _mock_register(tmp_store)
        polygon = [(10.0, 20.0), (110.0, 20.0), (110.0, 80.0), (10.0, 80.0)]

        det_id = tmp_store.save_detection(
            doc_id=doc_id,
            page=0,
            run_id="run_poly",
            element_type="notes_column",
            bbox=(10, 20, 110, 80),
            text_content="hello",
            features=sample_features,
            polygon=polygon,
        )

        dets = tmp_store.get_detections_for_page(doc_id, page=0)
        assert len(dets) == 1
        assert dets[0]["polygon"] == polygon

    def test_save_without_polygon_returns_none(
        self, tmp_store: CorrectionStore, sample_features: dict
    ) -> None:
        doc_id = _mock_register(tmp_store)
        tmp_store.save_detection(
            doc_id=doc_id,
            page=0,
            run_id="run_nopoly",
            element_type="header",
            bbox=(0, 0, 100, 50),
            text_content="x",
            features=sample_features,
        )
        dets = tmp_store.get_detections_for_page(doc_id, page=0)
        assert dets[0]["polygon"] is None

    def test_update_detection_polygon(
        self, tmp_store: CorrectionStore, sample_features: dict
    ) -> None:
        doc_id = _mock_register(tmp_store)
        det_id = tmp_store.save_detection(
            doc_id=doc_id,
            page=0,
            run_id="run_up",
            element_type="legend",
            bbox=(0, 0, 50, 50),
            text_content="leg",
            features=sample_features,
        )
        new_poly = [(5.0, 5.0), (45.0, 5.0), (45.0, 45.0), (5.0, 45.0)]
        tmp_store.update_detection_polygon(
            det_id, polygon=new_poly, bbox=(5, 5, 45, 45)
        )
        dets = tmp_store.get_detections_for_page(doc_id, page=0)
        assert dets[0]["polygon"] == new_poly
        assert dets[0]["bbox"] == (5.0, 5.0, 45.0, 45.0)


class TestTrainingRuns:
    """Tests for the training_runs table and related methods."""

    def test_save_and_retrieve_training_run(self, tmp_store: CorrectionStore) -> None:
        metrics = {
            "accuracy": 0.85,
            "f1_macro": 0.82,
            "f1_weighted": 0.84,
            "n_train": 100,
            "n_val": 20,
            "labels": ["header", "notes_column"],
            "per_class": {
                "header": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 10},
                "notes_column": {
                    "precision": 0.8,
                    "recall": 0.9,
                    "f1": 0.85,
                    "support": 10,
                },
            },
        }
        run_id = tmp_store.save_training_run(
            metrics, model_path="model.pkl", notes="test run"
        )
        assert run_id.startswith("run_")

        history = tmp_store.get_training_history()
        assert len(history) == 1
        assert history[0]["run_id"] == run_id
        assert history[0]["accuracy"] == 0.85
        assert history[0]["f1_macro"] == 0.82
        assert history[0]["f1_weighted"] == 0.84
        assert history[0]["n_train"] == 100
        assert history[0]["n_val"] == 20
        assert history[0]["labels"] == ["header", "notes_column"]
        assert "header" in history[0]["per_class"]
        assert history[0]["model_path"] == "model.pkl"
        assert history[0]["notes"] == "test run"

    def test_multiple_runs_ordered_newest_first(
        self, tmp_store: CorrectionStore
    ) -> None:
        m1 = {"accuracy": 0.7, "n_train": 10, "n_val": 5, "labels": [], "per_class": {}}
        m2 = {
            "accuracy": 0.9,
            "n_train": 50,
            "n_val": 10,
            "labels": [],
            "per_class": {},
        }
        tmp_store.save_training_run(m1, notes="first")
        tmp_store.save_training_run(m2, notes="second")

        history = tmp_store.get_training_history()
        assert len(history) == 2
        assert history[0]["notes"] == "second"
        assert history[1]["notes"] == "first"

    def test_empty_history(self, tmp_store: CorrectionStore) -> None:
        assert tmp_store.get_training_history() == []
