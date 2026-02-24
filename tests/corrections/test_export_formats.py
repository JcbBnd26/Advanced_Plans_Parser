"""Tests for plancheck.corrections.export_formats."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from plancheck.corrections.export_formats import (
    _get_corrected_detections,
    _get_page_dimensions,
    export_coco,
    export_voc,
)
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


def _make_features() -> dict:
    return {
        "font_size_pt": 8.0,
        "font_size_max_pt": 10.0,
        "font_size_min_pt": 6.0,
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
        "zone": "header",
        "neighbor_count": 2,
        "unique_word_ratio": 0.8,
        "uppercase_word_frac": 0.4,
        "avg_word_length": 5.0,
        "kw_notes_pattern": 0,
        "kw_header_pattern": 0,
        "kw_legend_pattern": 0,
        "kw_abbreviation_pattern": 0,
        "kw_revision_pattern": 0,
        "kw_title_block_pattern": 0,
        "kw_detail_pattern": 0,
    }


def _seed_corrected_data(store: CorrectionStore) -> tuple[str, str]:
    """Insert a detection + correction and return (doc_id, det_id)."""
    doc_id = _register_doc(store)
    det_id = store.save_detection(
        doc_id=doc_id,
        page=0,
        run_id="run_test",
        element_type="header",
        bbox=(100.0, 200.0, 400.0, 300.0),
        text_content="Some header text",
        features=_make_features(),
    )
    store.save_correction(
        doc_id=doc_id,
        page=0,
        correction_type="relabel",
        corrected_label="notes_column",
        corrected_bbox=(100.0, 200.0, 400.0, 300.0),
        detection_id=det_id,
        original_label="header",
        original_bbox=(100.0, 200.0, 400.0, 300.0),
        session_id="s1",
    )
    return doc_id, det_id


# ---------------------------------------------------------------------------
# _get_corrected_detections
# ---------------------------------------------------------------------------


class TestGetCorrectedDetections:
    def test_returns_corrected_detections(self, tmp_path: Path) -> None:
        store = CorrectionStore(db_path=tmp_path / "test.db")
        _seed_corrected_data(store)

        results = _get_corrected_detections(store)
        assert len(results) == 1
        assert results[0]["label"] == "notes_column"

    def test_excludes_deleted(self, tmp_path: Path) -> None:
        store = CorrectionStore(db_path=tmp_path / "test.db")
        doc_id = _register_doc(store)
        det_id = store.save_detection(
            doc_id=doc_id,
            page=0,
            run_id="run_test",
            element_type="header",
            bbox=(100, 200, 400, 300),
            text_content="to delete",
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

        results = _get_corrected_detections(store)
        assert len(results) == 0

    def test_empty_database(self, tmp_path: Path) -> None:
        store = CorrectionStore(db_path=tmp_path / "test.db")
        results = _get_corrected_detections(store)
        assert results == []


# ---------------------------------------------------------------------------
# export_coco
# ---------------------------------------------------------------------------


class TestExportCoco:
    def test_creates_annotations_json(self, tmp_path: Path) -> None:
        store = CorrectionStore(db_path=tmp_path / "test.db")
        _seed_corrected_data(store)

        out_dir = tmp_path / "coco_out"
        result_path = export_coco(store, out_dir)

        assert result_path.exists()
        assert result_path.name == "annotations.json"

        with open(result_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "images" in data
        assert "annotations" in data
        assert "categories" in data
        assert len(data["images"]) == 1
        assert len(data["annotations"]) == 1

    def test_bbox_format_xywh(self, tmp_path: Path) -> None:
        store = CorrectionStore(db_path=tmp_path / "test.db")
        _seed_corrected_data(store)

        out_path = export_coco(store, tmp_path / "coco")
        with open(out_path, encoding="utf-8") as f:
            data = json.load(f)

        ann = data["annotations"][0]
        # bbox should be [x, y, width, height]
        assert len(ann["bbox"]) == 4
        # For our test data: x0=100, y0=200, x1=400, y1=300
        # width=300, height=100
        assert ann["bbox"][0] == 100.0
        assert ann["bbox"][1] == 200.0
        assert ann["bbox"][2] == 300.0  # width
        assert ann["bbox"][3] == 100.0  # height

    def test_category_id_mapped(self, tmp_path: Path) -> None:
        store = CorrectionStore(db_path=tmp_path / "test.db")
        _seed_corrected_data(store)

        out_path = export_coco(store, tmp_path / "coco")
        with open(out_path, encoding="utf-8") as f:
            data = json.load(f)

        ann = data["annotations"][0]
        # notes_column → category_id 1
        assert ann["category_id"] == 1

    def test_empty_db_creates_empty_file(self, tmp_path: Path) -> None:
        store = CorrectionStore(db_path=tmp_path / "test.db")
        out_path = export_coco(store, tmp_path / "coco")

        with open(out_path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["annotations"] == []
        assert data["images"] == []


# ---------------------------------------------------------------------------
# export_voc
# ---------------------------------------------------------------------------


class TestExportVoc:
    def test_creates_xml_files(self, tmp_path: Path) -> None:
        store = CorrectionStore(db_path=tmp_path / "test.db")
        _seed_corrected_data(store)

        out_dir = tmp_path / "voc_out"
        result_dir = export_voc(store, out_dir)

        assert result_dir.exists()
        xml_files = list(result_dir.glob("*.xml"))
        assert len(xml_files) == 1

    def test_xml_content(self, tmp_path: Path) -> None:
        store = CorrectionStore(db_path=tmp_path / "test.db")
        _seed_corrected_data(store)

        out_dir = tmp_path / "voc_out"
        export_voc(store, out_dir)

        xml_files = list(out_dir.glob("*.xml"))
        tree = ET.parse(xml_files[0])
        root = tree.getroot()

        assert root.tag == "annotation"
        objects = root.findall("object")
        assert len(objects) == 1
        assert objects[0].find("name").text == "notes_column"

        bndbox = objects[0].find("bndbox")
        assert bndbox.find("xmin").text == "100"
        assert bndbox.find("ymin").text == "200"
        assert bndbox.find("xmax").text == "400"
        assert bndbox.find("ymax").text == "300"

    def test_empty_db_creates_no_files(self, tmp_path: Path) -> None:
        store = CorrectionStore(db_path=tmp_path / "test.db")
        out_dir = tmp_path / "voc_out"
        result_dir = export_voc(store, out_dir)

        assert result_dir.exists()
        xml_files = list(result_dir.glob("*.xml"))
        assert len(xml_files) == 0

    def test_multiple_pages_multiple_files(self, tmp_path: Path) -> None:
        store = CorrectionStore(db_path=tmp_path / "test.db")
        doc_id = _register_doc(store)

        for page in [0, 1]:
            det_id = store.save_detection(
                doc_id=doc_id,
                page=page,
                run_id="run_test",
                element_type="header",
                bbox=(100 + page * 100, 200, 400, 300),
                text_content=f"Page {page}",
                features=_make_features(),
            )
            store.save_correction(
                doc_id=doc_id,
                page=page,
                correction_type="accept",
                corrected_label="header",
                corrected_bbox=(100 + page * 100, 200, 400, 300),
                detection_id=det_id,
                session_id="s1",
            )

        out_dir = tmp_path / "voc_out"
        export_voc(store, out_dir)

        xml_files = list(out_dir.glob("*.xml"))
        assert len(xml_files) == 2

    def test_voc_uses_detection_dimensions(self, tmp_path: Path) -> None:
        """VOC size element should reflect the max detection bbox, not hardcoded values."""
        store = CorrectionStore(db_path=tmp_path / "test.db")
        doc_id = _register_doc(store)

        # Detection with a large bbox to set page dimensions
        det_id = store.save_detection(
            doc_id=doc_id,
            page=0,
            run_id="run_dim",
            element_type="header",
            bbox=(0.0, 0.0, 3000.0, 2000.0),
            text_content="Big page",
            features=_make_features(),
        )
        store.save_correction(
            doc_id=doc_id,
            page=0,
            correction_type="accept",
            corrected_label="header",
            corrected_bbox=(0.0, 0.0, 3000.0, 2000.0),
            detection_id=det_id,
        )

        out_dir = tmp_path / "voc_dim"
        export_voc(store, out_dir)

        xml_files = list(out_dir.glob("*.xml"))
        tree = ET.parse(xml_files[0])
        root = tree.getroot()
        size = root.find("size")
        assert size.find("width").text == "3000"
        assert size.find("height").text == "2000"
