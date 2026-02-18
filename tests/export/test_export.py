"""Tests for the export module."""

import csv
import json
from pathlib import Path

import pytest

from plancheck.export import (
    _bbox_str,
    export_abbreviations_csv,
    export_blocks_csv,
    export_from_manifest,
    export_legends_csv,
    export_notes_csv,
    export_page_results,
    export_page_summary_csv,
    export_revisions_csv,
    export_standard_details_csv,
)

# ── Helpers ────────────────────────────────────────────────────────────


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── _bbox_str ──────────────────────────────────────────────────────────


def test_bbox_str_none():
    assert _bbox_str(None) == ""


def test_bbox_str_empty_list():
    assert _bbox_str([]) == ""


def test_bbox_str_normal():
    assert _bbox_str([10, 20, 30.5, 40.123]) == "(10.0, 20.0, 30.5, 40.1)"


# ── export_page_summary_csv ───────────────────────────────────────────


def test_page_summary_csv_creates_file(tmp_path):
    out = tmp_path / "summary.csv"
    page_result = {
        "page": 1,
        "page_width": 612,
        "page_height": 792,
        "skew_degrees": 0.5,
        "counts": {"boxes": 100, "lines": 30, "blocks": 5},
        "stages": {"tocr": {"status": "ok"}, "vocr": {"status": "skipped"}},
    }
    export_page_summary_csv(page_result, out)
    assert out.exists()
    rows = _read_csv(out)
    assert len(rows) == 1
    assert rows[0]["page"] == "1"
    assert rows[0]["boxes"] == "100"
    assert rows[0]["tocr_status"] == "ok"


def test_page_summary_csv_appends(tmp_path):
    out = tmp_path / "summary.csv"
    p1 = {"page": 0, "counts": {}, "stages": {}}
    p2 = {"page": 1, "counts": {}, "stages": {}}
    export_page_summary_csv(p1, out)
    export_page_summary_csv(p2, out)
    rows = _read_csv(out)
    assert len(rows) == 2
    assert rows[0]["page"] == "0"
    assert rows[1]["page"] == "1"


def test_page_summary_csv_missing_keys(tmp_path):
    """Even with empty result dict the CSV should still be written."""
    out = tmp_path / "summary.csv"
    export_page_summary_csv({}, out)
    rows = _read_csv(out)
    assert len(rows) == 1
    assert rows[0]["page"] == "0"  # default


# ── export_notes_csv ──────────────────────────────────────────────────


def test_notes_csv_from_json(tmp_path):
    cols_data = [
        {
            "header_text": "GENERAL NOTES",
            "base_header_text": "GENERAL NOTES",
            "is_continuation": False,
            "column_group_id": "col_0",
            "continues_from": "",
            "notes_count": 8,
            "bbox": [10, 20, 300, 400],
        }
    ]
    cols_json = tmp_path / "columns.json"
    cols_json.write_text(json.dumps(cols_data))
    blocks_json = tmp_path / "blocks.json"
    blocks_json.write_text("[]")

    out = tmp_path / "notes.csv"
    export_notes_csv(cols_json, blocks_json, out, page_num=2)
    rows = _read_csv(out)
    assert len(rows) == 1
    assert rows[0]["header_text"] == "GENERAL NOTES"
    assert rows[0]["page"] == "2"
    assert rows[0]["notes_count"] == "8"


def test_notes_csv_empty(tmp_path):
    cols_json = tmp_path / "columns.json"
    cols_json.write_text("[]")
    blocks_json = tmp_path / "blocks.json"
    blocks_json.write_text("[]")
    out = tmp_path / "notes.csv"
    export_notes_csv(cols_json, blocks_json, out, page_num=0)
    # File should not have content rows (no header written for empty)
    assert not out.exists() or out.stat().st_size == 0


def test_notes_csv_missing_file(tmp_path):
    """Non-existent JSON path should produce empty output gracefully."""
    out = tmp_path / "notes.csv"
    export_notes_csv(tmp_path / "no.json", tmp_path / "nob.json", out)
    # Should not crash
    assert not out.exists() or out.stat().st_size == 0


# ── export_abbreviations_csv ─────────────────────────────────────────


def test_abbreviations_csv(tmp_path):
    data = [
        {
            "header_text": "ABBREVIATIONS",
            "entries": [
                {"code": "A/C", "meaning": "Air Conditioning"},
                {"code": "CLG", "meaning": "Ceiling"},
            ],
        }
    ]
    json_path = tmp_path / "abbrev.json"
    json_path.write_text(json.dumps(data))
    out = tmp_path / "abbreviations.csv"
    export_abbreviations_csv(json_path, out, page_num=1)
    rows = _read_csv(out)
    assert len(rows) == 2
    assert rows[0]["code"] == "A/C"
    assert rows[1]["meaning"] == "Ceiling"


def test_abbreviations_csv_missing_file(tmp_path):
    out = tmp_path / "abbreviations.csv"
    export_abbreviations_csv(tmp_path / "nope.json", out)
    assert not out.exists() or out.stat().st_size == 0


# ── export_legends_csv ───────────────────────────────────────────────


def test_legends_csv(tmp_path):
    data = [
        {
            "header_text": "LEGEND",
            "entries": [
                {"description": "Fire Extinguisher", "symbol_bbox": [1, 2, 3, 4]}
            ],
        }
    ]
    json_path = tmp_path / "legends.json"
    json_path.write_text(json.dumps(data))
    out = tmp_path / "legends.csv"
    export_legends_csv(json_path, out)
    rows = _read_csv(out)
    assert len(rows) == 1
    assert rows[0]["description"] == "Fire Extinguisher"
    assert rows[0]["symbol_bbox"] == "(1.0, 2.0, 3.0, 4.0)"


# ── export_standard_details_csv ──────────────────────────────────────


def test_standard_details_csv(tmp_path):
    data = [
        {
            "header_text": "STANDARD DETAILS",
            "entries": [
                {"sheet_number": "A-101", "description": "Wall Section"},
            ],
        }
    ]
    json_path = tmp_path / "details.json"
    json_path.write_text(json.dumps(data))
    out = tmp_path / "details.csv"
    export_standard_details_csv(json_path, out, page_num=3)
    rows = _read_csv(out)
    assert len(rows) == 1
    assert rows[0]["sheet_number"] == "A-101"
    assert rows[0]["page"] == "3"


# ── export_revisions_csv ────────────────────────────────────────────


def test_revisions_csv(tmp_path):
    data = [
        {
            "header_text": "REVISIONS",
            "entries": [
                {"number": "1", "description": "Initial Issue", "date": "2024-01-15"},
            ],
        }
    ]
    json_path = tmp_path / "revisions.json"
    json_path.write_text(json.dumps(data))
    out = tmp_path / "revisions.csv"
    export_revisions_csv(json_path, out)
    rows = _read_csv(out)
    assert len(rows) == 1
    assert rows[0]["number"] == "1"
    assert rows[0]["date"] == "2024-01-15"


# ── export_blocks_csv ───────────────────────────────────────────────


def test_blocks_csv(tmp_path):
    data = [
        {
            "bbox": [0, 0, 100, 200],
            "label": "header",
            "is_table": False,
            "is_notes": False,
            "lines": [{"text": "Hello"}, {"text": "World"}],
            "rows": [],
        }
    ]
    json_path = tmp_path / "blocks.json"
    json_path.write_text(json.dumps(data))
    out = tmp_path / "blocks.csv"
    export_blocks_csv(json_path, out, page_num=0)
    rows = _read_csv(out)
    assert len(rows) == 1
    assert rows[0]["label"] == "header"
    assert rows[0]["lines"] == "2"
    assert "Hello" in rows[0]["text_preview"]
    assert "World" in rows[0]["text_preview"]


def test_blocks_csv_with_rows(tmp_path):
    data = [
        {
            "bbox": [0, 0, 100, 200],
            "label": "",
            "is_table": True,
            "is_notes": False,
            "lines": [],
            "rows": [{"texts": ["cell1", "cell2"]}, {"texts": ["cell3"]}],
        }
    ]
    json_path = tmp_path / "blocks.json"
    json_path.write_text(json.dumps(data))
    out = tmp_path / "blocks.csv"
    export_blocks_csv(json_path, out)
    rows = _read_csv(out)
    assert len(rows) == 1
    assert rows[0]["rows"] == "2"
    assert "cell1" in rows[0]["text_preview"]


# ── export_page_results (integration) ───────────────────────────────


def test_export_page_results_creates_summary(tmp_path):
    """Minimal integration: at least the summary CSV is always created."""
    run_dir = tmp_path / "run_test"
    (run_dir / "exports").mkdir(parents=True)
    (run_dir / "artifacts").mkdir(parents=True)

    page_result = {
        "page": 0,
        "page_width": 612,
        "page_height": 792,
        "counts": {"boxes": 10},
        "stages": {},
        "artifacts": {},
    }
    exported = export_page_results(page_result, run_dir, "test_plan")
    assert "page_summary" in exported
    assert Path(exported["page_summary"]).exists()


def test_export_page_results_with_artifacts(tmp_path):
    """When JSON artifacts exist, CSVs are produced."""
    run_dir = tmp_path / "run_test"
    arts_dir = run_dir / "artifacts"
    arts_dir.mkdir(parents=True)
    (run_dir / "exports").mkdir(parents=True)

    pdf_stem = "test_plan"
    page_num = 0

    # Create abbreviations artifact
    abbrev_data = [{"header_text": "ABBR", "entries": [{"code": "X", "meaning": "Y"}]}]
    abbrev_path = arts_dir / f"{pdf_stem}_page_{page_num}_abbreviations.json"
    abbrev_path.write_text(json.dumps(abbrev_data))

    page_result = {
        "page": page_num,
        "counts": {},
        "stages": {},
        "artifacts": {
            "boxes_json": str(arts_dir / "something.json"),
            "abbreviations_json": str(abbrev_path),
        },
    }
    exported = export_page_results(page_result, run_dir, pdf_stem)
    assert "abbreviations" in exported
    rows = _read_csv(Path(exported["abbreviations"]))
    assert len(rows) == 1
    assert rows[0]["code"] == "X"


# ── export_from_manifest ────────────────────────────────────────────


def test_export_from_manifest(tmp_path):
    run_dir = tmp_path / "run_test"
    (run_dir / "exports").mkdir(parents=True)
    manifest = {
        "pdf_name": "my plan.pdf",
        "pages": [
            {
                "page": 0,
                "page_width": 612,
                "page_height": 792,
                "counts": {"boxes": 5},
                "stages": {},
                "artifacts": {},
            }
        ],
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    exported = export_from_manifest(manifest_path)
    assert "page_summary" in exported
    summary_rows = _read_csv(Path(exported["page_summary"]))
    assert len(summary_rows) == 1
    assert summary_rows[0]["boxes"] == "5"
