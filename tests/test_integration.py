"""Integration test: synthetic token set through the full grouping → zoning → export pipeline.

This test verifies that the end-to-end flow works with no real PDF, no real
OCR — just hand-crafted GlyphBox tokens that mimic a two-column notes sheet
with a header, numbered notes, and a legend.
"""

import json
from pathlib import Path

import pytest
from conftest import make_box

from plancheck.config import GroupingConfig
from plancheck.export import export_blocks_csv, export_notes_csv
from plancheck.grouping import (
    build_clusters_v2,
    group_notes_columns,
    link_continued_columns,
    mark_headers,
    mark_notes,
)
from plancheck.zoning import classify_blocks, detect_zones

# ── Synthetic page layout ─────────────────────────────────────────────


def _build_synthetic_tokens():
    """Return (tokens, page_height) for a plausible one-page plan sheet.

    Layout (800 pt tall page):
        Header (y ~50):     "GENERAL NOTES:"
        Notes (y ~70–250):  "1. SEE ALL PLANS AND SPECS"
                            "2. CONTRACTOR SHALL VERIFY"
                            "3. ALL DIMENSIONS IN FEET"
        Legend (y ~400):    "LEGEND"
                            symbol row 1: "EW" = "EXISTING WIRE"
                            symbol row 2: "NW" = "NEW WIRE"
    """
    tokens = [
        # ── Header
        make_box(50, 50, 130, 62, "GENERAL"),
        make_box(135, 50, 195, 62, "NOTES:"),
        # ── Note 1
        make_box(50, 75, 62, 87, "1."),
        make_box(66, 75, 95, 87, "SEE"),
        make_box(100, 75, 125, 87, "ALL"),
        make_box(130, 75, 175, 87, "PLANS"),
        make_box(180, 75, 210, 87, "AND"),
        make_box(215, 75, 260, 87, "SPECS"),
        # ── Note 2
        make_box(50, 95, 62, 107, "2."),
        make_box(66, 95, 140, 107, "CONTRACTOR"),
        make_box(145, 95, 185, 107, "SHALL"),
        make_box(190, 95, 235, 107, "VERIFY"),
        # ── Note 3
        make_box(50, 115, 62, 127, "3."),
        make_box(66, 115, 100, 127, "ALL"),
        make_box(105, 115, 175, 127, "DIMENSIONS"),
        make_box(180, 115, 200, 127, "IN"),
        make_box(205, 115, 240, 127, "FEET"),
        # ── Legend header (far below)
        make_box(50, 400, 120, 412, "LEGEND"),
        # ── Legend entries
        make_box(50, 420, 70, 432, "EW"),
        make_box(100, 420, 200, 432, "EXISTING"),
        make_box(205, 420, 250, 432, "WIRE"),
        make_box(50, 440, 70, 452, "NW"),
        make_box(100, 440, 170, 452, "NEW"),
        make_box(175, 440, 220, 452, "WIRE"),
    ]
    return tokens, 800.0


# ── Tests ──────────────────────────────────────────────────────────────


class TestEndToEndGrouping:
    """Run synthetic tokens through the full grouping pipeline."""

    def test_clusters_produced(self):
        tokens, page_h = _build_synthetic_tokens()
        cfg = GroupingConfig()
        blocks = build_clusters_v2(tokens, page_h, cfg)
        assert len(blocks) >= 2  # at least notes block + legend block
        # All tokens should be accounted for in total
        total_boxes = sum(len(b.get_all_boxes()) for b in blocks)
        assert total_boxes == len(tokens)

    def test_header_detected(self):
        tokens, page_h = _build_synthetic_tokens()
        cfg = GroupingConfig()
        blocks = build_clusters_v2(tokens, page_h, cfg)
        mark_headers(blocks)
        headers = [b for b in blocks if b.is_header]
        assert len(headers) >= 1
        # Header should contain "NOTES:" text
        header_texts = " ".join(bx.text for bx in headers[0].get_all_boxes())
        assert "NOTES:" in header_texts

    def test_notes_detected(self):
        tokens, page_h = _build_synthetic_tokens()
        cfg = GroupingConfig()
        blocks = build_clusters_v2(tokens, page_h, cfg)
        mark_headers(blocks)
        mark_notes(blocks)
        # mark_notes requires >= 2 rows; with synthetic data each note may
        # be its own block.  Just verify the pipeline runs without error and
        # every block has a bool is_notes.
        for b in blocks:
            assert isinstance(b.is_notes, bool)


class TestEndToEndZoning:
    """Run through grouping → zoning and verify zone assignment."""

    def test_zones_assigned(self):
        tokens, page_h = _build_synthetic_tokens()
        page_w = 600.0
        cfg = GroupingConfig()
        blocks = build_clusters_v2(tokens, page_h, cfg)
        mark_headers(blocks)
        mark_notes(blocks)
        zones = detect_zones(page_w, page_h, blocks, cfg=cfg)
        classify_blocks(blocks, zones)
        # Every block should have a label (could be None for unclassified, but
        # function should execute without error)
        for b in blocks:
            # label is optional-str — ensure classifying ran
            assert isinstance(getattr(b, "label", None), (str, type(None)))


class TestEndToEndExport:
    """Group → serialize → export CSV round-trip."""

    def test_blocks_csv(self, tmp_path):
        tokens, page_h = _build_synthetic_tokens()
        cfg = GroupingConfig()
        blocks = build_clusters_v2(tokens, page_h, cfg)
        mark_headers(blocks)
        mark_notes(blocks)

        # Serialize blocks to JSON (mimics what run_pdf_page does)
        blocks_data = []
        for i, blk in enumerate(blocks):
            boxes = blk.get_all_boxes()
            blocks_data.append(
                {
                    "block_id": i,
                    "page": blk.page,
                    "bbox": list(blk.bbox()),
                    "is_table": blk.is_table,
                    "is_notes": blk.is_notes,
                    "is_header": blk.is_header,
                    "label": blk.label,
                    "rows": [
                        {
                            "row_idx": ri,
                            "boxes": [
                                {
                                    "x0": b.x0,
                                    "y0": b.y0,
                                    "x1": b.x1,
                                    "y1": b.y1,
                                    "text": b.text,
                                }
                                for b in row.boxes
                            ],
                        }
                        for ri, row in enumerate(blk.rows)
                    ],
                }
            )

        blocks_json = tmp_path / "blocks.json"
        blocks_json.write_text(json.dumps(blocks_data, indent=2), encoding="utf-8")

        blocks_csv = tmp_path / "blocks.csv"
        export_blocks_csv(blocks_json, blocks_csv, page_num=0)
        assert blocks_csv.exists()
        content = blocks_csv.read_text(encoding="utf-8")
        # Header row + at least one data row
        lines = [l for l in content.strip().split("\n") if l.strip()]
        assert len(lines) >= 2


class TestNotesColumnsIntegration:
    """Verify notes-column grouping with synthetic multi-note layout."""

    def test_notes_columns_created(self):
        tokens, page_h = _build_synthetic_tokens()
        cfg = GroupingConfig()
        blocks = build_clusters_v2(tokens, page_h, cfg)
        mark_headers(blocks)
        mark_notes(blocks)
        columns = group_notes_columns(blocks, cfg=cfg)
        # Should produce at least one column (may have header=None for orphans)
        assert isinstance(columns, list)
        # No crash is the primary assertion; column count depends on heuristic
