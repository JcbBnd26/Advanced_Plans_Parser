"""Smoke tests for plancheck.export.reconcile_overlay – debug overlay rendering."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from PIL import Image

from plancheck.models import GlyphBox
from plancheck.reconcile.reconcile import MatchRecord, ReconcileResult

# ── helpers ────────────────────────────────────────────────────────────


def _box(x0, y0, x1, y1, text="", origin="text"):
    return GlyphBox(page=0, x0=x0, y0=y0, x1=x1, y1=y1, text=text, origin=origin)


def _empty_result() -> ReconcileResult:
    return ReconcileResult(
        added_tokens=[],
        all_ocr_tokens=[],
        matches=[],
        stats={},
    )


def _populated_result() -> ReconcileResult:
    """Result with added tokens, OCR tokens, matches, and injection log."""
    ocr1 = _box(10, 10, 30, 20, text="50%", origin="ocr_full")
    ocr2 = _box(50, 10, 70, 20, text="NOTES", origin="ocr_full")
    pdf1 = _box(10, 10, 25, 20, text="50", origin="text")
    added = _box(25, 10, 30, 20, text="%", origin="ocr")

    return ReconcileResult(
        added_tokens=[added],
        all_ocr_tokens=[ocr1, ocr2],
        matches=[
            MatchRecord(
                ocr_box=ocr1, pdf_box=pdf1, match_type="iou", ocr_confidence=0.95
            ),
            MatchRecord(
                ocr_box=ocr2, pdf_box=None, match_type="unmatched", ocr_confidence=0.8
            ),
        ],
        stats={
            "injection_log": [
                {
                    "anchors": [{"bbox": [10, 10, 25, 20], "text": "50"}],
                    "candidates": [
                        {"status": "accepted", "symbol": "%", "bbox": [25, 10, 30, 20]},
                        {
                            "status": "rejected",
                            "symbol": "/",
                            "bbox": [40, 10, 45, 20],
                            "reason": "no_context",
                        },
                    ],
                }
            ]
        },
    )


# ── draw_reconcile_debug tests ────────────────────────────────────────


class TestDrawReconcileDebug:
    """Smoke tests for draw_reconcile_debug."""

    def test_empty_result_no_background(self, tmp_path):
        from plancheck.export.reconcile_overlay import draw_reconcile_debug

        out = tmp_path / "debug.png"
        draw_reconcile_debug(
            result=_empty_result(),
            page_width=200.0,
            page_height=100.0,
            out_path=out,
            scale=1.0,
        )
        assert out.exists()
        img = Image.open(out)
        assert img.size == (200, 100)
        assert img.mode == "RGB"

    def test_empty_result_with_background(self, tmp_path):
        from plancheck.export.reconcile_overlay import draw_reconcile_debug

        bg = Image.new("RGB", (200, 100), (240, 240, 240))
        out = tmp_path / "debug_bg.png"
        draw_reconcile_debug(
            result=_empty_result(),
            page_width=200.0,
            page_height=100.0,
            out_path=out,
            scale=1.0,
            background=bg,
        )
        assert out.exists()

    def test_populated_result(self, tmp_path):
        """Full result with tokens, matches, and injection log renders without error."""
        from plancheck.export.reconcile_overlay import draw_reconcile_debug

        out = tmp_path / "debug_full.png"
        draw_reconcile_debug(
            result=_populated_result(),
            page_width=200.0,
            page_height=100.0,
            out_path=out,
            scale=2.0,
        )
        assert out.exists()
        img = Image.open(out)
        # With scale=2.0, canvas should be 400x200
        assert img.size == (400, 200)

    def test_scale_applied(self, tmp_path):
        from plancheck.export.reconcile_overlay import draw_reconcile_debug

        out = tmp_path / "debug_scaled.png"
        draw_reconcile_debug(
            result=_empty_result(),
            page_width=100.0,
            page_height=50.0,
            out_path=out,
            scale=3.0,
        )
        img = Image.open(out)
        assert img.size == (300, 150)

    def test_creates_parent_directory(self, tmp_path):
        from plancheck.export.reconcile_overlay import draw_reconcile_debug

        out = tmp_path / "subdir" / "nested" / "debug.png"
        draw_reconcile_debug(
            result=_empty_result(),
            page_width=100.0,
            page_height=50.0,
            out_path=out,
        )
        assert out.exists()

    def test_background_resized_if_mismatch(self, tmp_path):
        """Background image of different size should be resized to canvas."""
        from plancheck.export.reconcile_overlay import draw_reconcile_debug

        bg = Image.new("RGB", (50, 25))  # smaller than canvas
        out = tmp_path / "debug_resize.png"
        draw_reconcile_debug(
            result=_empty_result(),
            page_width=200.0,
            page_height=100.0,
            out_path=out,
            scale=1.0,
            background=bg,
        )
        img = Image.open(out)
        assert img.size == (200, 100)


# ── draw_symbol_overlay tests ─────────────────────────────────────────


class TestDrawSymbolOverlay:
    """Smoke tests for draw_symbol_overlay."""

    def test_empty_result_no_background(self, tmp_path):
        from plancheck.export.reconcile_overlay import draw_symbol_overlay

        out = tmp_path / "symbols.png"
        draw_symbol_overlay(
            result=_empty_result(),
            page_width=200.0,
            page_height=100.0,
            out_path=out,
        )
        assert out.exists()
        img = Image.open(out)
        assert img.mode == "RGB"

    def test_with_symbol_tokens(self, tmp_path):
        """Result containing symbol-bearing OCR tokens renders green boxes."""
        from plancheck.export.reconcile_overlay import draw_symbol_overlay

        out = tmp_path / "symbols_pop.png"
        draw_symbol_overlay(
            result=_populated_result(),
            page_width=200.0,
            page_height=100.0,
            out_path=out,
            scale=2.0,
            show_labels=True,
        )
        assert out.exists()
        img = Image.open(out)
        assert img.size == (400, 200)

    def test_no_labels(self, tmp_path):
        from plancheck.export.reconcile_overlay import draw_symbol_overlay

        out = tmp_path / "symbols_nolabel.png"
        draw_symbol_overlay(
            result=_populated_result(),
            page_width=200.0,
            page_height=100.0,
            out_path=out,
            show_labels=False,
        )
        assert out.exists()

    def test_custom_allowed_symbols(self, tmp_path):
        from plancheck.export.reconcile_overlay import draw_symbol_overlay

        out = tmp_path / "symbols_custom.png"
        draw_symbol_overlay(
            result=_populated_result(),
            page_width=200.0,
            page_height=100.0,
            out_path=out,
            allowed_symbols="°±",
        )
        assert out.exists()

    def test_with_background(self, tmp_path):
        from plancheck.export.reconcile_overlay import draw_symbol_overlay

        bg = Image.new("RGB", (200, 100), (200, 200, 200))
        out = tmp_path / "symbols_bg.png"
        draw_symbol_overlay(
            result=_populated_result(),
            page_width=200.0,
            page_height=100.0,
            out_path=out,
            background=bg,
        )
        assert out.exists()

    def test_creates_parent_directory(self, tmp_path):
        from plancheck.export.reconcile_overlay import draw_symbol_overlay

        out = tmp_path / "deep" / "path" / "symbols.png"
        draw_symbol_overlay(
            result=_empty_result(),
            page_width=100.0,
            page_height=50.0,
            out_path=out,
        )
        assert out.exists()
