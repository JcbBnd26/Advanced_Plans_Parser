"""Tests for plancheck.export.font_map and plancheck.export.sheet_recreation.

Covers:
  • strip_subset_prefix  – 6-uppercase-letter prefix removal
  • resolve_font          – PDF fontname → ReportLab font name
  • _effective_font_size  – font_size field vs bbox-height fallback
  • _resolve_color        – color_map lookup / plain-black default
  • draw_sheet_recreation – canvas interaction (state isolation, width fitting,
                            block boundaries, labels, footer, layers)
  • recreate_sheet        – reads artifact JSON, writes valid PDF, metadata,
                            block pass-through, layers, watermark
  • Coordinate transform  – rl_y = page_height - y1
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, call, patch

import pytest

from plancheck.export.font_map import resolve_font, strip_subset_prefix
from plancheck.export.sheet_recreation import (
    ORIGIN_COLORS,
    _block_color,
    _block_tag,
    _effective_font_size,
    _make_page_label,
    _resolve_color,
    _token_counts,
    draw_sheet_recreation,
    recreate_sheet,
)
from plancheck.models import BlockCluster, GlyphBox, RowBand

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _box(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    text: str = "WORD",
    page: int = 0,
    fontname: str = "ArialMT",
    font_size: float = 10.0,
    origin: str = "text",
) -> GlyphBox:
    return GlyphBox(
        page=page,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        text=text,
        fontname=fontname,
        font_size=font_size,
        origin=origin,
    )


# ---------------------------------------------------------------------------
# strip_subset_prefix
# ---------------------------------------------------------------------------


class TestStripSubsetPrefix:
    def test_removes_six_uppercase_plus(self):
        assert strip_subset_prefix("BCDFEE+ArialMT") == "ArialMT"

    def test_removes_all_uppercase_prefix(self):
        assert strip_subset_prefix("ABCXYZ+TimesNewRomanPSMT") == "TimesNewRomanPSMT"

    def test_no_prefix_unchanged(self):
        assert strip_subset_prefix("ArialMT") == "ArialMT"

    def test_mixed_case_prefix_not_removed(self):
        # Must be exactly 6 uppercase; lowercase prefix stays
        assert strip_subset_prefix("abcDEF+Arial") == "abcDEF+Arial"

    def test_empty_string(self):
        assert strip_subset_prefix("") == ""

    def test_cad_font_unchanged(self):
        assert strip_subset_prefix("RomanT") == "RomanT"

    def test_short_prefix_not_removed(self):
        # Only 5 uppercase letters before +
        assert strip_subset_prefix("ABCDE+Arial") == "ABCDE+Arial"


# ---------------------------------------------------------------------------
# resolve_font
# ---------------------------------------------------------------------------


class TestResolveFont:
    # Helvetica family
    def test_arial_maps_to_helvetica(self):
        assert resolve_font("ArialMT") == "Helvetica"

    def test_arial_with_subset_prefix(self):
        assert resolve_font("BCDFEE+ArialMT") == "Helvetica"

    def test_helvetica_maps_to_itself(self):
        assert resolve_font("Helvetica") == "Helvetica"

    def test_calibri_maps_to_helvetica(self):
        assert resolve_font("CalibriRegular") == "Helvetica"

    # Times family
    def test_times_new_roman(self):
        assert resolve_font("TimesNewRomanPSMT") == "Times-Roman"

    def test_georgia(self):
        assert resolve_font("Georgia") == "Times-Roman"

    # Courier family
    def test_courier(self):
        assert resolve_font("CourierNewPSMT") == "Courier"

    def test_consolas(self):
        assert resolve_font("Consolas") == "Courier"

    # CAD / SHX fonts
    def test_romant_maps_to_courier(self):
        assert resolve_font("RomanT") == "Courier"

    def test_simplex_maps_to_courier(self):
        assert resolve_font("Simplex") == "Courier"

    def test_txt_maps_to_courier(self):
        assert resolve_font("txt") == "Courier"

    def test_monotxt_maps_to_courier(self):
        assert resolve_font("monotxt") == "Courier"

    # Bold / italic modifiers
    def test_arial_bold(self):
        assert resolve_font("Arial-BoldMT") == "Helvetica-Bold"

    def test_arial_italic(self):
        assert resolve_font("Arial-ItalicMT") == "Helvetica-Oblique"

    def test_arial_bold_italic(self):
        assert resolve_font("Arial-BoldItalicMT") == "Helvetica-BoldOblique"

    def test_times_bold(self):
        assert resolve_font("TimesNewRomanPS-BoldMT") == "Times-Bold"

    def test_times_italic(self):
        assert resolve_font("TimesNewRomanPS-ItalicMT") == "Times-Italic"

    # Fallback
    def test_unknown_font_falls_back_to_helvetica(self):
        assert resolve_font("XYZCustomFont123") == "Helvetica"

    def test_empty_font_falls_back_to_helvetica(self):
        assert resolve_font("") == "Helvetica"

    def test_none_like_empty_string(self):
        # fontname="" is the default for VOCR tokens
        assert resolve_font("") == "Helvetica"


# ---------------------------------------------------------------------------
# _effective_font_size
# ---------------------------------------------------------------------------


class TestEffectiveFontSize:
    def test_uses_stored_font_size_when_positive(self):
        g = _box(0, 10, 72, 22, font_size=12.0)
        assert _effective_font_size(g) == pytest.approx(12.0)

    def test_fallback_to_bbox_height_when_zero(self):
        # bbox height = y1 - y0 = 22 - 10 = 12
        g = _box(0, 10, 72, 22, font_size=0.0)
        assert _effective_font_size(g) == pytest.approx(12.0)

    def test_floor_applied_on_tiny_size(self):
        g = _box(0, 0, 72, 1, font_size=1.0)  # 1pt < 4pt floor
        assert _effective_font_size(g) >= 4.0

    def test_cap_applied_on_huge_size(self):
        g = _box(0, 0, 72, 200, font_size=200.0)
        assert _effective_font_size(g) <= 72.0


# ---------------------------------------------------------------------------
# _resolve_color
# ---------------------------------------------------------------------------


class TestResolveColor:
    def test_plain_black_when_no_color_map(self):
        assert _resolve_color("text", None) == (0.0, 0.0, 0.0)

    def test_plain_black_for_any_origin_without_map(self):
        assert _resolve_color("ocr_full", None) == (0.0, 0.0, 0.0)

    def test_color_map_applied_for_known_origin(self):
        cmap = {"text": (1.0, 0.0, 0.0), "ocr_full": (0.0, 0.0, 1.0)}
        assert _resolve_color("text", cmap) == (1.0, 0.0, 0.0)
        assert _resolve_color("ocr_full", cmap) == (0.0, 0.0, 1.0)

    def test_color_map_falls_back_for_unknown_origin(self):
        cmap = {"text": (1.0, 0.0, 0.0)}
        # "ocr_full" not in map → default black
        assert _resolve_color("ocr_full", cmap) == (0.0, 0.0, 0.0)

    def test_origin_colors_constant_has_expected_keys(self):
        for key in ("text", "ocr_full", "reconcile"):
            assert key in ORIGIN_COLORS

    def test_origin_colors_are_valid_rgb(self):
        for origin, rgb in ORIGIN_COLORS.items():
            assert len(rgb) == 3, f"{origin} color should be 3-tuple"
            assert all(0.0 <= c <= 1.0 for c in rgb), f"{origin} color out of 0-1 range"


# ---------------------------------------------------------------------------
# draw_sheet_recreation  (canvas mock)
# ---------------------------------------------------------------------------


class TestDrawSheetRecreation:
    """Verify canvas interaction without writing real files."""

    def _make_mock_canvas(self) -> MagicMock:
        c = MagicMock()
        c.stringWidth.return_value = 50.0  # sensible default for width fitting
        return c

    def test_single_token_renders_text(self):
        c = self._make_mock_canvas()
        tokens = [_box(10, 5, 82, 17, text="HELLO", font_size=12.0)]
        # page_height=100 → rl_y = 100 - 17 = 83
        draw_sheet_recreation(c, page_width=500, page_height=100, tokens=tokens)
        c.beginText.assert_called_once_with(10.0, pytest.approx(83.0))
        tx = c.beginText.return_value
        tx.textOut.assert_called_once_with("HELLO")
        c.drawText.assert_called_once_with(tx)

    def test_coordinate_transform_rl_y(self):
        """rl_y = page_height - glyph.y1"""
        c = self._make_mock_canvas()
        g = _box(0, 20, 100, 30, text="A", font_size=10.0)
        draw_sheet_recreation(c, page_width=200, page_height=200, tokens=[g])
        # rl_y = 200 - 30 = 170
        c.beginText.assert_called_once_with(0.0, pytest.approx(170.0))

    def test_empty_text_glyphs_skipped(self):
        c = self._make_mock_canvas()
        tokens = [_box(0, 0, 10, 10, text="")]
        draw_sheet_recreation(c, 500, 500, tokens=tokens)
        c.beginText.assert_not_called()

    def test_empty_token_list(self):
        c = self._make_mock_canvas()
        draw_sheet_recreation(c, 500, 500, tokens=[])
        c.beginText.assert_not_called()
        c.showPage.assert_called_once()

    def test_show_page_called_once(self):
        c = self._make_mock_canvas()
        tokens = [_box(0, 0, 50, 10, text="X"), _box(60, 0, 100, 10, text="Y")]
        draw_sheet_recreation(c, 500, 500, tokens=tokens)
        c.showPage.assert_called_once()

    def test_set_page_size_called(self):
        c = self._make_mock_canvas()
        draw_sheet_recreation(c, page_width=1728, page_height=1152, tokens=[])
        c.setPageSize.assert_called_once_with((1728.0, 1152.0))

    def test_fill_color_set_for_each_token(self):
        c = self._make_mock_canvas()
        tokens = [
            _box(0, 0, 50, 10, text="A", origin="text"),
            _box(60, 0, 100, 10, text="B", origin="ocr_full"),
        ]
        draw_sheet_recreation(c, 500, 500, tokens=tokens, color_map=ORIGIN_COLORS)
        # Both tokens should trigger setFillColor on the text object
        tx = c.beginText.return_value
        assert tx.setFillColor.call_count == 2

    def test_plain_mode_always_black(self):
        c = self._make_mock_canvas()
        tokens = [
            _box(0, 0, 50, 10, text="A", origin="ocr_full"),
        ]
        draw_sheet_recreation(c, 500, 500, tokens=tokens, color_map=None)
        tx = c.beginText.return_value
        tx.setFillColor.assert_called_with((0.0, 0.0, 0.0))

    def test_font_resolved_without_error(self):
        """CAD font should not raise – falls back to Courier."""
        c = self._make_mock_canvas()
        tokens = [_box(0, 0, 50, 10, text="T", fontname="RomanT", font_size=8.0)]
        draw_sheet_recreation(c, 500, 500, tokens=tokens)
        # setFont called with "Courier"
        set_font_calls = [str(a[0]) for a in c.setFont.call_args_list]
        assert any("Courier" in fc for fc in set_font_calls)

    def test_unknown_font_falls_back_silently(self):
        """If ReportLab raises KeyError for an unknown font, Helvetica is used."""
        c = self._make_mock_canvas()
        # Simulate ReportLab raising KeyError on first setFont, succeeding on second
        c.setFont.side_effect = [KeyError("bad font"), None]
        tokens = [_box(0, 0, 50, 10, text="X", fontname="WeirdFont", font_size=10.0)]
        # Should not raise
        draw_sheet_recreation(c, 500, 500, tokens=tokens)
        assert c.setFont.call_count == 2
        # Second call is the fallback
        assert c.setFont.call_args_list[1] == call("Helvetica", 10.0)


# ---------------------------------------------------------------------------
# recreate_sheet  (integration – writes a real PDF)
# ---------------------------------------------------------------------------


class TestRecreateSheet:
    """Writes real artifact JSON and asserts a valid multi-page PDF is produced."""

    @staticmethod
    def _make_artifact(
        tmp_path: Path,
        pdf_stem: str,
        page_num: int,
        tokens: List[GlyphBox],
        page_w: float = 500.0,
        page_h: float = 400.0,
    ) -> Path:
        """Create a minimal *_extraction.json artifact file."""
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        # Minimal serialized format expected by deserialize_page
        data = {
            "page_width": page_w,
            "page_height": page_h,
            "tokens": [t.to_dict() for t in tokens],
            "blocks": [],
            "notes_columns": [],
        }
        fp = artifacts_dir / f"{pdf_stem}_page_{page_num}_extraction.json"
        fp.write_text(json.dumps(data), encoding="utf-8")
        return fp

    def test_creates_pdf_file(self, tmp_path):
        tokens = [_box(10, 5, 80, 15, text="AREA", font_size=10.0)]
        self._make_artifact(tmp_path, "Plans", 1, tokens)
        out = recreate_sheet(tmp_path)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_output_is_valid_pdf(self, tmp_path):
        tokens = [_box(10, 5, 80, 15, text="DETAIL", font_size=8.0)]
        self._make_artifact(tmp_path, "Plans", 1, tokens)
        out = recreate_sheet(tmp_path)
        header = out.read_bytes()[:5]
        assert header == b"%PDF-", f"Expected PDF header, got {header!r}"

    def test_default_output_path_in_exports(self, tmp_path):
        self._make_artifact(tmp_path, "MyPlans", 1, [_box(0, 0, 10, 5, text="X")])
        out = recreate_sheet(tmp_path)
        assert out.parent == tmp_path / "exports"
        assert out.name == "MyPlans_recreation.pdf"

    def test_explicit_out_path_respected(self, tmp_path):
        self._make_artifact(tmp_path, "A", 1, [_box(0, 0, 10, 5, text="Z")])
        explicit = tmp_path / "custom" / "out.pdf"
        out = recreate_sheet(tmp_path, out_path=explicit)
        assert out == explicit
        assert out.exists()

    def test_page_filter_includes_only_requested_pages(self, tmp_path):
        tokens = [_box(0, 0, 50, 10, text="P")]
        self._make_artifact(tmp_path, "S", 1, tokens)
        self._make_artifact(tmp_path, "S", 2, tokens)
        self._make_artifact(tmp_path, "S", 3, tokens)
        out = recreate_sheet(tmp_path, pages=[1, 3])
        assert out.exists()
        # We can't easily assert page count without a PDF reader,
        # but the file must exist and be non-empty
        assert out.stat().st_size > 0

    def test_missing_artifacts_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="artifacts"):
            recreate_sheet(tmp_path)

    def test_no_matching_pages_raises(self, tmp_path):
        self._make_artifact(tmp_path, "A", 1, [_box(0, 0, 10, 5, text="X")])
        with pytest.raises(FileNotFoundError):
            recreate_sheet(tmp_path, pages=[99])

    def test_multi_page_pdf_produced(self, tmp_path):
        tokens = [_box(0, 0, 50, 10, text="LINE")]
        for pg in (1, 2, 3):
            self._make_artifact(tmp_path, "Sheets", pg, tokens)
        out = recreate_sheet(tmp_path)
        # Multi-page PDF should be larger than single-page
        single_tmp = tmp_path / "single"
        single_tmp.mkdir()
        self._make_artifact(single_tmp, "Sheets", 1, tokens)
        single_out = recreate_sheet(single_tmp)
        assert out.stat().st_size > single_out.stat().st_size

    def test_empty_token_list_per_page(self, tmp_path):
        """Pages with no tokens should still produce a valid PDF."""
        self._make_artifact(tmp_path, "Empty", 1, [])
        out = recreate_sheet(tmp_path)
        assert out.read_bytes()[:5] == b"%PDF-"

    def test_origin_color_map_passed_through(self, tmp_path):
        """Smoke test: recreation with ORIGIN_COLORS color_map does not raise."""
        tokens = [
            _box(0, 0, 50, 10, text="T", origin="text"),
            _box(60, 0, 90, 10, text="O", origin="ocr_full"),
        ]
        self._make_artifact(tmp_path, "Col", 1, tokens)
        out = recreate_sheet(tmp_path, color_map=ORIGIN_COLORS)
        assert out.exists()


# ---------------------------------------------------------------------------
# _block_color / _block_tag helpers
# ---------------------------------------------------------------------------


def _block(
    x0: float = 0,
    y0: float = 0,
    x1: float = 100,
    y1: float = 50,
    is_header: bool = False,
    is_notes: bool = False,
    is_table: bool = False,
    label: str | None = None,
) -> BlockCluster:
    """Build a minimal BlockCluster with a single row for testing."""
    box = _box(x0, y0, x1, y1, text="X")
    row = RowBand(page=0, boxes=[box])
    return BlockCluster(
        page=0,
        rows=[row],
        is_header=is_header,
        is_notes=is_notes,
        is_table=is_table,
        label=label,
    )


class TestBlockColor:
    def test_header_block_returns_blue_gray(self):
        r, g, b = _block_color(_block(is_header=True))
        assert (r, g, b) == (0.3, 0.3, 0.8)

    def test_notes_block_returns_green_gray(self):
        r, g, b = _block_color(_block(is_notes=True))
        assert (r, g, b) == (0.2, 0.6, 0.2)

    def test_table_block_returns_orange_gray(self):
        r, g, b = _block_color(_block(is_table=True))
        assert (r, g, b) == (0.8, 0.5, 0.1)

    def test_plain_block_returns_light_gray(self):
        r, g, b = _block_color(_block())
        assert (r, g, b) == (0.75, 0.75, 0.75)

    def test_header_takes_priority_over_notes(self):
        """header flag checked first when multiple flags set."""
        blk = _block(is_header=True, is_notes=True)
        assert _block_color(blk) == (0.3, 0.3, 0.8)


class TestBlockTag:
    def test_header_tag(self):
        assert _block_tag(_block(is_header=True)) == "[HEADER]"

    def test_notes_tag(self):
        assert _block_tag(_block(is_notes=True)) == "[NOTES]"

    def test_table_tag(self):
        assert _block_tag(_block(is_table=True)) == "[TABLE]"

    def test_no_flags_no_label_returns_none(self):
        assert _block_tag(_block()) is None

    def test_custom_label_tag(self):
        assert _block_tag(_block(label="LEGEND")) == "[LEGEND]"

    def test_header_priority_over_label(self):
        blk = _block(is_header=True, label="OTHER")
        assert _block_tag(blk) == "[HEADER]"


# ---------------------------------------------------------------------------
# _token_counts / _make_page_label
# ---------------------------------------------------------------------------


class TestTokenCounts:
    def test_empty_list(self):
        assert _token_counts([]) == {}

    def test_single_origin(self):
        tokens = [_box(0, 0, 10, 10, origin="text")]
        assert _token_counts(tokens) == {"text": 1}

    def test_mixed_origins(self):
        tokens = [
            _box(0, 0, 10, 10, origin="text"),
            _box(20, 0, 30, 10, origin="ocr_full"),
            _box(40, 0, 50, 10, origin="text"),
            _box(60, 0, 70, 10, origin="reconcile"),
        ]
        c = _token_counts(tokens)
        assert c == {"text": 2, "ocr_full": 1, "reconcile": 1}


class TestMakePageLabel:
    def test_basic_label_parts(self):
        tokens = [_box(0, 0, 10, 10, origin="text")]
        label = _make_page_label(1, tokens, "Plans", "run_001")
        assert "Page 1" in label
        assert "Plans" in label
        assert "run_001" in label

    def test_token_count_in_label(self):
        tokens = [
            _box(0, 0, 10, 10, origin="text"),
            _box(20, 0, 30, 10, origin="ocr_full"),
        ]
        label = _make_page_label(5, tokens, "S", "R")
        assert "2 tokens" in label

    def test_origin_breakdown_in_label(self):
        tokens = [
            _box(0, 0, 10, 10, origin="text"),
            _box(20, 0, 30, 10, origin="ocr_full"),
            _box(40, 0, 50, 10, origin="reconcile"),
        ]
        label = _make_page_label(1, tokens, "P", "R")
        assert "TOCR" in label
        assert "VOCR" in label
        assert "reconcile" in label

    def test_empty_tokens_label(self):
        label = _make_page_label(1, [], "P", "R")
        assert "0 tokens" in label


# ---------------------------------------------------------------------------
# draw_sheet_recreation – enhanced feature tests
# ---------------------------------------------------------------------------


class TestDrawSheetRecreationEnhanced:
    """Tests for state isolation, width fitting, blocks, labels, footer, layers."""

    def _mock_canvas(self, string_width: float = 50.0) -> MagicMock:
        c = MagicMock()
        c.stringWidth.return_value = string_width
        return c

    # -- State isolation ------------------------------------------------

    def test_save_restore_per_token(self):
        """Each token is wrapped in saveState/restoreState."""
        c = self._mock_canvas()
        tokens = [_box(0, 0, 50, 10, text="A"), _box(60, 0, 100, 10, text="B")]
        draw_sheet_recreation(c, 500, 500, tokens=tokens)
        assert c.saveState.call_count == 2
        assert c.restoreState.call_count == 2

    def test_save_restore_order(self):
        """saveState comes before drawText which comes before restoreState."""
        c = self._mock_canvas()
        tokens = [_box(0, 0, 50, 10, text="X")]
        draw_sheet_recreation(c, 500, 500, tokens=tokens)
        calls = [m[0] for m in c.method_calls]
        save_idx = calls.index("saveState")
        draw_idx = calls.index("drawText")
        restore_idx = calls.index("restoreState")
        assert save_idx < draw_idx < restore_idx

    # -- Width fitting --------------------------------------------------

    def test_horiz_scale_called_when_width_differs(self):
        """setHorizScale called on text object when stringWidth differs from bbox width."""
        # bbox width = 100-0 = 100,  stringWidth returns 50
        # hscale = (100/50)*100 = 200  → within [50, 200] → applied
        c = self._mock_canvas(string_width=50.0)
        tokens = [_box(0, 0, 100, 10, text="WORD")]
        draw_sheet_recreation(c, 500, 500, tokens=tokens)
        tx = c.beginText.return_value
        tx.setHorizScale.assert_called_once()
        # hscale ≈ 200.0
        hscale = tx.setHorizScale.call_args[0][0]
        assert hscale == pytest.approx(200.0)

    def test_horiz_scale_skipped_when_out_of_bounds(self):
        """setHorizScale NOT called when ratio is beyond sanity bounds."""
        # bbox width=10, stringWidth=200 → hscale = (10/200)*100 = 5% < 50%
        c = self._mock_canvas(string_width=200.0)
        tokens = [_box(0, 0, 10, 10, text="TINY")]
        draw_sheet_recreation(c, 500, 500, tokens=tokens)
        tx = c.beginText.return_value
        tx.setHorizScale.assert_not_called()

    def test_horiz_scale_skipped_when_predicted_zero(self):
        """setHorizScale NOT called when stringWidth returns 0."""
        c = self._mock_canvas(string_width=0.0)
        tokens = [_box(0, 0, 50, 10, text="W")]
        draw_sheet_recreation(c, 500, 500, tokens=tokens)
        tx = c.beginText.return_value
        tx.setHorizScale.assert_not_called()

    def test_horiz_scale_skipped_when_bbox_zero_width(self):
        """setHorizScale NOT called when bounding box has zero width."""
        c = self._mock_canvas(string_width=50.0)
        tokens = [_box(10, 0, 10, 10, text="W")]  # x0 == x1
        draw_sheet_recreation(c, 500, 500, tokens=tokens)
        tx = c.beginText.return_value
        tx.setHorizScale.assert_not_called()

    # -- Block boundaries -----------------------------------------------

    def test_block_rect_drawn(self):
        """Block rectangle drawn when blocks provided with draw_blocks=True."""
        c = self._mock_canvas()
        blocks = [_block(10, 20, 200, 80)]
        draw_sheet_recreation(c, 500, 500, tokens=[], blocks=blocks)
        c.rect.assert_called_once()
        # rect args: x, y, width, height
        call_args = c.rect.call_args
        assert call_args[0][0] == pytest.approx(10.0)  # x
        assert call_args[0][2] == pytest.approx(190.0)  # width = 200-10
        assert call_args[0][3] == pytest.approx(60.0)  # height = 80-20

    def test_no_blocks_suppresses_drawing(self):
        """draw_blocks=False prevents block rendering."""
        c = self._mock_canvas()
        blocks = [_block(0, 0, 100, 50)]
        draw_sheet_recreation(c, 500, 500, tokens=[], blocks=blocks, draw_blocks=False)
        c.rect.assert_not_called()

    def test_block_colors_correct(self):
        """Different block types get different stroke colours."""
        c = self._mock_canvas()
        header = _block(0, 0, 100, 50, is_header=True)
        notes = _block(0, 60, 100, 100, is_notes=True)
        draw_sheet_recreation(c, 500, 500, tokens=[], blocks=[header, notes])
        # setStrokeColorRGB called for each block
        assert c.setStrokeColorRGB.call_count >= 2

    def test_degenerate_block_skipped(self):
        """Blocks with zero-area bbox are silently skipped."""
        c = self._mock_canvas()
        blocks = [_block(50, 50, 50, 50)]  # x0==x1, y0==y1
        draw_sheet_recreation(c, 500, 500, tokens=[], blocks=blocks)
        c.rect.assert_not_called()

    # -- Margin labels --------------------------------------------------

    def test_header_label_drawn(self):
        """[HEADER] tag drawn for header-classified blocks."""
        c = self._mock_canvas()
        blocks = [_block(10, 20, 200, 80, is_header=True)]
        draw_sheet_recreation(c, 500, 500, tokens=[], blocks=blocks)
        # drawString called at least once for the tag
        draw_calls = c.drawString.call_args_list
        label_calls = [
            dc for dc in draw_calls if len(dc[0]) >= 3 and "[HEADER]" in str(dc[0][2])
        ]
        assert len(label_calls) == 1

    def test_no_label_for_plain_block(self):
        """Plain blocks (no flags, no label) should NOT get a tag."""
        c = self._mock_canvas()
        blocks = [_block(10, 20, 200, 80)]
        draw_sheet_recreation(c, 500, 500, tokens=[], blocks=blocks)
        draw_calls = c.drawString.call_args_list
        tag_calls = [
            dc
            for dc in draw_calls
            if len(dc[0]) >= 3
            and isinstance(dc[0][2], str)
            and dc[0][2].startswith("[")
        ]
        assert len(tag_calls) == 0

    # -- Page footer ----------------------------------------------------

    def test_footer_drawn_when_label_provided(self):
        """Page footer text rendered at bottom when page_label given."""
        c = self._mock_canvas()
        draw_sheet_recreation(c, 500, 500, tokens=[], page_label="Page 1 | test")
        draw_calls = c.drawString.call_args_list
        footer_calls = [
            dc for dc in draw_calls if len(dc[0]) >= 3 and "Page 1" in str(dc[0][2])
        ]
        assert len(footer_calls) == 1

    def test_no_footer_when_label_none(self):
        """No footer text when page_label is None."""
        c = self._mock_canvas()
        draw_sheet_recreation(c, 500, 500, tokens=[], page_label=None)
        c.drawString.assert_not_called()

    # -- PDF layers -----------------------------------------------------

    def test_layers_mode_creates_ocgs(self):
        """use_layers=True triggers beginLayer/endLayer calls."""
        c = self._mock_canvas()
        tokens = [
            _box(0, 0, 50, 10, text="T", origin="text"),
            _box(60, 0, 100, 10, text="O", origin="ocr_full"),
        ]
        draw_sheet_recreation(c, 500, 500, tokens=tokens, use_layers=True)
        assert c.beginLayer.call_count >= 2  # at least text + ocr_full layers
        assert c.endLayer.call_count == c.beginLayer.call_count

    def test_layers_for_blocks_and_labels(self):
        """Block structure and labels get their own layers."""
        c = self._mock_canvas()
        blocks = [_block(0, 0, 100, 50, is_header=True)]
        draw_sheet_recreation(c, 500, 500, tokens=[], blocks=blocks, use_layers=True)
        layer_names = [str(ca[0][0]) for ca in c.beginLayer.call_args_list]
        assert "Block Structure" in layer_names
        assert "Labels" in layer_names

    def test_no_layers_when_disabled(self):
        """use_layers=False (default) → no beginLayer calls."""
        c = self._mock_canvas()
        tokens = [_box(0, 0, 50, 10, text="T")]
        draw_sheet_recreation(c, 500, 500, tokens=tokens, use_layers=False)
        c.beginLayer.assert_not_called()

    # -- Watermark ------------------------------------------------------

    @patch("plancheck.export.sheet_recreation.ImageReader")
    def test_watermark_drawn(self, mock_ir):
        """drawImage called when watermark_img provided."""
        c = self._mock_canvas()
        fake_img = MagicMock()
        draw_sheet_recreation(c, 500, 400, tokens=[], watermark_img=fake_img)
        c.drawImage.assert_called_once()
        mock_ir.assert_called_once_with(fake_img)

    def test_no_watermark_when_none(self):
        """drawImage not called without watermark_img."""
        c = self._mock_canvas()
        draw_sheet_recreation(c, 500, 400, tokens=[])
        c.drawImage.assert_not_called()

    @patch("plancheck.export.sheet_recreation.ImageReader")
    def test_watermark_layer_when_layers_enabled(self, _mock_ir):
        """Watermark gets its own layer when use_layers=True."""
        c = self._mock_canvas()
        fake_img = MagicMock()
        draw_sheet_recreation(
            c, 500, 400, tokens=[], watermark_img=fake_img, use_layers=True
        )
        layer_names = [str(ca[0][0]) for ca in c.beginLayer.call_args_list]
        assert "Original Page" in layer_names


# ---------------------------------------------------------------------------
# recreate_sheet – enhanced integration tests
# ---------------------------------------------------------------------------


class TestRecreateSheetEnhanced:
    """Integration tests for new recreate_sheet parameters."""

    @staticmethod
    def _make_artifact(
        tmp_path: Path,
        pdf_stem: str,
        page_num: int,
        tokens: List[GlyphBox],
        blocks: List[dict] | None = None,
        page_w: float = 500.0,
        page_h: float = 400.0,
    ) -> Path:
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "page_width": page_w,
            "page_height": page_h,
            "tokens": [t.to_dict() for t in tokens],
            "blocks": blocks or [],
            "notes_columns": [],
        }
        fp = artifacts_dir / f"{pdf_stem}_page_{page_num}_extraction.json"
        fp.write_text(json.dumps(data), encoding="utf-8")
        return fp

    def test_draw_blocks_false_produces_pdf(self, tmp_path):
        """draw_blocks=False still produces a valid PDF."""
        tokens = [_box(0, 0, 50, 10, text="T")]
        self._make_artifact(tmp_path, "P", 1, tokens)
        out = recreate_sheet(tmp_path, draw_blocks=False)
        assert out.read_bytes()[:5] == b"%PDF-"

    def test_use_layers_produces_pdf(self, tmp_path):
        """use_layers=True still produces a valid PDF."""
        tokens = [
            _box(0, 0, 50, 10, text="T", origin="text"),
            _box(60, 0, 100, 10, text="O", origin="ocr_full"),
        ]
        self._make_artifact(tmp_path, "L", 1, tokens)
        out = recreate_sheet(tmp_path, use_layers=True)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_pdf_metadata_set(self, tmp_path):
        """PDF is produced with correct metadata (spot-check via mock)."""
        tokens = [_box(0, 0, 50, 10, text="X")]
        self._make_artifact(tmp_path, "Meta", 1, tokens)
        with patch("plancheck.export.sheet_recreation.Canvas") as MockCanvas:
            mock_c = MagicMock()
            mock_c.stringWidth.return_value = 50.0
            MockCanvas.return_value = mock_c
            recreate_sheet(tmp_path)
            mock_c.setTitle.assert_called_once()
            assert "Meta" in mock_c.setTitle.call_args[0][0]
            mock_c.setAuthor.assert_called_once_with("Advanced Plan Parser")
            mock_c.setCreator.assert_called_once()

    def test_page_label_passed_to_draw(self, tmp_path):
        """Page footer label is generated and passed through."""
        tokens = [_box(0, 0, 50, 10, text="Y", origin="text")]
        self._make_artifact(tmp_path, "Lab", 1, tokens)
        with patch(
            "plancheck.export.sheet_recreation.draw_sheet_recreation"
        ) as mock_draw:
            with patch("plancheck.export.sheet_recreation.Canvas") as MockCanvas:
                MockCanvas.return_value = MagicMock()
                recreate_sheet(tmp_path)
                assert mock_draw.call_count == 1
                kwargs = mock_draw.call_args
                # page_label should be a non-empty string
                page_label = (
                    kwargs[1].get("page_label") or kwargs[0][7]
                    if len(kwargs[0]) > 7
                    else kwargs[1].get("page_label")
                )
                assert page_label is not None
                assert "Page 1" in page_label
