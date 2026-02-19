"""Tests for plancheck.overlay – visual overlay rendering.

Covers:
  • _get_color / _scale_point helpers
  • _draw_label font/alpha knobs
  • draw_lines_overlay with Span-based lines
  • draw_overlay for every element type, colour overrides, cfg knob wiring
  • Edge cases (empty inputs, zero-size regions, missing background)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import pytest
from PIL import Image

from plancheck.config import GroupingConfig
from plancheck.models import (
    AbbreviationEntry,
    AbbreviationRegion,
    BlockCluster,
    GlyphBox,
    LegendRegion,
    Line,
    MiscTitleRegion,
    NotesColumn,
    RevisionEntry,
    RevisionRegion,
    RowBand,
    Span,
    StandardDetailEntry,
    StandardDetailRegion,
)
from plancheck.export.overlay import (
    COLOR_KEYS,
    COLUMN_COLORS,
    DEFAULT_COLOR,
    LABEL_PREFIXES,
    _draw_label,
    _get_color,
    _scale_point,
    draw_lines_overlay,
    draw_overlay,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _box(x0, y0, x1, y1, text="", page=0) -> GlyphBox:
    return GlyphBox(page=page, x0=x0, y0=y0, x1=x1, y1=y1, text=text)


def _row(boxes: list[GlyphBox], page=0) -> RowBand:
    return RowBand(page=page, boxes=boxes)


def _block(
    boxes: list[GlyphBox],
    page=0,
    is_table=False,
    is_header=False,
    label: str | None = None,
) -> BlockCluster:
    """Build a BlockCluster from flat list of GlyphBoxes (all in one row)."""
    return BlockCluster(
        page=page,
        rows=[_row(boxes, page)],
        is_table=is_table,
        is_header=is_header,
        label=label,
    )


def _all_colors() -> dict[str, tuple]:
    """Return colour overrides that enable drawing for every element key."""
    return {key: (255, 0, 0, 200) for key in COLOR_KEYS}


# ===================================================================
# _get_color
# ===================================================================


class TestGetColor:
    def test_returns_none_without_overrides(self):
        assert _get_color(None, "glyph_boxes") is None

    def test_returns_none_for_missing_key(self):
        assert _get_color({"rows": (0, 255, 0, 128)}, "glyph_boxes") is None

    def test_returns_override_when_present(self):
        c = (10, 20, 30, 200)
        assert _get_color({"rows": c}, "rows") == c

    def test_empty_dict_returns_none(self):
        assert _get_color({}, "rows") is None


# ===================================================================
# _scale_point
# ===================================================================


class TestScalePoint:
    def test_identity(self):
        assert _scale_point(10.0, 20.0, 1.0) == (10.0, 20.0)

    def test_double(self):
        assert _scale_point(5.0, 3.0, 2.0) == (10.0, 6.0)

    def test_fraction(self):
        assert _scale_point(100.0, 50.0, 0.5) == (50.0, 25.0)

    def test_zero_scale(self):
        assert _scale_point(7.0, 9.0, 0.0) == (0.0, 0.0)


# ===================================================================
# _draw_label
# ===================================================================


class TestDrawLabel:
    @pytest.fixture()
    def canvas(self):
        from PIL import ImageDraw

        img = Image.new("RGBA", (200, 200), (255, 255, 255, 255))
        draw = ImageDraw.Draw(img, "RGBA")
        return img, draw

    def test_label_renders_no_crash(self, canvas):
        img, draw = canvas
        _draw_label(draw, 10.0, 30.0, "glyph_boxes", 1, (255, 0, 0, 200), 1.0)
        # Just verifying it doesn't raise

    def test_label_with_cfg(self, canvas):
        img, draw = canvas
        cfg = GroupingConfig(
            overlay_label_font_base=14,
            overlay_label_font_floor=10,
            overlay_label_bg_alpha=100,
        )
        _draw_label(draw, 10.0, 50.0, "rows", 5, (0, 255, 0, 200), 1.0, cfg=cfg)

    def test_unknown_prefix_falls_back_to_question_mark(self, canvas):
        img, draw = canvas
        _draw_label(draw, 10.0, 30.0, "unknown_type", 1, (0, 0, 255, 200), 1.0)


# ===================================================================
# draw_lines_overlay
# ===================================================================


class TestDrawLinesOverlay:
    def _make_tokens(self) -> list[GlyphBox]:
        return [
            _box(10, 100, 40, 110, "HELLO"),
            _box(45, 100, 80, 110, "WORLD"),
        ]

    def _make_lines(self, tokens: list[GlyphBox]) -> list[Line]:
        span = Span(token_indices=[0, 1])
        return [Line(line_id=0, page=0, token_indices=[0, 1], spans=[span])]

    def test_creates_png(self, tmp_path):
        tokens = self._make_tokens()
        lines = self._make_lines(tokens)
        out = tmp_path / "lines.png"
        draw_lines_overlay(200.0, 200.0, lines, tokens, out, scale=1.0)
        assert out.exists()
        img = Image.open(out)
        assert img.size == (200, 200)

    def test_with_background(self, tmp_path):
        tokens = self._make_tokens()
        lines = self._make_lines(tokens)
        bg = Image.new("RGB", (200, 200), (128, 128, 128))
        out = tmp_path / "lines_bg.png"
        draw_lines_overlay(200.0, 200.0, lines, tokens, out, scale=1.0, background=bg)
        assert out.exists()

    def test_with_cfg_knobs(self, tmp_path):
        tokens = self._make_tokens()
        lines = self._make_lines(tokens)
        cfg = GroupingConfig(overlay_span_outline_width=5)
        out = tmp_path / "lines_cfg.png"
        draw_lines_overlay(200.0, 200.0, lines, tokens, out, scale=1.0, cfg=cfg)
        assert out.exists()

    def test_empty_lines(self, tmp_path):
        out = tmp_path / "empty.png"
        draw_lines_overlay(100.0, 100.0, [], [], out, scale=1.0)
        assert out.exists()

    def test_line_with_no_token_indices_skipped(self, tmp_path):
        empty_line = Line(line_id=0, page=0, token_indices=[])
        out = tmp_path / "no_tok.png"
        draw_lines_overlay(100.0, 100.0, [empty_line], [], out, scale=1.0)
        assert out.exists()

    def test_scale_factor(self, tmp_path):
        tokens = self._make_tokens()
        lines = self._make_lines(tokens)
        out = tmp_path / "scaled.png"
        draw_lines_overlay(200.0, 200.0, lines, tokens, out, scale=2.0)
        img = Image.open(out)
        assert img.size == (400, 400)

    def test_background_resize(self, tmp_path):
        tokens = self._make_tokens()
        lines = self._make_lines(tokens)
        bg = Image.new("RGB", (100, 100), (200, 200, 200))
        out = tmp_path / "resized.png"
        draw_lines_overlay(200.0, 200.0, lines, tokens, out, scale=1.0, background=bg)
        img = Image.open(out)
        assert img.size == (200, 200)


# ===================================================================
# draw_overlay – core rendering
# ===================================================================


class TestDrawOverlayBasic:
    """Basic rendering and file-output tests."""

    def test_creates_png_minimal(self, tmp_path):
        out = tmp_path / "overlay.png"
        draw_overlay(100, 100, [], [], [], out, scale=1.0)
        assert out.exists()
        img = Image.open(out)
        assert img.size == (100, 100)

    def test_creates_png_with_background(self, tmp_path):
        bg = Image.new("RGB", (200, 200), (64, 64, 64))
        out = tmp_path / "overlay_bg.png"
        draw_overlay(200, 200, [], [], [], out, scale=1.0, background=bg)
        assert out.exists()

    def test_scale_doubles_image(self, tmp_path):
        out = tmp_path / "scaled.png"
        draw_overlay(100, 100, [], [], [], out, scale=2.0)
        img = Image.open(out)
        assert img.size == (200, 200)

    def test_background_resized_to_match(self, tmp_path):
        bg = Image.new("RGB", (50, 50), (0, 0, 0))
        out = tmp_path / "bg_resize.png"
        draw_overlay(100, 100, [], [], [], out, scale=1.0, background=bg)
        img = Image.open(out)
        assert img.size == (100, 100)


# ===================================================================
# draw_overlay – glyph boxes
# ===================================================================


class TestDrawOverlayGlyphs:
    def test_glyph_boxes_drawn_when_enabled(self, tmp_path):
        boxes = [_box(10, 10, 50, 20, "A")]
        out = tmp_path / "glyphs.png"
        colors = {"glyph_boxes": (255, 0, 0, 200)}
        draw_overlay(100, 100, boxes, [], [], out, color_overrides=colors)
        img = Image.open(out)
        # The red channel at the glyph border should differ from pure white
        px = img.getpixel((10, 10))
        assert px[0] > 200  # red channel present at border

    def test_glyph_boxes_not_drawn_without_override(self, tmp_path):
        boxes = [_box(10, 10, 50, 20)]
        out = tmp_path / "no_glyphs.png"
        draw_overlay(100, 100, boxes, [], [], out)
        img = Image.open(out)
        # Should be white-ish everywhere (no drawing)
        px = img.getpixel((10, 10))
        assert px[:3] == (255, 255, 255)

    def test_glyph_width_from_cfg(self, tmp_path):
        boxes = [_box(20, 20, 80, 40)]
        cfg = GroupingConfig(overlay_glyph_outline_width=5)
        out = tmp_path / "glyph_wide.png"
        draw_overlay(
            100,
            100,
            boxes,
            [],
            [],
            out,
            color_overrides={"glyph_boxes": (255, 0, 0, 255)},
            cfg=cfg,
        )
        assert out.exists()


# ===================================================================
# draw_overlay – rows
# ===================================================================


class TestDrawOverlayRows:
    def test_rows_drawn_when_enabled(self, tmp_path):
        b = _box(0, 0, 50, 10)
        rows = [_row([b])]
        out = tmp_path / "rows.png"
        draw_overlay(
            100, 100, [], rows, [], out, color_overrides={"rows": (0, 255, 0, 200)}
        )
        assert out.exists()

    def test_rows_not_drawn_without_color(self, tmp_path):
        b = _box(0, 0, 50, 10)
        rows = [_row([b])]
        out = tmp_path / "rows_no.png"
        draw_overlay(100, 100, [], rows, [], out)
        img = Image.open(out)
        assert img.getpixel((0, 0))[:3] == (255, 255, 255)


# ===================================================================
# draw_overlay – blocks (regular, table, header)
# ===================================================================


class TestDrawOverlayBlocks:
    def _blocks(self):
        regular = _block([_box(10, 10, 90, 20)])
        table = _block([_box(10, 30, 90, 40)], is_table=True)
        header = _block([_box(10, 50, 90, 60)], label="note_column_header")
        return [regular, table, header]

    def test_regular_block_drawn(self, tmp_path):
        out = tmp_path / "regular.png"
        draw_overlay(
            100,
            100,
            [],
            [],
            self._blocks(),
            out,
            color_overrides={"regular_blocks": (0, 0, 255, 200)},
        )
        assert out.exists()

    def test_table_block_filled(self, tmp_path):
        out = tmp_path / "table.png"
        draw_overlay(
            100,
            100,
            [],
            [],
            self._blocks(),
            out,
            color_overrides={"table_blocks": (0, 128, 0, 200)},
        )
        assert out.exists()

    def test_header_block_drawn(self, tmp_path):
        out = tmp_path / "header.png"
        draw_overlay(
            100,
            100,
            [],
            [],
            self._blocks(),
            out,
            color_overrides={"header_blocks": (200, 0, 200, 200)},
        )
        assert out.exists()

    def test_table_fill_alpha_from_cfg(self, tmp_path):
        cfg = GroupingConfig(overlay_table_fill_alpha=120)
        blocks = [_block([_box(10, 10, 90, 90)], is_table=True)]
        out = tmp_path / "table_alpha.png"
        draw_overlay(
            100,
            100,
            [],
            [],
            blocks,
            out,
            color_overrides={"table_blocks": (0, 200, 0, 255)},
            cfg=cfg,
        )
        img = Image.open(out)
        # Interior pixel should have partial green transparency
        px = img.getpixel((50, 50))
        assert px[1] > 0  # some green

    def test_block_width_from_cfg(self, tmp_path):
        cfg = GroupingConfig(overlay_block_outline_width=6)
        blocks = [_block([_box(10, 10, 90, 90)])]
        out = tmp_path / "block_wide.png"
        draw_overlay(
            100,
            100,
            [],
            [],
            blocks,
            out,
            color_overrides={"regular_blocks": (0, 0, 255, 255)},
            cfg=cfg,
        )
        assert out.exists()


# ===================================================================
# draw_overlay – notes columns
# ===================================================================


class TestDrawOverlayNotesColumns:
    def _notes_col(self):
        blk = _block([_box(10, 10, 90, 90)])
        return NotesColumn(page=0, notes_blocks=[blk])

    def test_notes_column_drawn(self, tmp_path):
        out = tmp_path / "notes.png"
        draw_overlay(
            100,
            100,
            [],
            [],
            [],
            out,
            notes_columns=[self._notes_col()],
            color_overrides={"notes_columns": (200, 200, 0, 200)},
        )
        assert out.exists()

    def test_empty_bbox_notes_column_skipped(self, tmp_path):
        col = NotesColumn(page=0)  # no blocks → bbox (0,0,0,0)
        out = tmp_path / "empty_notes.png"
        draw_overlay(
            100,
            100,
            [],
            [],
            [],
            out,
            notes_columns=[col],
            color_overrides={"notes_columns": (200, 200, 0, 200)},
        )
        img = Image.open(out)
        # nothing drawn at 0,0 besides white bg
        assert img.getpixel((50, 50))[:3] == (255, 255, 255)

    def test_region_width_from_cfg(self, tmp_path):
        cfg = GroupingConfig(overlay_region_outline_width=8)
        out = tmp_path / "notes_wide.png"
        draw_overlay(
            100,
            100,
            [],
            [],
            [],
            out,
            notes_columns=[self._notes_col()],
            color_overrides={"notes_columns": (200, 200, 0, 255)},
            cfg=cfg,
        )
        assert out.exists()


# ===================================================================
# draw_overlay – legend regions
# ===================================================================


class TestDrawOverlayLegendRegions:
    def _legend(self):
        hdr = _block([_box(10, 10, 90, 20, "LEGEND")])
        return LegendRegion(page=0, header=hdr, box_bbox=(5, 5, 95, 95))

    def test_legend_region_drawn(self, tmp_path):
        out = tmp_path / "legend.png"
        draw_overlay(
            100,
            100,
            [],
            [],
            [],
            out,
            legend_regions=[self._legend()],
            color_overrides={
                "legend_region": (255, 0, 255, 200),
                "legend_header": (128, 0, 128, 200),
            },
        )
        assert out.exists()

    def test_legend_no_header(self, tmp_path):
        lg = LegendRegion(page=0, box_bbox=(10, 10, 90, 90))
        out = tmp_path / "lg_nohdr.png"
        draw_overlay(
            100,
            100,
            [],
            [],
            [],
            out,
            legend_regions=[lg],
            color_overrides={"legend_region": (255, 0, 255, 200)},
        )
        assert out.exists()

    def test_zero_bbox_legend_skipped(self, tmp_path):
        lg = LegendRegion(page=0)  # bbox (0,0,0,0)
        out = tmp_path / "lg_zero.png"
        draw_overlay(
            100,
            100,
            [],
            [],
            [],
            out,
            legend_regions=[lg],
            color_overrides=_all_colors(),
        )
        assert out.exists()


# ===================================================================
# draw_overlay – abbreviation regions
# ===================================================================


class TestDrawOverlayAbbreviationRegions:
    def _abbrev(self):
        hdr = _block([_box(10, 10, 90, 20, "ABBREVIATIONS")])
        entry = AbbreviationEntry(
            page=0,
            code="BM",
            meaning="BENCHMARK",
            code_bbox=(10, 30, 30, 40),
            meaning_bbox=(40, 30, 90, 40),
        )
        return AbbreviationRegion(
            page=0, header=hdr, entries=[entry], box_bbox=(5, 5, 95, 95)
        )

    def test_abbreviation_full(self, tmp_path):
        out = tmp_path / "abbrev.png"
        draw_overlay(
            100,
            100,
            [],
            [],
            [],
            out,
            abbreviation_regions=[self._abbrev()],
            color_overrides={
                "abbreviation_region": (255, 128, 0, 200),
                "abbreviation_header": (200, 100, 0, 200),
                "abbreviation_code": (0, 200, 200, 200),
                "abbreviation_meaning": (0, 100, 200, 200),
                "abbreviation_line": (100, 100, 100, 200),
            },
        )
        assert out.exists()

    def test_abbreviation_entry_no_bboxes(self, tmp_path):
        entry = AbbreviationEntry(page=0, code="X", meaning="Y")
        ar = AbbreviationRegion(page=0, entries=[entry], box_bbox=(10, 10, 90, 90))
        out = tmp_path / "abbrev_nobox.png"
        draw_overlay(
            100,
            100,
            [],
            [],
            [],
            out,
            abbreviation_regions=[ar],
            color_overrides=_all_colors(),
        )
        assert out.exists()

    def test_abbreviation_no_header(self, tmp_path):
        ar = AbbreviationRegion(page=0, box_bbox=(10, 10, 90, 90))
        out = tmp_path / "abbrev_nohdr.png"
        draw_overlay(
            100,
            100,
            [],
            [],
            [],
            out,
            abbreviation_regions=[ar],
            color_overrides=_all_colors(),
        )
        assert out.exists()


# ===================================================================
# draw_overlay – revision regions
# ===================================================================


class TestDrawOverlayRevisionRegions:
    def _revision(self):
        hdr = _block([_box(10, 10, 90, 20, "REVISIONS")])
        entry = RevisionEntry(
            page=0,
            number="1",
            description="INITIAL",
            date="2025-01",
            row_bbox=(10, 30, 90, 40),
        )
        return RevisionRegion(
            page=0, header=hdr, entries=[entry], box_bbox=(5, 5, 95, 95)
        )

    def test_revision_full(self, tmp_path):
        out = tmp_path / "rev.png"
        draw_overlay(
            100,
            100,
            [],
            [],
            [],
            out,
            revision_regions=[self._revision()],
            color_overrides={
                "revision_region": (0, 200, 0, 200),
                "revision_header": (0, 128, 0, 200),
                "revision_entry": (0, 100, 100, 200),
            },
        )
        assert out.exists()

    def test_revision_entry_no_row_bbox(self, tmp_path):
        entry = RevisionEntry(page=0, number="2")
        rv = RevisionRegion(page=0, entries=[entry], box_bbox=(10, 10, 90, 90))
        out = tmp_path / "rev_norow.png"
        draw_overlay(
            100,
            100,
            [],
            [],
            [],
            out,
            revision_regions=[rv],
            color_overrides=_all_colors(),
        )
        assert out.exists()

    def test_revision_no_header(self, tmp_path):
        rv = RevisionRegion(page=0, box_bbox=(10, 10, 90, 90))
        out = tmp_path / "rev_nohdr.png"
        draw_overlay(
            100,
            100,
            [],
            [],
            [],
            out,
            revision_regions=[rv],
            color_overrides=_all_colors(),
        )
        assert out.exists()


# ===================================================================
# draw_overlay – misc title regions
# ===================================================================


class TestDrawOverlayMiscTitleRegions:
    def _misc_title(self):
        blk = _block([_box(10, 50, 90, 60, "TRANSPORTATION")])
        return MiscTitleRegion(
            page=0, text="TRANSPORTATION", text_block=blk, box_bbox=(10, 50, 90, 60)
        )

    def test_misc_title_box_drawn(self, tmp_path):
        out = tmp_path / "misc.png"
        draw_overlay(
            100,
            100,
            [],
            [],
            [],
            out,
            misc_title_regions=[self._misc_title()],
            color_overrides={"misc_title_region": (128, 128, 0, 200)},
        )
        assert out.exists()

    def test_misc_title_text_combined_box(self, tmp_path):
        """Combined bounding box drawn around overlapping glyph boxes."""
        mt = self._misc_title()
        boxes = [_box(10, 50, 50, 60, "TRANS"), _box(55, 50, 90, 60, "PORT")]
        out = tmp_path / "misc_combined.png"
        draw_overlay(
            100,
            100,
            boxes,
            [],
            [],
            out,
            misc_title_regions=[mt],
            color_overrides={
                "misc_title_text": (255, 255, 0, 200),
                "glyph_boxes": (200, 200, 200, 100),
            },
        )
        assert out.exists()

    def test_misc_title_same_line_detection(self, tmp_path):
        """Glyph boxes on the same y-line as misc_title should be included."""
        mt = MiscTitleRegion(page=0, text="T", box_bbox=(10, 50, 50, 60))
        # A box to the right, same y-band, within proximity
        boxes = [_box(55, 50, 90, 60, "EXT")]
        out = tmp_path / "misc_sameline.png"
        draw_overlay(
            100,
            100,
            boxes,
            [],
            [],
            out,
            misc_title_regions=[mt],
            color_overrides={"misc_title_text": (255, 255, 0, 200)},
        )
        assert out.exists()

    def test_same_line_proximity_from_cfg(self, tmp_path):
        """Proximity and overlap knobs from cfg are respected."""
        cfg = GroupingConfig(overlay_proximity_pts=5.0, overlay_same_line_overlap=0.9)
        mt = MiscTitleRegion(page=0, text="T", box_bbox=(10, 50, 50, 60))
        # Box far to the right – should NOT be included with tight proximity
        boxes = [_box(200, 50, 250, 60, "FAR")]
        out = tmp_path / "misc_tight.png"
        draw_overlay(
            300,
            100,
            boxes,
            [],
            [],
            out,
            misc_title_regions=[mt],
            color_overrides={"misc_title_text": (0, 255, 0, 200)},
            cfg=cfg,
        )
        assert out.exists()

    def test_zero_bbox_misc_title_skipped(self, tmp_path):
        mt = MiscTitleRegion(page=0)
        out = tmp_path / "misc_zero.png"
        draw_overlay(
            100,
            100,
            [],
            [],
            [],
            out,
            misc_title_regions=[mt],
            color_overrides=_all_colors(),
        )
        assert out.exists()


# ===================================================================
# draw_overlay – standard detail regions
# ===================================================================


class TestDrawOverlayStandardDetailRegions:
    def _detail(self):
        hdr_row = RowBand(page=0, boxes=[_box(10, 10, 90, 20, "STANDARD DETAILS")])
        hdr = BlockCluster(page=0, rows=[hdr_row])
        entry = StandardDetailEntry(
            page=0,
            sheet_number="C-101",
            description="PAVING DETAIL",
            sheet_bbox=(10, 40, 40, 50),
            description_bbox=(45, 40, 90, 50),
        )
        return StandardDetailRegion(
            page=0,
            header=hdr,
            subheader="CIVIL",
            subheader_bbox=(10, 25, 90, 35),
            entries=[entry],
            box_bbox=(5, 5, 95, 95),
        )

    def test_standard_detail_full(self, tmp_path):
        out = tmp_path / "sd.png"
        draw_overlay(
            100,
            100,
            [],
            [],
            [],
            out,
            standard_detail_regions=[self._detail()],
            color_overrides={
                "standard_detail_region": (200, 0, 0, 200),
                "standard_detail_header": (180, 0, 0, 200),
                "standard_detail_subheader": (160, 50, 50, 200),
                "standard_detail_sheet": (0, 0, 200, 200),
                "standard_detail_description": (0, 200, 200, 200),
            },
        )
        assert out.exists()

    def test_standard_detail_no_header(self, tmp_path):
        sd = StandardDetailRegion(page=0, box_bbox=(10, 10, 90, 90))
        out = tmp_path / "sd_nohdr.png"
        draw_overlay(
            100,
            100,
            [],
            [],
            [],
            out,
            standard_detail_regions=[sd],
            color_overrides=_all_colors(),
        )
        assert out.exists()

    def test_standard_detail_no_subheader(self, tmp_path):
        hdr_row = RowBand(page=0, boxes=[_box(10, 10, 90, 20, "HDR")])
        hdr = BlockCluster(page=0, rows=[hdr_row])
        sd = StandardDetailRegion(page=0, header=hdr, box_bbox=(5, 5, 95, 95))
        out = tmp_path / "sd_nosub.png"
        draw_overlay(
            100,
            100,
            [],
            [],
            [],
            out,
            standard_detail_regions=[sd],
            color_overrides=_all_colors(),
        )
        assert out.exists()

    def test_standard_detail_entry_no_bboxes(self, tmp_path):
        entry = StandardDetailEntry(page=0, sheet_number="X")
        sd = StandardDetailRegion(page=0, entries=[entry], box_bbox=(10, 10, 90, 90))
        out = tmp_path / "sd_nobbox.png"
        draw_overlay(
            100,
            100,
            [],
            [],
            [],
            out,
            standard_detail_regions=[sd],
            color_overrides=_all_colors(),
        )
        assert out.exists()

    def test_zero_bbox_detail_skipped(self, tmp_path):
        sd = StandardDetailRegion(page=0)
        out = tmp_path / "sd_zero.png"
        draw_overlay(
            100,
            100,
            [],
            [],
            [],
            out,
            standard_detail_regions=[sd],
            color_overrides=_all_colors(),
        )
        assert out.exists()


# ===================================================================
# draw_overlay – cfg knobs integration
# ===================================================================


class TestDrawOverlayCfgKnobs:
    """Verify that every config knob from GroupingConfig flows through."""

    def test_all_outline_widths_respected(self, tmp_path):
        cfg = GroupingConfig(
            overlay_glyph_outline_width=2,
            overlay_span_outline_width=4,
            overlay_block_outline_width=6,
            overlay_region_outline_width=8,
        )
        boxes = [_box(10, 10, 40, 20)]
        rows = [_row(boxes)]
        blocks = [_block(boxes)]
        hdr = _block([_box(10, 10, 90, 20, "LEGEND")])
        legends = [LegendRegion(page=0, header=hdr, box_bbox=(5, 5, 95, 95))]
        out = tmp_path / "all_widths.png"
        draw_overlay(
            100,
            100,
            boxes,
            rows,
            blocks,
            out,
            legend_regions=legends,
            color_overrides=_all_colors(),
            cfg=cfg,
        )
        assert out.exists()

    def test_default_cfg_matches_hardcoded_defaults(self, tmp_path):
        """Drawing with default GroupingConfig should be identical to cfg=None."""
        boxes = [_box(10, 10, 50, 20)]
        rows = [_row(boxes)]
        blocks = [_block(boxes)]
        colors = _all_colors()

        out1 = tmp_path / "default_none.png"
        draw_overlay(
            100, 100, boxes, rows, blocks, out1, color_overrides=colors, cfg=None
        )

        out2 = tmp_path / "default_cfg.png"
        draw_overlay(
            100,
            100,
            boxes,
            rows,
            blocks,
            out2,
            color_overrides=colors,
            cfg=GroupingConfig(),
        )

        img1 = Image.open(out1)
        img2 = Image.open(out2)
        assert img1.size == img2.size
        # Pixel-identical (same defaults)
        assert list(img1.getdata()) == list(img2.getdata())


# ===================================================================
# draw_overlay – combined stress test
# ===================================================================


class TestDrawOverlayStress:
    """Draw everything at once to ensure no crashes or interaction bugs."""

    def test_all_elements_at_once(self, tmp_path):
        boxes = [_box(10, 10, 40, 20, "A"), _box(50, 10, 80, 20, "B")]
        r = _row(boxes)
        blk = _block(boxes)
        tbl = _block([_box(10, 30, 90, 60)], is_table=True)
        hdr_blk = _block([_box(10, 70, 90, 80)], label="note_column_header")

        notes = [NotesColumn(page=0, notes_blocks=[blk])]
        legends = [
            LegendRegion(
                page=0,
                header=_block([_box(5, 5, 20, 10, "LG")]),
                box_bbox=(5, 5, 95, 95),
            )
        ]
        abbrev_entry = AbbreviationEntry(
            page=0,
            code="C",
            meaning="M",
            code_bbox=(10, 85, 20, 90),
            meaning_bbox=(30, 85, 60, 90),
        )
        abbrevs = [
            AbbreviationRegion(page=0, entries=[abbrev_entry], box_bbox=(5, 80, 95, 95))
        ]
        rev_entry = RevisionEntry(page=0, number="1", row_bbox=(10, 92, 90, 98))
        revisions = [
            RevisionRegion(page=0, entries=[rev_entry], box_bbox=(5, 90, 95, 100))
        ]
        misc = [MiscTitleRegion(page=0, text="T", box_bbox=(10, 50, 90, 60))]
        sd_entry = StandardDetailEntry(
            page=0, sheet_bbox=(10, 60, 30, 65), description_bbox=(35, 60, 90, 65)
        )
        standard = [
            StandardDetailRegion(page=0, entries=[sd_entry], box_bbox=(5, 55, 95, 70))
        ]

        out = tmp_path / "stress.png"
        draw_overlay(
            100,
            100,
            boxes,
            [r],
            [blk, tbl, hdr_blk],
            out,
            scale=2.0,
            notes_columns=notes,
            legend_regions=legends,
            abbreviation_regions=abbrevs,
            revision_regions=revisions,
            misc_title_regions=misc,
            standard_detail_regions=standard,
            color_overrides=_all_colors(),
            cfg=GroupingConfig(),
        )
        assert out.exists()
        img = Image.open(out)
        assert img.size == (200, 200)


# ===================================================================
# Generator / iterable safety
# ===================================================================


class TestDrawOverlayGeneratorSafety:
    """Verify that passing generators (single-use iterables) works correctly."""

    def test_boxes_as_generator(self, tmp_path):
        """Boxes passed as a generator should still render glyphs + misc-title combined."""
        mt = MiscTitleRegion(page=0, text="T", box_bbox=(10, 50, 50, 60))

        def gen_boxes():
            yield _box(15, 50, 45, 60, "TITLE")

        out = tmp_path / "gen_boxes.png"
        draw_overlay(
            100,
            100,
            gen_boxes(),
            iter([]),
            iter([]),
            out,
            misc_title_regions=[mt],
            color_overrides={
                "glyph_boxes": (255, 0, 0, 200),
                "misc_title_text": (0, 255, 0, 200),
            },
        )
        assert out.exists()

    def test_rows_as_generator(self, tmp_path):
        b = _box(0, 0, 50, 10)
        out = tmp_path / "gen_rows.png"
        draw_overlay(
            100,
            100,
            iter([]),
            iter([_row([b])]),
            iter([]),
            out,
            color_overrides={"rows": (0, 255, 0, 200)},
        )
        assert out.exists()

    def test_blocks_as_generator(self, tmp_path):
        blk = _block([_box(10, 10, 90, 20)])
        out = tmp_path / "gen_blocks.png"
        draw_overlay(
            100,
            100,
            iter([]),
            iter([]),
            iter([blk]),
            out,
            color_overrides={"regular_blocks": (0, 0, 255, 200)},
        )
        assert out.exists()


# ===================================================================
# Hoisted color lookup – no drawing when disabled
# ===================================================================


class TestDrawOverlayColorHoisting:
    """Confirm that hoisted colour lookups skip the loop entirely when None."""

    def test_no_glyph_drawing_without_override(self, tmp_path):
        boxes = [_box(10, 10, 90, 90)]
        out = tmp_path / "no_glyph.png"
        draw_overlay(100, 100, boxes, [], [], out, color_overrides={})
        img = Image.open(out)
        # Centre pixel stays white
        assert img.getpixel((50, 50))[:3] == (255, 255, 255)

    def test_no_row_drawing_without_override(self, tmp_path):
        rows = [_row([_box(10, 10, 90, 90)])]
        out = tmp_path / "no_row.png"
        draw_overlay(100, 100, [], rows, [], out, color_overrides={})
        img = Image.open(out)
        assert img.getpixel((50, 50))[:3] == (255, 255, 255)


# ===================================================================
# Module-level constants
# ===================================================================


class TestOverlayConstants:
    def test_default_color_is_none(self):
        assert DEFAULT_COLOR is None

    def test_color_keys_has_expected_entries(self):
        assert "glyph_boxes" in COLOR_KEYS
        assert "rows" in COLOR_KEYS
        assert "standard_detail_region" in COLOR_KEYS

    def test_label_prefixes_keys_subset_of_color_keys(self):
        for key in LABEL_PREFIXES:
            assert key in COLOR_KEYS, f"{key} not in COLOR_KEYS"

    def test_column_colors_are_rgba(self):
        for c in COLUMN_COLORS:
            assert len(c) == 4
            assert all(0 <= v <= 255 for v in c)
