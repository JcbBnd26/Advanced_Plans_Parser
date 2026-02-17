"""Tests for _structural_boxes.py — structural box detection, classification,
and semantic region growth."""

from __future__ import annotations

import pytest

from plancheck._structural_boxes import (
    BoxType,
    SemanticRegion,
    StructuralBox,
    _dedup_boxes,
    _detect_boxes_from_lines,
    _first_row_text,
    _grow_region_from_anchor,
    _label_from_header_text,
    _promote_sub_boxes,
    classify_structural_boxes,
    create_synthetic_regions,
    detect_semantic_regions,
    detect_structural_boxes,
    mask_blocks_by_structural_boxes,
)
from plancheck.models import BlockCluster, GlyphBox, GraphicElement, RowBand

# ── Helpers ────────────────────────────────────────────────────────────


def _rect(page, x0, y0, x1, y1, **kw):
    """Quick GraphicElement rect factory."""
    return GraphicElement(
        page=page, element_type="rect", x0=x0, y0=y0, x1=x1, y1=y1, **kw
    )


def _line_g(page, x0, y0, x1, y1):
    """Quick GraphicElement line factory."""
    return GraphicElement(page=page, element_type="line", x0=x0, y0=y0, x1=x1, y1=y1)


def _box(text: str, x0: float, y0: float, x1: float, y1: float, page: int = 0):
    """Build a single-row BlockCluster with one GlyphBox."""
    gb = GlyphBox(page=page, x0=x0, y0=y0, x1=x1, y1=y1, text=text, origin="test")
    row = RowBand(page=page, boxes=[gb])
    return BlockCluster(page=page, rows=[row])


# ══════════════════════════════════════════════════════════════════════
# 1. detect_structural_boxes
# ══════════════════════════════════════════════════════════════════════


class TestDetectStructuralBoxes:
    """Tests for detecting drawn rectangles from graphic elements."""

    def test_empty_graphics(self):
        result = detect_structural_boxes([], 612, 792)
        assert result == []

    def test_filters_tiny_rects(self):
        """Rects smaller than MIN_SIZE_PTS should be dropped."""
        tiny = _rect(0, 100, 100, 105, 105)  # 5×5 — too small
        result = detect_structural_boxes([tiny], 612, 792)
        assert len(result) == 0

    def test_keeps_large_rect(self):
        """A rect covering ~25% of the page should be kept."""
        big = _rect(0, 0, 0, 306, 396)  # half-width × half-height = 25%
        result = detect_structural_boxes([big], 612, 792)
        assert len(result) == 1
        assert result[0].x0 == 0 and result[0].x1 == 306

    def test_filters_below_area_frac(self):
        """Rects with area < min_area_frac should be dropped."""
        small = _rect(0, 100, 100, 120, 120)  # 20x20 = 400 pts² vs 612*792 = 484704
        result = detect_structural_boxes([small], 612, 792, min_area_frac=0.005)
        assert len(result) == 0

    def test_dedup_near_identical(self):
        """Two almost-identical rects should be deduped."""
        r1 = _rect(0, 10, 10, 300, 400)
        r2 = _rect(0, 11, 11, 301, 401)  # within 4pt tolerance
        result = detect_structural_boxes([r1, r2], 612, 792)
        assert len(result) == 1

    def test_curves_ignored(self):
        """Curves should not produce structural boxes."""
        curve = GraphicElement(
            page=0,
            element_type="curve",
            x0=0,
            y0=0,
            x1=300,
            y1=400,
            pts=[(0, 0), (150, 200), (300, 400)],
        )
        result = detect_structural_boxes([curve], 612, 792)
        assert len(result) == 0

    def test_sorted_by_area_descending(self):
        """Results should be sorted largest area first."""
        small = _rect(0, 0, 0, 100, 100)
        large = _rect(0, 0, 0, 400, 400)
        result = detect_structural_boxes([small, large], 612, 792)
        assert result[0].area() >= result[1].area()


# ══════════════════════════════════════════════════════════════════════
# 2. _detect_boxes_from_lines
# ══════════════════════════════════════════════════════════════════════


class TestDetectBoxesFromLines:
    """Tests for reconstructing rectangles from four separate lines."""

    def test_four_lines_form_box(self):
        """Four axis-aligned lines forming a closed rect should be detected."""
        # Top horizontal, bottom horizontal, left vertical, right vertical
        top = _line_g(0, 100, 100, 400, 100)
        bot = _line_g(0, 100, 500, 400, 500)
        left = _line_g(0, 100, 100, 100, 500)
        right = _line_g(0, 400, 100, 400, 500)
        result = _detect_boxes_from_lines([top, bot, left, right], 612, 792, 8.0)
        assert len(result) >= 1
        # Should have approximate dimensions 300×400
        box = result[0]
        assert abs(box.width() - 300) < 10
        assert abs(box.height() - 400) < 10

    def test_no_vertical_lines_no_box(self):
        """Without vertical lines, no box should be found."""
        top = _line_g(0, 100, 100, 400, 100)
        bot = _line_g(0, 100, 500, 400, 500)
        result = _detect_boxes_from_lines([top, bot], 612, 792, 8.0)
        assert len(result) == 0

    def test_short_lines_filtered(self):
        """Lines shorter than min_size should be ignored."""
        top = _line_g(0, 100, 100, 105, 100)  # 5pts wide
        bot = _line_g(0, 100, 500, 105, 500)
        left = _line_g(0, 100, 100, 100, 105)
        right = _line_g(0, 105, 100, 105, 105)
        result = _detect_boxes_from_lines([top, bot, left, right], 612, 792, 8.0)
        assert len(result) == 0


# ══════════════════════════════════════════════════════════════════════
# 3. _dedup_boxes
# ══════════════════════════════════════════════════════════════════════


class TestDedupBoxes:
    def test_empty(self):
        assert _dedup_boxes([]) == []

    def test_keeps_distinct(self):
        b1 = StructuralBox(page=0, x0=0, y0=0, x1=100, y1=100)
        b2 = StructuralBox(page=0, x0=200, y0=200, x1=400, y1=400)
        assert len(_dedup_boxes([b1, b2])) == 2

    def test_removes_near_duplicate(self):
        b1 = StructuralBox(page=0, x0=10, y0=10, x1=300, y1=400)
        b2 = StructuralBox(page=0, x0=12, y0=12, x1=302, y1=402)
        assert len(_dedup_boxes([b1, b2], tolerance=4.0)) == 1


# ══════════════════════════════════════════════════════════════════════
# 4. classify_structural_boxes
# ══════════════════════════════════════════════════════════════════════


class TestClassifyStructuralBoxes:
    """Test priority-based classification of structural boxes."""

    def test_page_border(self):
        """Box covering ≥80% of page area → page_border."""
        sb = StructuralBox(page=0, x0=0, y0=0, x1=600, y1=770)
        blks = [_box("some text", 50, 50, 200, 70)]
        classify_structural_boxes([sb], blks, 612, 792)
        assert sb.box_type == BoxType.page_border

    def test_title_block(self):
        """Tall, skinny box near right edge → title_block."""
        # height = 500 (63% page), width = 100, aspect 0.2, right edge at 97%
        sb = StructuralBox(page=0, x0=500, y0=100, x1=600, y1=600)
        classify_structural_boxes([sb], [], 612, 792)
        assert sb.box_type == BoxType.title_block

    def test_legend_keyword(self):
        """Box containing 'LEGEND' text → legend."""
        sb = StructuralBox(page=0, x0=300, y0=400, x1=550, y1=700)
        blk = _box("UTILITY LEGEND", 320, 410, 500, 430)
        classify_structural_boxes([sb], [blk], 612, 792)
        assert sb.box_type == BoxType.legend

    def test_notes_box_keyword(self):
        """Box containing 'GENERAL NOTES' → notes_box."""
        sb = StructuralBox(page=0, x0=10, y0=10, x1=300, y1=500)
        blk = _box("GENERAL NOTES:", 20, 20, 200, 40)
        classify_structural_boxes([sb], [blk], 612, 792)
        assert sb.box_type == BoxType.notes_box

    def test_unknown_fallback(self):
        """Box with no matching keywords → unknown."""
        sb = StructuralBox(page=0, x0=100, y0=100, x1=300, y1=300)
        blk = _box("random text here", 110, 110, 290, 130)
        classify_structural_boxes([sb], [blk], 612, 792)
        assert sb.box_type == BoxType.unknown

    def test_callout(self):
        """Small box with short all-caps text → callout."""
        sb = StructuralBox(page=0, x0=100, y0=100, x1=130, y1=120)
        blk = _box("A1", 105, 105, 125, 115)
        classify_structural_boxes([sb], [blk], 612, 792)
        assert sb.box_type == BoxType.callout

    def test_classification_priority(self):
        """Page border should win over keyword matches."""
        sb = StructuralBox(page=0, x0=0, y0=0, x1=612, y1=792)
        blk = _box("LEGEND", 50, 50, 200, 70)
        classify_structural_boxes([sb], [blk], 612, 792)
        assert sb.box_type == BoxType.page_border


# ══════════════════════════════════════════════════════════════════════
# 5. _promote_sub_boxes
# ══════════════════════════════════════════════════════════════════════


class TestPromoteSubBoxes:
    def test_promotes_unknown_below_legend(self):
        legend_box = StructuralBox(
            page=0,
            x0=100,
            y0=100,
            x1=400,
            y1=300,
            box_type=BoxType.legend,
            confidence=0.8,
        )
        unk_box = StructuralBox(
            page=0,
            x0=100,
            y0=305,
            x1=400,
            y1=500,
            box_type=BoxType.unknown,
            confidence=0.3,
        )
        _promote_sub_boxes([legend_box, unk_box], 612)
        assert unk_box.box_type == BoxType.legend
        assert unk_box.confidence == pytest.approx(0.8 * 0.7, abs=0.01)

    def test_does_not_promote_far_away(self):
        legend_box = StructuralBox(
            page=0,
            x0=100,
            y0=100,
            x1=400,
            y1=300,
            box_type=BoxType.legend,
            confidence=0.8,
        )
        unk_box = StructuralBox(
            page=0,
            x0=100,
            y0=400,
            x1=400,
            y1=500,
            box_type=BoxType.unknown,
            confidence=0.3,
        )
        _promote_sub_boxes([legend_box, unk_box], 612)
        # 100pt gap > 50pt threshold
        assert unk_box.box_type == BoxType.unknown


# ══════════════════════════════════════════════════════════════════════
# 6. create_synthetic_regions
# ══════════════════════════════════════════════════════════════════════


class TestCreateSyntheticRegions:
    def test_creates_region_from_header_anchor(self):
        """A block with 'GENERAL NOTES:' should generate a synthetic region."""
        header_blk = _box("GENERAL NOTES:", 100, 100, 300, 120)
        note1 = _box("1. First note text.", 100, 130, 300, 150)
        note2 = _box("2. Second note text.", 100, 160, 300, 180)
        blocks = [header_blk, note1, note2]

        synthetics = create_synthetic_regions(blocks, [], 612, 792)
        assert len(synthetics) >= 1
        assert synthetics[0].box_type == BoxType.notes_box
        assert synthetics[0].is_synthetic is True

    def test_skips_anchor_inside_classified_box(self):
        """Anchor already inside a classified box → no synthetic region."""
        header_blk = _box("LEGEND", 110, 110, 250, 130)
        blocks = [header_blk]
        sb = StructuralBox(
            page=0,
            x0=100,
            y0=100,
            x1=400,
            y1=500,
            box_type=BoxType.legend,
            confidence=0.8,
            contained_block_indices=[0],
        )
        synthetics = create_synthetic_regions(blocks, [sb], 612, 792)
        assert len(synthetics) == 0

    def test_abbreviations_anchor(self):
        """A 'ABBREVIATIONS' header should create a legend-type synthetic."""
        header_blk = _box("ABBREVIATIONS:", 100, 100, 300, 120)
        blocks = [header_blk]
        synthetics = create_synthetic_regions(blocks, [], 612, 792)
        assert len(synthetics) == 1
        assert synthetics[0].box_type == BoxType.legend


# ══════════════════════════════════════════════════════════════════════
# 7. _grow_region_from_anchor
# ══════════════════════════════════════════════════════════════════════


class TestGrowRegionFromAnchor:
    def test_grows_downward(self):
        """Region should grow downward to include nearby blocks."""
        header = _box("GENERAL NOTES:", 100, 100, 350, 120)
        note1 = _box("1. Note one.", 100, 130, 350, 150)
        note2 = _box("2. Note two.", 100, 160, 350, 180)
        blocks = [header, note1, note2]

        bbox = _grow_region_from_anchor(0, blocks, [], 612, 792)
        # Should include all three
        assert bbox[1] == pytest.approx(100, abs=1)
        assert bbox[3] == pytest.approx(180, abs=1)

    def test_stops_at_large_gap(self):
        """A vertical gap > max_gap should stop growth."""
        header = _box("GENERAL NOTES:", 100, 100, 350, 120)
        close = _box("1. Note one.", 100, 130, 350, 150)
        far = _box("2. Far away.", 100, 250, 350, 270)  # 100pt gap
        blocks = [header, close, far]

        bbox = _grow_region_from_anchor(0, blocks, [], 612, 792, max_gap=40)
        # Should NOT include 'far'
        assert bbox[3] == pytest.approx(150, abs=1)

    def test_stops_at_different_header(self):
        """Growth should stop at a different section header."""
        header = _box("GENERAL NOTES:", 100, 100, 350, 120)
        note = _box("1. Note one.", 100, 130, 350, 150)
        other_header = _box("LEGEND:", 100, 160, 350, 180)
        blocks = [header, note, other_header]

        bbox = _grow_region_from_anchor(0, blocks, [], 612, 792)
        # Should stop before LEGEND
        assert bbox[3] == pytest.approx(150, abs=1)

    def test_stops_at_title_block(self):
        """Growth should not extend into a title block structural box."""
        header = _box("NOTES:", 100, 600, 350, 620)
        note = _box("1. Note.", 100, 630, 350, 650)
        blocks = [header, note]
        title_sb = StructuralBox(
            page=0,
            x0=0,
            y0=680,
            x1=612,
            y1=792,
            box_type=BoxType.title_block,
        )
        bbox = _grow_region_from_anchor(0, blocks, [title_sb], 612, 792)
        assert bbox[3] <= 680

    def test_ignores_blocks_in_other_column(self):
        """Blocks far to the right should not be included."""
        header = _box("NOTES:", 100, 100, 250, 120)
        same_col = _box("1. Note.", 100, 130, 250, 150)
        other_col = _box("Random text.", 500, 130, 600, 150)  # way to the right
        blocks = [header, same_col, other_col]

        bbox = _grow_region_from_anchor(0, blocks, [], 612, 792, x_tolerance=80)
        # Should not include other_col
        assert bbox[2] <= 260


# ══════════════════════════════════════════════════════════════════════
# 8. mask_blocks_by_structural_boxes
# ══════════════════════════════════════════════════════════════════════


class TestMaskBlocksByStructuralBoxes:
    def test_masks_legend_blocks(self):
        """Blocks inside a legend box should be masked."""
        b1 = _box("LEGEND entry", 110, 110, 300, 130)
        b2 = _box("Note text", 10, 10, 200, 30)
        blocks = [b1, b2]
        sb = StructuralBox(
            page=0,
            x0=100,
            y0=100,
            x1=400,
            y1=500,
            box_type=BoxType.legend,
            contained_block_indices=[0],
        )
        masked = mask_blocks_by_structural_boxes(blocks, [sb])
        assert 0 in masked
        assert 1 not in masked

    def test_masks_title_block(self):
        sb = StructuralBox(
            page=0,
            x0=500,
            y0=600,
            x1=612,
            y1=792,
            box_type=BoxType.title_block,
            contained_block_indices=[2, 3],
        )
        masked = mask_blocks_by_structural_boxes(
            [None, None, None, None], [sb]  # type: ignore
        )
        assert 2 in masked and 3 in masked

    def test_does_not_mask_notes_box(self):
        """notes_box is not in the default exclude list."""
        sb = StructuralBox(
            page=0,
            x0=10,
            y0=10,
            x1=300,
            y1=500,
            box_type=BoxType.notes_box,
            contained_block_indices=[0, 1],
        )
        masked = mask_blocks_by_structural_boxes([None, None], [sb])  # type: ignore
        assert len(masked) == 0

    def test_custom_exclude_types(self):
        sb = StructuralBox(
            page=0,
            x0=10,
            y0=10,
            x1=300,
            y1=500,
            box_type=BoxType.notes_box,
            contained_block_indices=[0],
        )
        masked = mask_blocks_by_structural_boxes(
            [None], [sb], exclude_types=[BoxType.notes_box]  # type: ignore
        )
        assert 0 in masked


# ══════════════════════════════════════════════════════════════════════
# 9. detect_semantic_regions (integration)
# ══════════════════════════════════════════════════════════════════════


class TestDetectSemanticRegions:
    """Integration test for the full pipeline."""

    def test_drawn_box_with_notes(self):
        """A drawn rectangle around GENERAL NOTES should produce a struct box + region."""
        rect = _rect(0, 50, 50, 400, 600)
        header = _box("GENERAL NOTES:", 60, 60, 300, 80)
        note1 = _box("1. First note.", 60, 90, 300, 110)
        note2 = _box("2. Second note.", 60, 120, 300, 140)
        blocks = [header, note1, note2]

        sb_list, regions = detect_semantic_regions(blocks, [rect], 612, 792)
        # Should have at least one structural box
        assert len(sb_list) >= 1
        # Should have at least one semantic region
        assert len(regions) >= 1
        # The region should cover the notes
        r = regions[0]
        assert "NOTE" in r.label.upper()

    def test_no_graphics_creates_synthetic(self):
        """Without any drawn rects, synthetic regions should be created from anchors."""
        header = _box("LEGEND:", 100, 100, 250, 120)
        entry = _box("--- EXISTING FENCE", 100, 130, 300, 150)
        blocks = [header, entry]

        sb_list, regions = detect_semantic_regions(blocks, [], 612, 792)
        # Should have synthetic structural box
        synthetics = [sb for sb in sb_list if sb.is_synthetic]
        assert len(synthetics) >= 1
        # Should have a semantic region
        assert len(regions) >= 1
        assert "LEGEND" in regions[0].label.upper()

    def test_empty_page(self):
        sb_list, regions = detect_semantic_regions([], [], 612, 792)
        assert sb_list == []
        assert regions == []


# ══════════════════════════════════════════════════════════════════════
# 10. Helpers
# ══════════════════════════════════════════════════════════════════════


class TestHelpers:
    def test_first_row_text(self):
        blk = _box("GENERAL NOTES:", 10, 10, 200, 30)
        assert _first_row_text(blk) == "GENERAL NOTES:"

    def test_first_row_text_empty(self):
        blk = BlockCluster(page=0, rows=[])
        assert _first_row_text(blk) == ""

    def test_label_from_header_text(self):
        assert _label_from_header_text("GENERAL NOTES:") == "GENERAL NOTES"
        assert _label_from_header_text("LEGEND.") == "LEGEND"
        assert _label_from_header_text("") == "UNKNOWN"

    def test_structural_box_contains_point(self):
        sb = StructuralBox(page=0, x0=10, y0=10, x1=100, y1=100)
        assert sb.contains_point(50, 50)
        assert not sb.contains_point(0, 0)
        assert sb.contains_point(5, 5, pad=10)  # with padding

    def test_structural_box_properties(self):
        sb = StructuralBox(page=0, x0=10, y0=20, x1=110, y1=220)
        assert sb.width() == 100
        assert sb.height() == 200
        assert sb.area() == 20000
        assert sb.bbox() == (10, 20, 110, 220)
