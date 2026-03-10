"""Round-trip serialization tests for PageResult and DocumentResult.

Verifies that every nested data structure survives a to_dict → from_dict
round-trip with full fidelity.
"""

from __future__ import annotations

import json

import pytest

from plancheck.analysis.structural_boxes import (BoxType, SemanticRegion,
                                                 StructuralBox)
from plancheck.analysis.title_block import TitleBlockField, TitleBlockInfo
from plancheck.analysis.zoning import PageZone, ZoneTag
from plancheck.checks.semantic_checks import CheckResult
from plancheck.config import GroupingConfig
from plancheck.models import (AbbreviationEntry, AbbreviationRegion,
                              BlockCluster, GlyphBox, GraphicElement,
                              LegendEntry, LegendRegion, Line, MiscTitleRegion,
                              NotesColumn, RevisionEntry, RevisionRegion,
                              RowBand, Span, StandardDetailEntry,
                              StandardDetailRegion, SuspectRegion)
from plancheck.pipeline import DocumentResult, PageResult, StageResult
from plancheck.reconcile.reconcile import MatchRecord, ReconcileResult

# ── Helpers ────────────────────────────────────────────────────────────


def _box(x0=10, y0=100, x1=60, y1=112, text="HELLO", page=0):
    return GlyphBox(page=page, x0=x0, y0=y0, x1=x1, y1=y1, text=text)


def _make_tokens_blocks_columns(page=0):
    """Build a minimal but realistic set of tokens, blocks, and columns."""
    tokens = [
        _box(50, 100, 120, 112, "GENERAL", page),
        _box(130, 100, 190, 112, "NOTES:", page),
        _box(50, 120, 65, 132, "1.", page),
        _box(70, 120, 150, 132, "ALL WORK", page),
    ]
    spans_line0 = [Span(token_indices=[0, 1])]
    spans_line1 = [Span(token_indices=[2, 3])]
    line0 = Line(
        line_id=0, page=page, token_indices=[0, 1], baseline_y=106, spans=spans_line0
    )
    line1 = Line(
        line_id=1, page=page, token_indices=[2, 3], baseline_y=126, spans=spans_line1
    )

    header_block = BlockCluster(
        page=page, lines=[line0], _tokens=tokens, is_header=True, label="header"
    )
    header_block.populate_rows_from_lines()

    notes_block = BlockCluster(
        page=page, lines=[line1], _tokens=tokens, is_notes=True, label="note"
    )
    notes_block.populate_rows_from_lines()

    blocks = [header_block, notes_block]
    notes_col = NotesColumn(page=page, header=header_block, notes_blocks=[notes_block])

    return tokens, blocks, [notes_col]


def _make_full_page_result() -> PageResult:
    """Build a PageResult with at least one instance of every region type."""
    tokens, blocks, notes_columns = _make_tokens_blocks_columns(page=2)

    graphic = GraphicElement(
        page=2,
        element_type="rect",
        x0=10,
        y0=10,
        x1=100,
        y1=100,
        stroke_color=(0, 0, 0),
        fill_color=(1, 1, 1),
        linewidth=2.0,
        pts=[(10, 10), (100, 10), (100, 100), (10, 100)],
    )

    legend_entry = LegendEntry(
        page=2,
        symbol=graphic,
        symbol_bbox=(10, 10, 30, 30),
        description="Test symbol",
        description_bbox=(35, 10, 100, 30),
    )
    legend_region = LegendRegion(
        page=2,
        header=blocks[0],
        entries=[legend_entry],
        is_boxed=True,
        box_bbox=(5, 5, 200, 200),
        confidence=0.85,
    )

    abbrev_entry = AbbreviationEntry(
        page=2,
        code="AI",
        meaning="Area Inlet",
        code_bbox=(50, 200, 70, 212),
        meaning_bbox=(80, 200, 160, 212),
    )
    abbrev_region = AbbreviationRegion(
        page=2,
        header=blocks[0],
        entries=[abbrev_entry],
        is_boxed=False,
        confidence=0.9,
    )

    revision_entry = RevisionEntry(
        page=2,
        number="1",
        description="Initial release",
        date="2026-01-15",
        row_bbox=(100, 300, 400, 320),
    )
    revision_region = RevisionRegion(
        page=2,
        header=blocks[0],
        entries=[revision_entry],
        is_boxed=True,
        box_bbox=(90, 290, 410, 330),
        confidence=0.95,
    )

    std_entry = StandardDetailEntry(
        page=2,
        sheet_number="SS-1",
        description="Standard curb detail",
        sheet_bbox=(50, 400, 100, 412),
        description_bbox=(110, 400, 300, 412),
    )
    std_region = StandardDetailRegion(
        page=2,
        header=blocks[0],
        subheader="SHALL BE USED ON THIS PROJECT:",
        subheader_bbox=(50, 380, 300, 396),
        entries=[std_entry],
        is_boxed=True,
        box_bbox=(40, 370, 310, 420),
        confidence=0.88,
    )

    misc_title = MiscTitleRegion(
        page=2,
        text="OKLAHOMA DOT",
        text_block=blocks[0],
        is_boxed=True,
        box_bbox=(10, 500, 200, 530),
        confidence=0.7,
    )

    structural_box = StructuralBox(
        page=2,
        x0=20,
        y0=20,
        x1=500,
        y1=700,
        box_type=BoxType.notes_box,
        confidence=0.92,
        contained_block_indices=[0, 1],
        contained_text="GENERAL NOTES",
        is_synthetic=False,
        polygon=[(20, 20), (500, 20), (500, 700), (20, 700)],
    )

    semantic_region = SemanticRegion(
        page=2,
        label="GENERAL NOTES",
        x0=20,
        y0=20,
        x1=500,
        y1=700,
        anchor_block=blocks[0],
        child_blocks=[blocks[1]],
        confidence=0.91,
    )

    title_block = TitleBlockInfo(
        page=2,
        bbox=(100, 1300, 2400, 1500),
        confidence=0.98,
        raw_text="PROJECT NAME: Test Project | SHEET: C-1",
        fields=[
            TitleBlockField(
                label="project_name",
                value="Test Project",
                bbox=(200, 1350, 500, 1370),
                confidence=0.95,
            ),
            TitleBlockField(
                label="sheet_number",
                value="C-1",
                bbox=(600, 1350, 700, 1370),
                confidence=0.99,
            ),
        ],
    )

    page_zone = PageZone(
        tag=ZoneTag.page,
        x0=0,
        y0=0,
        x1=2448,
        y1=1584,
        confidence=1.0,
        children=[
            PageZone(
                tag=ZoneTag.drawing, x0=20, y0=50, x1=2000, y1=1300, confidence=0.8
            ),
            PageZone(tag=ZoneTag.title_block, x0=100, y0=1300, x1=2400, y1=1500),
        ],
    )

    check_result = CheckResult(
        check_id="ABBREV_DUP",
        severity="warning",
        message="Duplicate abbreviation 'AI'",
        page=2,
        bbox=(50, 200, 160, 212),
        details={"code": "AI", "count": 2},
    )

    ocr_token = _box(200, 200, 250, 212, "OCR_WORD", 2)
    match_record = MatchRecord(
        ocr_box=ocr_token,
        pdf_box=tokens[0],
        match_type="iou",
        ocr_confidence=0.87,
    )
    reconcile_result = ReconcileResult(
        added_tokens=[ocr_token],
        all_ocr_tokens=[ocr_token],
        matches=[match_record],
        stats={"total_matches": 1, "iou_matches": 1},
    )

    stage_ingest = StageResult(
        stage="ingest",
        enabled=True,
        ran=True,
        status="success",
        duration_ms=150,
        counts={"pages": 1},
    )
    stage_tocr = StageResult(
        stage="tocr",
        enabled=True,
        ran=True,
        status="success",
        duration_ms=300,
        counts={"tokens": 4},
    )

    return PageResult(
        page=2,
        page_width=2448.0,
        page_height=1584.0,
        skew_degrees=0.15,
        page_quality=0.87,
        stages={"ingest": stage_ingest, "tocr": stage_tocr},
        tokens=tokens,
        blocks=blocks,
        notes_columns=notes_columns,
        graphics=[graphic],
        structural_boxes=[structural_box],
        semantic_regions=[semantic_region],
        abbreviation_regions=[abbrev_region],
        legend_regions=[legend_region],
        revision_regions=[revision_region],
        standard_detail_regions=[std_region],
        misc_title_regions=[misc_title],
        title_blocks=[title_block],
        page_zones=[page_zone],
        layout_predictions=[{"label": "header", "confidence": 0.9}],
        drift_warnings=["font_distribution_shift"],
        semantic_findings=[check_result],
        ocr_tokens=[ocr_token],
        ocr_confs=[0.87],
        reconcile_result=reconcile_result,
    )


# ── Individual model round-trip tests ──────────────────────────────────


class TestGraphicElementRoundTrip:
    def test_basic(self):
        g = GraphicElement(
            page=1,
            element_type="rect",
            x0=10,
            y0=20,
            x1=100,
            y1=200,
            stroke_color=(0, 0, 0),
            fill_color=(1, 0, 0),
            linewidth=2.5,
            pts=[(10, 20), (100, 200)],
        )
        restored = GraphicElement.from_dict(g.to_dict())
        assert restored.page == g.page
        assert restored.element_type == g.element_type
        assert restored.bbox() == pytest.approx(g.bbox(), abs=0.01)
        assert restored.stroke_color == g.stroke_color
        assert restored.fill_color == g.fill_color
        assert restored.linewidth == pytest.approx(g.linewidth, abs=0.01)
        assert len(restored.pts) == len(g.pts)

    def test_none_colors(self):
        g = GraphicElement(page=0, element_type="line", x0=0, y0=0, x1=50, y1=50)
        restored = GraphicElement.from_dict(g.to_dict())
        assert restored.stroke_color is None
        assert restored.fill_color is None
        assert restored.pts is None


class TestLegendEntryRoundTrip:
    def test_with_symbol(self):
        symbol = GraphicElement(page=1, element_type="rect", x0=10, y0=10, x1=30, y1=30)
        entry = LegendEntry(
            page=1,
            symbol=symbol,
            symbol_bbox=(10, 10, 30, 30),
            description="Test",
            description_bbox=(35, 10, 100, 30),
        )
        restored = LegendEntry.from_dict(entry.to_dict())
        assert restored.description == "Test"
        assert restored.symbol_bbox == (10, 10, 30, 30)
        assert restored.symbol is not None
        assert restored.symbol.element_type == "rect"


class TestAbbreviationEntryRoundTrip:
    def test_basic(self):
        entry = AbbreviationEntry(
            page=0,
            code="AI",
            meaning="Area Inlet",
            code_bbox=(10, 10, 30, 20),
            meaning_bbox=(35, 10, 100, 20),
        )
        restored = AbbreviationEntry.from_dict(entry.to_dict())
        assert restored.code == "AI"
        assert restored.meaning == "Area Inlet"
        assert restored.code_bbox == entry.code_bbox


class TestRevisionEntryRoundTrip:
    def test_basic(self):
        entry = RevisionEntry(
            page=0,
            number="1",
            description="Initial",
            date="2026-01-15",
            row_bbox=(10, 10, 200, 30),
        )
        restored = RevisionEntry.from_dict(entry.to_dict())
        assert restored.number == "1"
        assert restored.date == "2026-01-15"
        assert restored.row_bbox == entry.row_bbox


class TestStandardDetailEntryRoundTrip:
    def test_basic(self):
        entry = StandardDetailEntry(
            page=0,
            sheet_number="SS-1",
            description="Curb detail",
            sheet_bbox=(10, 10, 50, 20),
            description_bbox=(55, 10, 200, 20),
        )
        restored = StandardDetailEntry.from_dict(entry.to_dict())
        assert restored.sheet_number == "SS-1"
        assert restored.description == "Curb detail"


class TestStructuralBoxRoundTrip:
    def test_basic(self):
        sb = StructuralBox(
            page=1,
            x0=10,
            y0=10,
            x1=500,
            y1=700,
            box_type=BoxType.title_block,
            confidence=0.92,
            contained_block_indices=[0, 1, 2],
            contained_text="Title",
            is_synthetic=True,
            polygon=[(10, 10), (500, 10), (500, 700), (10, 700)],
        )
        restored = StructuralBox.from_dict(sb.to_dict())
        assert restored.box_type == BoxType.title_block
        assert restored.confidence == pytest.approx(0.92, abs=0.001)
        assert restored.contained_block_indices == [0, 1, 2]
        assert restored.is_synthetic is True
        assert len(restored.polygon) == 4

    def test_unknown_box_type(self):
        d = {"page": 0, "bbox": [0, 0, 100, 100], "box_type": "nonexistent"}
        restored = StructuralBox.from_dict(d)
        assert restored.box_type == BoxType.unknown


class TestPageZoneRoundTrip:
    def test_nested(self):
        zone = PageZone(
            tag=ZoneTag.page,
            x0=0,
            y0=0,
            x1=2448,
            y1=1584,
            children=[
                PageZone(tag=ZoneTag.drawing, x0=20, y0=50, x1=2000, y1=1300),
                PageZone(tag=ZoneTag.notes, x0=2050, y0=50, x1=2400, y1=1300),
            ],
        )
        restored = PageZone.from_dict(zone.to_dict())
        assert restored.tag == ZoneTag.page
        assert len(restored.children) == 2
        assert restored.children[0].tag == ZoneTag.drawing
        assert restored.children[1].tag == ZoneTag.notes


class TestCheckResultRoundTrip:
    def test_basic(self):
        cr = CheckResult(
            check_id="TEST_01",
            severity="warning",
            message="Test finding",
            page=3,
            bbox=(10, 10, 100, 50),
            details={"key": "value"},
        )
        restored = CheckResult.from_dict(cr.to_dict())
        assert restored.check_id == "TEST_01"
        assert restored.severity == "warning"
        assert restored.bbox == (10, 10, 100, 50)
        assert restored.details == {"key": "value"}


class TestTitleBlockInfoRoundTrip:
    def test_basic(self):
        tb = TitleBlockInfo(
            page=1,
            bbox=(100, 1300, 2400, 1500),
            confidence=0.98,
            raw_text="Sheet C-1",
            fields=[
                TitleBlockField("project_name", "Test", (200, 1350, 500, 1370), 0.95),
                TitleBlockField("sheet_number", "C-1", None, 0.99),
            ],
        )
        restored = TitleBlockInfo.from_dict(tb.to_dict())
        assert restored.page == 1
        assert restored.confidence == pytest.approx(0.98, abs=0.01)
        assert len(restored.fields) == 2
        assert restored.get("project_name") == "Test"
        assert restored.get("sheet_number") == "C-1"
        assert restored.fields[0].bbox is not None
        assert restored.fields[1].bbox is None


class TestReconcileResultRoundTrip:
    def test_basic(self):
        token = _box(10, 10, 50, 20, "test", 0)
        mr = MatchRecord(
            ocr_box=token, pdf_box=token, match_type="iou", ocr_confidence=0.9
        )
        rr = ReconcileResult(
            added_tokens=[token],
            all_ocr_tokens=[token],
            matches=[mr],
            stats={"total": 1},
        )
        restored = ReconcileResult.from_dict(rr.to_dict())
        assert len(restored.added_tokens) == 1
        assert len(restored.matches) == 1
        assert restored.matches[0].match_type == "iou"
        assert restored.stats == {"total": 1}


class TestSuspectRegionRoundTrip:
    def test_basic(self):
        sr = SuspectRegion(
            page=1,
            x0=10,
            y0=20,
            x1=100,
            y1=40,
            word_text="CONC",
            context="CONC WALK",
            reason="fused_word",
            source_label="header",
            block_index=3,
        )
        restored = SuspectRegion.from_dict(sr.to_dict())
        assert restored.word_text == "CONC"
        assert restored.reason == "fused_word"
        assert restored.bbox() == pytest.approx(sr.bbox(), abs=0.01)


class TestRowBandRoundTrip:
    def test_basic(self):
        rb = RowBand(
            page=0, boxes=[_box(10, 10, 50, 20, "A"), _box(60, 10, 100, 20, "B")]
        )
        restored = RowBand.from_dict(rb.to_dict())
        assert len(restored.boxes) == 2
        assert restored.boxes[0].text == "A"


class TestStageResultRoundTrip:
    def test_full(self):
        sr = StageResult(
            stage="tocr",
            enabled=True,
            ran=True,
            status="success",
            skip_reason=None,
            duration_ms=350,
            counts={"tokens": 100},
            inputs={"has_pdf": True},
            outputs={"deduped": 95},
            error=None,
        )
        restored = StageResult.from_dict(sr.to_dict())
        assert restored.stage == "tocr"
        assert restored.ran is True
        assert restored.duration_ms == 350
        assert restored.counts == {"tokens": 100}

    def test_failed(self):
        sr = StageResult(
            stage="vocr",
            enabled=True,
            ran=True,
            status="failed",
            error={"type": "RuntimeError", "message": "OCR failed"},
        )
        restored = StageResult.from_dict(sr.to_dict())
        assert restored.status == "failed"
        assert restored.error["type"] == "RuntimeError"


# ── PageResult round-trip ──────────────────────────────────────────────


class TestPageResultRoundTrip:
    def test_full_round_trip(self):
        pr = _make_full_page_result()
        data = pr.to_dict()

        # Should be JSON-serializable
        json_str = json.dumps(data)
        data_back = json.loads(json_str)

        restored = PageResult.from_dict(data_back)

        # Scalars
        assert restored.page == 2
        assert restored.page_width == pytest.approx(2448.0)
        assert restored.page_height == pytest.approx(1584.0)
        assert restored.skew_degrees == pytest.approx(0.15, abs=0.001)
        assert restored.page_quality == pytest.approx(0.87, abs=0.001)

    def test_core_artefact_counts(self):
        pr = _make_full_page_result()
        restored = PageResult.from_dict(pr.to_dict())

        assert len(restored.tokens) == len(pr.tokens)
        assert len(restored.blocks) == len(pr.blocks)
        assert len(restored.notes_columns) == len(pr.notes_columns)

    def test_token_text_preserved(self):
        pr = _make_full_page_result()
        restored = PageResult.from_dict(pr.to_dict())

        orig_texts = sorted(t.text for t in pr.tokens)
        rest_texts = sorted(t.text for t in restored.tokens)
        assert orig_texts == rest_texts

    def test_block_bbox_preserved(self):
        pr = _make_full_page_result()
        restored = PageResult.from_dict(pr.to_dict())

        for orig, rest in zip(pr.blocks, restored.blocks):
            assert rest.bbox() == pytest.approx(orig.bbox(), abs=0.01)

    def test_graphics_preserved(self):
        pr = _make_full_page_result()
        restored = PageResult.from_dict(pr.to_dict())

        assert len(restored.graphics) == 1
        assert restored.graphics[0].element_type == "rect"
        assert restored.graphics[0].stroke_color == (0, 0, 0)

    def test_region_counts(self):
        pr = _make_full_page_result()
        restored = PageResult.from_dict(pr.to_dict())

        assert len(restored.abbreviation_regions) == 1
        assert len(restored.legend_regions) == 1
        assert len(restored.revision_regions) == 1
        assert len(restored.standard_detail_regions) == 1
        assert len(restored.misc_title_regions) == 1
        assert len(restored.structural_boxes) == 1
        assert len(restored.semantic_regions) == 1
        assert len(restored.title_blocks) == 1

    def test_region_entries(self):
        pr = _make_full_page_result()
        restored = PageResult.from_dict(pr.to_dict())

        assert len(restored.abbreviation_regions[0].entries) == 1
        assert restored.abbreviation_regions[0].entries[0].code == "AI"

        assert len(restored.legend_regions[0].entries) == 1
        assert restored.legend_regions[0].entries[0].description == "Test symbol"

        assert len(restored.revision_regions[0].entries) == 1
        assert restored.revision_regions[0].entries[0].number == "1"

        assert len(restored.standard_detail_regions[0].entries) == 1
        assert restored.standard_detail_regions[0].entries[0].sheet_number == "SS-1"

    def test_title_block_fields(self):
        pr = _make_full_page_result()
        restored = PageResult.from_dict(pr.to_dict())

        tb = restored.title_blocks[0]
        assert tb.page == 2
        assert tb.get("project_name") == "Test Project"
        assert tb.get("sheet_number") == "C-1"

    def test_page_zones_nested(self):
        pr = _make_full_page_result()
        restored = PageResult.from_dict(pr.to_dict())

        assert len(restored.page_zones) == 1
        root = restored.page_zones[0]
        assert root.tag == ZoneTag.page
        assert len(root.children) == 2

    def test_stages_preserved(self):
        pr = _make_full_page_result()
        restored = PageResult.from_dict(pr.to_dict())

        assert "ingest" in restored.stages
        assert "tocr" in restored.stages
        assert restored.stages["ingest"].status == "success"
        assert restored.stages["tocr"].duration_ms == 300

    def test_semantic_findings(self):
        pr = _make_full_page_result()
        restored = PageResult.from_dict(pr.to_dict())

        assert len(restored.semantic_findings) == 1
        assert restored.semantic_findings[0].check_id == "ABBREV_DUP"

    def test_ocr_artefacts(self):
        pr = _make_full_page_result()
        restored = PageResult.from_dict(pr.to_dict())

        assert restored.ocr_tokens is not None
        assert len(restored.ocr_tokens) == 1
        assert restored.ocr_confs == [0.87]

    def test_reconcile_result(self):
        pr = _make_full_page_result()
        restored = PageResult.from_dict(pr.to_dict())

        assert restored.reconcile_result is not None
        assert len(restored.reconcile_result.added_tokens) == 1
        assert restored.reconcile_result.stats["total_matches"] == 1

    def test_layout_predictions_and_drift(self):
        pr = _make_full_page_result()
        restored = PageResult.from_dict(pr.to_dict())

        assert restored.layout_predictions == [{"label": "header", "confidence": 0.9}]
        assert restored.drift_warnings == ["font_distribution_shift"]

    def test_empty_page_result(self):
        pr = PageResult(page=0, page_width=100, page_height=200)
        restored = PageResult.from_dict(pr.to_dict())
        assert restored.page == 0
        assert len(restored.tokens) == 0
        assert len(restored.blocks) == 0
        assert restored.reconcile_result is None

    def test_json_serializable(self):
        """Ensure to_dict() output is fully JSON-serializable."""
        pr = _make_full_page_result()
        data = pr.to_dict()
        # This should not raise
        json_str = json.dumps(data, default=str)
        assert len(json_str) > 100


# ── DocumentResult round-trip ──────────────────────────────────────────


class TestDocumentResultRoundTrip:
    def test_basic(self):
        pr = _make_full_page_result()
        doc = DocumentResult(
            pdf_path=None,
            pages=[pr],
            document_findings=[
                CheckResult(check_id="DOC_01", severity="info", message="Test")
            ],
            config=GroupingConfig(),
        )
        data = doc.to_dict()
        json_str = json.dumps(data, default=str)
        data_back = json.loads(json_str)
        restored = DocumentResult.from_dict(data_back)

        assert len(restored.pages) == 1
        assert restored.pages[0].page == 2
        assert len(restored.document_findings) == 1
        assert restored.document_findings[0].check_id == "DOC_01"

    def test_with_config(self):
        cfg = GroupingConfig(enable_vocr=True, llm_provider="anthropic")
        doc = DocumentResult(pages=[], config=cfg)
        restored = DocumentResult.from_dict(doc.to_dict())
        assert restored.config is not None
        assert restored.config.enable_vocr is True
        assert restored.config.llm_provider == "anthropic"
