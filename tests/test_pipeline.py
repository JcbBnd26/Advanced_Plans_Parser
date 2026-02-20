"""Tests for plancheck.pipeline — gating, stage results, fingerprinting, PageResult, DocumentResult."""

import pytest

from plancheck.config import GroupingConfig
from plancheck.pipeline import (
    STAGE_ORDER,
    DocumentResult,
    PageResult,
    SkipReason,
    StageResult,
    _run_document_checks,
    gate,
    input_fingerprint,
)


class TestStageOrder:
    def test_all_stages_present(self):
        assert "ingest" in STAGE_ORDER
        assert "tocr" in STAGE_ORDER
        assert "vocrpp" in STAGE_ORDER
        assert "vocr" in STAGE_ORDER
        assert "reconcile" in STAGE_ORDER

    def test_correct_order(self):
        assert STAGE_ORDER.index("ingest") < STAGE_ORDER.index("tocr")
        assert STAGE_ORDER.index("tocr") < STAGE_ORDER.index("vocr")
        assert STAGE_ORDER.index("vocr") < STAGE_ORDER.index("reconcile")


class TestGate:
    def test_ingest_always_runs(self):
        cfg = GroupingConfig()
        should_run, reason = gate("ingest", cfg)
        assert should_run is True
        assert reason is None

    def test_tocr_disabled(self):
        cfg = GroupingConfig(enable_tocr=False)
        should_run, reason = gate("tocr", cfg)
        assert should_run is False
        assert reason == SkipReason.disabled_by_config.value

    def test_tocr_enabled(self):
        cfg = GroupingConfig(enable_tocr=True)
        should_run, _ = gate("tocr", cfg)
        assert should_run is True

    def test_vocr_disabled(self):
        cfg = GroupingConfig(enable_vocr=False)
        should_run, reason = gate("vocr", cfg)
        assert should_run is False
        assert reason == SkipReason.disabled_by_config.value

    def test_vocrpp_requires_vocr(self):
        cfg = GroupingConfig(enable_ocr_preprocess=True, enable_vocr=False)
        should_run, reason = gate("vocrpp", cfg)
        assert should_run is False

    def test_reconcile_requires_vocr(self):
        cfg = GroupingConfig(enable_ocr_reconcile=True, enable_vocr=False)
        should_run, reason = gate("reconcile", cfg)
        assert should_run is False

    def test_unknown_stage(self):
        cfg = GroupingConfig()
        should_run, reason = gate("nonexistent", cfg)
        assert should_run is False
        assert reason == SkipReason.not_applicable.value


class TestStageResult:
    def test_to_dict_minimal(self):
        sr = StageResult(stage="test")
        d = sr.to_dict()
        assert d["stage"] == "test"
        assert d["status"] == "skipped"
        assert d["ran"] is False

    def test_to_dict_full(self):
        sr = StageResult(
            stage="tocr",
            enabled=True,
            ran=True,
            status="success",
            duration_ms=42,
            counts={"tokens": 100},
            error=None,
        )
        d = sr.to_dict()
        assert d["status"] == "success"
        assert d["duration_ms"] == 42
        assert d["counts"]["tokens"] == 100

    def test_to_dict_with_error(self):
        sr = StageResult(stage="vocr", error={"message": "boom"})
        d = sr.to_dict()
        assert d["error"]["message"] == "boom"


class TestInputFingerprint:
    def test_same_inputs_same_hash(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake pdf content")
        cfg = GroupingConfig()
        fp1 = input_fingerprint(pdf, [0, 1], cfg)
        fp2 = input_fingerprint(pdf, [0, 1], cfg)
        assert fp1 == fp2

    def test_different_pages_different_hash(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake pdf content")
        cfg = GroupingConfig()
        fp1 = input_fingerprint(pdf, [0], cfg)
        fp2 = input_fingerprint(pdf, [0, 1], cfg)
        assert fp1 != fp2

    def test_different_config_different_hash(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake pdf content")
        cfg1 = GroupingConfig(iou_prune=0.5)
        cfg2 = GroupingConfig(iou_prune=0.9)
        fp1 = input_fingerprint(pdf, [0], cfg1)
        fp2 = input_fingerprint(pdf, [0], cfg2)
        assert fp1 != fp2


# ══════════════════════════════════════════════════════════════════════
# PageResult and DocumentResult
# ══════════════════════════════════════════════════════════════════════


class TestPageResult:
    def test_defaults(self):
        pr = PageResult()
        assert pr.page == 0
        assert pr.tokens == []
        assert pr.blocks == []
        assert pr.semantic_findings == []

    def test_to_summary_dict(self):
        pr = PageResult(page=1, page_width=612, page_height=792)
        d = pr.to_summary_dict()
        assert d["page"] == 1
        assert d["page_width"] == 612
        assert "counts" in d
        assert d["counts"]["tokens"] == 0

    def test_summary_dict_with_findings(self):
        from plancheck.checks.semantic_checks import CheckResult

        finding = CheckResult(check_id="TEST", severity="warning", message="msg")
        pr = PageResult(page=0, semantic_findings=[finding])
        d = pr.to_summary_dict()
        assert d["counts"]["semantic_findings"] == 1
        assert len(d["semantic_findings"]) == 1


class TestDocumentResult:
    def test_defaults(self):
        dr = DocumentResult()
        assert dr.pages == []
        assert dr.document_findings == []
        assert dr.total_findings() == 0

    def test_total_findings(self):
        from plancheck.checks.semantic_checks import CheckResult

        f1 = CheckResult(check_id="A", severity="error", message="a")
        f2 = CheckResult(check_id="B", severity="warning", message="b")
        pr = PageResult(page=0, semantic_findings=[f1])
        dr = DocumentResult(pages=[pr], document_findings=[f2])
        assert dr.total_findings() == 2

    def test_to_summary_dict(self):
        dr = DocumentResult()
        d = dr.to_summary_dict()
        assert d["pages_processed"] == 0
        assert d["document_findings"] == 0


# ══════════════════════════════════════════════════════════════════════
# Cross-page document checks
# ══════════════════════════════════════════════════════════════════════


def _make_tb(sheet_number, project_name="Test"):
    from plancheck.analysis.title_block import TitleBlockField, TitleBlockInfo

    return TitleBlockInfo(
        page=0,
        fields=[
            TitleBlockField(label="sheet_number", value=sheet_number),
            TitleBlockField(label="project_name", value=project_name),
        ],
    )


def _make_abbrev_entry(code, meaning, page=0):
    from plancheck.models import AbbreviationEntry

    return AbbreviationEntry(page=page, code=code, meaning=meaning)


def _make_abbrev_region(entries, page=0):
    from plancheck.models import AbbreviationRegion

    return AbbreviationRegion(page=page, entries=entries)


class TestDocumentChecks:
    def test_empty_pages(self):
        assert _run_document_checks([]) == []

    def test_duplicate_sheet_numbers(self):
        pr1 = PageResult(page=0, title_blocks=[_make_tb("C-1")])
        pr2 = PageResult(page=1, title_blocks=[_make_tb("C-1")])
        findings = _run_document_checks([pr1, pr2])
        ids = {f.check_id for f in findings}
        assert "DOC_DUP_SHEET" in ids

    def test_unique_sheet_numbers_ok(self):
        pr1 = PageResult(page=0, title_blocks=[_make_tb("C-1")])
        pr2 = PageResult(page=1, title_blocks=[_make_tb("C-2")])
        findings = _run_document_checks([pr1, pr2])
        dup_findings = [f for f in findings if f.check_id == "DOC_DUP_SHEET"]
        assert dup_findings == []

    def test_abbreviation_conflict(self):
        pr1 = PageResult(
            page=0,
            abbreviation_regions=[
                _make_abbrev_region([_make_abbrev_entry("AI", "AREA INLET")])
            ],
        )
        pr2 = PageResult(
            page=1,
            abbreviation_regions=[
                _make_abbrev_region([_make_abbrev_entry("AI", "AIR INTAKE")])
            ],
        )
        findings = _run_document_checks([pr1, pr2])
        ids = {f.check_id for f in findings}
        assert "DOC_ABBREV_CONFLICT" in ids

    def test_consistent_abbreviations_ok(self):
        pr1 = PageResult(
            page=0,
            abbreviation_regions=[
                _make_abbrev_region([_make_abbrev_entry("AI", "AREA INLET")])
            ],
        )
        pr2 = PageResult(
            page=1,
            abbreviation_regions=[
                _make_abbrev_region([_make_abbrev_entry("AI", "AREA INLET")])
            ],
        )
        findings = _run_document_checks([pr1, pr2])
        conflict = [f for f in findings if f.check_id == "DOC_ABBREV_CONFLICT"]
        assert conflict == []

    def test_inconsistent_title_blocks(self):
        pr1 = PageResult(page=0, title_blocks=[_make_tb("C-1")])
        pr2 = PageResult(page=1, title_blocks=[])  # no title block
        findings = _run_document_checks([pr1, pr2])
        ids = {f.check_id for f in findings}
        assert "DOC_TITLE_INCONSISTENT" in ids

    def test_all_missing_title_blocks_no_flag(self):
        """If NO page has title blocks, don't flag inconsistency."""
        pr1 = PageResult(page=0, title_blocks=[])
        pr2 = PageResult(page=1, title_blocks=[])
        findings = _run_document_checks([pr1, pr2])
        inconsistent = [f for f in findings if f.check_id == "DOC_TITLE_INCONSISTENT"]
        assert inconsistent == []

    def test_legend_conflict(self):
        from plancheck.models import GraphicElement, LegendEntry, LegendRegion

        sym = GraphicElement(page=0, element_type="rect", x0=10, y0=10, x1=20, y1=20)
        e1 = LegendEntry(
            page=0, symbol=sym, symbol_bbox=(10, 10, 20, 20), description="Storm Sewer"
        )
        e2 = LegendEntry(
            page=1,
            symbol=sym,
            symbol_bbox=(10, 10, 20, 20),
            description="Sanitary Sewer",
        )
        pr1 = PageResult(page=0, legend_regions=[LegendRegion(page=0, entries=[e1])])
        pr2 = PageResult(page=1, legend_regions=[LegendRegion(page=1, entries=[e2])])
        findings = _run_document_checks([pr1, pr2])
        ids = {f.check_id for f in findings}
        assert "DOC_LEGEND_CONFLICT" in ids

    def test_consistent_legends_ok(self):
        from plancheck.models import GraphicElement, LegendEntry, LegendRegion

        sym = GraphicElement(page=0, element_type="rect", x0=10, y0=10, x1=20, y1=20)
        e1 = LegendEntry(
            page=0, symbol=sym, symbol_bbox=(10, 10, 20, 20), description="Storm Sewer"
        )
        e2 = LegendEntry(
            page=1, symbol=sym, symbol_bbox=(10, 10, 20, 20), description="Storm Sewer"
        )
        pr1 = PageResult(page=0, legend_regions=[LegendRegion(page=0, entries=[e1])])
        pr2 = PageResult(page=1, legend_regions=[LegendRegion(page=1, entries=[e2])])
        findings = _run_document_checks([pr1, pr2])
        assert all(f.check_id != "DOC_LEGEND_CONFLICT" for f in findings)

    def test_revision_gap(self):
        from plancheck.models import RevisionEntry, RevisionRegion

        r1 = RevisionEntry(page=0, number="1", description="Initial")
        r3 = RevisionEntry(page=0, number="3", description="Update")
        pr = PageResult(
            page=0,
            revision_regions=[RevisionRegion(page=0, entries=[r1, r3])],
        )
        findings = _run_document_checks([pr])
        ids = {f.check_id for f in findings}
        assert "DOC_REVISION_GAP" in ids
        gap_f = [f for f in findings if f.check_id == "DOC_REVISION_GAP"][0]
        assert 2 in gap_f.details["missing"]

    def test_revision_sequence_ok(self):
        from plancheck.models import RevisionEntry, RevisionRegion

        entries = [
            RevisionEntry(page=0, number="1"),
            RevisionEntry(page=0, number="2"),
            RevisionEntry(page=0, number="3"),
        ]
        pr = PageResult(
            page=0,
            revision_regions=[RevisionRegion(page=0, entries=entries)],
        )
        findings = _run_document_checks([pr])
        assert all(f.check_id != "DOC_REVISION_GAP" for f in findings)

    def test_low_quality_flagged(self):
        """Pages significantly below average quality are flagged."""
        pr1 = PageResult(page=0, page_quality=0.90)
        pr2 = PageResult(page=1, page_quality=0.85)
        pr3 = PageResult(page=2, page_quality=0.20)  # Low outlier
        findings = _run_document_checks([pr1, pr2, pr3])
        ids = {f.check_id for f in findings}
        assert "DOC_LOW_QUALITY" in ids

    def test_uniform_quality_ok(self):
        """Uniform quality scores don't trigger the check."""
        pr1 = PageResult(page=0, page_quality=0.85)
        pr2 = PageResult(page=1, page_quality=0.80)
        pr3 = PageResult(page=2, page_quality=0.82)
        findings = _run_document_checks([pr1, pr2, pr3])
        assert all(f.check_id != "DOC_LOW_QUALITY" for f in findings)
