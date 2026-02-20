"""Tests for plancheck.semantic_checks module."""

from __future__ import annotations

import pytest

from plancheck.checks.semantic_checks import (
    CheckResult,
    _parse_date,
    check_abbreviation_duplicates,
    check_abbreviation_empty,
    check_abbreviations_undefined,
    check_legend_empty,
    check_legend_no_header,
    check_notes_cross_references,
    check_notes_no_header,
    check_notes_numbering,
    check_revision_date_order,
    check_revision_empty,
    check_revision_missing_number,
    check_standard_detail_duplicates,
    check_standard_detail_missing_desc,
    check_title_block_fields,
    check_title_block_missing,
    run_all_checks,
)
from plancheck.models import (
    AbbreviationEntry,
    AbbreviationRegion,
    BlockCluster,
    GlyphBox,
    LegendEntry,
    LegendRegion,
    NotesColumn,
    RevisionEntry,
    RevisionRegion,
    RowBand,
    StandardDetailEntry,
    StandardDetailRegion,
)

# ── Helpers ────────────────────────────────────────────────────────────


def _box(x0, y0, x1, y1, text="", page=0):
    return GlyphBox(page=page, x0=x0, y0=y0, x1=x1, y1=y1, text=text)


def _block(texts, page=0, is_notes=False, is_header=False):
    """Build a BlockCluster from [(x0, y0, x1, y1, text), ...]."""
    from collections import defaultdict

    row_map = defaultdict(list)
    for x0, y0, x1, y1, text in texts:
        row_map[y0].append(_box(x0, y0, x1, y1, text, page=page))
    rows = [RowBand(page=page, boxes=boxes) for _, boxes in sorted(row_map.items())]
    blk = BlockCluster(page=page, rows=rows, is_notes=is_notes, is_header=is_header)
    return blk


def _notes_column(note_numbers, header_text="NOTES", page=0):
    """Build a NotesColumn with notes blocks numbered as given."""
    header_blk = _block([(0, 0, 100, 12, header_text)], page=page, is_header=True)
    notes_blocks = []
    for i, num in enumerate(note_numbers):
        y = 20 + i * 20
        blk = _block(
            [
                (10, y, 30, y + 10, f"{num}."),
                (35, y, 200, y + 10, f"Note text for item {num}"),
                (10, y + 12, 200, y + 22, "Continuation of note text."),
            ],
            page=page,
            is_notes=True,
        )
        notes_blocks.append(blk)
    return NotesColumn(page=page, header=header_blk, notes_blocks=notes_blocks)


def _abbrev_region(entries_dict, page=0, is_boxed=True):
    """Build an AbbreviationRegion from {code: meaning} dict."""
    entries = [
        AbbreviationEntry(
            page=page,
            code=code,
            meaning=meaning,
            code_bbox=(0, 0, 30, 10),
            meaning_bbox=(40, 0, 200, 10),
        )
        for code, meaning in entries_dict.items()
    ]
    header = _block([(0, 0, 100, 12, "ABBREVIATIONS")], page=page, is_header=True)
    return AbbreviationRegion(
        page=page,
        header=header,
        entries=entries,
        is_boxed=is_boxed,
    )


def _revision_region(entries_list, page=0):
    """Build a RevisionRegion from [(number, description, date), ...]."""
    entries = [
        RevisionEntry(
            page=page,
            number=num,
            description=desc,
            date=date,
            row_bbox=(0, i * 15, 300, (i + 1) * 15),
        )
        for i, (num, desc, date) in enumerate(entries_list)
    ]
    header = _block([(0, 0, 100, 12, "REVISIONS")], page=page, is_header=True)
    return RevisionRegion(page=page, header=header, entries=entries, is_boxed=True)


def _std_detail_region(entries_list, page=0):
    """Build a StandardDetailRegion from [(sheet_number, description), ...]."""
    entries = [
        StandardDetailEntry(
            page=page,
            sheet_number=sheet,
            description=desc,
            sheet_bbox=(0, i * 15, 50, (i + 1) * 15),
        )
        for i, (sheet, desc) in enumerate(entries_list)
    ]
    header = _block([(0, 0, 200, 12, "STANDARD DETAILS")], page=page, is_header=True)
    return StandardDetailRegion(page=page, header=header, entries=entries)


# ══════════════════════════════════════════════════════════════════════
# 1. Notes numbering
# ══════════════════════════════════════════════════════════════════════


class TestNotesNumbering:
    def test_sequential_notes_clean(self):
        """No findings for properly numbered notes (1, 2, 3)."""
        col = _notes_column([1, 2, 3])
        findings = check_notes_numbering([col], [], page=1)
        assert findings == []

    def test_gap_detected(self):
        """Missing note number produces NOTES_GAP."""
        col = _notes_column([1, 2, 4])
        findings = check_notes_numbering([col], [], page=1)
        ids = [f.check_id for f in findings]
        assert "NOTES_GAP" in ids
        gap = next(f for f in findings if f.check_id == "NOTES_GAP")
        assert 3 in gap.details["missing"]

    def test_duplicate_detected(self):
        """Duplicate note number produces NOTES_DUP."""
        col = _notes_column([1, 2, 2, 3])
        findings = check_notes_numbering([col], [], page=1)
        ids = [f.check_id for f in findings]
        assert "NOTES_DUP" in ids

    def test_out_of_order(self):
        """Notes not in ascending order produce NOTES_ORDER."""
        col = _notes_column([1, 3, 2])
        findings = check_notes_numbering([col], [], page=1)
        ids = [f.check_id for f in findings]
        assert "NOTES_ORDER" in ids

    def test_empty_column_skipped(self):
        """Column with no numbered notes produces no findings."""
        col = NotesColumn(page=0, header=None, notes_blocks=[])
        findings = check_notes_numbering([col], [], page=0)
        assert findings == []

    def test_single_note(self):
        """Single note is never a gap or ordering error."""
        col = _notes_column([1])
        findings = check_notes_numbering([col], [], page=1)
        assert findings == []

    def test_large_gap(self):
        """Large gap (1, 10) correctly reports all missing numbers."""
        col = _notes_column([1, 10])
        findings = check_notes_numbering([col], [], page=1)
        gap = next(f for f in findings if f.check_id == "NOTES_GAP")
        assert gap.details["missing"] == list(range(2, 10))

    def test_severity(self):
        """Gap is warning, duplicate is error."""
        col = _notes_column([1, 2, 2, 5])
        findings = check_notes_numbering([col], [], page=1)
        for f in findings:
            if f.check_id == "NOTES_DUP":
                assert f.severity == "error"
            if f.check_id == "NOTES_GAP":
                assert f.severity == "warning"


# ══════════════════════════════════════════════════════════════════════
# 2. Abbreviation duplicates
# ══════════════════════════════════════════════════════════════════════


class TestAbbreviationDuplicates:
    def test_no_duplicates(self):
        """No findings when all codes are unique."""
        region = _abbrev_region({"AI": "AREA INLET", "BOC": "BACK OF CURB"})
        findings = check_abbreviation_duplicates([region], page=1)
        assert findings == []

    def test_same_meaning_is_ok(self):
        """Same code with same meaning (across regions) is not flagged."""
        r1 = _abbrev_region({"AI": "AREA INLET"})
        r2 = _abbrev_region({"AI": "AREA INLET"})
        findings = check_abbreviation_duplicates([r1, r2], page=1)
        assert findings == []

    def test_conflicting_meanings(self):
        """Same code with different meanings is an error."""
        r1 = _abbrev_region({"AI": "AREA INLET"})
        r2 = _abbrev_region({"AI": "AS INDICATED"})
        findings = check_abbreviation_duplicates([r1, r2], page=1)
        assert len(findings) == 1
        assert findings[0].check_id == "ABBREV_DUP"
        assert findings[0].severity == "error"

    def test_case_insensitive(self):
        """Codes are compared case-insensitively."""
        r1 = _abbrev_region({"ai": "AREA INLET"})
        r2 = _abbrev_region({"AI": "As Indicated"})
        findings = check_abbreviation_duplicates([r1, r2], page=1)
        assert len(findings) == 1

    def test_empty_regions(self):
        findings = check_abbreviation_duplicates([], page=1)
        assert findings == []


# ══════════════════════════════════════════════════════════════════════
# 3. Revision date ordering
# ══════════════════════════════════════════════════════════════════════


class TestRevisionDateOrder:
    def test_ascending_ok(self):
        """Ascending dates produce no findings."""
        region = _revision_region(
            [
                ("1", "First", "01/15/2024"),
                ("2", "Second", "03/20/2024"),
                ("3", "Third", "06/01/2024"),
            ]
        )
        findings = check_revision_date_order([region], page=1)
        assert findings == []

    def test_descending_ok(self):
        """Descending dates produce no findings (newest first)."""
        region = _revision_region(
            [
                ("3", "Third", "06/01/2024"),
                ("2", "Second", "03/20/2024"),
                ("1", "First", "01/15/2024"),
            ]
        )
        findings = check_revision_date_order([region], page=1)
        assert findings == []

    def test_out_of_order(self):
        """Mixed-order dates produce REV_DATE_ORDER warning."""
        region = _revision_region(
            [
                ("1", "First", "01/15/2024"),
                ("2", "Second", "06/01/2024"),
                ("3", "Third", "03/20/2024"),  # ← out of order
            ]
        )
        findings = check_revision_date_order([region], page=1)
        assert len(findings) == 1
        assert findings[0].check_id == "REV_DATE_ORDER"
        assert findings[0].severity == "warning"

    def test_single_entry_skipped(self):
        """Single revision entry can't be out of order."""
        region = _revision_region([("1", "Only", "01/15/2024")])
        findings = check_revision_date_order([region], page=1)
        assert findings == []

    def test_unparseable_dates_skipped(self):
        """Entries with unparseable dates are silently skipped."""
        region = _revision_region(
            [
                ("1", "First", "TBD"),
                ("2", "Second", "N/A"),
            ]
        )
        findings = check_revision_date_order([region], page=1)
        assert findings == []

    def test_mixed_formats(self):
        """Different date formats are handled."""
        region = _revision_region(
            [
                ("1", "First", "01/15/2024"),
                ("2", "Second", "2024-03-20"),
            ]
        )
        findings = check_revision_date_order([region], page=1)
        assert findings == []  # Ascending is fine

    def test_two_digit_year(self):
        """Two-digit years are interpreted as 20xx."""
        region = _revision_region(
            [
                ("1", "First", "01/15/24"),
                ("2", "Second", "03/20/24"),
            ]
        )
        findings = check_revision_date_order([region], page=1)
        assert findings == []


# ══════════════════════════════════════════════════════════════════════
# 4. Abbreviation used but not defined
# ══════════════════════════════════════════════════════════════════════


class TestAbbreviationsUndefined:
    def test_all_defined(self):
        """No findings when all abbreviations in notes are defined."""
        abbrev = _abbrev_region({"AI": "AREA INLET", "BOC": "BACK OF CURB"})
        col = _notes_column([1, 2])
        # Replace note text with known abbreviations
        for blk in col.notes_blocks:
            blk.rows[0].boxes[1] = _box(35, 0, 200, 10, "AI per BOC specs")
        findings = check_abbreviations_undefined([abbrev], [col], [], page=1)
        assert findings == []

    def test_undefined_flagged(self):
        """Undefined abbreviation-like tokens produce ABBREV_UNDEF."""
        abbrev = _abbrev_region({"AI": "AREA INLET"})
        col = _notes_column([1])
        col.notes_blocks[0].rows[0].boxes[1] = _box(
            35, 0, 200, 10, "See TYP and RCP details"
        )
        findings = check_abbreviations_undefined([abbrev], [col], [], page=1)
        codes = [f.details["token"] for f in findings]
        assert "RCP" in codes  # Not defined, not a common word

    def test_common_words_excluded(self):
        """Common English words are not flagged."""
        abbrev = _abbrev_region({"AI": "AREA INLET"})
        col = _notes_column([1])
        col.notes_blocks[0].rows[0].boxes[1] = _box(
            35, 0, 200, 10, "THE PLAN FOR THIS PROJECT"
        )
        findings = check_abbreviations_undefined([abbrev], [col], [], page=1)
        assert findings == []

    def test_no_abbreviation_table_skips(self):
        """When no abbreviation table exists, check is skipped entirely."""
        col = _notes_column([1])
        findings = check_abbreviations_undefined([], [col], [], page=1)
        assert findings == []

    def test_severity_is_info(self):
        """Undefined abbreviations are informational, not errors."""
        abbrev = _abbrev_region({"AI": "AREA INLET"})
        col = _notes_column([1])
        col.notes_blocks[0].rows[0].boxes[1] = _box(
            35, 0, 200, 10, "Install RCP per XYZ standard"
        )
        findings = check_abbreviations_undefined([abbrev], [col], [], page=1)
        for f in findings:
            assert f.severity == "info"


# ══════════════════════════════════════════════════════════════════════
# 5. Standard detail duplicates
# ══════════════════════════════════════════════════════════════════════


class TestStandardDetailDuplicates:
    def test_no_duplicates(self):
        region = _std_detail_region(
            [
                ("SS-1", "STORM SEWER PLAN"),
                ("SS-2", "STORM SEWER PROFILE"),
            ]
        )
        findings = check_standard_detail_duplicates([region], page=1)
        assert findings == []

    def test_duplicate_detected(self):
        region = _std_detail_region(
            [
                ("SS-1", "STORM SEWER PLAN"),
                ("SS-1", "STORM SEWER PLAN (REVISED)"),
            ]
        )
        findings = check_standard_detail_duplicates([region], page=1)
        assert len(findings) == 1
        assert findings[0].check_id == "STDDET_DUP"

    def test_case_insensitive(self):
        region = _std_detail_region(
            [
                ("ss-1", "STORM SEWER PLAN"),
                ("SS-1", "STORM SEWER PLAN"),
            ]
        )
        findings = check_standard_detail_duplicates([region], page=1)
        assert len(findings) == 1

    def test_across_regions(self):
        r1 = _std_detail_region([("621-1", "GUARDRAIL DETAILS")])
        r2 = _std_detail_region([("621-1", "GUARDRAIL REVISION")])
        findings = check_standard_detail_duplicates([r1, r2], page=1)
        assert len(findings) == 1


# ══════════════════════════════════════════════════════════════════════
# 6. Date parsing
# ══════════════════════════════════════════════════════════════════════


class TestDateParsing:
    def test_slash_mdy(self):
        dt = _parse_date("01/15/2024")
        assert dt is not None
        assert dt.month == 1 and dt.day == 15 and dt.year == 2024

    def test_dash_mdy(self):
        dt = _parse_date("01-15-2024")
        assert dt is not None
        assert dt.month == 1 and dt.day == 15

    def test_iso(self):
        dt = _parse_date("2024-03-20")
        assert dt is not None
        assert dt.month == 3 and dt.day == 20

    def test_two_digit_year(self):
        dt = _parse_date("01/15/24")
        assert dt is not None
        assert dt.year == 2024

    def test_empty_returns_none(self):
        assert _parse_date("") is None
        assert _parse_date("   ") is None
        assert _parse_date("TBD") is None

    def test_month_name(self):
        dt = _parse_date("15 Jan 2024")
        assert dt is not None
        assert dt.month == 1

    def test_month_name_long(self):
        dt = _parse_date("January 15, 2024")
        assert dt is not None
        assert dt.month == 1 and dt.day == 15 and dt.year == 2024


# ══════════════════════════════════════════════════════════════════════
# 7. run_all_checks orchestrator
# ══════════════════════════════════════════════════════════════════════


class TestRunAllChecks:
    def test_empty_inputs_produce_no_findings(self):
        findings = run_all_checks(page=1)
        # With no structural boxes at all, the only finding should be
        # TITLE_MISSING (warns that title block may be absent).
        assert len(findings) == 1
        assert findings[0].check_id == "TITLE_MISSING"

    def test_combines_findings(self):
        """Multiple checks produce combined findings."""
        col = _notes_column([1, 2, 2, 4])  # dup + gap
        abbrev = _abbrev_region({"AI": "AREA INLET"})
        findings = run_all_checks(
            notes_columns=[col],
            abbreviation_regions=[abbrev],
            page=1,
        )
        ids = {f.check_id for f in findings}
        assert "NOTES_DUP" in ids
        assert "NOTES_GAP" in ids

    def test_check_result_to_dict(self):
        r = CheckResult(
            check_id="TEST",
            severity="warning",
            message="Test message",
            page=1,
            bbox=(0, 0, 100, 100),
            details={"key": "value"},
        )
        d = r.to_dict()
        assert d["check_id"] == "TEST"
        assert d["bbox"] == [0, 0, 100, 100]
        assert d["details"]["key"] == "value"

    def test_check_result_no_optional(self):
        r = CheckResult(check_id="TEST", severity="info", message="msg")
        d = r.to_dict()
        assert "bbox" not in d
        assert "details" not in d


# ══════════════════════════════════════════════════════════════════════
# 8. New checks: legend_empty, legend_no_header
# ══════════════════════════════════════════════════════════════════════


class TestCheckLegendEmpty:
    def test_legend_with_entries_no_finding(self):
        region = LegendRegion(
            page=0,
            entries=[LegendEntry(page=0, description="Item A")],
        )
        assert check_legend_empty([region]) == []

    def test_legend_empty_flagged(self):
        region = LegendRegion(page=0, entries=[])
        findings = check_legend_empty([region])
        assert len(findings) == 1
        assert findings[0].check_id == "LEGEND_EMPTY"

    def test_multiple_legends_mixed(self):
        good = LegendRegion(page=0, entries=[LegendEntry(page=0, description="X")])
        empty = LegendRegion(page=0, entries=[])
        findings = check_legend_empty([good, empty])
        assert len(findings) == 1


class TestCheckLegendNoHeader:
    def test_legend_with_header_ok(self):
        blk = _block([(10, 10, 100, 22, "LEGEND")])
        region = LegendRegion(
            page=0,
            header=blk,
            entries=[LegendEntry(page=0, description="X")],
        )
        assert check_legend_no_header([region]) == []

    def test_legend_no_header_flagged(self):
        region = LegendRegion(
            page=0,
            header=None,
            entries=[LegendEntry(page=0, description="X")],
        )
        findings = check_legend_no_header([region])
        assert len(findings) == 1
        assert findings[0].check_id == "LEGEND_NO_HEADER"

    def test_legend_no_header_no_entries_ok(self):
        """No entries + no header → not flagged (already caught by LEGEND_EMPTY)."""
        region = LegendRegion(page=0, header=None, entries=[])
        assert check_legend_no_header([region]) == []


# ══════════════════════════════════════════════════════════════════════
# 9. Notes column no header
# ══════════════════════════════════════════════════════════════════════


class TestCheckNotesNoHeader:
    def test_notes_with_header_ok(self):
        hdr = _block([(10, 10, 100, 22, "GENERAL NOTES")])
        nblk = _block([(10, 30, 100, 42, "1. Do stuff")])
        col = NotesColumn(page=0, header=hdr, notes_blocks=[nblk])
        assert check_notes_no_header([col]) == []

    def test_notes_no_header_flagged(self):
        nblk = _block([(10, 30, 100, 42, "1. Do stuff")])
        col = NotesColumn(page=0, header=None, notes_blocks=[nblk])
        findings = check_notes_no_header([col])
        assert len(findings) == 1
        assert findings[0].check_id == "NOTES_NO_HEADER"


# ══════════════════════════════════════════════════════════════════════
# 10. Revision empty, revision missing number
# ══════════════════════════════════════════════════════════════════════


class TestCheckRevisionEmpty:
    def test_revision_with_entries_ok(self):
        region = RevisionRegion(
            page=0,
            entries=[
                RevisionEntry(page=0, number="1", description="Init", date="01/01/2024")
            ],
        )
        assert check_revision_empty([region]) == []

    def test_revision_empty_flagged(self):
        region = RevisionRegion(page=0, entries=[])
        findings = check_revision_empty([region])
        assert len(findings) == 1
        assert findings[0].check_id == "REV_EMPTY"


class TestCheckRevisionMissingNumber:
    def test_entries_with_numbers_ok(self):
        region = RevisionRegion(
            page=0,
            entries=[
                RevisionEntry(
                    page=0, number="1", description="Init", date="01/01/2024"
                ),
            ],
        )
        assert check_revision_missing_number([region]) == []

    def test_entry_missing_number_flagged(self):
        region = RevisionRegion(
            page=0,
            entries=[
                RevisionEntry(
                    page=0, number="", description="Added stuff", date="02/02/2024"
                ),
            ],
        )
        findings = check_revision_missing_number([region])
        assert len(findings) == 1
        assert findings[0].check_id == "REV_NO_NUMBER"

    def test_entry_only_whitespace_number(self):
        region = RevisionRegion(
            page=0,
            entries=[
                RevisionEntry(page=0, number="  ", description="Fix", date=""),
            ],
        )
        findings = check_revision_missing_number([region])
        assert len(findings) == 1


# ══════════════════════════════════════════════════════════════════════
# 11. Abbreviation empty
# ══════════════════════════════════════════════════════════════════════


class TestCheckAbbreviationEmpty:
    def test_region_with_entries_ok(self):
        region = _abbrev_region({"AI": "AREA INLET"})
        assert check_abbreviation_empty([region]) == []

    def test_region_empty_flagged(self):
        region = AbbreviationRegion(page=0, entries=[])
        findings = check_abbreviation_empty([region])
        assert len(findings) == 1
        assert findings[0].check_id == "ABBREV_EMPTY"


# ══════════════════════════════════════════════════════════════════════
# 12. Standard detail missing description
# ══════════════════════════════════════════════════════════════════════


class TestCheckStandardDetailMissingDesc:
    def test_entries_with_descriptions_ok(self):
        region = StandardDetailRegion(
            page=0,
            entries=[
                StandardDetailEntry(
                    page=0, sheet_number="SS-1", description="Steel detail"
                ),
            ],
        )
        assert check_standard_detail_missing_desc([region]) == []

    def test_entry_missing_description_flagged(self):
        region = StandardDetailRegion(
            page=0,
            entries=[
                StandardDetailEntry(page=0, sheet_number="SS-1", description=""),
            ],
        )
        findings = check_standard_detail_missing_desc([region])
        assert len(findings) == 1
        assert findings[0].check_id == "STDDET_NO_DESC"


# ══════════════════════════════════════════════════════════════════════
# 13. Title block checks
# ══════════════════════════════════════════════════════════════════════


class TestCheckTitleBlockMissing:
    def test_no_boxes_at_all(self):
        findings = check_title_block_missing(None, page=0)
        assert len(findings) == 1
        assert findings[0].check_id == "TITLE_MISSING"

    def test_boxes_but_no_title_block(self):
        class _Box:
            box_type = "unknown"

        findings = check_title_block_missing([_Box()], page=0)
        assert len(findings) == 1

    def test_title_block_present(self):
        from plancheck.analysis.structural_boxes import BoxType

        class _Box:
            box_type = BoxType.title_block

        findings = check_title_block_missing([_Box()], page=0)
        assert findings == []


class TestCheckTitleBlockFields:
    def test_no_title_blocks_ok(self):
        assert check_title_block_fields(None) == []

    def test_complete_title_block_ok(self):
        from plancheck.analysis.title_block import TitleBlockField, TitleBlockInfo

        tb = TitleBlockInfo(
            page=0,
            fields=[
                TitleBlockField(label="sheet_number", value="C-1"),
                TitleBlockField(label="date", value="01/01/2024"),
            ],
        )
        assert check_title_block_fields([tb]) == []

    def test_missing_sheet_number(self):
        from plancheck.analysis.title_block import TitleBlockField, TitleBlockInfo

        tb = TitleBlockInfo(
            page=0,
            bbox=(0, 0, 100, 100),
            fields=[
                TitleBlockField(label="date", value="01/01/2024"),
            ],
        )
        findings = check_title_block_fields([tb])
        ids = {f.check_id for f in findings}
        assert "TITLE_NO_SHEET_NUMBER" in ids

    def test_missing_date(self):
        from plancheck.analysis.title_block import TitleBlockField, TitleBlockInfo

        tb = TitleBlockInfo(
            page=0,
            bbox=(0, 0, 100, 100),
            fields=[
                TitleBlockField(label="sheet_number", value="C-1"),
            ],
        )
        findings = check_title_block_fields([tb])
        ids = {f.check_id for f in findings}
        assert "TITLE_NO_DATE" in ids


# ══════════════════════════════════════════════════════════════════════
# 14. Notes cross-references
# ══════════════════════════════════════════════════════════════════════


class TestCheckNotesCrossReferences:
    def test_valid_reference(self):
        col = _notes_column([1, 2, 3])
        blk = _block([(10, 200, 200, 212, "SEE NOTE 2")])
        findings = check_notes_cross_references([col], [blk])
        assert findings == []

    def test_undefined_reference(self):
        col = _notes_column([1, 2, 3])
        blk = _block([(10, 200, 200, 212, "SEE NOTE 5")])
        findings = check_notes_cross_references([col], [blk])
        assert len(findings) == 1
        assert findings[0].check_id == "NOTES_XREF"
        assert findings[0].details["referenced_note"] == 5

    def test_no_columns_no_findings(self):
        blk = _block([(10, 200, 200, 212, "SEE NOTE 5")])
        findings = check_notes_cross_references([], [blk])
        assert findings == []


# ══════════════════════════════════════════════════════════════════════
# 15. run_all_checks with new params
# ══════════════════════════════════════════════════════════════════════


class TestRunAllChecksExtended:
    def test_title_block_missing_included(self):
        """run_all_checks should report TITLE_MISSING when no boxes."""
        findings = run_all_checks(
            structural_boxes=[],
            page=1,
        )
        ids = {f.check_id for f in findings}
        assert "TITLE_MISSING" in ids

    def test_legend_empty_included(self):
        region = LegendRegion(page=0, entries=[])
        findings = run_all_checks(legend_regions=[region])
        ids = {f.check_id for f in findings}
        assert "LEGEND_EMPTY" in ids
