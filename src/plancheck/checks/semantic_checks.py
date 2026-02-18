"""Semantic checks for construction-plan pages.

Each check receives the structured data already extracted by the pipeline
(regions, notes columns, blocks, stage results, quality metadata) and
returns a list of CheckResult findings.  The checks are deterministic —
no LLM calls, no heuristics beyond what's already in the detectors.

Usage::

    from plancheck.semantic_checks import run_all_checks
    findings = run_all_checks(page_data)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ── CheckResult data class ───────────────────────────────────────────


@dataclass
class CheckResult:
    """A single finding from a semantic check."""

    check_id: str  # e.g. "ABBREV_DUP"
    severity: str  # "error" | "warning" | "info"
    message: str  # Human-readable description
    page: int = 0
    bbox: Optional[Tuple[float, float, float, float]] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "check_id": self.check_id,
            "severity": self.severity,
            "message": self.message,
            "page": self.page,
        }
        if self.bbox:
            d["bbox"] = list(self.bbox)
        if self.details:
            d["details"] = self.details
        return d


# ── Individual checks ────────────────────────────────────────────────

# 1. Notes numbering ─────────────────────────────────────────────────
_NOTE_NUM_RE = re.compile(r"^(\d+)\.")


def check_notes_numbering(
    notes_columns: Sequence[Any],
    blocks: Sequence[Any],
    page: int = 0,
) -> List[CheckResult]:
    """Verify note numbers are sequential within each notes column.

    Detects:
    - Gaps (e.g. 1, 2, 4 — missing 3)
    - Duplicates (e.g. 1, 2, 2, 3)
    - Non-sequential ordering (e.g. 1, 3, 2)
    """
    findings: List[CheckResult] = []

    for col in notes_columns:
        col_header = col.header_text() if hasattr(col, "header_text") else ""

        # Collect note numbers from notes_blocks in order
        numbers: List[int] = []
        note_bboxes: List[Tuple[float, float, float, float]] = []
        for blk in col.notes_blocks:
            if not blk.rows:
                continue
            first_row = blk.rows[0]
            texts = [
                b.text for b in sorted(first_row.boxes, key=lambda b: b.x0) if b.text
            ]
            row_text = " ".join(texts).strip()
            m = _NOTE_NUM_RE.match(row_text)
            if m:
                numbers.append(int(m.group(1)))
                note_bboxes.append(blk.bbox())

        if not numbers:
            continue

        # Check for duplicates
        seen: Dict[int, int] = {}
        for idx, num in enumerate(numbers):
            if num in seen:
                findings.append(
                    CheckResult(
                        check_id="NOTES_DUP",
                        severity="error",
                        message=(f"Duplicate note #{num} in '{col_header or 'notes'}'"),
                        page=page,
                        bbox=note_bboxes[idx] if idx < len(note_bboxes) else None,
                        details={"column_header": col_header, "number": num},
                    )
                )
            seen[num] = idx

        # Check for gaps (expected: 1..max)
        expected = set(range(1, max(numbers) + 1))
        actual = set(numbers)
        missing = sorted(expected - actual)
        if missing:
            findings.append(
                CheckResult(
                    check_id="NOTES_GAP",
                    severity="warning",
                    message=(f"Missing note(s) {missing} in '{col_header or 'notes'}'"),
                    page=page,
                    details={
                        "column_header": col_header,
                        "missing": missing,
                        "found": sorted(actual),
                    },
                )
            )

        # Check monotonic ordering
        for i in range(1, len(numbers)):
            if numbers[i] < numbers[i - 1]:
                findings.append(
                    CheckResult(
                        check_id="NOTES_ORDER",
                        severity="warning",
                        message=(
                            f"Note #{numbers[i]} appears after #{numbers[i-1]} "
                            f"in '{col_header or 'notes'}'"
                        ),
                        page=page,
                        bbox=note_bboxes[i] if i < len(note_bboxes) else None,
                        details={
                            "column_header": col_header,
                            "out_of_order": numbers[i],
                            "previous": numbers[i - 1],
                        },
                    )
                )
                break  # One ordering error per column is enough

    return findings


# 2. Abbreviation duplicates ─────────────────────────────────────────


def check_abbreviation_duplicates(
    abbreviation_regions: Sequence[Any],
    page: int = 0,
) -> List[CheckResult]:
    """Detect abbreviation codes defined more than once with different meanings.

    Processes all abbreviation regions on a page as a single namespace.
    """
    findings: List[CheckResult] = []

    code_meanings: Dict[str, List[str]] = {}
    code_bboxes: Dict[str, Tuple[float, float, float, float]] = {}

    for region in abbreviation_regions:
        for entry in region.entries:
            code = entry.code.strip().upper()
            meaning = entry.meaning.strip().upper()
            if not code:
                continue
            code_meanings.setdefault(code, []).append(meaning)
            if code not in code_bboxes and entry.code_bbox:
                code_bboxes[code] = entry.code_bbox

    for code, meanings in code_meanings.items():
        unique = set(meanings)
        if len(unique) > 1:
            findings.append(
                CheckResult(
                    check_id="ABBREV_DUP",
                    severity="error",
                    message=(
                        f"Abbreviation '{code}' has conflicting definitions: "
                        f"{sorted(unique)}"
                    ),
                    page=page,
                    bbox=code_bboxes.get(code),
                    details={"code": code, "meanings": sorted(unique)},
                )
            )

    return findings


# 3. Revision date ordering ──────────────────────────────────────────

_DATE_PATTERNS = [
    (re.compile(r"(\d{1,2})/(\d{1,2})/(\d{2,4})"), "MDY"),  # MM/DD/YYYY
    (re.compile(r"(\d{1,2})-(\d{1,2})-(\d{2,4})"), "MDY"),  # MM-DD-YYYY
    (re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})"), "YMD"),  # YYYY-MM-DD
    (re.compile(r"(\d{1,2})\s+\w+\s+(\d{4})"), "DMonY"),  # 12 Jan 2024
    (re.compile(r"\w+\s+(\d{1,2}),?\s+(\d{4})"), "MonDY"),  # Jan 12, 2024
]


def _parse_date(text: str) -> Optional[datetime]:
    """Try to parse a date string using common US engineering formats."""
    text = text.strip()
    if not text:
        return None

    # MM/DD/YYYY or MM-DD-YYYY
    for pattern, fmt in _DATE_PATTERNS[:2]:
        m = pattern.search(text)
        if m:
            month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if year < 100:
                year += 2000
            try:
                return datetime(year, month, day)
            except ValueError:
                continue

    # YYYY-MM-DD
    m = _DATE_PATTERNS[2][0].search(text)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # Try free-form with dateutil-style month names
    month_names = {
        "JAN": 1,
        "FEB": 2,
        "MAR": 3,
        "APR": 4,
        "MAY": 5,
        "JUN": 6,
        "JUL": 7,
        "AUG": 8,
        "SEP": 9,
        "OCT": 10,
        "NOV": 11,
        "DEC": 12,
        "JANUARY": 1,
        "FEBRUARY": 2,
        "MARCH": 3,
        "APRIL": 4,
        "JUNE": 6,
        "JULY": 7,
        "AUGUST": 8,
        "SEPTEMBER": 9,
        "OCTOBER": 10,
        "NOVEMBER": 11,
        "DECEMBER": 12,
    }
    parts = re.findall(r"[A-Za-z]+|\d+", text.upper())
    nums = [int(p) for p in parts if p.isdigit()]
    months = [month_names[p] for p in parts if p in month_names]

    if months and len(nums) >= 2:
        month = months[0]
        # Determine which num is day vs year
        if nums[-1] > 31:  # last number is year
            year = nums[-1] if nums[-1] > 100 else nums[-1] + 2000
            day = nums[0]
        else:
            day = nums[0]
            year = nums[1] if nums[1] > 100 else nums[1] + 2000
        try:
            return datetime(year, month, day)
        except ValueError:
            pass

    return None


def check_revision_date_order(
    revision_regions: Sequence[Any],
    page: int = 0,
) -> List[CheckResult]:
    """Verify revision entries have chronologically ordered dates.

    Revisions are typically listed top-to-bottom from newest to oldest,
    but in some plans it's oldest-first.  We flag any revision where
    the date is *neither* monotonically increasing nor decreasing.
    """
    findings: List[CheckResult] = []

    for region in revision_regions:
        if len(region.entries) < 2:
            continue

        dates: List[Tuple[int, datetime, str]] = []
        for idx, entry in enumerate(region.entries):
            dt = _parse_date(entry.date)
            if dt:
                dates.append((idx, dt, entry.date))

        if len(dates) < 2:
            continue

        # Check monotonic (ascending or descending)
        just_dates = [d[1] for d in dates]
        is_asc = all(a <= b for a, b in zip(just_dates, just_dates[1:]))
        is_desc = all(a >= b for a, b in zip(just_dates, just_dates[1:]))

        if not is_asc and not is_desc:
            # Find the first violation
            for i in range(1, len(dates)):
                prev_idx, prev_dt, prev_raw = dates[i - 1]
                curr_idx, curr_dt, curr_raw = dates[i]
                if not (prev_dt <= curr_dt) and not (prev_dt >= curr_dt):
                    continue  # equal dates OK
                # Determine direction from first pair
                if dates[0][1] <= dates[1][1]:
                    # Ascending expected
                    if curr_dt < prev_dt:
                        findings.append(
                            CheckResult(
                                check_id="REV_DATE_ORDER",
                                severity="warning",
                                message=(
                                    f"Revision dates out of order: "
                                    f"'{prev_raw}' then '{curr_raw}'"
                                ),
                                page=page,
                                bbox=(
                                    region.entries[curr_idx].row_bbox
                                    if region.entries[curr_idx].row_bbox
                                    else None
                                ),
                                details={
                                    "index": curr_idx,
                                    "expected_direction": "ascending",
                                    "prev_date": prev_raw,
                                    "curr_date": curr_raw,
                                },
                            )
                        )
                        break
                else:
                    # Descending expected
                    if curr_dt > prev_dt:
                        findings.append(
                            CheckResult(
                                check_id="REV_DATE_ORDER",
                                severity="warning",
                                message=(
                                    f"Revision dates out of order: "
                                    f"'{prev_raw}' then '{curr_raw}'"
                                ),
                                page=page,
                                bbox=(
                                    region.entries[curr_idx].row_bbox
                                    if region.entries[curr_idx].row_bbox
                                    else None
                                ),
                                details={
                                    "index": curr_idx,
                                    "expected_direction": "descending",
                                    "prev_date": prev_raw,
                                    "curr_date": curr_raw,
                                },
                            )
                        )
                        break

    return findings


# 4. Abbreviation used but not defined ────────────────────────────────

_ABBREV_WORD_RE = re.compile(r"\b([A-Z]{2,})\b")
_COMMON_WORDS = frozenset(
    {
        "THE",
        "AND",
        "FOR",
        "NOT",
        "ALL",
        "ARE",
        "BUT",
        "CAN",
        "HAD",
        "HAS",
        "HER",
        "HIS",
        "HOW",
        "ITS",
        "MAY",
        "NEW",
        "NOW",
        "OLD",
        "OUR",
        "OUT",
        "OWN",
        "SAY",
        "SHE",
        "TOO",
        "USE",
        "WAY",
        "WHO",
        "BOY",
        "DID",
        "GET",
        "HIM",
        "LET",
        "PUT",
        "RUN",
        "SET",
        "TOP",
        "BIG",
        "END",
        "FAR",
        "RUN",
        "TRY",
        "ASK",
        "MAN",
        "DAY",
        "ALSO",
        "MADE",
        "EACH",
        "BACK",
        "BEEN",
        "BOTH",
        "COME",
        "DOES",
        "DOWN",
        "EVEN",
        "FIND",
        "GIVE",
        "GOOD",
        "HAVE",
        "HERE",
        "INTO",
        "JUST",
        "KEEP",
        "KNOW",
        "LAST",
        "LIKE",
        "LINE",
        "LIST",
        "LONG",
        "LOOK",
        "MAKE",
        "MANY",
        "MORE",
        "MOST",
        "MUCH",
        "MUST",
        "NAME",
        "NEXT",
        "ONLY",
        "OPEN",
        "OVER",
        "PART",
        "PLAN",
        "PAGE",
        "SAME",
        "SHOW",
        "SIDE",
        "SOME",
        "SUCH",
        "TAKE",
        "TELL",
        "THAN",
        "THAT",
        "THEM",
        "THEN",
        "THIS",
        "TIME",
        "TURN",
        "TYPE",
        "UPON",
        "VERY",
        "WANT",
        "WELL",
        "WENT",
        "WERE",
        "WHAT",
        "WHEN",
        "WILL",
        "WITH",
        "WORK",
        "YEAR",
        "YOUR",
        "FROM",
        "THEY",
        "BEEN",
        "CALL",
        "SAID",
        "USED",
        "CONT",
        "NOTES",
        "NOTE",
        "SITE",
        "SEE",
        "PER",
        "REF",
        "YES",
        "SHEET",
        "DATE",
        "SCALE",
        "PLAN",
        "PLANS",
        "STATE",
        "ROAD",
        "DRAWING",
        "GENERAL",
        "PROJECT",
        "OKLAHOMA",
        "DEPARTMENT",
        "TRANSPORTATION",
        "STANDARD",
        "DETAILS",
        "LEGEND",
        "REVISION",
        "REVISIONS",
        "ABBREVIATIONS",
        "CONTINUED",
        "TOTAL",
        "AREA",
        # Common uppercase plan terms that aren't abbreviations
        "SECTION",
        "TYPICAL",
        "NORTH",
        "SOUTH",
        "EAST",
        "WEST",
        "COUNTY",
        "CITY",
        "TOWN",
        "HIGHWAY",
        "STREET",
        "AVENUE",
        "PROPOSED",
        "EXISTING",
        "REMOVED",
        "CONSTRUCTION",
        "ENGINEER",
    }
)


def check_abbreviations_undefined(
    abbreviation_regions: Sequence[Any],
    notes_columns: Sequence[Any],
    blocks: Sequence[Any],
    page: int = 0,
) -> List[CheckResult]:
    """Detect abbreviation-like tokens in notes that aren't in the abbreviation table.

    Only flags uppercase 2+ letter tokens found in notes blocks that appear
    to be abbreviations (not common English words) and aren't defined.
    """
    findings: List[CheckResult] = []

    # Build set of defined abbreviations
    defined = set()
    for region in abbreviation_regions:
        for entry in region.entries:
            defined.add(entry.code.strip().upper())

    if not defined:
        # No abbreviation table to check against — skip
        return findings

    # Collect all uppercase tokens from notes blocks
    used_abbrevs: Dict[str, Tuple[float, float, float, float]] = {}
    for col in notes_columns:
        for blk in col.notes_blocks:
            for row in blk.rows:
                for box in row.boxes:
                    for m in _ABBREV_WORD_RE.finditer(box.text):
                        word = m.group(1)
                        if word not in _COMMON_WORDS and word not in defined:
                            if word not in used_abbrevs:
                                used_abbrevs[word] = box.bbox()

    for word, bbox in sorted(used_abbrevs.items()):
        findings.append(
            CheckResult(
                check_id="ABBREV_UNDEF",
                severity="info",
                message=f"'{word}' used in notes but not in abbreviation table",
                page=page,
                bbox=bbox,
                details={"token": word},
            )
        )

    return findings


# 5. Standard detail duplicate sheet numbers ──────────────────────────


def check_standard_detail_duplicates(
    standard_detail_regions: Sequence[Any],
    page: int = 0,
) -> List[CheckResult]:
    """Detect duplicate sheet numbers in standard details."""
    findings: List[CheckResult] = []

    sheet_map: Dict[str, List[str]] = {}
    sheet_bboxes: Dict[str, Tuple[float, float, float, float]] = {}

    for region in standard_detail_regions:
        for entry in region.entries:
            sheet = entry.sheet_number.strip().upper()
            desc = entry.description.strip()
            if not sheet:
                continue
            sheet_map.setdefault(sheet, []).append(desc)
            if sheet not in sheet_bboxes and entry.sheet_bbox:
                sheet_bboxes[sheet] = entry.sheet_bbox

    for sheet, descs in sheet_map.items():
        if len(descs) > 1:
            findings.append(
                CheckResult(
                    check_id="STDDET_DUP",
                    severity="warning",
                    message=f"Standard detail '{sheet}' listed {len(descs)} times",
                    page=page,
                    bbox=sheet_bboxes.get(sheet),
                    details={"sheet_number": sheet, "descriptions": descs},
                )
            )

    return findings


# ── Orchestrator ─────────────────────────────────────────────────────


def run_all_checks(
    *,
    notes_columns: Sequence[Any] | None = None,
    abbreviation_regions: Sequence[Any] | None = None,
    revision_regions: Sequence[Any] | None = None,
    standard_detail_regions: Sequence[Any] | None = None,
    legend_regions: Sequence[Any] | None = None,
    misc_title_regions: Sequence[Any] | None = None,
    blocks: Sequence[Any] | None = None,
    page: int = 0,
) -> List[CheckResult]:
    """Run all semantic checks and return a combined list of findings.

    Each parameter is optional — checks that require missing inputs are
    silently skipped, so callers can supply only what they have.
    """
    notes_columns = notes_columns or []
    abbreviation_regions = abbreviation_regions or []
    revision_regions = revision_regions or []
    standard_detail_regions = standard_detail_regions or []
    legend_regions = legend_regions or []
    misc_title_regions = misc_title_regions or []
    blocks = blocks or []

    findings: List[CheckResult] = []

    # 1. Notes numbering
    if notes_columns:
        findings.extend(check_notes_numbering(notes_columns, blocks, page=page))

    # 2. Abbreviation duplicates
    if abbreviation_regions:
        findings.extend(check_abbreviation_duplicates(abbreviation_regions, page=page))

    # 3. Revision date ordering
    if revision_regions:
        findings.extend(check_revision_date_order(revision_regions, page=page))

    # 4. Abbreviation used-but-undefined
    if abbreviation_regions and notes_columns:
        findings.extend(
            check_abbreviations_undefined(
                abbreviation_regions,
                notes_columns,
                blocks,
                page=page,
            )
        )

    # 5. Standard detail duplicates
    if standard_detail_regions:
        findings.extend(
            check_standard_detail_duplicates(
                standard_detail_regions,
                page=page,
            )
        )

    return findings
