"""Notes-related semantic checks.

Checks for notes column numbering, headers, and cross-references.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from .registry import register_check
from .result import CheckResult

# ── Regex patterns ───────────────────────────────────────────────────

_NOTE_NUM_RE = re.compile(r"^(\d+)\.")
_NOTE_XREF_RE = re.compile(r"(?:SEE|REFER\s+TO)\s+NOTE\s+#?(\d+)", re.I)


# ── Check functions ──────────────────────────────────────────────────


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


def check_notes_no_header(
    notes_columns: Sequence[Any],
    page: int = 0,
) -> List[CheckResult]:
    """Flag notes columns without a header block."""
    findings: List[CheckResult] = []
    for col in notes_columns:
        if not col.header and col.notes_blocks:
            findings.append(
                CheckResult(
                    check_id="NOTES_NO_HEADER",
                    severity="warning",
                    message="Notes column has notes blocks but no header",
                    page=page,
                    bbox=col.bbox(),
                )
            )
    return findings


def check_notes_cross_references(
    notes_columns: Sequence[Any],
    blocks: Sequence[Any],
    page: int = 0,
) -> List[CheckResult]:
    """Detect cross-references to notes that don't exist.

    Scans all blocks for patterns like "SEE NOTE 5" and verifies that
    note #5 actually exists in a notes column on the same page.
    """
    findings: List[CheckResult] = []

    # Collect all defined note numbers
    defined_notes: set[int] = set()
    for col in notes_columns:
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
                defined_notes.add(int(m.group(1)))

    if not defined_notes and not notes_columns:
        return findings

    # Scan all blocks for cross-references
    for blk in blocks:
        for row in blk.rows:
            for box in row.boxes:
                for m in _NOTE_XREF_RE.finditer(box.text):
                    ref_num = int(m.group(1))
                    if ref_num not in defined_notes:
                        findings.append(
                            CheckResult(
                                check_id="NOTES_XREF",
                                severity="warning",
                                message=f"Reference to note #{ref_num} but note not found",
                                page=page,
                                bbox=box.bbox(),
                                details={"referenced_note": ref_num},
                            )
                        )

    return findings


# ── Registration wrappers ────────────────────────────────────────────


@register_check
def _reg_check_notes_numbering(*, notes_columns=None, blocks=None, page=0, **_):
    if not notes_columns:
        return []
    return check_notes_numbering(notes_columns, blocks or [], page=page)


@register_check
def _reg_check_notes_no_header(*, notes_columns=None, page=0, **_):
    if not notes_columns:
        return []
    return check_notes_no_header(notes_columns, page=page)


@register_check
def _reg_check_notes_cross_references(*, notes_columns=None, blocks=None, page=0, **_):
    if not notes_columns or not blocks:
        return []
    return check_notes_cross_references(notes_columns, blocks, page=page)
