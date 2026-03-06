"""Abbreviation-related semantic checks.

Checks for duplicate definitions, undefined abbreviations, and empty tables.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from .registry import register_check
from .result import CheckResult

# ── Regex patterns and common word filters ───────────────────────────

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


# ── Check functions ──────────────────────────────────────────────────


def check_abbreviation_duplicates(
    abbreviation_regions: Sequence[Any],
    page: int = 0,
) -> List[CheckResult]:
    """Detect abbreviation codes defined more than once with different meanings.

    Processes all abbreviation regions on a page as a single namespace.
    Same code with identical meanings is fine; different meanings → error.
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


def check_abbreviation_empty(
    abbreviation_regions: Sequence[Any],
    page: int = 0,
) -> List[CheckResult]:
    """Flag abbreviation regions with a header but no entries."""
    findings: List[CheckResult] = []
    for region in abbreviation_regions:
        if not region.entries:
            findings.append(
                CheckResult(
                    check_id="ABBREV_EMPTY",
                    severity="info",
                    message=(
                        f"Abbreviation region '{region.header_text() or 'ABBREVIATIONS'}' "
                        f"has no entries"
                    ),
                    page=page,
                    bbox=region.bbox(),
                    details={"header": region.header_text()},
                )
            )
    return findings


# ── Registration wrappers ────────────────────────────────────────────


@register_check
def _reg_check_abbreviation_duplicates(*, abbreviation_regions=None, page=0, **_):
    if not abbreviation_regions:
        return []
    return check_abbreviation_duplicates(abbreviation_regions, page=page)


@register_check
def _reg_check_abbreviations_undefined(
    *, abbreviation_regions=None, notes_columns=None, blocks=None, page=0, **_
):
    if not abbreviation_regions or not notes_columns:
        return []
    return check_abbreviations_undefined(
        abbreviation_regions, notes_columns, blocks or [], page=page
    )


@register_check
def _reg_check_abbreviation_empty(*, abbreviation_regions=None, page=0, **_):
    if not abbreviation_regions:
        return []
    return check_abbreviation_empty(abbreviation_regions, page=page)
