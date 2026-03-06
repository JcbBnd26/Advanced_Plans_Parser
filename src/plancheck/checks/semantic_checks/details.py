"""Standard detail-related semantic checks.

Checks for duplicate sheet numbers and missing descriptions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from .registry import register_check
from .result import CheckResult

# ── Check functions ──────────────────────────────────────────────────


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


def check_standard_detail_missing_desc(
    standard_detail_regions: Sequence[Any],
    page: int = 0,
) -> List[CheckResult]:
    """Flag standard detail entries that have a sheet number but no description."""
    findings: List[CheckResult] = []
    for region in standard_detail_regions:
        for entry in region.entries:
            if entry.sheet_number.strip() and not entry.description.strip():
                findings.append(
                    CheckResult(
                        check_id="STDDET_NO_DESC",
                        severity="warning",
                        message=(
                            f"Standard detail '{entry.sheet_number}' has no description"
                        ),
                        page=page,
                        bbox=entry.bbox(),
                        details={"sheet_number": entry.sheet_number},
                    )
                )
    return findings


# ── Registration wrappers ────────────────────────────────────────────


@register_check
def _reg_check_standard_detail_duplicates(*, standard_detail_regions=None, page=0, **_):
    if not standard_detail_regions:
        return []
    return check_standard_detail_duplicates(standard_detail_regions, page=page)


@register_check
def _reg_check_standard_detail_missing_desc(
    *, standard_detail_regions=None, page=0, **_
):
    if not standard_detail_regions:
        return []
    return check_standard_detail_missing_desc(standard_detail_regions, page=page)
