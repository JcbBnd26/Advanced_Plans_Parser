"""Revision-related semantic checks.

Checks for date ordering, empty regions, and missing revision numbers.
"""

from __future__ import annotations

from typing import Any, List, Sequence

from .helpers import _parse_date
from .registry import register_check
from .result import CheckResult

# ── Check functions ──────────────────────────────────────────────────


def check_revision_date_order(
    revision_regions: Sequence[Any],
    page: int = 0,
) -> List[CheckResult]:
    """Verify revision entries have chronologically ordered dates.

    Revisions are typically listed top-to-bottom from newest to oldest,
    but in some plans it's oldest-first.  We flag any revision where
    the date is *neither* monotonically increasing nor decreasing.
    """
    from datetime import datetime

    findings: List[CheckResult] = []

    for region in revision_regions:
        if len(region.entries) < 2:
            continue

        dates: List[tuple[int, datetime, str]] = []
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


def check_revision_empty(
    revision_regions: Sequence[Any],
    page: int = 0,
) -> List[CheckResult]:
    """Flag revision regions detected with no entries."""
    findings: List[CheckResult] = []
    for region in revision_regions:
        if not region.entries:
            findings.append(
                CheckResult(
                    check_id="REV_EMPTY",
                    severity="info",
                    message=(
                        f"Revision region '{region.header_text() or 'REVISIONS'}' "
                        f"has no entries"
                    ),
                    page=page,
                    bbox=region.bbox(),
                    details={"header": region.header_text()},
                )
            )
    return findings


def check_revision_missing_number(
    revision_regions: Sequence[Any],
    page: int = 0,
) -> List[CheckResult]:
    """Flag revision entries that have a date/description but no revision number."""
    findings: List[CheckResult] = []
    for region in revision_regions:
        for entry in region.entries:
            has_content = entry.description.strip() or entry.date.strip()
            if has_content and not entry.number.strip():
                findings.append(
                    CheckResult(
                        check_id="REV_NO_NUMBER",
                        severity="warning",
                        message="Revision entry has content but no revision number",
                        page=page,
                        bbox=entry.bbox(),
                        details={
                            "description": entry.description,
                            "date": entry.date,
                        },
                    )
                )
    return findings


# ── Registration wrappers ────────────────────────────────────────────


@register_check
def _reg_check_revision_date_order(*, revision_regions=None, page=0, **_):
    if not revision_regions:
        return []
    return check_revision_date_order(revision_regions, page=page)


@register_check
def _reg_check_revision_empty(*, revision_regions=None, page=0, **_):
    if not revision_regions:
        return []
    return check_revision_empty(revision_regions, page=page)


@register_check
def _reg_check_revision_missing_number(*, revision_regions=None, page=0, **_):
    if not revision_regions:
        return []
    return check_revision_missing_number(revision_regions, page=page)
