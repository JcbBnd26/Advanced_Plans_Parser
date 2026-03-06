"""Legend-related semantic checks.

Checks for empty legends and missing headers.
"""

from __future__ import annotations

from typing import Any, List, Sequence

from .registry import register_check
from .result import CheckResult

# ── Check functions ──────────────────────────────────────────────────


def check_legend_empty(
    legend_regions: Sequence[Any],
    page: int = 0,
) -> List[CheckResult]:
    """Flag legend regions that have zero entries."""
    findings: List[CheckResult] = []
    for region in legend_regions:
        if not region.entries:
            findings.append(
                CheckResult(
                    check_id="LEGEND_EMPTY",
                    severity="warning",
                    message=(
                        f"Legend '{region.header_text() or 'LEGEND'}' detected "
                        f"but contains no entries"
                    ),
                    page=page,
                    bbox=region.bbox(),
                    details={"header": region.header_text()},
                )
            )
    return findings


def check_legend_no_header(
    legend_regions: Sequence[Any],
    page: int = 0,
) -> List[CheckResult]:
    """Flag legend regions that have entries but no header block."""
    findings: List[CheckResult] = []
    for region in legend_regions:
        if region.entries and not region.header:
            findings.append(
                CheckResult(
                    check_id="LEGEND_NO_HEADER",
                    severity="info",
                    message="Legend has entries but no header block",
                    page=page,
                    bbox=region.bbox(),
                )
            )
    return findings


# ── Registration wrappers ────────────────────────────────────────────


@register_check
def _reg_check_legend_empty(*, legend_regions=None, page=0, **_):
    if not legend_regions:
        return []
    return check_legend_empty(legend_regions, page=page)


@register_check
def _reg_check_legend_no_header(*, legend_regions=None, page=0, **_):
    if not legend_regions:
        return []
    return check_legend_no_header(legend_regions, page=page)
