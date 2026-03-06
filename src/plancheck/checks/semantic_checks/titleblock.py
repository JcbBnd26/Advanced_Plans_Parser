"""Title block-related semantic checks.

Checks for missing title blocks and incomplete fields.
"""

from __future__ import annotations

from typing import Any, List, Sequence

from .registry import register_check
from .result import CheckResult

# ── Check functions ──────────────────────────────────────────────────


def check_title_block_missing(
    structural_boxes: Sequence[Any] | None = None,
    page: int = 0,
) -> List[CheckResult]:
    """Flag pages with no detected title block."""
    findings: List[CheckResult] = []
    if not structural_boxes:
        findings.append(
            CheckResult(
                check_id="TITLE_MISSING",
                severity="warning",
                message="No structural boxes detected — title block may be missing",
                page=page,
            )
        )
        return findings

    from plancheck.analysis.structural_boxes import BoxType

    has_tb = any(
        getattr(sb, "box_type", None) == BoxType.title_block for sb in structural_boxes
    )
    if not has_tb:
        findings.append(
            CheckResult(
                check_id="TITLE_MISSING",
                severity="warning",
                message="No title block detected on this page",
                page=page,
            )
        )
    return findings


def check_title_block_fields(
    title_blocks: Sequence[Any] | None = None,
    page: int = 0,
) -> List[CheckResult]:
    """Flag title blocks missing critical fields (sheet number, date)."""
    findings: List[CheckResult] = []
    if not title_blocks:
        return findings

    required = {"sheet_number", "date"}
    for tb in title_blocks:
        present = {f.label for f in tb.fields}
        for key in required:
            if key not in present:
                label_nice = key.replace("_", " ").title()
                findings.append(
                    CheckResult(
                        check_id=f"TITLE_NO_{key.upper()}",
                        severity="warning",
                        message=f"Title block is missing '{label_nice}'",
                        page=page,
                        bbox=tb.bbox,
                        details={"missing_field": key},
                    )
                )
    return findings


# ── Registration wrappers ────────────────────────────────────────────


@register_check
def _reg_check_title_block_missing(*, structural_boxes=None, page=0, **_):
    return check_title_block_missing(structural_boxes or [], page=page)


@register_check
def _reg_check_title_block_fields(*, title_blocks=None, page=0, **_):
    if not title_blocks:
        return []
    return check_title_block_fields(title_blocks, page=page)
