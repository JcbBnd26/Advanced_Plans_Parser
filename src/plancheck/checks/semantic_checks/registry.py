"""Check registry and orchestration for semantic checks.

This module provides the infrastructure for registering and running
semantic checks across construction-plan pages.
"""

from __future__ import annotations

from typing import Any, Callable, List, Sequence

from .helpers import _adjusted_severity
from .result import CheckResult

# ── Check registry ────────────────────────────────────────────────────

_CHECK_REGISTRY: list[Callable] = []


def register_check(fn: Callable) -> Callable:
    """Decorator: register a check function for use in :func:`run_all_checks`.

    Registered functions must accept the full set of keyword arguments that
    ``run_all_checks`` passes (they should use ``**_`` to ignore unknown ones).
    """
    _CHECK_REGISTRY.append(fn)
    return fn


# ── Orchestrator ─────────────────────────────────────────────────────


def run_all_checks(
    *,
    notes_columns: Sequence[Any] | None = None,
    abbreviation_regions: Sequence[Any] | None = None,
    revision_regions: Sequence[Any] | None = None,
    standard_detail_regions: Sequence[Any] | None = None,
    legend_regions: Sequence[Any] | None = None,
    misc_title_regions: Sequence[Any] | None = None,
    structural_boxes: Sequence[Any] | None = None,
    title_blocks: Sequence[Any] | None = None,
    blocks: Sequence[Any] | None = None,
    page: int = 0,
    mean_ocr_confidence: float = 1.0,
) -> List[CheckResult]:
    """Run all semantic checks and return a combined list of findings.

    Each parameter is optional — checks that require missing inputs are
    silently skipped, so callers can supply only what they have.

    When *mean_ocr_confidence* is below the attenuation threshold (0.6),
    finding severities are automatically downgraded (error→warning,
    warning→info) to reflect reduced trust in OCR-derived data.
    """
    kwargs = dict(
        notes_columns=notes_columns or [],
        abbreviation_regions=abbreviation_regions or [],
        revision_regions=revision_regions or [],
        standard_detail_regions=standard_detail_regions or [],
        legend_regions=legend_regions or [],
        misc_title_regions=misc_title_regions or [],
        structural_boxes=structural_boxes or [],
        title_blocks=title_blocks or [],
        blocks=blocks or [],
        page=page,
        mean_ocr_confidence=mean_ocr_confidence,
    )
    findings: List[CheckResult] = []
    for check_fn in _CHECK_REGISTRY:
        findings.extend(check_fn(**kwargs))

    # ── Post-process: attenuate severity on low OCR confidence ──────
    if mean_ocr_confidence < 1.0:
        for f in findings:
            f.severity = _adjusted_severity(f.severity, mean_ocr_confidence)

    return findings
