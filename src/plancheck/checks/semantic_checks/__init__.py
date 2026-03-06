"""Semantic checks for construction-plan pages.

Each check receives the structured data already extracted by the pipeline
(regions, notes columns, blocks, stage results, quality metadata) and
returns a list of CheckResult findings.  The checks are deterministic —
no LLM calls, no heuristics beyond what's already in the detectors.

Usage::

    from plancheck.checks.semantic_checks import run_all_checks
    findings = run_all_checks(page_data)
"""

# Import check modules to trigger registration of @register_check decorators
from . import abbreviations, details, legends, notes, revisions, titleblock
from .abbreviations import (
    check_abbreviation_duplicates,
    check_abbreviation_empty,
    check_abbreviations_undefined,
)
from .details import (
    check_standard_detail_duplicates,
    check_standard_detail_missing_desc,
)
from .helpers import _adjusted_severity, _parse_date
from .legends import check_legend_empty, check_legend_no_header

# Re-export public check functions for direct use
from .notes import (
    check_notes_cross_references,
    check_notes_no_header,
    check_notes_numbering,
)
from .registry import register_check, run_all_checks

# Import core types first
from .result import CheckResult
from .revisions import (
    check_revision_date_order,
    check_revision_empty,
    check_revision_missing_number,
)
from .titleblock import check_title_block_fields, check_title_block_missing

__all__ = [
    # Core types
    "CheckResult",
    "register_check",
    "run_all_checks",
    # Notes checks
    "check_notes_numbering",
    "check_notes_no_header",
    "check_notes_cross_references",
    # Abbreviation checks
    "check_abbreviation_duplicates",
    "check_abbreviations_undefined",
    "check_abbreviation_empty",
    # Revision checks
    "check_revision_date_order",
    "check_revision_empty",
    "check_revision_missing_number",
    # Legend checks
    "check_legend_empty",
    "check_legend_no_header",
    # Detail checks
    "check_standard_detail_duplicates",
    "check_standard_detail_missing_desc",
    # Title block checks
    "check_title_block_missing",
    "check_title_block_fields",
]
