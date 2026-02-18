"""Backward compatibility — imports moved to plancheck.analysis.standard_details."""

from .analysis.standard_details import *  # noqa: F401,F403
from .analysis.standard_details import (  # noqa: F401
    _has_inline_entries,
    _is_standard_detail_header,
    _parse_standard_detail_entries,
    _parse_standard_detail_entries_inline,
    _parse_standard_detail_entries_two_column,
    detect_standard_detail_regions,
)
