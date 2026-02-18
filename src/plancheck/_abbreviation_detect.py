"""Backward compatibility — imports moved to plancheck.analysis.abbreviations."""

from .analysis.abbreviations import *  # noqa: F401,F403
from .analysis.abbreviations import (
    _is_abbreviation_header,  # noqa: F401
    _parse_abbreviation_entries,
    _parse_abbreviation_entries_two_column,
    detect_abbreviation_regions,
)
