"""Backward compatibility — imports moved to plancheck.analysis.revisions."""

from .analysis.revisions import *  # noqa: F401,F403
from .analysis.revisions import (
    _has_revision_column_headers,  # noqa: F401
    _is_column_header_row,
    _is_revision_header,
    _parse_revision_entries,
    _parse_revision_row,
    detect_revision_regions,
)
