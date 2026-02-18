"""Backward compatibility — imports moved to plancheck.analysis.structural_boxes."""

from .analysis.structural_boxes import *  # noqa: F401,F403
from .analysis.structural_boxes import (
    _dedup_boxes,  # noqa: F401
    _detect_boxes_from_lines,
    _first_row_text,
    _grow_region_from_anchor,
    _label_from_header_text,
    _promote_sub_boxes,
)
