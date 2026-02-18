"""Backward compatibility — imports moved to plancheck.analysis.legends."""

from .analysis.legends import *  # noqa: F401,F403
from .analysis.legends import (
    _detect_legend_columns,  # noqa: F401
    _is_legend_header,
    _pair_symbols_with_text,
    detect_abbreviation_regions,
    detect_legend_regions,
    detect_misc_title_regions,
    detect_revision_regions,
    detect_standard_detail_regions,
    extract_graphics,
    filter_graphics_outside_regions,
)
