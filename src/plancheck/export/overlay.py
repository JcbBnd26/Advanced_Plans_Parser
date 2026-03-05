"""Backward-compatible re-exports for overlay functions.

This module is a thin façade that imports from the overlays/ sub-package.
For new code, import directly from plancheck.export.overlays.
"""

from .overlays import (
    COLOR_KEYS,
    COLUMN_COLORS,
    DEFAULT_COLOR,
    LABEL_PREFIXES,
    draw_columns_overlay,
    draw_lines_overlay,
    draw_overlay,
)
from .overlays.colors import _draw_label, _get_color, _scale_point

__all__ = [
    "draw_overlay",
    "draw_columns_overlay",
    "draw_lines_overlay",
    "COLOR_KEYS",
    "DEFAULT_COLOR",
    "LABEL_PREFIXES",
    "COLUMN_COLORS",
    # Underscore helpers re-exported for test compatibility
    "_draw_label",
    "_get_color",
    "_scale_point",
]
