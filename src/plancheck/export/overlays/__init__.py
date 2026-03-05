"""Overlay rendering sub-package.

This package contains:
  - colors.py: Color constants and shared drawing utilities
  - detection.py: Detection overlay (draw_overlay and element drawing helpers)
  - structural.py: Structural overlays (draw_columns_overlay, draw_lines_overlay)
"""

from .colors import (
    COLOR_KEYS,
    COLUMN_COLORS,
    DEFAULT_COLOR,
    LABEL_PREFIXES,
    _draw_label,
    _draw_rect_or_polygon,
    _get_color,
    _header_to_prefix,
    _scale_point,
)
from .detection import draw_overlay
from .structural import draw_columns_overlay, draw_lines_overlay

__all__ = [
    # Main public functions
    "draw_overlay",
    "draw_columns_overlay",
    "draw_lines_overlay",
    # Color constants (for external consumers)
    "COLOR_KEYS",
    "DEFAULT_COLOR",
    "LABEL_PREFIXES",
    "COLUMN_COLORS",
]
