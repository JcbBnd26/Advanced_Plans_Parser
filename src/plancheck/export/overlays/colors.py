"""Color constants and shared drawing utilities for overlay rendering."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from PIL import ImageDraw, ImageFont

from ...config import GroupingConfig

# Default color: None means don't draw unless explicitly selected
DEFAULT_COLOR = None

# Color keys for different element types
COLOR_KEYS = [
    "glyph_boxes",
    "rows",
    "lines",  # New: line-based grouping
    "spans",  # New: span visualization (colored by col_id)
    "regular_blocks",
    "table_blocks",
    "header_blocks",
    "notes_columns",
    "legend_region",
    "legend_header",
    "abbreviation_region",
    "abbreviation_header",
    "abbreviation_code",
    "abbreviation_meaning",
    "abbreviation_line",
    "revision_region",
    "revision_header",
    "revision_entry",
    "misc_title_region",
    "misc_title_text",
    "standard_detail_region",
    "standard_detail_header",
    "standard_detail_subheader",
    "standard_detail_sheet",
    "standard_detail_description",
]

# Label prefixes for each element type
LABEL_PREFIXES = {
    "glyph_boxes": "G",
    "rows": "R",
    "lines": "L",  # New: line-based grouping
    "spans": "S",  # New: spans within lines
    "regular_blocks": "B",
    "table_blocks": "T",
    "header_blocks": "H",
    "notes_columns": "NC",
    "legend_region": "LR",
    "legend_header": "LH",
    "abbreviation_region": "AR",
    "abbreviation_header": "AH",
    "revision_region": "RV",
    "standard_detail_region": "SD",
    "misc_title_text": "MT",
}

# Palette for column-id colored spans
COLUMN_COLORS = [
    (255, 0, 0, 180),  # Red
    (0, 0, 255, 180),  # Blue
    (0, 180, 0, 180),  # Green
    (255, 165, 0, 180),  # Orange
    (128, 0, 128, 180),  # Purple
    (0, 200, 200, 180),  # Cyan
    (255, 105, 180, 180),  # Pink
    (139, 69, 19, 180),  # Brown
]


def _get_color(color_overrides: Optional[Dict[str, tuple]], key: str) -> tuple | None:
    """Get color for a key, using override if provided, otherwise None (don't draw)."""
    if color_overrides and key in color_overrides:
        return color_overrides[key]
    return DEFAULT_COLOR


def _scale_point(x: float, y: float, scale: float) -> Tuple[float, float]:
    """Scale (x, y) by *scale* for overlay rendering."""
    return (x * scale, y * scale)


def _draw_rect_or_polygon(
    draw: ImageDraw.ImageDraw,
    obj: Any,
    bbox: Tuple[float, float, float, float],
    scale: float,
    outline: tuple,
    width: int,
) -> None:
    """Draw a rectangle or polygon outline, depending on whether *obj* has a ``polygon`` attribute.

    If ``obj.polygon`` is a non-empty list of ``(x, y)`` vertices the
    shape is drawn as a polygon; otherwise a plain axis-aligned rectangle
    is drawn from *bbox*.
    """
    poly = getattr(obj, "polygon", None)
    if poly:
        scaled = [(px * scale, py * scale) for px, py in poly]
        draw.polygon(scaled, outline=outline, fill=None)
        # Pillow's polygon() doesn't support `width`, so overlay the
        # outline multiple times for thicker strokes.
        for _w in range(1, width):
            draw.polygon(scaled, outline=outline, fill=None)
    else:
        x0, y0, x1, y1 = bbox
        sx0, sy0 = _scale_point(x0, y0, scale)
        sx1, sy1 = _scale_point(x1, y1, scale)
        draw.rectangle([(sx0, sy0), (sx1, sy1)], outline=outline, width=width)


def _draw_label(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    element_type: str,
    index: int | str,
    color: tuple,
    scale: float,
    cfg: GroupingConfig | None = None,
) -> None:
    """Draw a small label at the top-left corner of an element.

    Args:
        draw: PIL ImageDraw object
        x: Scaled x position (top-left corner)
        y: Scaled y position (top-left corner)
        element_type: Key from LABEL_PREFIXES
        index: 1-based index or full label string (e.g. "GN1.2")
        color: RGBA tuple for the label color
        scale: Current scale factor
        cfg: Optional config for font/alpha knobs
    """
    _font_base = cfg.overlay_label_font_base if cfg else 10
    _font_floor = cfg.overlay_label_font_floor if cfg else 8
    _bg_alpha = cfg.overlay_label_bg_alpha if cfg else 200

    if isinstance(index, str):
        label = index  # Caller provided full label (e.g. "GN1.2")
    else:
        prefix = LABEL_PREFIXES.get(element_type, "?")
        label = f"{prefix}{index}"

    # Use a small font size scaled appropriately
    font_size = max(_font_floor, int(_font_base * scale))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Draw background rectangle for readability
    bbox = draw.textbbox((x, y - font_size - 2), label, font=font)
    # Expand bbox slightly for padding
    bg_bbox = (bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1)
    draw.rectangle(bg_bbox, fill=(255, 255, 255, _bg_alpha))

    # Draw the label text
    text_color = (color[0], color[1], color[2]) if len(color) >= 3 else (0, 0, 0)
    draw.text((x, y - font_size - 2), label, fill=text_color, font=font)


def _header_to_prefix(header_text: str) -> str:
    """Derive a short prefix from header text using initials of main words.

    Generic suffix words like "NOTES" are excluded so the prefix captures
    the *subject* of the column rather than its type.

    Examples:
        "GENERAL NOTES:"                    → "G"
        "CAST IN PLACE CONCRETE - GENERAL:" → "CIPC"
        "EROSION CONTROL NOTES - GENERAL:"  → "EC"
        "SIDEWALK/CURB RAMP NOTES:"         → "SCR"
        "ODOT STANDARD DETAILS:"            → "OSD"
    """
    import re as _re

    # Words to exclude from initials (generic column-type suffixes)
    _stop_words = {"NOTES", "NOTE"}

    text = header_text.strip().rstrip(":")
    # Take part before " - " if present (strip subtitle like "- GENERAL")
    if " - " in text:
        text = text.split(" - ")[0].strip()
    # Split on spaces, slashes
    words = _re.split(r"[\s/]+", text)
    words = [w for w in words if w and w.upper() not in _stop_words]
    if not words:
        return "NC"
    # Take first letter of each word
    initials = "".join(w[0] for w in words).upper()
    return initials or "NC"
