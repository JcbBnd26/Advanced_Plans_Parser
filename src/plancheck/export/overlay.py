from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from ..config import GroupingConfig
from ..models import (
    AbbreviationRegion,
    BlockCluster,
    GlyphBox,
    LegendRegion,
    Line,
    MiscTitleRegion,
    NotesColumn,
    RevisionRegion,
    RowBand,
    Span,
    StandardDetailRegion,
)

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


def _get_color(color_overrides: Optional[Dict[str, tuple]], key: str) -> tuple | None:
    """Get color for a key, using override if provided, otherwise None (don't draw)."""
    if color_overrides and key in color_overrides:
        return color_overrides[key]
    return DEFAULT_COLOR


def _scale_point(x: float, y: float, scale: float) -> Tuple[float, float]:
    """Scale (x, y) by *scale* for overlay rendering."""
    return (x * scale, y * scale)


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


def draw_columns_overlay(
    page_width: float,
    page_height: float,
    blocks: List[BlockCluster],
    tokens: List[GlyphBox],
    out_path: Path,
    scale: float = 1.0,
    background: Image.Image | None = None,
    cfg: GroupingConfig | None = None,
) -> None:
    """Render block outlines colour-coded by semantic type.

    No histogram column bands are drawn.  Each block is outlined and
    labelled with its semantic type: H=header, N=notes, T=table, B=regular.

    Args:
        page_width:  Page width in PDF points
        page_height: Page height in PDF points
        blocks:  BlockClusters (with lines populated)
        tokens:  Full token list (needed for bbox computation)
        out_path: Destination PNG path
        scale:   PDF-to-pixel scale factor
        background: Optional background image
        cfg:     GroupingConfig (for font sizing)
    """
    img_w = int(page_width * scale)
    img_h = int(page_height * scale)

    if background is not None:
        img = background.convert("RGBA")
        if img.size != (img_w, img_h):
            img = img.resize((img_w, img_h))
    else:
        img = Image.new("RGBA", (img_w, img_h), (255, 255, 255, 255))

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    _font_base = cfg.overlay_label_font_base if cfg else 10
    _font_floor = cfg.overlay_label_font_floor if cfg else 8
    _block_w = cfg.overlay_block_outline_width if cfg else 3

    font_size = max(_font_floor, int(_font_base * scale))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Colour map by semantic type
    _type_colors = {
        "H": (255, 0, 0, 220),  # red for headers
        "N": (0, 0, 255, 220),  # blue for notes
        "T": (255, 165, 0, 220),  # orange for tables
        "B": (0, 180, 0, 150),  # green for regular blocks
    }

    # Draw block outlines colour-coded by type
    for blk in blocks:
        bb = blk.bbox()
        if bb == (0, 0, 0, 0):
            continue

        # Determine block type label
        if blk.is_header:
            tag = "H"
        elif blk.is_notes:
            tag = "N"
        elif blk.is_table:
            tag = "T"
        else:
            tag = "B"

        outline = _type_colors.get(tag, _type_colors["B"])

        bx0, by0, bx1, by1 = bb
        sx0, sy0 = int(bx0 * scale), int(by0 * scale)
        sx1, sy1 = int(bx1 * scale), int(by1 * scale)
        draw.rectangle([(sx0, sy0), (sx1, sy1)], outline=outline, width=_block_w)

        # Small label
        label = tag
        bbox_txt = draw.textbbox((sx0, sy0 - font_size - 2), label, font=font)
        bg = (bbox_txt[0] - 1, bbox_txt[1] - 1, bbox_txt[2] + 1, bbox_txt[3] + 1)
        draw.rectangle(bg, fill=(255, 255, 255, 220))
        draw.text(
            (sx0, sy0 - font_size - 2),
            label,
            fill=(outline[0], outline[1], outline[2]),
            font=font,
        )

    img = Image.alpha_composite(img, overlay)
    img.save(out_path, format="PNG")


def draw_lines_overlay(
    page_width: float,
    page_height: float,
    lines: Iterable[Line],
    tokens: List[GlyphBox],
    out_path: Path,
    scale: float = 1.0,
    background: Image.Image | None = None,
    span_color: tuple = (0, 0, 255, 200),  # Blue outlines for spans
    cfg: GroupingConfig | None = None,
) -> None:
    """Render Span outlines as an overlay PNG for debugging the row-truth layer.

    Args:
        page_width: Page width in points
        page_height: Page height in points
        lines: Lines from build_lines()
        tokens: Original token list
        out_path: Output path for PNG
        scale: Scale factor for rendering
        background: Optional background image
        span_color: RGBA color for span outlines
    """
    img_w = int(page_width * scale)
    img_h = int(page_height * scale)

    if background is not None:
        img = background.convert("RGBA")
        if img.size != (img_w, img_h):
            img = img.resize((img_w, img_h))
    else:
        img = Image.new("RGBA", (img_w, img_h), (255, 255, 255, 255))

    draw = ImageDraw.Draw(img, "RGBA")

    _font_base = cfg.overlay_label_font_base if cfg else 10
    _font_floor = cfg.overlay_label_font_floor if cfg else 8
    _bg_alpha = cfg.overlay_label_bg_alpha if cfg else 200
    _span_w = cfg.overlay_span_outline_width if cfg else 2

    # Cache font once instead of loading per-span
    font_size = max(_font_floor, int(_font_base * scale))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for line in lines:
        if not line.token_indices:
            continue

        # Draw span outlines labeled L{line_id}.S{span_pos}
        if line.spans:
            for span_pos, span in enumerate(line.spans):
                if not span.token_indices:
                    continue

                sb = span.bbox(tokens)
                ssx0, ssy0 = _scale_point(sb[0], sb[1], scale)
                ssx1, ssy1 = _scale_point(sb[2], sb[3], scale)
                draw.rectangle(
                    [(ssx0, ssy0), (ssx1, ssy1)], outline=span_color, width=_span_w
                )

                # Label as L{line_id}.S{span_pos} so it maps to the data
                label = f"L{line.line_id}.S{span_pos}"
                bbox_txt = draw.textbbox((ssx0, ssy0 - font_size - 2), label, font=font)
                bg = (
                    bbox_txt[0] - 1,
                    bbox_txt[1] - 1,
                    bbox_txt[2] + 1,
                    bbox_txt[3] + 1,
                )
                draw.rectangle(bg, fill=(255, 255, 255, _bg_alpha))
                draw.text(
                    (ssx0, ssy0 - font_size - 2),
                    label,
                    fill=(span_color[0], span_color[1], span_color[2]),
                    font=font,
                )

    img.save(out_path, format="PNG")


def _draw_glyph_boxes(
    draw: ImageDraw.ImageDraw,
    boxes: List[GlyphBox],
    scale: float,
    color_overrides: Optional[Dict[str, tuple]],
    cfg: GroupingConfig | None,
    glyph_w: int,
) -> None:
    """Draw individual glyph-box outlines."""
    glyph_color = _get_color(color_overrides, "glyph_boxes")
    if not glyph_color:
        return
    for glyph_idx, b in enumerate(boxes, start=1):
        sx0, sy0 = _scale_point(b.x0, b.y0, scale)
        sx1, sy1 = _scale_point(b.x1, b.y1, scale)
        draw.rectangle([(sx0, sy0), (sx1, sy1)], outline=glyph_color, width=glyph_w)
        _draw_label(
            draw, sx0, sy0, "glyph_boxes", glyph_idx, glyph_color, scale, cfg=cfg
        )


def _draw_misc_title_text(
    draw: ImageDraw.ImageDraw,
    boxes: List[GlyphBox],
    misc_title_bboxes: List[Tuple[float, float, float, float]],
    page_width: float,
    page_height: float,
    scale: float,
    color_overrides: Optional[Dict[str, tuple]],
    same_line_overlap: float,
    proximity: float,
    span_w: int,
) -> None:
    """Draw combined text boxes for misc-title regions."""
    for mx0, my0, mx1, my1 in misc_title_bboxes:

        def _matches_this_title(b, _mx0=mx0, _my0=my0, _mx1=mx1, _my1=my1):
            """Return True if box *b* overlaps or neighbours this title region."""
            if not (b.x1 < _mx0 or b.x0 > _mx1 or b.y1 < _my0 or b.y0 > _my1):
                return True
            y_overlap = min(b.y1, _my1) - max(b.y0, _my0)
            box_height = b.y1 - b.y0
            if box_height > 0 and y_overlap / box_height > same_line_overlap:
                if b.x0 <= _mx1 + proximity and b.x1 >= _mx0 - proximity:
                    return True
            return False

        overlapping = [b for b in boxes if _matches_this_title(b)]
        if overlapping:
            combined_x0 = max(0, min(b.x0 for b in overlapping))
            combined_y0 = max(0, min(b.y0 for b in overlapping))
            combined_x1 = min(page_width, max(b.x1 for b in overlapping))
            combined_y1 = min(page_height, max(b.y1 for b in overlapping))
            color = _get_color(color_overrides, "misc_title_text")
            if color:
                draw.rectangle(
                    [
                        _scale_point(combined_x0, combined_y0, scale),
                        _scale_point(combined_x1, combined_y1, scale),
                    ],
                    outline=color,
                    width=span_w,
                )


def _draw_rows(
    draw: ImageDraw.ImageDraw,
    rows: List[RowBand],
    scale: float,
    color_overrides: Optional[Dict[str, tuple]],
    cfg: GroupingConfig | None,
    span_w: int,
) -> None:
    """Draw row-band outlines."""
    row_color = _get_color(color_overrides, "rows")
    if not row_color:
        return
    for row_idx, r in enumerate(rows, start=1):
        x0, y0, x1, y1 = r.bbox()
        sx0, sy0 = _scale_point(x0, y0, scale)
        sx1, sy1 = _scale_point(x1, y1, scale)
        draw.rectangle([(sx0, sy0), (sx1, sy1)], outline=row_color, width=span_w)
        _draw_label(draw, sx0, sy0, "rows", row_idx, row_color, scale, cfg=cfg)


def _draw_blocks(
    draw: ImageDraw.ImageDraw,
    blocks: List[BlockCluster],
    scale: float,
    color_overrides: Optional[Dict[str, tuple]],
    cfg: GroupingConfig | None,
    block_w: int,
    table_fill_alpha: int,
) -> None:
    """Draw block outlines colour-coded by semantic type."""
    for block_idx, blk in enumerate(blocks, start=1):
        x0, y0, x1, y1 = blk.bbox()
        sx0, sy0 = _scale_point(x0, y0, scale)
        sx1, sy1 = _scale_point(x1, y1, scale)
        if getattr(blk, "label", None) == "note_column_header":
            color = _get_color(color_overrides, "header_blocks")
            if color:
                draw.rectangle([(sx0, sy0), (sx1, sy1)], outline=color, width=block_w)
                _draw_label(
                    draw, sx0, sy0, "header_blocks", block_idx, color, scale, cfg=cfg
                )
        elif blk.is_table:
            color = _get_color(color_overrides, "table_blocks")
            if color:
                fill_color = (
                    (color[0], color[1], color[2], table_fill_alpha)
                    if len(color) >= 3
                    else color
                )
                draw.rectangle(
                    [(sx0, sy0), (sx1, sy1)],
                    fill=fill_color,
                    outline=color,
                    width=block_w,
                )
                _draw_label(
                    draw, sx0, sy0, "table_blocks", block_idx, color, scale, cfg=cfg
                )
        else:
            color = _get_color(color_overrides, "regular_blocks")
            if color:
                draw.rectangle([(sx0, sy0), (sx1, sy1)], outline=color, width=block_w)
                _draw_label(
                    draw, sx0, sy0, "regular_blocks", block_idx, color, scale, cfg=cfg
                )


def _draw_notes_columns(
    draw: ImageDraw.ImageDraw,
    notes_columns: Iterable[NotesColumn] | None,
    scale: float,
    color_overrides: Optional[Dict[str, tuple]],
    cfg: GroupingConfig | None,
    region_w: int,
) -> None:
    """Draw notes-column outlines with smart header-derived labels."""
    if not notes_columns:
        return
    notes_columns_list = list(notes_columns)
    col_labels: list[str] = []
    prefix_counters: dict[str, int] = {}
    group_prefix: dict[str, str] = {}
    group_sub_counter: dict[str, int] = {}
    for col in notes_columns_list:
        grp = col.column_group_id
        if col.header is not None:
            hdr_text = col.header_text()
            prefix = _header_to_prefix(hdr_text) if hdr_text else "NC"
            count = prefix_counters.get(prefix, 0) + 1
            prefix_counters[prefix] = count
            label = f"{prefix}{count}"
            if grp is not None:
                group_prefix[grp] = label
                group_sub_counter[grp] = 2
            col_labels.append(label)
        elif grp is not None and grp in group_prefix:
            parent_label = group_prefix[grp]
            sub = group_sub_counter.get(grp, 2)
            group_sub_counter[grp] = sub + 1
            col_labels.append(f"{parent_label}.{sub}")
        else:
            prefix = "NC"
            count = prefix_counters.get(prefix, 0) + 1
            prefix_counters[prefix] = count
            col_labels.append(f"{prefix}{count}")

    for col_idx, col in enumerate(notes_columns_list):
        x0, y0, x1, y1 = col.bbox()
        if x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0:
            continue
        color = _get_color(color_overrides, "notes_columns")
        if color:
            sx0, sy0 = _scale_point(x0, y0, scale)
            sx1, sy1 = _scale_point(x1, y1, scale)
            draw.rectangle([(sx0, sy0), (sx1, sy1)], outline=color, width=region_w)
            _draw_label(
                draw,
                sx0,
                sy0,
                "notes_columns",
                col_labels[col_idx],
                color,
                scale,
                cfg=cfg,
            )


def _draw_legend_regions(
    draw: ImageDraw.ImageDraw,
    legend_regions: Iterable[LegendRegion] | None,
    scale: float,
    color_overrides: Optional[Dict[str, tuple]],
    cfg: GroupingConfig | None,
    region_w: int,
    block_w: int,
) -> None:
    """Draw legend region and header outlines."""
    if not legend_regions:
        return
    for legend_idx, legend in enumerate(legend_regions, start=1):
        x0, y0, x1, y1 = legend.bbox()
        if x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0:
            continue
        color = _get_color(color_overrides, "legend_region")
        if color:
            sx0, sy0 = _scale_point(x0, y0, scale)
            sx1, sy1 = _scale_point(x1, y1, scale)
            draw.rectangle([(sx0, sy0), (sx1, sy1)], outline=color, width=region_w)
            _draw_label(
                draw, sx0, sy0, "legend_region", legend_idx, color, scale, cfg=cfg
            )
        if legend.header:
            hx0, hy0, hx1, hy1 = legend.header.bbox()
            color = _get_color(color_overrides, "legend_header")
            if color:
                sx0, sy0 = _scale_point(hx0, hy0, scale)
                sx1, sy1 = _scale_point(hx1, hy1, scale)
                draw.rectangle([(sx0, sy0), (sx1, sy1)], outline=color, width=block_w)
                _draw_label(
                    draw, sx0, sy0, "legend_header", legend_idx, color, scale, cfg=cfg
                )


def _draw_abbreviation_regions(
    draw: ImageDraw.ImageDraw,
    abbreviation_regions: Iterable[AbbreviationRegion] | None,
    scale: float,
    color_overrides: Optional[Dict[str, tuple]],
    cfg: GroupingConfig | None,
    region_w: int,
    block_w: int,
    span_w: int,
) -> None:
    """Draw abbreviation region outlines, headers, entries, and connecting lines."""
    if not abbreviation_regions:
        return
    for abbrev_idx, abbrev in enumerate(abbreviation_regions, start=1):
        x0, y0, x1, y1 = abbrev.bbox()
        if x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0:
            continue
        color = _get_color(color_overrides, "abbreviation_region")
        if color:
            sx0, sy0 = _scale_point(x0, y0, scale)
            sx1, sy1 = _scale_point(x1, y1, scale)
            draw.rectangle([(sx0, sy0), (sx1, sy1)], outline=color, width=region_w)
            _draw_label(
                draw, sx0, sy0, "abbreviation_region", abbrev_idx, color, scale, cfg=cfg
            )
        if abbrev.header:
            hx0, hy0, hx1, hy1 = abbrev.header.bbox()
            color = _get_color(color_overrides, "abbreviation_header")
            if color:
                sx0, sy0 = _scale_point(hx0, hy0, scale)
                sx1, sy1 = _scale_point(hx1, hy1, scale)
                draw.rectangle([(sx0, sy0), (sx1, sy1)], outline=color, width=block_w)
                _draw_label(
                    draw,
                    sx0,
                    sy0,
                    "abbreviation_header",
                    abbrev_idx,
                    color,
                    scale,
                    cfg=cfg,
                )
        for entry in abbrev.entries:
            if entry.code_bbox:
                cx0, cy0, cx1, cy1 = entry.code_bbox
                color = _get_color(color_overrides, "abbreviation_code")
                if color:
                    draw.rectangle(
                        [_scale_point(cx0, cy0, scale), _scale_point(cx1, cy1, scale)],
                        outline=color,
                        width=span_w,
                    )
            if entry.meaning_bbox:
                mx0, my0, mx1, my1 = entry.meaning_bbox
                color = _get_color(color_overrides, "abbreviation_meaning")
                if color:
                    draw.rectangle(
                        [_scale_point(mx0, my0, scale), _scale_point(mx1, my1, scale)],
                        outline=color,
                        width=span_w,
                    )
            if entry.code_bbox and entry.meaning_bbox:
                cx0, cy0, cx1, cy1 = entry.code_bbox
                mx0, my0, mx1, my1 = entry.meaning_bbox
                code_right = (cx1, (cy0 + cy1) / 2)
                meaning_left = (mx0, (my0 + my1) / 2)
                color = _get_color(color_overrides, "abbreviation_line")
                if color:
                    draw.line(
                        [
                            _scale_point(*code_right, scale),
                            _scale_point(*meaning_left, scale),
                        ],
                        fill=color,
                        width=span_w,
                    )


def _draw_revision_regions(
    draw: ImageDraw.ImageDraw,
    revision_regions: Iterable[RevisionRegion] | None,
    scale: float,
    color_overrides: Optional[Dict[str, tuple]],
    cfg: GroupingConfig | None,
    region_w: int,
    block_w: int,
    span_w: int,
) -> None:
    """Draw revision region outlines, headers, and entry rows."""
    if not revision_regions:
        return
    for rev_idx, revision in enumerate(revision_regions, start=1):
        x0, y0, x1, y1 = revision.bbox()
        if x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0:
            continue
        color = _get_color(color_overrides, "revision_region")
        if color:
            sx0, sy0 = _scale_point(x0, y0, scale)
            sx1, sy1 = _scale_point(x1, y1, scale)
            draw.rectangle([(sx0, sy0), (sx1, sy1)], outline=color, width=region_w)
            _draw_label(
                draw, sx0, sy0, "revision_region", rev_idx, color, scale, cfg=cfg
            )
        if revision.header:
            hx0, hy0, hx1, hy1 = revision.header.bbox()
            color = _get_color(color_overrides, "revision_header")
            if color:
                sx0, sy0 = _scale_point(hx0, hy0, scale)
                sx1, sy1 = _scale_point(hx1, hy1, scale)
                draw.rectangle([(sx0, sy0), (sx1, sy1)], outline=color, width=block_w)
        for entry in revision.entries:
            if entry.row_bbox:
                rx0, ry0, rx1, ry1 = entry.row_bbox
                color = _get_color(color_overrides, "revision_entry")
                if color:
                    draw.rectangle(
                        [_scale_point(rx0, ry0, scale), _scale_point(rx1, ry1, scale)],
                        outline=color,
                        width=span_w,
                    )


def _draw_misc_title_regions(
    draw: ImageDraw.ImageDraw,
    misc_title_regions: Iterable[MiscTitleRegion] | None,
    scale: float,
    color_overrides: Optional[Dict[str, tuple]],
    block_w: int,
) -> None:
    """Draw misc-title region outlines."""
    if not misc_title_regions:
        return
    for misc_title in misc_title_regions:
        x0, y0, x1, y1 = misc_title.bbox()
        if x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0:
            continue
        color = _get_color(color_overrides, "misc_title_region")
        if color:
            draw.rectangle(
                [_scale_point(x0, y0, scale), _scale_point(x1, y1, scale)],
                outline=color,
                width=block_w,
            )


def _draw_standard_detail_regions(
    draw: ImageDraw.ImageDraw,
    standard_detail_regions: Iterable[StandardDetailRegion] | None,
    scale: float,
    color_overrides: Optional[Dict[str, tuple]],
    cfg: GroupingConfig | None,
    region_w: int,
    block_w: int,
    span_w: int,
) -> None:
    """Draw standard-detail region outlines, headers, subheaders, and entries."""
    if not standard_detail_regions:
        return
    for detail_idx, detail in enumerate(standard_detail_regions, start=1):
        x0, y0, x1, y1 = detail.bbox()
        if x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0:
            continue
        color = _get_color(color_overrides, "standard_detail_region")
        if color:
            sx0, sy0 = _scale_point(x0, y0, scale)
            sx1, sy1 = _scale_point(x1, y1, scale)
            draw.rectangle([(sx0, sy0), (sx1, sy1)], outline=color, width=region_w)
            _draw_label(
                draw,
                sx0,
                sy0,
                "standard_detail_region",
                detail_idx,
                color,
                scale,
                cfg=cfg,
            )
        if detail.header and detail.header.rows:
            first_row = detail.header.rows[0]
            hx0, hy0, hx1, hy1 = first_row.bbox()
            color = _get_color(color_overrides, "standard_detail_header")
            if color:
                sx0, sy0 = _scale_point(hx0, hy0, scale)
                sx1, sy1 = _scale_point(hx1, hy1, scale)
                draw.rectangle([(sx0, sy0), (sx1, sy1)], outline=color, width=block_w)
        if detail.subheader_bbox:
            sx0, sy0, sx1, sy1 = detail.subheader_bbox
            color = _get_color(color_overrides, "standard_detail_subheader")
            if color:
                draw.rectangle(
                    [_scale_point(sx0, sy0, scale), _scale_point(sx1, sy1, scale)],
                    outline=color,
                    width=span_w,
                )
        for entry in detail.entries:
            if entry.sheet_bbox:
                sx0, sy0, sx1, sy1 = entry.sheet_bbox
                color = _get_color(color_overrides, "standard_detail_sheet")
                if color:
                    draw.rectangle(
                        [_scale_point(sx0, sy0, scale), _scale_point(sx1, sy1, scale)],
                        outline=color,
                        width=span_w,
                    )
            if entry.description_bbox:
                dx0, dy0, dx1, dy1 = entry.description_bbox
                color = _get_color(color_overrides, "standard_detail_description")
                if color:
                    draw.rectangle(
                        [_scale_point(dx0, dy0, scale), _scale_point(dx1, dy1, scale)],
                        outline=color,
                        width=span_w,
                    )


def draw_overlay(
    page_width: float,
    page_height: float,
    boxes: Iterable[GlyphBox],
    rows: Iterable[RowBand],
    blocks: Iterable[BlockCluster],
    out_path: Path,
    scale: float = 1.0,
    background: Image.Image | None = None,
    notes_columns: Iterable[NotesColumn] | None = None,
    legend_regions: Iterable[LegendRegion] | None = None,
    abbreviation_regions: Iterable[AbbreviationRegion] | None = None,
    revision_regions: Iterable[RevisionRegion] | None = None,
    misc_title_regions: Iterable[MiscTitleRegion] | None = None,
    standard_detail_regions: Iterable[StandardDetailRegion] | None = None,
    color_overrides: Optional[Dict[str, tuple]] = None,
    cfg: GroupingConfig | None = None,
) -> None:
    """Render grouping stages as an overlay PNG for quick visual QA.

    If `background` is provided, it will be used as the base (should match page dims * scale).
    If `color_overrides` is provided, it maps element keys to RGBA tuples.
    Any element not in overrides will use DEFAULT_COLOR (None = don't draw).
    If `cfg` is provided, overlay widths, font sizes, and alphas are read from it.

    Drawing is delegated to per-element-type helpers (``_draw_glyph_boxes``,
    ``_draw_blocks``, ``_draw_notes_columns``, etc.) for maintainability.
    """

    # Resolve config knobs (fall back to defaults if cfg is None)
    _glyph_w = cfg.overlay_glyph_outline_width if cfg else 1
    _span_w = cfg.overlay_span_outline_width if cfg else 2
    _block_w = cfg.overlay_block_outline_width if cfg else 3
    _region_w = cfg.overlay_region_outline_width if cfg else 4
    _table_fill_alpha = cfg.overlay_table_fill_alpha if cfg else 60
    _same_line_overlap = cfg.overlay_same_line_overlap if cfg else 0.5
    _proximity = cfg.overlay_proximity_pts if cfg else 50.0

    # Materialise iterables once so generators aren't exhausted on re-use.
    boxes = list(boxes)
    rows = list(rows)
    blocks = list(blocks)

    img_w = int(page_width * scale)
    img_h = int(page_height * scale)
    if background is not None:
        img = background.convert("RGBA")
        if img.size != (img_w, img_h):
            img = img.resize((img_w, img_h))
    else:
        img = Image.new("RGBA", (img_w, img_h), (255, 255, 255, 255))

    draw = ImageDraw.Draw(img, "RGBA")

    # Build set of misc_title bboxes for combined-text overlays
    misc_title_bboxes = []
    if misc_title_regions:
        for mt in misc_title_regions:
            misc_title_bboxes.append(mt.bbox())

    # Delegate to per-element-type helpers
    _draw_glyph_boxes(draw, boxes, scale, color_overrides, cfg, _glyph_w)
    _draw_misc_title_text(
        draw,
        boxes,
        misc_title_bboxes,
        page_width,
        page_height,
        scale,
        color_overrides,
        _same_line_overlap,
        _proximity,
        _span_w,
    )
    _draw_rows(draw, rows, scale, color_overrides, cfg, _span_w)
    _draw_blocks(draw, blocks, scale, color_overrides, cfg, _block_w, _table_fill_alpha)
    _draw_notes_columns(draw, notes_columns, scale, color_overrides, cfg, _region_w)
    _draw_legend_regions(
        draw,
        legend_regions,
        scale,
        color_overrides,
        cfg,
        _region_w,
        _block_w,
    )
    _draw_abbreviation_regions(
        draw,
        abbreviation_regions,
        scale,
        color_overrides,
        cfg,
        _region_w,
        _block_w,
        _span_w,
    )
    _draw_revision_regions(
        draw,
        revision_regions,
        scale,
        color_overrides,
        cfg,
        _region_w,
        _block_w,
        _span_w,
    )
    _draw_misc_title_regions(draw, misc_title_regions, scale, color_overrides, _block_w)
    _draw_standard_detail_regions(
        draw,
        standard_detail_regions,
        scale,
        color_overrides,
        cfg,
        _region_w,
        _block_w,
        _span_w,
    )

    img.save(out_path, format="PNG")
