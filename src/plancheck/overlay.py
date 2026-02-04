from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from PIL import Image, ImageDraw

from .models import (
    AbbreviationRegion,
    BlockCluster,
    GlyphBox,
    LegendRegion,
    MiscTitleRegion,
    NotesColumn,
    RevisionRegion,
    RowBand,
    StandardDetailRegion,
)

# Default color: None means don't draw unless explicitly selected
DEFAULT_COLOR = None

# Color keys for different element types
COLOR_KEYS = [
    "glyph_boxes",
    "rows",
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
    return (x * scale, y * scale)


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
) -> None:
    """Render grouping stages as an overlay PNG for quick visual QA.

    If `background` is provided, it will be used as the base (should match page dims * scale).
    If `color_overrides` is provided, it maps element keys to RGBA tuples.
    Any element not in overrides will use DEFAULT_COLOR (light gray).
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

    # Build set of misc_title bboxes to exclude glyph boxes from
    misc_title_bboxes = []
    if misc_title_regions:
        for mt in misc_title_regions:
            misc_title_bboxes.append(mt.bbox())

    # Build set of standard_detail bboxes to exclude blocks from
    standard_detail_bboxes = []
    if standard_detail_regions:
        for sd in standard_detail_regions:
            standard_detail_bboxes.append(sd.bbox())

    def _inside_misc_title(bx0, by0, bx1, by1):
        """Check if a box is inside any misc_title region."""
        for mx0, my0, mx1, my1 in misc_title_bboxes:
            if bx0 >= mx0 and by0 >= my0 and bx1 <= mx1 and by1 <= my1:
                return True
        return False

    def _overlaps_misc_title(bx0, by0, bx1, by1):
        """Check if a box overlaps any misc_title region."""
        for mx0, my0, mx1, my1 in misc_title_bboxes:
            if not (bx1 < mx0 or bx0 > mx1 or by1 < my0 or by0 > my1):
                return True
        return False

    def _overlaps_standard_detail(bx0, by0, bx1, by1):
        """Check if a box overlaps any standard_detail region."""
        for sx0, sy0, sx1, sy1 in standard_detail_bboxes:
            if not (bx1 < sx0 or bx0 > sx1 or by1 < sy0 or by0 > sy1):
                return True
        return False

    # Build set of abbreviation bboxes to exclude blocks from
    abbreviation_bboxes = []
    if abbreviation_regions:
        for ab in abbreviation_regions:
            abbreviation_bboxes.append(ab.bbox())

    def _overlaps_abbreviation(bx0, by0, bx1, by1):
        """Check if a box overlaps any abbreviation region."""
        for ax0, ay0, ax1, ay1 in abbreviation_bboxes:
            if not (bx1 < ax0 or bx0 > ax1 or by1 < ay0 or by0 > ay1):
                return True
        return False

    # Build set of legend bboxes to exclude overlays inside legend regions
    legend_bboxes = []
    if legend_regions:
        for lg in legend_regions:
            legend_bboxes.append(lg.bbox())

    def _overlaps_legend(bx0, by0, bx1, by1):
        """Check if a box overlaps any legend region."""
        for lx0, ly0, lx1, ly1 in legend_bboxes:
            if not (bx1 < lx0 or bx0 > lx1 or by1 < ly0 or by0 > ly1):
                return True
        return False

    def _same_line_as_misc_title(bx0, by0, bx1, by1):
        """Check if a box is on the same line (y-band) as any misc_title region.

        This catches text that extends beyond the detected box but is part of
        the same title line (e.g., TRANSPORTATION extending past the rounded box).
        """
        for mx0, my0, mx1, my1 in misc_title_bboxes:
            # Check if the y-ranges overlap significantly (same line)
            y_overlap = min(by1, my1) - max(by0, my0)
            box_height = by1 - by0
            if box_height > 0 and y_overlap / box_height > 0.5:
                # On the same line - check if it's close to or overlapping x range
                # Include boxes that touch or are very close to the misc_title
                if bx0 <= mx1 + 50 and bx1 >= mx0 - 50:  # within 50 pts
                    return True
        return False

    # Glyph boxes - draw ALL boxes when selected (no region skipping)
    for b in boxes:
        color = _get_color(color_overrides, "glyph_boxes")
        if color:
            draw.rectangle(
                [
                    _scale_point(b.x0, b.y0, scale),
                    _scale_point(b.x1, b.y1, scale),
                ],
                outline=color,
                width=1,
            )

    # Draw ONE combined box for all text inside each misc_title region
    for mx0, my0, mx1, my1 in misc_title_bboxes:
        # Find all glyph boxes that overlap with this misc_title OR are on the same line
        def _matches_this_title(b):
            # Direct overlap
            if not (b.x1 < mx0 or b.x0 > mx1 or b.y1 < my0 or b.y0 > my1):
                return True
            # Same line check
            y_overlap = min(b.y1, my1) - max(b.y0, my0)
            box_height = b.y1 - b.y0
            if box_height > 0 and y_overlap / box_height > 0.5:
                if b.x0 <= mx1 + 50 and b.x1 >= mx0 - 50:
                    return True
            return False

        overlapping = [b for b in boxes if _matches_this_title(b)]
        if overlapping:
            # Compute combined bounding box for the actual text
            # Clip to page bounds since PDF extraction can extend past page edge
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
                    width=2,
                )

    # Rows - draw ALL rows when selected (no region skipping)
    for r in rows:
        x0, y0, x1, y1 = r.bbox()
        color = _get_color(color_overrides, "rows")
        if color:
            draw.rectangle(
                [
                    _scale_point(x0, y0, scale),
                    _scale_point(x1, y1, scale),
                ],
                outline=color,
                width=2,
            )

    # Blocks - draw ALL blocks when selected (no region skipping)
    for blk in blocks:
        x0, y0, x1, y1 = blk.bbox()
        if getattr(blk, "label", None) == "note_column_header":
            # Outline header blocks
            color = _get_color(color_overrides, "header_blocks")
            if color:
                draw.rectangle(
                    [
                        _scale_point(x0, y0, scale),
                        _scale_point(x1, y1, scale),
                    ],
                    outline=color,
                    width=3,
                )
        elif blk.is_table:
            color = _get_color(color_overrides, "table_blocks")
            if color:
                fill_color = (
                    (color[0], color[1], color[2], 60) if len(color) >= 3 else color
                )
                draw.rectangle(
                    [
                        _scale_point(x0, y0, scale),
                        _scale_point(x1, y1, scale),
                    ],
                    fill=fill_color,
                    outline=color,
                    width=3,
                )
        else:
            color = _get_color(color_overrides, "regular_blocks")
            if color:
                draw.rectangle(
                    [
                        _scale_point(x0, y0, scale),
                        _scale_point(x1, y1, scale),
                    ],
                    outline=color,
                    width=3,
                )

    # Notes columns - draw ALL when selected (no region skipping)
    if notes_columns:
        for col in notes_columns:
            x0, y0, x1, y1 = col.bbox()
            if x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0:
                continue  # Skip empty columns
            color = _get_color(color_overrides, "notes_columns")
            if color:
                draw.rectangle(
                    [
                        _scale_point(x0, y0, scale),
                        _scale_point(x1, y1, scale),
                    ],
                    outline=color,
                    width=4,
                )

    # Legend regions
    if legend_regions:
        for legend in legend_regions:
            # Draw outer legend region box
            x0, y0, x1, y1 = legend.bbox()
            if x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0:
                continue
            color = _get_color(color_overrides, "legend_region")
            if color:
                draw.rectangle(
                    [
                        _scale_point(x0, y0, scale),
                        _scale_point(x1, y1, scale),
                    ],
                    outline=color,
                    width=4,
                )

            # Draw legend header
            if legend.header:
                hx0, hy0, hx1, hy1 = legend.header.bbox()
                color = _get_color(color_overrides, "legend_header")
                if color:
                    draw.rectangle(
                        [
                            _scale_point(hx0, hy0, scale),
                            _scale_point(hx1, hy1, scale),
                        ],
                        outline=color,
                        width=3,
                    )

    # Abbreviation regions (exclusion zones)
    if abbreviation_regions:
        for abbrev in abbreviation_regions:
            # Draw outer abbreviation region box
            x0, y0, x1, y1 = abbrev.bbox()
            if x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0:
                continue
            color = _get_color(color_overrides, "abbreviation_region")
            if color:
                draw.rectangle(
                    [
                        _scale_point(x0, y0, scale),
                        _scale_point(x1, y1, scale),
                    ],
                    outline=color,
                    width=4,
                )

            # Draw abbreviation header
            if abbrev.header:
                hx0, hy0, hx1, hy1 = abbrev.header.bbox()
                color = _get_color(color_overrides, "abbreviation_header")
                if color:
                    draw.rectangle(
                        [
                            _scale_point(hx0, hy0, scale),
                            _scale_point(hx1, hy1, scale),
                        ],
                        outline=color,
                        width=3,
                    )

            # Draw each abbreviation entry
            for entry in abbrev.entries:
                # Draw code box
                if entry.code_bbox:
                    cx0, cy0, cx1, cy1 = entry.code_bbox
                    color = _get_color(color_overrides, "abbreviation_code")
                    if color:
                        draw.rectangle(
                            [
                                _scale_point(cx0, cy0, scale),
                                _scale_point(cx1, cy1, scale),
                            ],
                            outline=color,
                            width=2,
                        )

                # Draw meaning box
                if entry.meaning_bbox:
                    mx0, my0, mx1, my1 = entry.meaning_bbox
                    color = _get_color(color_overrides, "abbreviation_meaning")
                    if color:
                        draw.rectangle(
                            [
                                _scale_point(mx0, my0, scale),
                                _scale_point(mx1, my1, scale),
                            ],
                            outline=color,
                            width=2,
                        )

                # Draw connecting line between code and meaning
                if entry.code_bbox and entry.meaning_bbox:
                    cx0, cy0, cx1, cy1 = entry.code_bbox
                    mx0, my0, mx1, my1 = entry.meaning_bbox
                    # Line from right-center of code to left-center of meaning
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
                            width=2,
                        )

    # Revision regions
    if revision_regions:
        for revision in revision_regions:
            # Draw outer revision region box
            x0, y0, x1, y1 = revision.bbox()
            if x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0:
                continue
            color = _get_color(color_overrides, "revision_region")
            if color:
                draw.rectangle(
                    [
                        _scale_point(x0, y0, scale),
                        _scale_point(x1, y1, scale),
                    ],
                    outline=color,
                    width=4,
                )

            # Draw revision header
            if revision.header:
                hx0, hy0, hx1, hy1 = revision.header.bbox()
                color = _get_color(color_overrides, "revision_header")
                if color:
                    draw.rectangle(
                        [
                            _scale_point(hx0, hy0, scale),
                            _scale_point(hx1, hy1, scale),
                        ],
                        outline=color,
                        width=3,
                    )

            # Draw each revision entry row
            for entry in revision.entries:
                if entry.row_bbox:
                    rx0, ry0, rx1, ry1 = entry.row_bbox
                    color = _get_color(color_overrides, "revision_entry")
                    if color:
                        draw.rectangle(
                            [
                                _scale_point(rx0, ry0, scale),
                                _scale_point(rx1, ry1, scale),
                            ],
                            outline=color,
                            width=2,
                        )

    # Misc title regions
    if misc_title_regions:
        for misc_title in misc_title_regions:
            x0, y0, x1, y1 = misc_title.bbox()
            if x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0:
                continue
            color = _get_color(color_overrides, "misc_title_region")
            if color:
                draw.rectangle(
                    [
                        _scale_point(x0, y0, scale),
                        _scale_point(x1, y1, scale),
                    ],
                    outline=color,
                    width=3,
                )

    # Standard detail regions
    if standard_detail_regions:
        for detail in standard_detail_regions:
            # Draw outer region box
            x0, y0, x1, y1 = detail.bbox()
            if x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0:
                continue
            color = _get_color(color_overrides, "standard_detail_region")
            if color:
                draw.rectangle(
                    [
                        _scale_point(x0, y0, scale),
                        _scale_point(x1, y1, scale),
                    ],
                    outline=color,
                    width=4,
                )

            # Draw header (just the first row - the actual header text)
            if detail.header and detail.header.rows:
                first_row = detail.header.rows[0]
                hx0, hy0, hx1, hy1 = first_row.bbox()
                color = _get_color(color_overrides, "standard_detail_header")
                if color:
                    draw.rectangle(
                        [
                            _scale_point(hx0, hy0, scale),
                            _scale_point(hx1, hy1, scale),
                        ],
                        outline=color,
                        width=3,
                    )

            # Draw subheader (if present)
            if detail.subheader_bbox:
                sx0, sy0, sx1, sy1 = detail.subheader_bbox
                color = _get_color(color_overrides, "standard_detail_subheader")
                if color:
                    draw.rectangle(
                        [
                            _scale_point(sx0, sy0, scale),
                            _scale_point(sx1, sy1, scale),
                        ],
                        outline=color,
                        width=2,
                    )

            # Draw each entry
            for entry in detail.entries:
                # Draw sheet number box
                if entry.sheet_bbox:
                    sx0, sy0, sx1, sy1 = entry.sheet_bbox
                    color = _get_color(color_overrides, "standard_detail_sheet")
                    if color:
                        draw.rectangle(
                            [
                                _scale_point(sx0, sy0, scale),
                                _scale_point(sx1, sy1, scale),
                            ],
                            outline=color,
                            width=2,
                        )

                # Draw description box
                if entry.description_bbox:
                    dx0, dy0, dx1, dy1 = entry.description_bbox
                    color = _get_color(color_overrides, "standard_detail_description")
                    if color:
                        draw.rectangle(
                            [
                                _scale_point(dx0, dy0, scale),
                                _scale_point(dx1, dy1, scale),
                            ],
                            outline=color,
                            width=2,
                        )

    img.save(out_path)
