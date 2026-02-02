from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

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
) -> None:
    """Render grouping stages as an overlay PNG for quick visual QA.

    If `background` is provided, it will be used as the base (should match page dims * scale).
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

    # Glyph boxes in light gray.
    # For misc_title regions, collect overlapping boxes and draw ONE combined box.
    # Also include boxes on the same line that are part of the same title.
    misc_title_glyph_boxes = []  # Collect boxes that overlap with misc_title
    for b in boxes:
        if _overlaps_misc_title(b.x0, b.y0, b.x1, b.y1) or _same_line_as_misc_title(
            b.x0, b.y0, b.x1, b.y1
        ):
            misc_title_glyph_boxes.append(b)
            continue  # Don't draw individually
        draw.rectangle(
            [
                _scale_point(b.x0, b.y0, scale),
                _scale_point(b.x1, b.y1, scale),
            ],
            outline=(0, 255, 0, 200),  # LIME - glyph boxes
            width=1,
        )

    # Draw ONE combined gray box for all text inside each misc_title region
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

        overlapping = [b for b in misc_title_glyph_boxes if _matches_this_title(b)]
        if overlapping:
            # Compute combined bounding box for the actual text
            # Clip to page bounds since PDF extraction can extend past page edge
            combined_x0 = max(0, min(b.x0 for b in overlapping))
            combined_y0 = max(0, min(b.y0 for b in overlapping))
            combined_x1 = min(page_width, max(b.x1 for b in overlapping))
            combined_y1 = min(page_height, max(b.y1 for b in overlapping))
            draw.rectangle(
                [
                    _scale_point(combined_x0, combined_y0, scale),
                    _scale_point(combined_x1, combined_y1, scale),
                ],
                outline=(255, 255, 0, 255),  # YELLOW - misc_title combined text
                width=2,
            )

    # Rows in light gray - skip those that overlap with or are on the same line as misc_title regions.
    for r in rows:
        x0, y0, x1, y1 = r.bbox()
        if _overlaps_misc_title(x0, y0, x1, y1) or _same_line_as_misc_title(
            x0, y0, x1, y1
        ):
            continue
        draw.rectangle(
            [
                _scale_point(x0, y0, scale),
                _scale_point(x1, y1, scale),
            ],
            outline=(0, 0, 255, 200),  # BLUE - rows
            width=2,
        )

    # Blocks: headers in purple, tables in yellow, others in red.
    # Skip blocks that overlap with or are on the same line as misc_title regions.
    for blk in blocks:
        x0, y0, x1, y1 = blk.bbox()
        if _overlaps_misc_title(x0, y0, x1, y1) or _same_line_as_misc_title(
            x0, y0, x1, y1
        ):
            continue
        if getattr(blk, "label", None) == "note_column_header":
            # Outline header blocks in purple
            draw.rectangle(
                [
                    _scale_point(x0, y0, scale),
                    _scale_point(x1, y1, scale),
                ],
                outline=(128, 0, 128, 220),  # PURPLE - header blocks
                width=3,
            )
        elif blk.is_table:
            draw.rectangle(
                [
                    _scale_point(x0, y0, scale),
                    _scale_point(x1, y1, scale),
                ],
                fill=(255, 214, 102, 60),
                outline=(255, 165, 0, 220),  # ORANGE - table blocks
                width=3,
            )
        else:
            draw.rectangle(
                [
                    _scale_point(x0, y0, scale),
                    _scale_point(x1, y1, scale),
                ],
                outline=(255, 0, 0, 220),  # RED - regular blocks
                width=3,
            )

    # Notes columns in green (outer bounding box around header + all notes blocks)
    if notes_columns:
        for col in notes_columns:
            x0, y0, x1, y1 = col.bbox()
            if x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0:
                continue  # Skip empty columns
            draw.rectangle(
                [
                    _scale_point(x0, y0, scale),
                    _scale_point(x1, y1, scale),
                ],
                outline=(0, 128, 0, 220),  # DARK GREEN - notes columns
                width=4,
            )

    # Legend regions in cyan (outer box) with orange symbol boxes
    if legend_regions:
        for legend in legend_regions:
            # Draw outer legend region box in cyan
            x0, y0, x1, y1 = legend.bbox()
            if x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0:
                continue
            draw.rectangle(
                [
                    _scale_point(x0, y0, scale),
                    _scale_point(x1, y1, scale),
                ],
                outline=(0, 255, 255, 220),  # CYAN - legend regions
                width=4,
            )

            # Draw legend header in darker cyan
            if legend.header:
                hx0, hy0, hx1, hy1 = legend.header.bbox()
                draw.rectangle(
                    [
                        _scale_point(hx0, hy0, scale),
                        _scale_point(hx1, hy1, scale),
                    ],
                    outline=(0, 150, 180, 255),  # darker cyan
                    width=3,
                )

            # Draw each legend entry
            for entry in legend.entries:
                # Draw symbol box in orange
                if entry.symbol_bbox:
                    sx0, sy0, sx1, sy1 = entry.symbol_bbox
                    draw.rectangle(
                        [
                            _scale_point(sx0, sy0, scale),
                            _scale_point(sx1, sy1, scale),
                        ],
                        outline=(255, 152, 0, 255),  # orange
                        width=2,
                    )

                # Draw description box in light orange
                if entry.description_bbox:
                    dx0, dy0, dx1, dy1 = entry.description_bbox
                    draw.rectangle(
                        [
                            _scale_point(dx0, dy0, scale),
                            _scale_point(dx1, dy1, scale),
                        ],
                        outline=(255, 193, 100, 200),  # light orange
                        width=2,
                    )

                # Draw connecting line between symbol and description
                if entry.symbol_bbox and entry.description_bbox:
                    sx0, sy0, sx1, sy1 = entry.symbol_bbox
                    dx0, dy0, dx1, dy1 = entry.description_bbox
                    # Line from right-center of symbol to left-center of description
                    sym_right = (sx1, (sy0 + sy1) / 2)
                    desc_left = (dx0, (dy0 + dy1) / 2)
                    draw.line(
                        [
                            _scale_point(*sym_right, scale),
                            _scale_point(*desc_left, scale),
                        ],
                        fill=(255, 152, 0, 150),  # semi-transparent orange
                        width=1,
                    )

    # Abbreviation regions in magenta (these are exclusion zones - no graphics)
    if abbreviation_regions:
        for abbrev in abbreviation_regions:
            # Draw outer abbreviation region box in magenta
            x0, y0, x1, y1 = abbrev.bbox()
            if x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0:
                continue
            draw.rectangle(
                [
                    _scale_point(x0, y0, scale),
                    _scale_point(x1, y1, scale),
                ],
                outline=(255, 0, 255, 200),  # MAGENTA - abbreviation regions
                width=4,
            )

            # Draw abbreviation header in darker magenta
            if abbrev.header:
                hx0, hy0, hx1, hy1 = abbrev.header.bbox()
                draw.rectangle(
                    [
                        _scale_point(hx0, hy0, scale),
                        _scale_point(hx1, hy1, scale),
                    ],
                    outline=(180, 0, 180, 255),  # darker magenta
                    width=3,
                )

            # Draw each abbreviation entry - code in pink, meaning in light pink
            for entry in abbrev.entries:
                # Draw code box in pink
                if entry.code_bbox:
                    cx0, cy0, cx1, cy1 = entry.code_bbox
                    draw.rectangle(
                        [
                            _scale_point(cx0, cy0, scale),
                            _scale_point(cx1, cy1, scale),
                        ],
                        outline=(255, 100, 150, 220),  # pink
                        width=2,
                    )

                # Draw meaning box in light pink
                if entry.meaning_bbox:
                    mx0, my0, mx1, my1 = entry.meaning_bbox
                    draw.rectangle(
                        [
                            _scale_point(mx0, my0, scale),
                            _scale_point(mx1, my1, scale),
                        ],
                        outline=(255, 180, 200, 180),  # light pink
                        width=2,
                    )

    # Revision regions in teal (title block element)
    if revision_regions:
        for revision in revision_regions:
            # Draw outer revision region box in teal
            x0, y0, x1, y1 = revision.bbox()
            if x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0:
                continue
            draw.rectangle(
                [
                    _scale_point(x0, y0, scale),
                    _scale_point(x1, y1, scale),
                ],
                outline=(0, 128, 128, 220),  # TEAL - revision regions
                width=4,
            )

            # Draw revision header in darker teal
            if revision.header:
                hx0, hy0, hx1, hy1 = revision.header.bbox()
                draw.rectangle(
                    [
                        _scale_point(hx0, hy0, scale),
                        _scale_point(hx1, hy1, scale),
                    ],
                    outline=(0, 100, 100, 255),  # darker teal
                    width=3,
                )

            # Draw each revision entry row
            for entry in revision.entries:
                if entry.row_bbox:
                    rx0, ry0, rx1, ry1 = entry.row_bbox
                    draw.rectangle(
                        [
                            _scale_point(rx0, ry0, scale),
                            _scale_point(rx1, ry1, scale),
                        ],
                        outline=(100, 180, 180, 180),  # light teal
                        width=2,
                    )

    # Misc title regions in pink (title block element to exclude)
    if misc_title_regions:
        for misc_title in misc_title_regions:
            x0, y0, x1, y1 = misc_title.bbox()
            if x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0:
                continue
            draw.rectangle(
                [
                    _scale_point(x0, y0, scale),
                    _scale_point(x1, y1, scale),
                ],
                outline=(255, 105, 180, 220),  # PINK - misc_title box
                width=3,
            )

    # Standard detail regions in dark cyan
    if standard_detail_regions:
        for detail in standard_detail_regions:
            # Draw outer region box
            x0, y0, x1, y1 = detail.bbox()
            if x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0:
                continue
            draw.rectangle(
                [
                    _scale_point(x0, y0, scale),
                    _scale_point(x1, y1, scale),
                ],
                outline=(0, 139, 139, 220),  # DARK CYAN - standard detail regions
                width=4,
            )

            # Draw header in darker cyan
            if detail.header:
                hx0, hy0, hx1, hy1 = detail.header.bbox()
                draw.rectangle(
                    [
                        _scale_point(hx0, hy0, scale),
                        _scale_point(hx1, hy1, scale),
                    ],
                    outline=(0, 100, 100, 255),  # darker cyan
                    width=3,
                )

            # Draw each entry - sheet number in teal, description in light teal
            for entry in detail.entries:
                # Draw sheet number box
                if entry.sheet_bbox:
                    sx0, sy0, sx1, sy1 = entry.sheet_bbox
                    draw.rectangle(
                        [
                            _scale_point(sx0, sy0, scale),
                            _scale_point(sx1, sy1, scale),
                        ],
                        outline=(0, 180, 180, 220),  # teal
                        width=2,
                    )

                # Draw description box
                if entry.description_bbox:
                    dx0, dy0, dx1, dy1 = entry.description_bbox
                    draw.rectangle(
                        [
                            _scale_point(dx0, dy0, scale),
                            _scale_point(dx1, dy1, scale),
                        ],
                        outline=(100, 200, 200, 180),  # light teal
                        width=2,
                    )

    img.save(out_path)
