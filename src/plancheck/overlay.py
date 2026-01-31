from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image, ImageDraw

from .models import BlockCluster, GlyphBox, RowBand


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

    # Glyph boxes in light gray.
    for b in boxes:
        draw.rectangle(
            [
                _scale_point(b.x0, b.y0, scale),
                _scale_point(b.x1, b.y1, scale),
            ],
            outline=(180, 180, 180, 180),  # light gray
            width=1,
        )

    # Rows in light gray.
    for r in rows:
        x0, y0, x1, y1 = r.bbox()
        draw.rectangle(
            [
                _scale_point(x0, y0, scale),
                _scale_point(x1, y1, scale),
            ],
            outline=(180, 180, 180, 180),  # light gray
            width=2,
        )

    # Blocks: headers in purple, tables in yellow, others in red.
    for blk in blocks:
        x0, y0, x1, y1 = blk.bbox()
        if getattr(blk, "label", None) == "note_column_header":
            # Outline header blocks in purple
            draw.rectangle(
                [
                    _scale_point(x0, y0, scale),
                    _scale_point(x1, y1, scale),
                ],
                outline=(128, 0, 128, 220),  # purple
                width=3,
            )
        elif blk.is_table:
            draw.rectangle(
                [
                    _scale_point(x0, y0, scale),
                    _scale_point(x1, y1, scale),
                ],
                fill=(255, 214, 102, 60),
                outline=(231, 76, 60, 220),
                width=3,
            )
        else:
            draw.rectangle(
                [
                    _scale_point(x0, y0, scale),
                    _scale_point(x1, y1, scale),
                ],
                outline=(231, 76, 60, 220),
                width=3,
            )

    img.save(out_path)
