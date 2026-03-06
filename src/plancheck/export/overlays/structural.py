"""Structural overlay rendering: columns and lines debug visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageDraw, ImageFont

from ...config import GroupingConfig
from ...models import BlockCluster, GlyphBox, Line
from .colors import _scale_point


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
