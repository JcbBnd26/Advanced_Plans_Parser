"""Pure overlay rendering engine (no tkinter dependency).

Provides ``render_overlay()`` — a pure function that runs the TOCR pipeline
on a single page, composites the requested annotation layers onto a
rasterised page background, and returns a ``PIL.Image``.  LLMs and CLI
scripts can call this directly without importing tkinter.

Layer drawing helpers (``_draw_green_layer``, ``_draw_red_layer``, etc.)
and label builders live here so they can be unit-tested independently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pdfplumber
from PIL import Image, ImageDraw, ImageFont

from plancheck import (
    BlockCluster,
    GlyphBox,
    GroupingConfig,
    build_clusters_v2,
    nms_prune,
)
from plancheck.analysis.abbreviations import detect_abbreviation_regions
from plancheck.analysis.graphics import extract_graphics
from plancheck.analysis.legends import detect_legend_regions
from plancheck.analysis.misc_titles import detect_misc_title_regions
from plancheck.analysis.revisions import detect_revision_regions
from plancheck.analysis.standard_details import detect_standard_detail_regions
from plancheck.analysis.structural_boxes import (
    classify_structural_boxes,
    detect_semantic_regions,
    detect_structural_boxes,
)
from plancheck.analysis.zoning import detect_zones
from plancheck.export.overlay import _header_to_prefix
from plancheck.export.page_data import deserialize_page, serialize_page
from plancheck.grouping import group_notes_columns, link_continued_columns
from plancheck.tocr.extract import extract_tocr_from_page

from ..utils.run_utils import latest_overlays_dir
from ..utils.run_utils import scale as _scale

# ---------------------------------------------------------------------------
# Font helper
# ---------------------------------------------------------------------------


def _load_font(scale: float) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", max(10, int(10 * scale / 2.78)))
    except OSError:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Label builders  (isolated from drawing so they can be unit-tested later)
# ---------------------------------------------------------------------------


def _build_note_labels(
    blocks: list[BlockCluster],
) -> dict[int, str]:
    """Build ``{block_index: display_label}`` for notes blocks."""
    prefix_counters: dict[str, int] = {}
    current_prefix = "N"
    note_seq = 0
    labels: dict[int, str] = {}
    for i, blk in enumerate(blocks):
        lbl = getattr(blk, "label", None)
        if lbl == "note_column_header":
            if blk.rows and blk.rows[0].boxes:
                hdr_text = " ".join(b.text for b in blk.rows[0].boxes)
            else:
                hdr_text = " ".join(b.text for b in blk.get_all_boxes())
            prefix = _header_to_prefix(hdr_text) if hdr_text else "N"
            count = prefix_counters.get(prefix, 0) + 1
            prefix_counters[prefix] = count
            current_prefix = f"{prefix}{count}"
            note_seq = 0
        elif getattr(blk, "is_notes", False):
            note_seq += 1
            labels[i] = f"{current_prefix}-n{note_seq}"
    return labels


def _build_header_labels(
    blocks: list[BlockCluster],
) -> dict[int, str]:
    """Build ``{block_index: display_label}`` for header / sub-header blocks."""
    header_idxs = [
        (i, blk)
        for i, blk in enumerate(blocks)
        if getattr(blk, "label", None)
        in ("note_column_header", "note_column_subheader")
    ]
    prefix_counters: dict[str, int] = {}
    parent_lbl: dict[int, str] = {}
    sub_counters: dict[int, int] = {}
    labels: dict[int, str] = {}
    for i, blk in header_idxs:
        pidx = getattr(blk, "parent_block_index", None)
        if pidx is not None:
            pl = parent_lbl.get(pidx, f"B{pidx}")
            seq = sub_counters.get(pidx, 1)
            sub_counters[pidx] = seq + 1
            labels[i] = f"{pl}.{seq}"
        else:
            if blk.rows and blk.rows[0].boxes:
                hdr_text = " ".join(b.text for b in blk.rows[0].boxes)
            else:
                hdr_text = " ".join(b.text for b in blk.get_all_boxes())
            prefix = _header_to_prefix(hdr_text) if hdr_text else "H"
            count = prefix_counters.get(prefix, 0) + 1
            prefix_counters[prefix] = count
            lbl = f"{prefix}{count}"
            labels[i] = lbl
            parent_lbl[i] = lbl
    return labels


# ---------------------------------------------------------------------------
# Layer drawing
# ---------------------------------------------------------------------------

_GREEN = (30, 160, 30, 230)
_GREEN_BG = (30, 130, 30, 180)
_PURPLE = (128, 0, 128, 220)
_PURPLE_BG = (100, 0, 100, 180)
_RED = (220, 30, 30, 230)
_RED_BG = (220, 30, 30, 180)
_LABEL_FG = (255, 255, 255, 255)

# Additional layer colours
_CYAN = (0, 180, 220, 200)
_CYAN_BG = (0, 140, 180, 180)
_ORANGE = (230, 140, 20, 200)
_ORANGE_BG = (200, 110, 15, 180)
_YELLOW = (220, 200, 0, 180)
_YELLOW_BG = (180, 160, 0, 160)
_TEAL = (0, 150, 130, 200)
_TEAL_BG = (0, 120, 100, 180)
_PINK = (220, 80, 150, 200)
_PINK_BG = (180, 60, 120, 180)
_GRAY = (140, 140, 140, 160)
_GRAY_BG = (110, 110, 110, 140)


def _draw_labeled_rect(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[float, float, float, float],
    label_str: str,
    outline: tuple[int, ...],
    label_bg: tuple[int, ...],
    scale: float,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    line_w: int = 2,
) -> None:
    x0, y0, x1, y1 = bbox
    sx0, sy0 = _scale(x0, y0, scale)
    sx1, sy1 = _scale(x1, y1, scale)
    draw.rectangle([(sx0, sy0), (sx1, sy1)], outline=outline, width=line_w)
    if label_str:
        tb = draw.textbbox((0, 0), label_str, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = 2
        lx = sx0
        ly = max(0, sy0 - th - pad * 2 - 2)
        draw.rectangle(
            [(lx, ly), (lx + tw + pad * 2, ly + th + pad * 2)],
            fill=label_bg,
        )
        draw.text((lx + pad, ly + pad), label_str, fill=_LABEL_FG, font=font)


def _draw_green_layer(
    draw: ImageDraw.ImageDraw,
    blocks: list[BlockCluster],
    scale: float,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    color: tuple[int, ...] = _GREEN,
    label_bg: tuple[int, ...] = _GREEN_BG,
) -> None:
    """Draw green rectangles around notes blocks."""
    labels = _build_note_labels(blocks)
    for i, blk in enumerate(blocks):
        if not getattr(blk, "is_notes", False):
            continue
        dl = labels.get(i, f"n{i}")
        if blk.rows and blk.rows[0].boxes:
            preview = " ".join(
                b.text for b in sorted(blk.rows[0].boxes, key=lambda b: b.x0)
            )
        else:
            preview = " ".join(b.text for b in blk.get_all_boxes())
        if len(preview) > 40:
            preview = preview[:37] + "..."
        _draw_labeled_rect(
            draw,
            blk.bbox(),
            f"{dl}: {preview}",
            color,
            label_bg,
            scale,
            font,
            2,
        )


def _draw_purple_layer(
    draw: ImageDraw.ImageDraw,
    blocks: list[BlockCluster],
    notes_columns,
    scale: float,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    color: tuple[int, ...] = _PURPLE,
    label_bg: tuple[int, ...] = _PURPLE_BG,
) -> None:
    """Draw purple rectangles around notes columns."""
    prefix_counters: dict[str, int] = {}
    group_prefix: dict[str, str] = {}
    group_sub_counter: dict[str, int] = {}
    col_labels: list[str] = []
    for col in notes_columns:
        grp = col.column_group_id
        if col.header is not None:
            hdr_text = col.header_text()
            prefix = _header_to_prefix(hdr_text) if hdr_text else "NC"
            count = prefix_counters.get(prefix, 0) + 1
            prefix_counters[prefix] = count
            lbl = f"{prefix}{count}"
            if grp is not None:
                group_prefix[grp] = lbl
                group_sub_counter[grp] = 2
            col_labels.append(lbl)
        elif grp is not None and grp in group_prefix:
            pl = group_prefix[grp]
            sub = group_sub_counter.get(grp, 2)
            group_sub_counter[grp] = sub + 1
            col_labels.append(f"{pl}.{sub}")
        else:
            prefix = "NC"
            count = prefix_counters.get(prefix, 0) + 1
            prefix_counters[prefix] = count
            col_labels.append(f"NC{count}")

    for i, col in enumerate(notes_columns):
        bbox = col.bbox()
        hdr_boxes = col.header.get_all_boxes() if col.header else []
        hdr = " ".join(b.text for b in hdr_boxes) if hdr_boxes else "(no header)"
        if len(hdr) > 40:
            hdr = hdr[:37] + "..."
        label_str = f"{col_labels[i]}: {hdr} [{len(col.notes_blocks)}]"
        _draw_labeled_rect(
            draw,
            bbox,
            label_str,
            color,
            label_bg,
            scale,
            font,
            3,
        )


def _draw_red_layer(
    draw: ImageDraw.ImageDraw,
    blocks: list[BlockCluster],
    scale: float,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    color: tuple[int, ...] = _RED,
    label_bg: tuple[int, ...] = _RED_BG,
) -> None:
    """Draw red rectangles around header blocks."""
    labels = _build_header_labels(blocks)
    for i, blk in enumerate(blocks):
        if i not in labels:
            continue
        if blk.rows and blk.rows[0].boxes:
            hdr_text = " ".join(b.text for b in blk.rows[0].boxes)
        else:
            hdr_text = " ".join(b.text for b in blk.get_all_boxes())
        dl = labels[i]
        lbl = f"{dl}: {hdr_text}"
        if len(lbl) > 60:
            lbl = lbl[:57] + "..."
        _draw_labeled_rect(
            draw,
            blk.bbox(),
            lbl,
            color,
            label_bg,
            scale,
            font,
            3,
        )


# ---------------------------------------------------------------------------
# Additional layer drawing functions
# ---------------------------------------------------------------------------


def _draw_glyph_boxes(
    draw: ImageDraw.ImageDraw,
    tokens: list,
    scale: float,
    font,
    color: tuple[int, ...] = _GRAY,
) -> None:
    """Draw thin outlines around individual glyph boxes."""
    for t in tokens:
        sx0, sy0 = _scale(t.x0, t.y0, scale)
        sx1, sy1 = _scale(t.x1, t.y1, scale)
        draw.rectangle([(sx0, sy0), (sx1, sy1)], outline=color, width=1)


def _draw_block_outlines(
    draw: ImageDraw.ImageDraw,
    blocks: list[BlockCluster],
    scale: float,
    font,
) -> None:
    """Draw all block bounding boxes, colour-coded by type."""
    for i, blk in enumerate(blocks):
        bbox = blk.bbox()
        lbl = getattr(blk, "label", None) or ""
        is_table = getattr(blk, "is_table", False)
        is_notes = getattr(blk, "is_notes", False)
        is_header = lbl in ("note_column_header", "note_column_subheader")

        if is_header:
            color = _RED
        elif is_notes:
            color = _GREEN
        elif is_table:
            color = _ORANGE
        else:
            color = _CYAN

        sx0, sy0 = _scale(bbox[0], bbox[1], scale)
        sx1, sy1 = _scale(bbox[2], bbox[3], scale)
        draw.rectangle([(sx0, sy0), (sx1, sy1)], outline=color, width=1)
        tag = f"B{i}"
        if lbl:
            tag += f" [{lbl}]"
        tb = draw.textbbox((0, 0), tag, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.rectangle(
            [(sx0, max(0, sy0 - th - 4)), (sx0 + tw + 4, max(0, sy0 - 2))],
            fill=(*color[:3], 140),
        )
        draw.text(
            (sx0 + 2, max(0, sy0 - th - 2)),
            tag,
            fill=_LABEL_FG,
            font=font,
        )


def _draw_structural_boxes_layer(
    draw: ImageDraw.ImageDraw,
    structural_boxes: list,
    scale: float,
    font,
    color: tuple[int, ...] = _ORANGE,
    label_bg: tuple[int, ...] = _ORANGE_BG,
) -> None:
    """Draw structural boxes detected from PDF graphics."""
    for sb in structural_boxes:
        cls = getattr(sb, "classification", None) or "unknown"
        bbox = (
            sb.bbox
            if hasattr(sb, "bbox") and not callable(sb.bbox)
            else (sb.x0, sb.y0, sb.x1, sb.y1)
        )
        _draw_labeled_rect(draw, bbox, f"SB:{cls}", color, label_bg, scale, font, 2)


def _draw_zone_layer(
    draw: ImageDraw.ImageDraw,
    zones: list,
    scale: float,
    font,
) -> None:
    """Draw detected page zones (title_block, notes, border, etc.)."""
    zone_colors = {
        "title_block": _RED,
        "notes": _GREEN,
        "border": _GRAY,
        "page": _CYAN,
        "drawing": _TEAL,
    }
    zone_bgs = {
        "title_block": _RED_BG,
        "notes": _GREEN_BG,
        "border": _GRAY_BG,
        "page": _CYAN_BG,
        "drawing": _TEAL_BG,
    }
    for z in zones:
        tag = getattr(z, "zone_type", None) or getattr(z, "tag", "?")
        tag_str = str(tag.value) if hasattr(tag, "value") else str(tag)
        color = zone_colors.get(tag_str, _YELLOW)
        bg = zone_bgs.get(tag_str, _YELLOW_BG)
        bbox = (z.x0, z.y0, z.x1, z.y1)
        _draw_labeled_rect(draw, bbox, f"Zone:{tag_str}", color, bg, scale, font, 3)


def _draw_legend_layer(
    draw: ImageDraw.ImageDraw,
    legend_regions: list,
    scale: float,
    font,
    color: tuple[int, ...] = _TEAL,
    label_bg: tuple[int, ...] = _TEAL_BG,
) -> None:
    """Draw legend regions."""
    for i, lr in enumerate(legend_regions):
        bbox = (
            lr.bbox()
            if callable(getattr(lr, "bbox", None))
            else (lr.x0, lr.y0, lr.x1, lr.y1)
        )
        entry_count = len(getattr(lr, "entries", []))
        _draw_labeled_rect(
            draw, bbox, f"Legend[{entry_count}]", color, label_bg, scale, font, 2
        )


def _draw_abbreviation_layer(
    draw: ImageDraw.ImageDraw,
    abbr_regions: list,
    scale: float,
    font,
    color: tuple[int, ...] = _PINK,
    label_bg: tuple[int, ...] = _PINK_BG,
) -> None:
    """Draw abbreviation regions."""
    for i, ar in enumerate(abbr_regions):
        bbox = (
            ar.bbox()
            if callable(getattr(ar, "bbox", None))
            else (ar.x0, ar.y0, ar.x1, ar.y1)
        )
        entry_count = len(getattr(ar, "entries", []))
        _draw_labeled_rect(
            draw, bbox, f"Abbr[{entry_count}]", color, label_bg, scale, font, 2
        )


def _draw_revision_layer(
    draw: ImageDraw.ImageDraw,
    rev_regions: list,
    scale: float,
    font,
    color: tuple[int, ...] = _YELLOW,
    label_bg: tuple[int, ...] = _YELLOW_BG,
) -> None:
    """Draw revision regions."""
    for i, rr in enumerate(rev_regions):
        bbox = (
            rr.bbox()
            if callable(getattr(rr, "bbox", None))
            else (rr.x0, rr.y0, rr.x1, rr.y1)
        )
        entry_count = len(getattr(rr, "entries", []))
        _draw_labeled_rect(
            draw, bbox, f"Rev[{entry_count}]", color, label_bg, scale, font, 2
        )


def _draw_std_detail_layer(
    draw: ImageDraw.ImageDraw,
    detail_regions: list,
    scale: float,
    font,
    color: tuple[int, ...] = _CYAN,
    label_bg: tuple[int, ...] = _CYAN_BG,
) -> None:
    """Draw standard-detail regions."""
    for i, dr in enumerate(detail_regions):
        bbox = (
            dr.bbox()
            if callable(getattr(dr, "bbox", None))
            else (dr.x0, dr.y0, dr.x1, dr.y1)
        )
        entry_count = len(getattr(dr, "entries", []))
        _draw_labeled_rect(
            draw, bbox, f"StdDet[{entry_count}]", color, label_bg, scale, font, 2
        )


# ---------------------------------------------------------------------------
# Public API: render_overlay
# ---------------------------------------------------------------------------


def render_overlay(
    pdf_path: str | Path,
    page_idx: int = 0,
    *,
    cfg: GroupingConfig | dict | None = None,
    layers: dict[str, bool] | None = None,
    resolution: int = 200,
    json_path: str | Path | None = None,
) -> Image.Image:
    """Run TOCR pipeline and composite requested overlay layers.

    Parameters
    ----------
    pdf_path : path to the PDF (needed for background raster)
    page_idx : zero-based page index
    cfg : GroupingConfig or dict of overrides (applied on top of defaults)
    layers : ``{"green": True, "purple": True, "red": True, ...}``
             New layer keys: ``glyph_boxes``, ``block_outlines``,
             ``structural``, ``zones``, ``legends``, ``abbreviations``,
             ``revisions``, ``std_details``
    resolution : render DPI
    json_path : read pre-computed extraction JSON instead of re-running pipeline

    Returns
    -------
    PIL.Image.Image – RGBA composited overlay
    """
    pdf_path = Path(pdf_path)
    if layers is None:
        layers = {"green": True, "purple": True, "red": True}

    # Resolve config
    if isinstance(cfg, dict):
        real_cfg = GroupingConfig()
        for k, v in cfg.items():
            if hasattr(real_cfg, k):
                setattr(real_cfg, k, type(getattr(real_cfg, k))(v))
        cfg = real_cfg
    elif cfg is None:
        cfg = GroupingConfig()

    # ── Pipeline data ────────────────────────────────────────────────
    graphics = None  # pre-extracted graphics for analysis layers
    if json_path is not None:
        raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
        tokens, blocks, notes_columns, page_w, page_h = deserialize_page(raw)
        # Still need a single open for background raster (+ optional graphics)
        needs_analysis = any(
            layers.get(k, False)
            for k in (
                "structural",
                "zones",
                "legends",
                "abbreviations",
                "revisions",
                "std_details",
            )
        )
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_idx]
            bg = page.to_image(resolution=resolution).original.copy()
            if needs_analysis:
                from plancheck.analysis.graphics import extract_graphics_from_data

                graphics = extract_graphics_from_data(
                    page_idx,
                    list(page.lines),
                    list(page.rects),
                    list(page.curves),
                )
    else:
        # Single open: TOCR + background + graphics all at once
        needs_analysis = any(
            layers.get(k, False)
            for k in (
                "structural",
                "zones",
                "legends",
                "abbreviations",
                "revisions",
                "std_details",
            )
        )
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_idx]
            result = extract_tocr_from_page(page, page_idx, cfg, mode="minimal")
            tokens = result.tokens
            page_w = result.page_width
            page_h = result.page_height
            bg = page.to_image(resolution=resolution).original.copy()
            if needs_analysis:
                from plancheck.analysis.graphics import extract_graphics_from_data

                graphics = extract_graphics_from_data(
                    page_idx,
                    list(page.lines),
                    list(page.rects),
                    list(page.curves),
                )
        tokens = nms_prune(tokens, cfg.iou_prune)
        blocks = build_clusters_v2(tokens, cfg)
        notes_columns = group_notes_columns(blocks, cfg=cfg)
        link_continued_columns(notes_columns, blocks=blocks, cfg=cfg)

    scale = resolution / 72.0
    img = bg.convert("RGBA")
    img_w, img_h = int(page_w * scale), int(page_h * scale)
    if img.size != (img_w, img_h):
        img = img.resize((img_w, img_h))

    draw = ImageDraw.Draw(img, "RGBA")
    font = _load_font(scale)

    # ── Composite layers ─────────────────────────────────────────────
    # New: glyph boxes (draw first so they're behind block outlines)
    if layers.get("glyph_boxes", False):
        _draw_glyph_boxes(draw, tokens, scale, font)

    # New: block outlines (all blocks colour-coded by type)
    if layers.get("block_outlines", False):
        _draw_block_outlines(draw, blocks, scale, font)

    # Original three layers
    if layers.get("green", False):
        _draw_green_layer(draw, blocks, scale, font)
    if layers.get("purple", False):
        _draw_purple_layer(draw, blocks, notes_columns, scale, font)
    if layers.get("red", False):
        _draw_red_layer(draw, blocks, scale, font)

    # New analysis layers (use pre-extracted graphics from single open)
    needs_analysis = any(
        layers.get(k, False)
        for k in (
            "structural",
            "zones",
            "legends",
            "abbreviations",
            "revisions",
            "std_details",
        )
    )
    if needs_analysis:
        if graphics is None:
            graphics = []

        if layers.get("structural", False):
            try:
                sboxes = detect_structural_boxes(graphics, page_w, page_h)
                classify_structural_boxes(
                    sboxes, blocks, page_w, page_h, config=cfg.analysis
                )
                _draw_structural_boxes_layer(draw, sboxes, scale, font)
            except Exception:  # noqa: BLE001 — layer rendering is best-effort
                pass

        # Exclusion zones for downstream detectors
        exclusion_zones = []

        if layers.get("legends", False):
            try:
                lregs = detect_legend_regions(
                    blocks,
                    graphics,
                    page_w,
                    page_h,
                    exclusion_zones=exclusion_zones,
                    cfg=cfg,
                )
                _draw_legend_layer(draw, lregs, scale, font)
            except Exception:  # noqa: BLE001 — layer rendering is best-effort
                pass

        if layers.get("abbreviations", False):
            try:
                aregs = detect_abbreviation_regions(
                    blocks, graphics, page_w, page_h, cfg=cfg
                )
                _draw_abbreviation_layer(draw, aregs, scale, font)
            except Exception:  # noqa: BLE001 — layer rendering is best-effort
                pass

        if layers.get("revisions", False):
            try:
                rregs = detect_revision_regions(
                    blocks,
                    graphics,
                    page_w,
                    page_h,
                    exclusion_zones=exclusion_zones,
                    cfg=cfg,
                )
                _draw_revision_layer(draw, rregs, scale, font)
            except Exception:  # noqa: BLE001 — layer rendering is best-effort
                pass

        if layers.get("std_details", False):
            try:
                dregs = detect_standard_detail_regions(
                    blocks,
                    graphics,
                    page_w,
                    page_h,
                    exclusion_zones=exclusion_zones,
                    cfg=cfg,
                )
                _draw_std_detail_layer(draw, dregs, scale, font)
            except Exception:  # noqa: BLE001 — layer rendering is best-effort
                pass

        if layers.get("zones", False):
            try:
                zones = detect_zones(
                    page_w, page_h, blocks, notes_columns=notes_columns, cfg=cfg
                )
                _draw_zone_layer(draw, zones, scale, font)
            except Exception:  # noqa: BLE001 — layer rendering is best-effort
                pass

    return img


# ---------------------------------------------------------------------------
# CLI entry point (for standalone use / LLM scripting)
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Unified overlay viewer")
    parser.add_argument("pdf", type=Path, help="PDF file")
    parser.add_argument("--page", type=int, default=0, help="Zero-based page index")
    parser.add_argument("--resolution", type=int, default=200)
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--json", type=Path, default=None, metavar="EXTRACTION_JSON")
    parser.add_argument(
        "--layers",
        type=str,
        default="green,purple,red",
        help="Comma-separated layer names to enable (green, purple, red)",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="JSON dict of GroupingConfig overrides, e.g. '{\"tocr_x_tolerance\": 2.0}'",
    )
    args = parser.parse_args()

    layer_set = {l.strip() for l in args.layers.split(",")}
    layers = {k: (k in layer_set) for k in ("green", "purple", "red")}

    cfg_overrides = json.loads(args.cfg) if args.cfg else None

    img = render_overlay(
        args.pdf,
        args.page,
        cfg=cfg_overrides,
        layers=layers,
        resolution=args.resolution,
        json_path=args.json,
    )

    out = args.out
    if out is None:
        out = latest_overlays_dir() / f"page_{args.page}_debug_overlay.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out))
    print(f"Overlay saved: {out} ({img.width}×{img.height})")


if __name__ == "__main__":
    main()
