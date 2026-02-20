"""Unified visual-debug overlay viewer.

Provides two public surfaces:

1. ``render_overlay(pdf_path, page_idx, ...)`` – pure function that runs the
   TOCR pipeline with the supplied *cfg*, composites the requested layers
   (green notes / purple columns / red headers) onto the rasterised page
   background, and returns a ``PIL.Image``.  An LLM can call this directly.

2. ``OverlayViewerTab(notebook)`` – a ``ttk.Frame`` that plugs into the
   existing tkinter GUI as a new tab, exposing PDF / page / DPI pickers,
   layer toggles, and live ``GroupingConfig`` knob editors.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

# Ensure plancheck is importable when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

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

# Analysis / detection imports (lazy-safe: modules exist alongside plancheck)
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scale(x: float, y: float, s: float) -> tuple[int, int]:
    return int(x * s), int(y * s)


def _load_font(scale: float) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", max(10, int(10 * scale / 2.78)))
    except OSError:
        return ImageFont.load_default()


def _latest_overlays_dir() -> Path:
    runs_dir = Path("runs")
    if runs_dir.is_dir():
        run_dirs = sorted(runs_dir.iterdir(), reverse=True)
        if run_dirs:
            d = run_dirs[0] / "overlays"
            d.mkdir(parents=True, exist_ok=True)
            return d
    return Path(".")


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
    # Build labels for columns
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
    if json_path is not None:
        raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
        tokens, blocks, notes_columns, page_w, page_h = deserialize_page(raw)
    else:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_idx]
            result = extract_tocr_from_page(page, page_idx, cfg, mode="minimal")
            tokens = result.tokens
            page_w = result.page_width
            page_h = result.page_height
        tokens = nms_prune(tokens, cfg.iou_prune)
        blocks = build_clusters_v2(tokens, page_h, cfg)
        notes_columns = group_notes_columns(blocks, cfg=cfg)
        link_continued_columns(notes_columns, blocks=blocks, cfg=cfg)

    # ── Background raster ────────────────────────────────────────────
    with pdfplumber.open(pdf_path) as pdf:
        bg = pdf.pages[page_idx].to_image(resolution=resolution).original.copy()

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

    # New analysis layers (require graphics extraction)
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
        try:
            graphics = extract_graphics(str(pdf_path), page_idx)
        except Exception:
            graphics = []

        if layers.get("structural", False):
            try:
                sboxes = detect_structural_boxes(graphics, page_w, page_h)
                classify_structural_boxes(sboxes, blocks, page_w, page_h)
                _draw_structural_boxes_layer(draw, sboxes, scale, font)
            except Exception:
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
            except Exception:
                pass

        if layers.get("abbreviations", False):
            try:
                aregs = detect_abbreviation_regions(
                    blocks, graphics, page_w, page_h, cfg=cfg
                )
                _draw_abbreviation_layer(draw, aregs, scale, font)
            except Exception:
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
            except Exception:
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
            except Exception:
                pass

        if layers.get("zones", False):
            try:
                zones = detect_zones(
                    page_w, page_h, blocks, notes_columns=notes_columns, cfg=cfg
                )
                _draw_zone_layer(draw, zones, scale, font)
            except Exception:
                pass

    return img


# ---------------------------------------------------------------------------
# GroupingConfig knob metadata (for GUI spinboxes)
# ---------------------------------------------------------------------------

# Fields exposed in the viewer knobs panel.  Only TOCR + geometry knobs —
# VOCR / reconcile / preprocessing knobs are irrelevant for TOCR-stage
# visual debugging.
_KNOB_FIELDS: list[str] = [
    # Geometry core
    "iou_prune",
    "horizontal_tol_mult",
    "vertical_tol_mult",
    "row_gap_mult",
    "block_gap_mult",
    "max_block_height_mult",
    "row_split_gap_mult",
    "column_gap_mult",
    "gutter_width_mult",
    "max_column_width_mult",
    "max_row_width_mult",
    "table_regular_tol",
    "span_gap_mult",
    "content_band_top",
    "content_band_bottom",
    # TOCR extraction
    "tocr_x_tolerance",
    "tocr_y_tolerance",
    "tocr_dedup_iou",
    "tocr_margin_pts",
    "tocr_mojibake_threshold",
]


# ---------------------------------------------------------------------------
# Tkinter tab
# ---------------------------------------------------------------------------


def _import_tk():
    """Lazy-import tkinter so headless ``render_overlay`` works without it."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    return tk, ttk, filedialog, messagebox


class OverlayViewerTab:
    """Visual Debug tab that plugs into the existing tkinter Notebook."""

    def __init__(self, notebook, gui_state=None) -> None:
        tk, ttk, filedialog, messagebox = _import_tk()
        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.state = gui_state

        self.frame = ttk.Frame(notebook)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(2, weight=1)  # image canvas row expands
        notebook.add(self.frame, text="Visual Debug")

        self._pdf_path: Path | None = None
        self._last_img: Image.Image | None = None
        self._photo = None  # prevent GC
        self._render_thread: threading.Thread | None = None
        self._zoom: float = 1.0  # current zoom factor
        self._zoom_levels = [
            0.1,
            0.15,
            0.2,
            0.25,
            0.33,
            0.5,
            0.67,
            0.75,
            1.0,
            1.25,
            1.5,
            2.0,
            3.0,
            4.0,
        ]

        self._build_controls(notebook)
        self._build_image_canvas()
        self._build_status_bar()

    # ── Controls ─────────────────────────────────────────────────────

    def _build_controls(self, notebook) -> None:
        tk, ttk = self.tk, self.ttk
        pad = {"padx": 8, "pady": 4}

        # --- Top bar: PDF + page + DPI ---
        top = ttk.LabelFrame(self.frame, text="Source", padding=6)
        top.grid(row=0, column=0, sticky="ew", **pad)
        top.columnconfigure(1, weight=1)

        ttk.Button(top, text="Select PDF...", command=self._pick_pdf).grid(
            row=0, column=0, padx=(0, 4)
        )
        self._pdf_label_var = tk.StringVar(value="No file selected")
        ttk.Label(top, textvariable=self._pdf_label_var, foreground="gray").grid(
            row=0, column=1, sticky="w"
        )

        ttk.Label(top, text="Page (0-based):").grid(row=0, column=2, padx=(12, 2))
        self._page_var = tk.StringVar(value="0")
        ttk.Spinbox(top, textvariable=self._page_var, from_=0, to=999, width=5).grid(
            row=0, column=3
        )

        ttk.Label(top, text="DPI:").grid(row=0, column=4, padx=(12, 2))
        self._dpi_var = tk.StringVar(value="150")
        ttk.Spinbox(
            top,
            textvariable=self._dpi_var,
            values=(72, 100, 150, 200, 300),
            width=5,
            state="readonly",
        ).grid(row=0, column=5)

        # --- Layer toggles + render button ---
        mid = ttk.LabelFrame(self.frame, text="Layers & Knobs", padding=6)
        mid.grid(row=1, column=0, sticky="ew", **pad)
        mid.columnconfigure(3, weight=1)

        # Core layer toggles (row 0)
        self._green_var = tk.BooleanVar(value=True)
        self._purple_var = tk.BooleanVar(value=True)
        self._red_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(mid, text="Green: notes", variable=self._green_var).grid(
            row=0, column=0, sticky="w", padx=(0, 6)
        )
        ttk.Checkbutton(mid, text="Purple: columns", variable=self._purple_var).grid(
            row=0, column=1, sticky="w", padx=(0, 6)
        )
        ttk.Checkbutton(mid, text="Red: headers", variable=self._red_var).grid(
            row=0, column=2, sticky="w", padx=(0, 6)
        )

        # Extended layer toggles (row 0 continued + row 0b)
        self._glyph_var = tk.BooleanVar(value=False)
        self._blocks_var = tk.BooleanVar(value=False)
        self._structural_var = tk.BooleanVar(value=False)
        self._zones_var = tk.BooleanVar(value=False)
        self._legends_var = tk.BooleanVar(value=False)
        self._abbrev_var = tk.BooleanVar(value=False)
        self._revisions_var = tk.BooleanVar(value=False)
        self._stddet_var = tk.BooleanVar(value=False)

        ext_layer_frame = ttk.Frame(mid)
        ext_layer_frame.grid(row=0, column=3, sticky="w", padx=(6, 0))

        ext_layers_btn = ttk.Menubutton(ext_layer_frame, text="More Layers ▾")
        ext_layers_btn.pack(side="left")
        ext_menu = tk.Menu(ext_layers_btn, tearoff=False)
        ext_layers_btn["menu"] = ext_menu
        ext_menu.add_checkbutton(label="Glyph Boxes", variable=self._glyph_var)
        ext_menu.add_checkbutton(label="Block Outlines", variable=self._blocks_var)
        ext_menu.add_separator()
        ext_menu.add_checkbutton(
            label="Structural Boxes", variable=self._structural_var
        )
        ext_menu.add_checkbutton(label="Zones", variable=self._zones_var)
        ext_menu.add_separator()
        ext_menu.add_checkbutton(label="Legends", variable=self._legends_var)
        ext_menu.add_checkbutton(label="Abbreviations", variable=self._abbrev_var)
        ext_menu.add_checkbutton(label="Revisions", variable=self._revisions_var)
        ext_menu.add_checkbutton(label="Standard Details", variable=self._stddet_var)

        self._render_btn = ttk.Button(mid, text="Render", command=self._on_render)
        self._render_btn.grid(row=0, column=4, padx=(8, 0))

        ttk.Button(mid, text="Save PNG...", command=self._on_save).grid(
            row=0, column=5, padx=(4, 0)
        )

        ttk.Button(mid, text="Load from Run...", command=self._load_from_run).grid(
            row=0, column=6, padx=(4, 0)
        )

        # --- Zoom controls ---
        zoom_frame = ttk.Frame(mid)
        zoom_frame.grid(row=0, column=7, padx=(12, 0))

        ttk.Button(zoom_frame, text="−", width=2, command=self._zoom_out).grid(
            row=0, column=0
        )
        self._zoom_label_var = tk.StringVar(value="100%")
        ttk.Label(
            zoom_frame, textvariable=self._zoom_label_var, width=6, anchor="center"
        ).grid(row=0, column=1, padx=2)
        ttk.Button(zoom_frame, text="+", width=2, command=self._zoom_in).grid(
            row=0, column=2
        )
        ttk.Button(zoom_frame, text="Fit", width=3, command=self._zoom_fit).grid(
            row=0, column=3, padx=(4, 0)
        )
        ttk.Button(zoom_frame, text="1:1", width=3, command=self._zoom_reset).grid(
            row=0, column=4, padx=(2, 0)
        )

        # --- Knobs (collapsible scroll area) ---
        knob_toggle_frame = ttk.Frame(mid)
        knob_toggle_frame.grid(row=1, column=0, columnspan=6, sticky="ew", pady=(6, 0))

        self._knobs_visible = tk.BooleanVar(value=False)
        self._knob_toggle_btn = ttk.Checkbutton(
            knob_toggle_frame,
            text="Show GroupingConfig knobs",
            variable=self._knobs_visible,
            command=self._toggle_knobs,
        )
        self._knob_toggle_btn.grid(row=0, column=0, sticky="w")

        ttk.Button(
            knob_toggle_frame, text="Reset defaults", command=self._reset_knobs
        ).grid(row=0, column=1, padx=(8, 0))

        # Scrollable knob frame (initially hidden)
        self._knob_container = ttk.Frame(mid)
        # Not gridded until toggled visible.

        self._knob_canvas = tk.Canvas(
            self._knob_container, height=140, highlightthickness=0
        )
        knob_sb = ttk.Scrollbar(
            self._knob_container, orient="vertical", command=self._knob_canvas.yview
        )
        self._knob_inner = ttk.Frame(self._knob_canvas)
        self._knob_inner.bind(
            "<Configure>",
            lambda e: self._knob_canvas.configure(
                scrollregion=self._knob_canvas.bbox("all")
            ),
        )
        self._knob_canvas.create_window((0, 0), window=self._knob_inner, anchor="nw")
        self._knob_canvas.configure(yscrollcommand=knob_sb.set)
        self._knob_canvas.pack(side="left", fill="both", expand=True)
        knob_sb.pack(side="right", fill="y")

        # Populate knob widgets
        defaults = GroupingConfig()
        self._knob_vars: dict[str, tk.StringVar] = {}
        for row_i, name in enumerate(_KNOB_FIELDS):
            default_val = getattr(defaults, name)
            ttk.Label(self._knob_inner, text=name, width=28, anchor="w").grid(
                row=row_i, column=0, sticky="w", padx=(2, 6)
            )
            sv = tk.StringVar(value=str(default_val))
            self._knob_vars[name] = sv
            ttk.Entry(self._knob_inner, textvariable=sv, width=10).grid(
                row=row_i, column=1, sticky="w"
            )
            ttk.Label(
                self._knob_inner, text=f"(default: {default_val})", foreground="gray"
            ).grid(row=row_i, column=2, sticky="w", padx=(6, 0))

    def _toggle_knobs(self) -> None:
        if self._knobs_visible.get():
            self._knob_container.grid(
                row=2, column=0, columnspan=6, sticky="nsew", pady=(4, 0)
            )
        else:
            self._knob_container.grid_remove()

    def _reset_knobs(self) -> None:
        defaults = GroupingConfig()
        for name, sv in self._knob_vars.items():
            sv.set(str(getattr(defaults, name)))

    # ── Image canvas ─────────────────────────────────────────────────

    def _build_image_canvas(self) -> None:
        tk, ttk = self.tk, self.ttk

        canvas_frame = ttk.Frame(self.frame)
        canvas_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=4)
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self._canvas = tk.Canvas(canvas_frame, bg="#2b2b2b", highlightthickness=0)
        self._h_scroll = ttk.Scrollbar(
            canvas_frame, orient="horizontal", command=self._canvas.xview
        )
        self._v_scroll = ttk.Scrollbar(
            canvas_frame, orient="vertical", command=self._canvas.yview
        )
        self._canvas.configure(
            xscrollcommand=self._h_scroll.set,
            yscrollcommand=self._v_scroll.set,
        )
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._v_scroll.grid(row=0, column=1, sticky="ns")
        self._h_scroll.grid(row=1, column=0, sticky="ew")

        self._canvas_img_id = None

        # Mouse-wheel zoom (bind on canvas)
        self._canvas.bind("<MouseWheel>", self._on_mousewheel_zoom)
        # Also allow standard scroll when not zooming (Shift+wheel = horizontal)
        self._canvas.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)

    # ── Status bar ───────────────────────────────────────────────────

    def _build_status_bar(self) -> None:
        tk, ttk = self.tk, self.ttk
        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(
            self.frame,
            textvariable=self._status_var,
            relief="sunken",
            anchor="w",
            padding=(6, 2),
        ).grid(row=3, column=0, sticky="ew")

    # ── Actions ──────────────────────────────────────────────────────

    def _pick_pdf(self) -> None:
        f = self.filedialog.askopenfilename(
            title="Select PDF",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")],
            initialdir=Path(__file__).parent.parent / "input",
        )
        if f:
            self._pdf_path = Path(f)
            self._pdf_label_var.set(self._pdf_path.name)

    def _collect_cfg(self) -> GroupingConfig:
        cfg = GroupingConfig()
        for name, sv in self._knob_vars.items():
            raw = sv.get().strip()
            if not raw:
                continue
            default_val = getattr(cfg, name)
            try:
                if isinstance(default_val, bool):
                    setattr(cfg, name, raw.lower() in ("1", "true", "yes"))
                elif isinstance(default_val, int):
                    setattr(cfg, name, int(raw))
                elif isinstance(default_val, float):
                    setattr(cfg, name, float(raw))
                else:
                    setattr(cfg, name, raw)
            except (ValueError, TypeError):
                pass  # keep default
        return cfg

    def _on_render(self) -> None:
        if self._pdf_path is None:
            self.messagebox.showwarning("No PDF", "Select a PDF file first.")
            return
        if self._render_thread and self._render_thread.is_alive():
            return  # already rendering

        self._status_var.set("Rendering...")
        self._render_btn.config(state="disabled")

        page_idx = int(self._page_var.get())
        resolution = int(self._dpi_var.get())
        cfg = self._collect_cfg()
        layers = {
            "green": self._green_var.get(),
            "purple": self._purple_var.get(),
            "red": self._red_var.get(),
            "glyph_boxes": self._glyph_var.get(),
            "block_outlines": self._blocks_var.get(),
            "structural": self._structural_var.get(),
            "zones": self._zones_var.get(),
            "legends": self._legends_var.get(),
            "abbreviations": self._abbrev_var.get(),
            "revisions": self._revisions_var.get(),
            "std_details": self._stddet_var.get(),
        }
        pdf_path = self._pdf_path

        def worker():
            t0 = time.perf_counter()
            try:
                img = render_overlay(
                    pdf_path,
                    page_idx,
                    cfg=cfg,
                    layers=layers,
                    resolution=resolution,
                )
                elapsed = time.perf_counter() - t0
                self.frame.after(0, lambda: self._show_image(img, elapsed))
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                self.frame.after(0, lambda: self._render_error(str(exc), elapsed))

        self._render_thread = threading.Thread(target=worker, daemon=True)
        self._render_thread.start()

    def _show_image(self, img: Image.Image, elapsed: float) -> None:
        self._last_img = img
        self._zoom = 1.0
        self._apply_zoom()
        self._status_var.set(f"Rendered {img.width}×{img.height} in {elapsed:.1f}s")
        self._render_btn.config(state="normal")

    def _apply_zoom(self) -> None:
        """Redraw the canvas image at the current zoom level."""
        from PIL import ImageTk

        if self._last_img is None:
            return
        img = self._last_img
        z = self._zoom
        disp_w = max(1, int(img.width * z))
        disp_h = max(1, int(img.height * z))
        resized = img.resize((disp_w, disp_h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        if self._canvas_img_id is not None:
            self._canvas.delete(self._canvas_img_id)
        self._canvas_img_id = self._canvas.create_image(
            0, 0, anchor="nw", image=self._photo
        )
        self._canvas.configure(scrollregion=(0, 0, disp_w, disp_h))
        self._zoom_label_var.set(f"{int(z * 100)}%")

    # ── Zoom helpers ─────────────────────────────────────────────────

    def _zoom_in(self) -> None:
        if self._last_img is None:
            return
        # Find next level above current zoom
        for lvl in self._zoom_levels:
            if lvl > self._zoom + 0.001:
                self._zoom = lvl
                self._apply_zoom()
                return

    def _zoom_out(self) -> None:
        if self._last_img is None:
            return
        # Find next level below current zoom
        for lvl in reversed(self._zoom_levels):
            if lvl < self._zoom - 0.001:
                self._zoom = lvl
                self._apply_zoom()
                return

    def _zoom_reset(self) -> None:
        if self._last_img is None:
            return
        self._zoom = 1.0
        self._apply_zoom()

    def _zoom_fit(self) -> None:
        """Fit the image within the visible canvas area."""
        if self._last_img is None:
            return
        self._canvas.update_idletasks()
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        iw, ih = self._last_img.width, self._last_img.height
        self._zoom = min(cw / iw, ch / ih)
        self._apply_zoom()

    def _on_mousewheel_zoom(self, event) -> None:
        """Ctrl-free mouse wheel zoom (standard zoom behaviour)."""
        if self._last_img is None:
            return
        # On Windows event.delta is typically ±120
        if event.delta > 0:
            self._zoom_in()
        else:
            self._zoom_out()

    def _on_shift_mousewheel(self, event) -> None:
        """Shift+wheel for horizontal scroll."""
        self._canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

    def _render_error(self, msg: str, elapsed: float) -> None:
        self._status_var.set(f"Error ({elapsed:.1f}s): {msg}")
        self._render_btn.config(state="normal")

    def _on_save(self) -> None:
        if self._last_img is None:
            self.messagebox.showinfo("Nothing to save", "Render an overlay first.")
            return
        path = self.filedialog.asksaveasfilename(
            title="Save overlay PNG",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")],
            initialdir=_latest_overlays_dir(),
            initialfile=f"page_{self._page_var.get()}_debug_overlay.png",
        )
        if path:
            self._last_img.save(path)
            self._status_var.set(f"Saved: {path}")

    def _load_from_run(self) -> None:
        """Load pre-computed extraction JSON from a run directory and render."""
        run_dir = self.filedialog.askdirectory(
            title="Select Run Directory",
            initialdir=str(Path(__file__).resolve().parent.parent.parent / "runs"),
        )
        if not run_dir:
            return
        run_dir = Path(run_dir)
        artifacts = run_dir / "artifacts"
        if not artifacts.is_dir():
            self.messagebox.showwarning(
                "No Artifacts", f"No artifacts/ folder in {run_dir.name}."
            )
            return

        # Find extraction JSON files
        jsons = sorted(artifacts.glob("page_*_extraction.json"))
        if not jsons:
            jsons = sorted(artifacts.glob("*extraction*.json"))
        if not jsons:
            self.messagebox.showwarning(
                "No Data", "No extraction JSON found in artifacts/."
            )
            return

        # If multiple pages, let user pick
        if len(jsons) == 1:
            json_path = jsons[0]
        else:
            # Simple page picker dialog
            page_names = [j.stem for j in jsons]
            pick_win = self.tk.Toplevel(self.frame)
            pick_win.title("Select extraction file")
            pick_win.geometry("400x300")
            lb = self.tk.Listbox(pick_win, selectmode="single")
            for n in page_names:
                lb.insert("end", n)
            lb.pack(fill="both", expand=True, padx=10, pady=10)
            chosen = [None]

            def on_ok():
                sel = lb.curselection()
                if sel:
                    chosen[0] = jsons[sel[0]]
                pick_win.destroy()

            self.ttk.Button(pick_win, text="OK", command=on_ok).pack(pady=5)
            pick_win.grab_set()
            pick_win.wait_window()
            json_path = chosen[0]
            if json_path is None:
                return

        # Find PDF path from manifest
        manifest_path = run_dir / "manifest.json"
        pdf_path = None
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                pdf_path_str = manifest.get("pdf_path") or manifest.get("pdf")
                if pdf_path_str:
                    pdf_path = Path(pdf_path_str)
                    if not pdf_path.is_file():
                        pdf_path = None
            except Exception:
                pass

        if pdf_path is None:
            f = self.filedialog.askopenfilename(
                title="Select matching PDF (for background raster)",
                filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")],
            )
            if not f:
                return
            pdf_path = Path(f)

        self._pdf_path = pdf_path
        self._pdf_label_var.set(pdf_path.name)
        self._status_var.set(f"Rendering from {json_path.name}...")
        self._render_btn.config(state="disabled")

        page_idx = int(self._page_var.get())
        resolution = int(self._dpi_var.get())
        cfg = self._collect_cfg()
        layers = {
            "green": self._green_var.get(),
            "purple": self._purple_var.get(),
            "red": self._red_var.get(),
            "glyph_boxes": self._glyph_var.get(),
            "block_outlines": self._blocks_var.get(),
            "structural": self._structural_var.get(),
            "zones": self._zones_var.get(),
            "legends": self._legends_var.get(),
            "abbreviations": self._abbrev_var.get(),
            "revisions": self._revisions_var.get(),
            "std_details": self._stddet_var.get(),
        }

        def worker():
            t0 = time.perf_counter()
            try:
                img = render_overlay(
                    pdf_path,
                    page_idx,
                    cfg=cfg,
                    layers=layers,
                    resolution=resolution,
                    json_path=str(json_path),
                )
                elapsed = time.perf_counter() - t0
                self.frame.after(0, lambda: self._show_image(img, elapsed))
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                self.frame.after(0, lambda: self._render_error(str(exc), elapsed))

        self._render_thread = threading.Thread(target=worker, daemon=True)
        self._render_thread.start()


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
        out = _latest_overlays_dir() / f"page_{args.page}_debug_overlay.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out))
    print(f"Overlay saved: {out} ({img.width}×{img.height})")


if __name__ == "__main__":
    main()
