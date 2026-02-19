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
    layers : ``{"green": True, "purple": True, "red": True}``
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
    if layers.get("green", False):
        _draw_green_layer(draw, blocks, scale, font)
    if layers.get("purple", False):
        _draw_purple_layer(draw, blocks, notes_columns, scale, font)
    if layers.get("red", False):
        _draw_red_layer(draw, blocks, scale, font)

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

    def __init__(self, notebook) -> None:
        tk, ttk, filedialog, messagebox = _import_tk()
        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox

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

        self._green_var = tk.BooleanVar(value=True)
        self._purple_var = tk.BooleanVar(value=True)
        self._red_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(mid, text="Green: notes blocks", variable=self._green_var).grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        ttk.Checkbutton(mid, text="Purple: columns", variable=self._purple_var).grid(
            row=0, column=1, sticky="w", padx=(0, 8)
        )
        ttk.Checkbutton(mid, text="Red: headers", variable=self._red_var).grid(
            row=0, column=2, sticky="w", padx=(0, 8)
        )

        self._render_btn = ttk.Button(mid, text="Render", command=self._on_render)
        self._render_btn.grid(row=0, column=4, padx=(8, 0))

        ttk.Button(mid, text="Save PNG...", command=self._on_save).grid(
            row=0, column=5, padx=(4, 0)
        )

        # --- Zoom controls ---
        zoom_frame = ttk.Frame(mid)
        zoom_frame.grid(row=0, column=6, padx=(12, 0))

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
