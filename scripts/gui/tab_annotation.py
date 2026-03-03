"""Tab 6 – Annotation: interactive detection correction UI.

Renders the PDF page with clickable, editable detection boxes.
Every correction is persisted to ``CorrectionStore`` immediately.
"""

from __future__ import annotations

import copy
import json
import threading
import tkinter as tk
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, simpledialog, ttk
from typing import Any, List, Optional
from uuid import uuid4

from PIL import Image, ImageTk

from plancheck.analysis.box_merge import merge_boxes, polygon_bbox
from plancheck.corrections.classifier import ElementClassifier
from plancheck.corrections.features import featurize_region
from plancheck.corrections.store import CorrectionStore
from plancheck.ingest.ingest import (extract_text_in_bbox,
                                     extract_text_in_polygon, point_in_polygon)

from .widgets import LogPanel, StatusBar
from .worker import PipelineWorker


def _reshape_bbox_from_handle(
    orig_bbox: tuple[float, float, float, float],
    handle: str,
    px: float,
    py: float,
    *,
    min_size: float = 1.0,
) -> tuple[float, float, float, float]:
    """Compute a resized bbox when dragging a named handle.

    Coordinates are in PDF space.
    """
    ox0, oy0, ox1, oy1 = orig_bbox
    nx0, ny0, nx1, ny1 = ox0, oy0, ox1, oy1

    if "w" in handle:
        nx0 = min(px, ox1 - min_size)
    if "e" in handle:
        nx1 = max(px, ox0 + min_size)
    if "n" in handle:
        ny0 = min(py, oy1 - min_size)
    if "s" in handle:
        ny1 = max(py, oy0 + min_size)

    return (nx0, ny0, nx1, ny1)


def _scale_polygon_to_bbox(
    orig_bbox: tuple[float, float, float, float],
    polygon: list[tuple[float, float]],
    new_bbox: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    """Scale polygon points from orig_bbox into new_bbox.

    This keeps each point's relative position within the bbox.
    """
    ox0, oy0, ox1, oy1 = orig_bbox
    nx0, ny0, nx1, ny1 = new_bbox

    ow = max(ox1 - ox0, 1e-6)
    oh = max(oy1 - oy0, 1e-6)
    nw = nx1 - nx0
    nh = ny1 - ny0

    scaled: list[tuple[float, float]] = []
    for px, py in polygon:
        u = (px - ox0) / ow
        v = (py - oy0) / oh
        scaled.append((nx0 + u * nw, ny0 + v * nh))
    return scaled

# ── CanvasBox ──────────────────────────────────────────────────────────


@dataclass
class CanvasBox:
    """Tracks a single detection drawn on the canvas."""

    detection_id: str
    element_type: str
    confidence: float | None
    text_content: str
    features: dict
    # bbox in PDF points (x0, y0, x1, y1)
    pdf_bbox: tuple[float, float, float, float]
    # Canvas item IDs (set after drawing)
    rect_id: int = 0
    label_id: int = 0
    conf_dot_id: int = 0
    handle_ids: list[int] = field(default_factory=list)
    # State
    selected: bool = False
    corrected: bool = False
    # Polygon outline for merged boxes: list[(x, y)] in PDF points, or None
    polygon: list[tuple[float, float]] | None = None
    # Detection IDs that were merged into this box (for undo)
    merged_from: list[str] | None = None
    # Group membership
    group_id: str | None = None
    is_group_root: bool = False


# ── Handle positions ───────────────────────────────────────────────────

HANDLE_POSITIONS = ("nw", "n", "ne", "e", "se", "s", "sw", "w")
HANDLE_SIZE = 5  # half-size in pixels


# ── Main Tab ───────────────────────────────────────────────────────────


class AnnotationTab:
    """Interactive annotation tab for detection correction.

    Renders the PDF page with coloured detection boxes.  Supports
    click-to-select, relabel, reshape, add, and reject.  Every action
    is persisted to a :class:`CorrectionStore` immediately on confirm.
    """

    LABEL_COLORS: dict[str, str] = {
        "notes_column": "#1ea01e",
        "notes_block": "#2ecc40",
        "header": "#dc1e1e",
        "abbreviations": "#e05096",
        "legend": "#009682",
        "revision": "#dcc800",
        "standard_detail": "#0090dc",
        "title_block": "#8c00c8",
        "misc_title": "#ff8c00",
    }

    ELEMENT_TYPES: list[str] = list(LABEL_COLORS.keys())

    def __init__(self, notebook: ttk.Notebook, gui_state: Any) -> None:
        self.notebook = notebook
        self.state = gui_state
        self.root = notebook.winfo_toplevel()

        self.frame = ttk.Frame(notebook)
        self.frame.columnconfigure(0, weight=3)
        self.frame.columnconfigure(1, weight=0)
        self.frame.rowconfigure(1, weight=1)
        notebook.add(self.frame, text="ML Trainer")

        # ── State ──────────────────────────────────────────────────
        self._pdf_path: Path | None = None
        self._doc_id: str | None = None
        self._run_id: str = ""
        self._page: int = 0
        self._page_count: int = 0
        self._resolution: int = 150
        self._scale: float = self._resolution / 72.0
        self._zoom: float = 1.0  # additional zoom on top of _scale
        self._bg_image: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._detections: list[dict] = []
        self._canvas_boxes: list[CanvasBox] = []
        self._selected_box: CanvasBox | None = None
        self._multi_selected: list[CanvasBox] = []
        self._mode: str = "select"

        self._draw_start: tuple[float, float] | None = None
        self._draw_rect_id: int | None = None
        self._lasso_start: tuple[float, float] | None = None
        self._lasso_rect_id: int | None = None
        self._lasso_shift: bool = False
        self._lasso_word: bool = False
        self._word_click_candidate_rid: int | None = None
        self._session_id: str = uuid4().hex[:8]
        self._session_count: int = 0
        self._store = CorrectionStore()
        self._worker: PipelineWorker | None = None
        self._classifier = ElementClassifier()

        # Shutdown safety: avoid scheduling after() onto a destroyed root
        self._closing: bool = False

        # Cooperative cancellation for background training
        self._train_cancel_event = threading.Event()
        self._train_gen: int = 0

        # Undo / redo stacks
        self._undo_stack: list[dict] = []
        self._redo_stack: list[dict] = []

        # Drag-handle state
        self._drag_handle: str | None = None
        self._drag_orig_bbox: tuple[float, float, float, float] | None = None
        self._drag_orig_polygon: list[tuple[float, float]] | None = None

        # Move-drag state (click inside already-selected box to drag)
        self._move_dragging: bool = False
        self._move_start_pdf: tuple[float, float] | None = None
        self._move_orig_bbox: tuple[float, float, float, float] | None = None
        self._move_orig_polygon: list[tuple[float, float]] | None = None

        # Pan state
        self._pan_start: tuple[int, int] | None = None

        # Mousewheel scroll state for the inspector panel (avoid unbind_all)
        self._insp_wheel_active: bool = False

        # Clipboard for box copy/paste
        self._copied_box_template: dict | None = None

        # Box groups (WBS hierarchy)
        self._groups: dict[str, dict] = {}
        self._group_link_ids: list[int] = []

        # Word overlay state
        self._word_overlay_on: bool = False
        self._word_overlay_ids: list[int] = []
        self._word_overlay_items: dict[int, dict[str, Any]] = {}
        self._selected_word_rids: set[int] = set()

        # Filter state
        self._filter_label_vars: dict[str, tk.BooleanVar] = {}
        self._filter_conf_min: tk.DoubleVar = tk.DoubleVar(value=0.0)
        self._filter_uncorrected_only: tk.BooleanVar = tk.BooleanVar(value=False)
        self._filter_frame: ttk.Frame | None = None
        self._active_filter_color_type: str | None = None
        self._filter_color_btn: ttk.Button | None = None

        # Model metrics cache
        self._last_metrics: dict | None = None

        self._build_ui()

        try:
            self.root.bind("<Destroy>", lambda e: setattr(self, "_closing", True), add="+")
        except Exception:
            pass

        # Subscribe to GuiState events
        self.state.subscribe("pdf_changed", self._on_pdf_changed)
        self.state.subscribe("run_completed", self._on_run_completed)

    def request_cancel(self) -> None:
        """Best-effort cancel of background model training."""
        self._train_cancel_event.set()
        self._train_gen += 1

    # ── Helpers: read-only text / copy menu ────────────────────────

    @staticmethod
    def _make_text_readonly(widget: tk.Text) -> None:
        """Allow selection & Ctrl-C in a Text widget but block edits."""

        def _on_key(event: tk.Event) -> str | None:
            # Allow Ctrl+C / Ctrl+A (copy / select-all)
            if event.state & 0x4 and event.keysym.lower() in ("c", "a"):
                return None
            # Allow navigation keys
            if event.keysym in (
                "Left",
                "Right",
                "Up",
                "Down",
                "Home",
                "End",
                "Prior",
                "Next",
                "Shift_L",
                "Shift_R",
                "Control_L",
                "Control_R",
                "Tab",
            ):
                return None
            return "break"

        widget.bind("<Key>", _on_key)
        # Block right-click paste / middle-click paste
        widget.bind("<<Paste>>", lambda e: "break")
        widget.bind("<Button-2>", lambda e: "break")

    def _add_copy_menu(self, widget: tk.Widget) -> None:
        """Attach a right-click *Copy* context menu to any widget."""
        menu = tk.Menu(widget, tearoff=0)

        def _copy() -> None:
            text = ""
            if isinstance(widget, (ttk.Label, tk.Label)):
                text = widget.cget("text")
            elif isinstance(widget, (ttk.Entry, tk.Entry)):
                text = widget.get()
            elif isinstance(widget, tk.Text):
                try:
                    text = widget.get("sel.first", "sel.last")
                except tk.TclError:
                    text = widget.get("1.0", "end-1c")
            self.root.clipboard_clear()
            self.root.clipboard_append(text)

        menu.add_command(label="Copy", command=_copy)
        widget.bind("<Button-3>", lambda e: menu.tk_popup(e.x_root, e.y_root))

    @staticmethod
    def _tooltip(widget: tk.Widget, text: str, delay: int = 500) -> None:
        """Attach a hover tooltip to *widget*."""
        tip_win: tk.Toplevel | None = None
        after_id: str | None = None

        def _show(event: tk.Event) -> None:
            nonlocal tip_win, after_id

            def _create() -> None:
                nonlocal tip_win
                if tip_win is not None:
                    return
                x = event.x_root + 12
                y = event.y_root + 10
                tip_win = tk.Toplevel(widget)
                tip_win.wm_overrideredirect(True)
                tip_win.wm_geometry(f"+{x}+{y}")
                lbl = tk.Label(
                    tip_win,
                    text=text,
                    background="#ffffe0",
                    foreground="#000",
                    relief="solid",
                    borderwidth=1,
                    font=("TkDefaultFont", 9),
                    wraplength=300,
                    justify="left",
                )
                lbl.pack()

            after_id = widget.after(delay, _create)

        def _hide(_event: tk.Event) -> None:
            nonlocal tip_win, after_id
            if after_id is not None:
                widget.after_cancel(after_id)
                after_id = None
            if tip_win is not None:
                tip_win.destroy()
                tip_win = None

        widget.bind("<Enter>", _show, add="+")
        widget.bind("<Leave>", _hide, add="+")

    # ── UI construction ────────────────────────────────────────────

    def _build_ui(self) -> None:
        pad = {"padx": 6, "pady": 3}

        # ── Top bar ───────────────────────────────────────────────
        top = ttk.Frame(self.frame)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", **pad)

        ttk.Label(top, text="PDF:").pack(side="left")
        self._pdf_label = ttk.Label(top, text="(none)", width=40)
        self._pdf_label.pack(side="left", padx=4)
        self._add_copy_menu(self._pdf_label)
        self._tooltip(self._pdf_label, "Currently loaded PDF file")

        _btn_browse = ttk.Button(top, text="Browse…", command=self._browse_pdf)
        _btn_browse.pack(side="left", padx=2)
        self._tooltip(_btn_browse, "Open a PDF file for annotation")

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)

        ttk.Label(top, text="Page:").pack(side="left")
        self._page_var = tk.IntVar(value=0)
        self._page_spin = ttk.Spinbox(
            top,
            from_=0,
            to=0,
            width=5,
            textvariable=self._page_var,
        )
        self._page_spin.pack(side="left", padx=2)
        # Navigate when Enter is pressed in the spinbox
        self._page_spin.bind("<Return>", self._on_page_spin_enter)
        self._tooltip(self._page_spin, "Page number (press Enter to navigate)")

        self._page_count_label = ttk.Label(top, text="/ ?")
        self._page_count_label.pack(side="left", padx=2)
        self._tooltip(self._page_count_label, "Total page count")

        _btn_prev = ttk.Button(top, text="◀", width=2, command=self._on_prev_page)
        _btn_prev.pack(side="left", padx=1)
        self._tooltip(_btn_prev, "Previous page (Ctrl+←)")
        _btn_next = ttk.Button(top, text="▶", width=2, command=self._on_next_page)
        _btn_next.pack(side="left", padx=1)
        self._tooltip(_btn_next, "Next page (Ctrl+→)")

        ttk.Label(top, text="DPI:").pack(side="left", padx=(8, 0))
        self._dpi_var = tk.IntVar(value=self._resolution)
        self._dpi_spin = ttk.Spinbox(
            top,
            from_=72,
            to=600,
            increment=50,
            width=5,
            textvariable=self._dpi_var,
        )
        self._dpi_spin.pack(side="left", padx=2)
        self._tooltip(self._dpi_spin, "Render resolution in dots per inch")

        _btn_zoom_out = ttk.Button(
            top, text="\u2212", width=2, command=lambda: self._apply_zoom(1 / 1.2)
        )
        _btn_zoom_out.pack(side="left", padx=1)
        self._tooltip(_btn_zoom_out, "Zoom out (-)")
        _btn_zoom_in = ttk.Button(
            top, text="+", width=2, command=lambda: self._apply_zoom(1.2)
        )
        _btn_zoom_in.pack(side="left", padx=1)
        self._tooltip(_btn_zoom_in, "Zoom in (+)")
        _btn_fit = ttk.Button(top, text="Fit", width=3, command=self._fit_to_window)
        _btn_fit.pack(side="left", padx=2)
        self._tooltip(_btn_fit, "Zoom to fit the full page in view (F)")

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)

        self._word_overlay_var = tk.BooleanVar(value=False)
        self._btn_words = ttk.Checkbutton(
            top,
            text="Words",
            variable=self._word_overlay_var,
            command=self._toggle_word_overlay,
        )
        self._btn_words.pack(side="left", padx=2)
        self._tooltip(
            self._btn_words,
            "Toggle light-gray boxes around every word pdfplumber identified (W)",
        )

        # ── Canvas (left) ─────────────────────────────────────────
        canvas_frame = ttk.Frame(self.frame)
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self._canvas = tk.Canvas(
            canvas_frame,
            bg="#2b2b2b",
            highlightthickness=0,
        )
        self._canvas.grid(row=0, column=0, sticky="nsew")

        # Scrollbars
        h_scroll = ttk.Scrollbar(
            canvas_frame, orient="horizontal", command=self._canvas.xview
        )
        h_scroll.grid(row=1, column=0, sticky="ew")
        v_scroll = ttk.Scrollbar(
            canvas_frame, orient="vertical", command=self._canvas.yview
        )
        v_scroll.grid(row=0, column=1, sticky="ns")
        self._canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        # Canvas bindings
        self._canvas.bind("<Control-Button-1>", self._on_word_click)
        self._canvas.bind("<Button-1>", self._on_canvas_click)
        self._canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self._canvas.bind("<MouseWheel>", self._on_mousewheel)
        self._canvas.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)
        # Right-click context menu
        self._canvas.bind("<Button-3>", self._on_canvas_right_click)
        # Middle-button pan
        self._canvas.bind("<Button-2>", self._on_pan_start)
        self._canvas.bind("<B2-Motion>", self._on_pan_motion)
        self._canvas.bind("<ButtonRelease-2>", self._on_pan_end)

        # ── Inspector (right) ─────────────────────────────────────
        inspector_outer = ttk.LabelFrame(self.frame, text="Inspector", width=260)
        inspector_outer.grid(row=1, column=1, sticky="nsew", padx=6, pady=3)
        inspector_outer.grid_propagate(False)
        inspector_outer.columnconfigure(0, weight=1)
        inspector_outer.rowconfigure(0, weight=1)

        # Scrollable wrapper inside the inspector
        self._insp_canvas = tk.Canvas(
            inspector_outer, highlightthickness=0, borderwidth=0
        )
        insp_sb = ttk.Scrollbar(
            inspector_outer, orient="vertical", command=self._insp_canvas.yview
        )
        inspector = ttk.Frame(self._insp_canvas)
        inspector.columnconfigure(1, weight=1)

        self._insp_canvas_window = self._insp_canvas.create_window(
            (0, 0), window=inspector, anchor="nw"
        )
        self._insp_canvas.configure(yscrollcommand=insp_sb.set)
        self._insp_canvas.grid(row=0, column=0, sticky="nsew")
        insp_sb.grid(row=0, column=1, sticky="ns")

        def _insp_inner_configure(event: tk.Event) -> None:
            self._insp_canvas.configure(scrollregion=self._insp_canvas.bbox("all"))

        inspector.bind("<Configure>", _insp_inner_configure)

        def _insp_canvas_configure(event: tk.Event) -> None:
            self._insp_canvas.itemconfig(self._insp_canvas_window, width=event.width)

        self._insp_canvas.bind("<Configure>", _insp_canvas_configure)

        # Mouse-wheel scrolling for the inspector panel
        self._insp_canvas.bind(
            "<Enter>",
            lambda e: setattr(self, "_insp_wheel_active", True),
        )
        self._insp_canvas.bind(
            "<Leave>",
            lambda e: setattr(self, "_insp_wheel_active", False),
        )

        def _on_inspector_mousewheel(event) -> None:
            if (
                not self._insp_wheel_active
                or not self._insp_canvas.winfo_ismapped()
            ):
                return
            self._insp_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # Bind once globally; handler is gated by _insp_wheel_active
        self.root.bind_all("<MouseWheel>", _on_inspector_mousewheel, add="+")

        row = 0
        ttk.Label(inspector, text="ID:").grid(
            row=row, column=0, sticky="w", padx=4, pady=2
        )
        self._insp_id = ttk.Label(inspector, text="—")
        self._insp_id.grid(row=row, column=1, sticky="w", padx=4, pady=2)
        self._add_copy_menu(self._insp_id)
        self._tooltip(self._insp_id, "Unique detection identifier")

        row += 1
        ttk.Label(inspector, text="Type:").grid(
            row=row, column=0, sticky="w", padx=4, pady=2
        )
        self._type_var = tk.StringVar()
        type_row_frame = ttk.Frame(inspector)
        type_row_frame.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        self._type_combo = ttk.Combobox(
            type_row_frame,
            textvariable=self._type_var,
            values=self.ELEMENT_TYPES,
            width=14,
        )
        self._type_combo.pack(side="left")
        self._type_combo.bind("<Return>", self._on_type_entered)
        self._tooltip(self._type_combo, "Element type — select or type a new one")
        _btn_add_type = ttk.Button(
            type_row_frame, text="+", width=2, command=self._on_add_element_type
        )
        _btn_add_type.pack(side="left", padx=2)
        self._tooltip(_btn_add_type, "Register a new element type")

        row += 1
        ttk.Label(inspector, text="Conf:").grid(
            row=row, column=0, sticky="w", padx=4, pady=2
        )
        self._insp_conf = ttk.Label(inspector, text="—")
        self._insp_conf.grid(row=row, column=1, sticky="w", padx=4, pady=2)
        self._add_copy_menu(self._insp_conf)
        self._tooltip(self._insp_conf, "Detection confidence score")

        row += 1
        ttk.Label(inspector, text="Text:").grid(
            row=row, column=0, sticky="nw", padx=4, pady=2
        )
        text_frame = ttk.Frame(inspector)
        text_frame.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        text_frame.columnconfigure(0, weight=1)
        self._insp_text = tk.Text(
            text_frame, width=24, height=6, wrap="word", state="disabled"
        )
        self._insp_text.grid(row=0, column=0, sticky="nsew")
        _insp_text_sb = ttk.Scrollbar(
            text_frame, orient="vertical", command=self._insp_text.yview
        )
        _insp_text_sb.grid(row=0, column=1, sticky="ns")
        self._insp_text.configure(yscrollcommand=_insp_text_sb.set)
        self._add_copy_menu(self._insp_text)
        self._tooltip(self._insp_text, "Extracted text content of the selected element")
        _btn_rescan = ttk.Button(
            text_frame, text="Rescan ↻", width=10, command=self._on_rescan_text
        )
        _btn_rescan.grid(row=1, column=0, sticky="w", pady=(2, 0))
        self._tooltip(
            _btn_rescan,
            "Re-extract text from the PDF under this box's current position",
        )

        # ── Buttons ───────────────────────────────────────────────
        row += 1
        btn_frame = ttk.Frame(inspector)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=6)

        _btn_accept = ttk.Button(btn_frame, text="Accept ✓", command=self._on_accept)
        _btn_accept.pack(side="left", padx=3)
        self._tooltip(_btn_accept, "Confirm this detection's label is correct (A)")
        _btn_relabel = ttk.Button(btn_frame, text="Relabel", command=self._on_relabel)
        _btn_relabel.pack(side="left", padx=3)
        self._tooltip(
            _btn_relabel, "Change this detection's type to the selected label (R)"
        )
        _btn_delete = ttk.Button(btn_frame, text="Reject ✗", command=self._on_delete)
        _btn_delete.pack(side="left", padx=3)
        self._tooltip(_btn_delete, "Mark this detection as a false positive (D)")

        # ── Merge button ──────────────────────────────────────────
        row += 1
        batch_frame = ttk.Frame(inspector)
        batch_frame.grid(row=row, column=0, columnspan=2, pady=2)
        _btn_merge = ttk.Button(batch_frame, text="Merge ⊞", command=self._on_merge)
        _btn_merge.pack(side="left", padx=3)
        self._tooltip(_btn_merge, "Merge selected boxes/words into one detection (M)")
        row += 1
        self._multi_label = ttk.Label(inspector, text="", foreground="blue")
        self._multi_label.grid(row=row, column=0, columnspan=2, sticky="w", padx=4)

        # ── Model suggestion ──────────────────────────────────────
        row += 1
        self._suggest_frame = ttk.Frame(inspector)
        self._suggest_frame.grid(
            row=row, column=0, columnspan=2, sticky="ew", padx=4, pady=2
        )
        self._suggest_label = ttk.Label(
            self._suggest_frame, text="", foreground="#0060c0"
        )
        self._suggest_label.pack(side="left")
        self._suggest_btn = ttk.Button(
            self._suggest_frame, text="Apply", command=self._apply_suggestion
        )
        self._tooltip(self._suggest_btn, "Apply the model's suggested label")
        self._model_suggestion: str | None = None

        # ── Group membership ──────────────────────────────────────
        row += 1
        ttk.Label(inspector, text="Group:").grid(
            row=row, column=0, sticky="nw", padx=4, pady=2
        )
        group_frame = ttk.Frame(inspector)
        group_frame.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        self._insp_group_label = ttk.Label(group_frame, text="—", wraplength=180)
        self._insp_group_label.pack(side="top", anchor="w")
        self._add_copy_menu(self._insp_group_label)
        self._tooltip(
            self._insp_group_label,
            "Group this box belongs to (parent/child hierarchy)",
        )

        group_btn_frame = ttk.Frame(group_frame)
        group_btn_frame.pack(side="top", anchor="w", pady=(2, 0))
        self._btn_create_group = ttk.Button(
            group_btn_frame, text="Create Group", command=self._on_create_group
        )
        self._btn_create_group.pack(side="left", padx=(0, 3))
        self._tooltip(
            self._btn_create_group,
            "Make this box the parent of a new group (G)",
        )

        self._btn_add_to_group = ttk.Button(
            group_btn_frame, text="Add to Group", command=self._on_add_to_group
        )
        self._btn_add_to_group.pack(side="left", padx=(0, 3))
        self._tooltip(
            self._btn_add_to_group,
            "Add selected / multi-selected boxes as children of the active group",
        )
        self._btn_add_to_group.pack_forget()

        self._btn_remove_group = ttk.Button(
            group_btn_frame, text="Remove", command=self._on_remove_from_group
        )
        self._btn_remove_group.pack(side="left", padx=(0, 3))
        self._tooltip(
            self._btn_remove_group,
            "Remove this box from its group (deletes group if parent)",
        )
        self._btn_remove_group.pack_forget()

        # ── Mode selector ─────────────────────────────────────────
        row += 1
        ttk.Separator(inspector).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=4
        )

        row += 1
        ttk.Label(inspector, text="Mode:", font=("TkDefaultFont", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4
        )

        row += 1
        self._mode_var = tk.StringVar(value="select")
        ttk.Radiobutton(
            inspector,
            text="Select / Edit",
            variable=self._mode_var,
            value="select",
            command=self._on_mode_change,
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=12)

        row += 1
        ttk.Radiobutton(
            inspector,
            text="Add New Element",
            variable=self._mode_var,
            value="add",
            command=self._on_mode_change,
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=12)

        # ── Filters ───────────────────────────────────────────────
        row += 1
        ttk.Separator(inspector).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=4
        )
        row += 1
        ttk.Label(inspector, text="Filters:", font=("TkDefaultFont", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4
        )

        row += 1
        filter_btns = ttk.Frame(inspector)
        filter_btns.grid(row=row, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 0))
        ttk.Button(
            filter_btns,
            text="Show All",
            command=self._select_all_filter_types,
            width=10,
        ).pack(side="left", padx=(0, 4))
        ttk.Button(
            filter_btns,
            text="Hide All",
            command=self._deselect_all_filter_types,
            width=10,
        ).pack(side="left")
        self._filter_color_btn = ttk.Button(
            filter_btns,
            text="Pick Color",
            command=self._choose_active_filter_color,
            width=18,
        )
        self._filter_color_btn.pack(side="left", padx=(8, 0))

        row += 1
        self._filter_frame = ttk.Frame(inspector)
        self._filter_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=8)
        if self.ELEMENT_TYPES:
            self._active_filter_color_type = self.ELEMENT_TYPES[0]
        self._rebuild_filter_controls()

        row += 1
        conf_frame = ttk.Frame(inspector)
        conf_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=4, pady=2)
        ttk.Label(conf_frame, text="Min conf:").pack(side="left")
        ttk.Scale(
            conf_frame,
            from_=0.0,
            to=1.0,
            variable=self._filter_conf_min,
            orient="horizontal",
            command=lambda _: self._apply_filters(),
        ).pack(side="left", fill="x", expand=True)

        row += 1
        ttk.Checkbutton(
            inspector,
            text="Uncorrected only",
            variable=self._filter_uncorrected_only,
            command=self._apply_filters,
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=8)

        # ── Session info ──────────────────────────────────────────
        row += 1
        ttk.Separator(inspector).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=4
        )

        row += 1
        self._session_label = ttk.Label(inspector, text="Session: 0 saved")
        self._session_label.grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4, pady=2
        )
        self._add_copy_menu(self._session_label)
        self._tooltip(
            self._session_label, "Number of corrections saved in this session"
        )

        # ── Page Elements ─────────────────────────────────────────
        row += 1
        ttk.Separator(inspector).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=4
        )
        row += 1
        ttk.Label(
            inspector, text="Page Elements:", font=("TkDefaultFont", 9, "bold")
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=4)
        row += 1
        self._page_elements_label = ttk.Label(
            inspector, text="(no page loaded)", foreground="gray",
            font=("TkDefaultFont", 8),
        )
        self._page_elements_label.grid(
            row=row, column=0, columnspan=2, sticky="w", padx=8, pady=2
        )
        self._add_copy_menu(self._page_elements_label)
        self._tooltip(
            self._page_elements_label,
            "Element types and counts on the current page",
        )

        # ── Model Performance ─────────────────────────────────────
        row += 1
        ttk.Separator(inspector).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=4
        )
        row += 1
        ttk.Label(inspector, text="Model:", font=("TkDefaultFont", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4
        )

        row += 1
        model_btns = ttk.Frame(inspector)
        model_btns.grid(row=row, column=0, columnspan=2, padx=4, pady=2)
        _btn_train = ttk.Button(
            model_btns, text="Train Model", command=self._on_train_model
        )
        _btn_train.pack(side="left", padx=3)
        self._tooltip(
            _btn_train, "Train the element classifier on accepted/relabeled corrections"
        )
        _btn_metrics = ttk.Button(
            model_btns, text="Metrics", command=self._on_show_metrics
        )
        _btn_metrics.pack(side="left", padx=3)
        self._tooltip(
            _btn_metrics, "Show accuracy and per-class metrics of the trained model"
        )
        _btn_history = ttk.Button(
            model_btns, text="History", command=self._on_show_training_history
        )
        _btn_history.pack(side="left", padx=3)
        self._tooltip(
            _btn_history, "Show all past training runs with accuracy and F1 trends"
        )
        _btn_importance = ttk.Button(
            model_btns, text="Importance", command=self._on_show_feature_importance
        )
        _btn_importance.pack(side="left", padx=3)
        self._tooltip(_btn_importance, "Show feature importance from the trained model")

        row += 1
        self._model_status_label = ttk.Label(
            inspector, text="No model trained", foreground="gray"
        )
        self._model_status_label.grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4, pady=2
        )
        self._add_copy_menu(self._model_status_label)
        self._tooltip(self._model_status_label, "Current model training status")

        # ── Annotation Stats ──────────────────────────────────────
        row += 1
        ttk.Separator(inspector).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=4
        )
        row += 1
        ttk.Label(inspector, text="Stats:", font=("TkDefaultFont", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4
        )
        row += 1
        self._stats_label = ttk.Label(
            inspector, text="", foreground="gray", font=("TkDefaultFont", 8)
        )
        self._stats_label.grid(
            row=row, column=0, columnspan=2, sticky="w", padx=8, pady=2
        )
        self._add_copy_menu(self._stats_label)
        self._tooltip(self._stats_label, "Annotation counts and coverage statistics")
        row += 1
        _btn_stats = ttk.Button(
            inspector, text="Refresh Stats", command=self._refresh_stats
        )
        _btn_stats.grid(row=row, column=0, padx=4, pady=2)
        self._tooltip(_btn_stats, "Recalculate annotation statistics from the database")
        _btn_clear_runs = ttk.Button(
            inspector, text="Clear Old Runs", command=self._on_clear_old_runs
        )
        _btn_clear_runs.grid(row=row, column=1, padx=4, pady=2)
        self._tooltip(
            _btn_clear_runs,
            "Remove detection data from old pipeline runs "
            "(preserves corrections and ML training data)",
        )

        # ── Active Learning ───────────────────────────────────────
        row += 1
        ttk.Separator(inspector).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=4
        )
        row += 1
        _btn_suggest = ttk.Button(
            inspector, text="Suggest Next Page", command=self._on_suggest_next
        )
        _btn_suggest.grid(row=row, column=0, columnspan=2, padx=4, pady=2)
        self._tooltip(
            _btn_suggest, "Navigate to the page that would benefit most from annotation"
        )

        # ── Snapshots ─────────────────────────────────────────────
        row += 1
        snap_btns = ttk.Frame(inspector)
        snap_btns.grid(row=row, column=0, columnspan=2, padx=4, pady=2)
        _btn_snap = ttk.Button(snap_btns, text="Snapshot", command=self._on_snapshot)
        _btn_snap.pack(side="left", padx=3)
        self._tooltip(
            _btn_snap, "Save a timestamped backup of the corrections database"
        )
        _btn_restore = ttk.Button(
            snap_btns, text="Restore…", command=self._on_restore_snapshot
        )
        _btn_restore.pack(side="left", padx=3)
        self._tooltip(_btn_restore, "Restore corrections from a previous snapshot")

        # ── Keyboard legend ───────────────────────────────────────
        row += 1
        ttk.Separator(inspector).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=4
        )
        row += 1
        kb_text = (
            "Shortcuts: A=Accept  D=Reject\n"
            "R=Relabel  M=Merge\n"
            "Esc=Deselect  ←→ Cycle\n"
            "F=Fit  +/- Zoom  Scroll=Pan\n"
            "Ctrl+Z/Y Undo/Redo\n"
            "Shift+Click Multi-select boxes\n"
            "Ctrl+Click Select words\n"
            "Ctrl+A Select all\n"
            "Ctrl+C Copy box/word text\n"
            "Ctrl+V Paste box\n"
            "G=Group  L=Link Column\n"
            "W=Words  Right-click: menu"
        )
        ttk.Label(
            inspector, text=kb_text, foreground="gray", font=("TkDefaultFont", 8)
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=4)

        # ── Status bar ────────────────────────────────────────────
        self._status = ttk.Label(self.frame, text="Ready", relief="sunken", anchor="w")
        self._status.grid(row=2, column=0, columnspan=2, sticky="ew", padx=2)
        self._add_copy_menu(self._status)
        self._tooltip(self._status, "Current status and last operation result")

        # ── Progress bar ──────────────────────────────────────────
        self._progress_var = tk.DoubleVar(value=0)
        self._progress = ttk.Progressbar(
            self.frame, variable=self._progress_var, maximum=100
        )
        self._progress.grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=2, pady=(0, 2)
        )
        self._progress.grid_remove()  # hidden by default

        # ── Global key bindings ───────────────────────────────────
        self.root.bind("<Key-a>", self._key_accept)
        self.root.bind("<Key-d>", self._key_delete)
        self.root.bind("<Key-r>", self._key_relabel)
        self.root.bind("<Escape>", self._key_deselect)
        self.root.bind("<Left>", self._key_prev_box)
        self.root.bind("<Right>", self._key_next_box)
        self.root.bind("<plus>", self._key_zoom_in)
        self.root.bind("<minus>", self._key_zoom_out)
        self.root.bind("<equal>", self._key_zoom_in)  # = / + on US keyboard
        self.root.bind("<Key-f>", self._key_fit)
        self.root.bind("<Control-z>", self._key_undo)
        self.root.bind("<Control-y>", self._key_redo)
        self.root.bind("<Control-a>", self._key_select_all)
        self.root.bind("<Key-m>", self._key_merge)
        self.root.bind("<Control-Left>", lambda e: self._on_prev_page())
        self.root.bind("<Control-Right>", lambda e: self._on_next_page())
        self.root.bind("<Control-c>", self._key_copy_box)
        self.root.bind("<Control-v>", self._key_paste_box)
        self.root.bind("<Key-g>", self._key_group)
        self.root.bind("<Key-l>", self._key_link_column)
        self.root.bind("<Key-w>", self._key_toggle_words)

        # Load any persisted element types from data/label_registry.json.
        # This must run after the inspector/filter UI exists because
        # _register_element_type updates comboboxes and filter controls.
        self._load_element_types_from_registry()

        # Initialize model status
        self._update_model_status()

    # ── PDF selection ──────────────────────────────────────────────

    def _browse_pdf(self) -> None:
        f = filedialog.askopenfilename(
            title="Select PDF",
            filetypes=[("PDF", "*.pdf"), ("All", "*.*")],
            initialdir=str(Path("input")),
        )
        if f:
            self.state.set_pdf(Path(f))

    def _on_pdf_changed(self) -> None:
        self._pdf_path = self.state.pdf_path
        if self._pdf_path:
            self._pdf_label.configure(text=self._pdf_path.name)
            # Count pages
            try:
                import pdfplumber

                with pdfplumber.open(self._pdf_path) as pdf:
                    self._page_count = len(pdf.pages)
            except Exception:
                self._page_count = 0
            # Reset page selection to first page
            self._page_var.set(0)
            self._page_spin.configure(to=max(0, self._page_count - 1))
            self._page_count_label.configure(text=f"/ {self._page_count}")
            # Clear prior state
            self._canvas_boxes.clear()
            self._selected_box = None
            self._multi_selected.clear()
            self._canvas.delete("all")
            # Render first page preview
            self._navigate_to_page()
        else:
            self._pdf_label.configure(text="(none)")
            self._page_count = 0
            self._page_count_label.configure(text="/ ?")
            self._page_spin.configure(to=999)

    def _on_run_completed(self) -> None:
        """Re-load the current page after a pipeline run finishes.

        The pipeline now persists detections to the correction store,
        so refreshing the page will pick up the fresh results.
        """
        if self._pdf_path:
            self._navigate_to_page()

    # ── Mode ───────────────────────────────────────────────────────

    def _on_mode_change(self) -> None:
        self._mode = self._mode_var.get()
        if self._mode == "add":
            self._canvas.config(cursor="crosshair")
            self._deselect()
        else:
            self._canvas.config(cursor="")

    # ── Dynamic element types ──────────────────────────────────────

    def _normalize_element_type_name(self, name: str) -> str:
        return name.strip().lower().replace(" ", "_")

    def _label_registry_path(self) -> Path:
        # scripts/gui/tab_annotation.py -> repo root is parent.parent.parent
        return (
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / "label_registry.json"
        )

    def _load_label_registry_json(self) -> dict:
        path = self._label_registry_path()
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"version": "1.0", "label_registry": []}
        except Exception:
            return {"version": "1.0", "label_registry": []}

    def _save_label_registry_json(self, data: dict) -> None:
        path = self._label_registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
        tmp.replace(path)

    def _persist_element_type_to_registry(
        self, *, label: str, display_name: str, color: str
    ) -> None:
        data = self._load_label_registry_json()
        reg = data.get("label_registry")
        if not isinstance(reg, list):
            reg = []
            data["label_registry"] = reg

        existing: dict | None = None
        for entry in reg:
            if isinstance(entry, dict) and entry.get("label") == label:
                existing = entry
                break

        if existing is None:
            reg.append(
                {
                    "label": label,
                    "display_name": display_name,
                    "color": color,
                    "description": "",
                    "aliases": [],
                    "expected_zones": [],
                    "text_patterns": [],
                }
            )
        else:
            existing["display_name"] = display_name
            existing["color"] = color

        if "version" not in data:
            data["version"] = "1.0"
        self._save_label_registry_json(data)

    def _load_element_types_from_registry(self) -> None:
        data = self._load_label_registry_json()
        reg = data.get("label_registry", [])
        if not isinstance(reg, list):
            return
        for entry in reg:
            if not isinstance(entry, dict):
                continue
            label = entry.get("label", "")
            color = entry.get("color", "")
            if not label:
                continue
            if isinstance(color, str) and color.startswith("#") and len(color) == 7:
                self._register_element_type(label, color=color)
            else:
                self._register_element_type(label)

    def _register_element_type(self, name: str, *, color: str | None = None) -> None:
        """Register a new element type (optionally with explicit color).

        Updates LABEL_COLORS, ELEMENT_TYPES, the type combo boxes,
        and the filter checkboxes.
        """
        name = self._normalize_element_type_name(name)
        if not name or name in self.LABEL_COLORS:
            return

        # Auto-assign a distinct color from a palette
        _palette = [
            "#e6194b",
            "#3cb44b",
            "#4363d8",
            "#f58231",
            "#911eb4",
            "#42d4f4",
            "#f032e6",
            "#bfef45",
            "#fabebe",
            "#469990",
            "#9a6324",
            "#800000",
            "#aaffc3",
            "#808000",
            "#000075",
        ]
        if not (isinstance(color, str) and color.startswith("#") and len(color) == 7):
            idx = len(self.LABEL_COLORS) % len(_palette)
            color = _palette[idx]

        self.LABEL_COLORS[name] = color
        if name not in self.ELEMENT_TYPES:
            self.ELEMENT_TYPES.append(name)

        # Update combo boxes
        self._type_combo.configure(values=self.ELEMENT_TYPES)

        if name not in self._filter_label_vars:
            self._filter_label_vars[name] = tk.BooleanVar(value=True)
        self._rebuild_filter_controls()

    def _rebuild_filter_controls(self) -> None:
        """Rebuild per-type filter rows with checkboxes and clickable labels."""
        if self._filter_frame is None:
            return

        for child in self._filter_frame.winfo_children():
            child.destroy()

        if self._active_filter_color_type not in self.ELEMENT_TYPES:
            self._active_filter_color_type = self.ELEMENT_TYPES[0] if self.ELEMENT_TYPES else None

        for i, etype in enumerate(self.ELEMENT_TYPES):
            if etype not in self._filter_label_vars:
                self._filter_label_vars[etype] = tk.BooleanVar(value=True)

            is_active = etype == self._active_filter_color_type
            row_bg = "SystemHighlight" if is_active else self._filter_frame.winfo_toplevel().cget("bg")
            row_fg = "SystemHighlightText" if is_active else "SystemWindowText"

            row_frame = tk.Frame(self._filter_frame, bg=row_bg)
            row_frame.grid(row=i // 2, column=i % 2, sticky="w", pady=1, padx=(0, 8))

            cb = tk.Checkbutton(
                row_frame,
                variable=self._filter_label_vars[etype],
                command=self._apply_filters,
                bg=row_bg,
                activebackground=row_bg,
                highlightthickness=0,
                borderwidth=0,
            )
            cb.pack(side="left")

            lbl = tk.Label(
                row_frame,
                text=etype,
                bg=row_bg,
                fg=row_fg,
                cursor="hand2",
                padx=4,
            )
            lbl.pack(side="left")
            lbl.bind("<Button-1>", lambda _e, label=etype: self._set_active_filter_color_type(label))

        self._update_filter_color_button_label()

    def _set_active_filter_color_type(self, element_type: str) -> None:
        """Set the active element type target for the shared color picker."""
        if element_type not in self.ELEMENT_TYPES:
            return
        self._active_filter_color_type = element_type
        self._rebuild_filter_controls()

    def _update_filter_color_button_label(self) -> None:
        """Refresh shared color button text for the active element type."""
        if self._filter_color_btn is None:
            return
        if self._active_filter_color_type:
            self._filter_color_btn.configure(
                text=f"Pick Color: {self._active_filter_color_type}"
            )
            self._filter_color_btn.state(["!disabled"])
        else:
            self._filter_color_btn.configure(text="Pick Color")
            self._filter_color_btn.state(["disabled"])

    def _choose_active_filter_color(self) -> None:
        """Prompt for color of the currently selected filter label type."""
        element_type = self._active_filter_color_type
        if not element_type:
            return
        current = self.LABEL_COLORS.get(element_type, "#888888")
        _rgb, chosen = colorchooser.askcolor(
            color=current,
            title=f"Choose color for {element_type}",
            parent=self.root,
        )
        if not chosen:
            return

        self.LABEL_COLORS[element_type] = chosen
        self._rebuild_filter_controls()
        self._draw_all_boxes()
        self._apply_filters()

    def _select_all_filter_types(self) -> None:
        """Enable all element-type filter checkboxes."""
        for var in self._filter_label_vars.values():
            var.set(True)
        self._apply_filters()

    def _deselect_all_filter_types(self) -> None:
        """Disable all element-type filter checkboxes."""
        for var in self._filter_label_vars.values():
            var.set(False)
        self._apply_filters()

    def _on_add_element_type(self) -> None:
        """Prompt user to add a new element type."""
        name = simpledialog.askstring(
            "New Element Type",
            "Enter new element type name:",
            parent=self.root,
        )
        if name:
            self._register_element_type(name)
            self._type_var.set(name.strip().lower().replace(" ", "_"))
            self._status.configure(
                text=f"Added element type: {name.strip().lower().replace(' ', '_')}"
            )

    def _on_type_entered(self, _event: Any = None) -> None:
        """Handle Enter key in the type combo — register if new."""
        name = self._type_var.get().strip()
        if name and name not in self.ELEMENT_TYPES:
            self._register_element_type(name)
            self._status.configure(text=f"Added element type: {name}")

    def _deduplicate_boxes(self) -> None:
        """Remove near-duplicate canvas boxes (same type, IoU > 0.8).

        When the pipeline produces overlapping detections of the same
        element_type (e.g. two "header" boxes covering the same area),
        keep the one with the higher detection_id (most recently saved)
        and drop the other.
        """
        if len(self._canvas_boxes) < 2:
            return

        def _iou(a: tuple, b: tuple) -> float:
            x0 = max(a[0], b[0])
            y0 = max(a[1], b[1])
            x1 = min(a[2], b[2])
            y1 = min(a[3], b[3])
            inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
            area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0.0

        to_remove: set[int] = set()  # indices to remove
        n = len(self._canvas_boxes)
        for i in range(n):
            if i in to_remove:
                continue
            a = self._canvas_boxes[i]
            for j in range(i + 1, n):
                if j in to_remove:
                    continue
                b = self._canvas_boxes[j]
                if a.element_type != b.element_type:
                    continue
                if _iou(a.pdf_bbox, b.pdf_bbox) > 0.8:
                    # Keep the one with the higher detection_id
                    if (a.detection_id or 0) >= (b.detection_id or 0):
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break  # 'a' is removed, stop comparing

        if to_remove:
            self._canvas_boxes = [
                cb for idx, cb in enumerate(self._canvas_boxes) if idx not in to_remove
            ]

    # ── Rendering ──────────────────────────────────────────────────

    def _effective_scale(self) -> float:
        """Combined DPI scale × zoom factor."""
        return self._scale * self._zoom

    def _render_background(self) -> None:
        """Display the background image on the canvas."""
        if self._bg_image is None:
            return

        eff = self._effective_scale()
        # Scale relative to the base DPI rendering
        if self._zoom != 1.0:
            new_w = int(self._bg_image.width * self._zoom)
            new_h = int(self._bg_image.height * self._zoom)
            display = self._bg_image.resize((new_w, new_h), Image.LANCZOS)
        else:
            display = self._bg_image

        self._photo = ImageTk.PhotoImage(display)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self._canvas.configure(scrollregion=(0, 0, display.width, display.height))

    def _load_groups_for_page(self) -> None:
        """Load saved box groups from the DB for the current page."""
        self._groups.clear()
        if not self._doc_id:
            return
        for g in self._store.get_groups_for_page(self._doc_id, self._page):
            gid = g["group_id"]
            self._groups[gid] = {
                "label": g["group_label"],
                "root_detection_id": g["root_detection_id"],
                "members": [m["detection_id"] for m in g["members"]],
            }
            # Update canvas boxes
            for cb in self._canvas_boxes:
                if cb.detection_id in self._groups[gid]["members"]:
                    cb.group_id = gid
                    cb.is_group_root = cb.detection_id == g["root_detection_id"]

    def _draw_all_boxes(self) -> None:
        """Redraw every detection box on the canvas."""
        # Remove existing box items (but not the background image)
        self._canvas.delete("det_box")
        self._canvas.delete("det_label")
        self._canvas.delete("det_handle")
        for cbox in self._canvas_boxes:
            self._draw_box(cbox)
        # Refresh word overlay if active
        if self._word_overlay_on:
            self._draw_word_overlay()

    def _confidence_color(self, conf: float | None) -> str:
        """Return a hex color for the confidence badge.

        Green (>0.9) → yellow (0.5–0.9) → red (<0.5) → grey (None).
        """
        if conf is None:
            return "#888888"
        if conf >= 0.9:
            return "#28a745"
        if conf >= 0.5:
            # Interpolate yellow-ish
            return "#d4a017"
        return "#dc3545"

    def _draw_box(self, cbox: CanvasBox) -> None:
        """Draw (or redraw) a single CanvasBox on the canvas."""
        # Remove old items if they exist
        if cbox.rect_id:
            self._canvas.delete(cbox.rect_id)
        if cbox.label_id:
            self._canvas.delete(cbox.label_id)
        if cbox.conf_dot_id:
            self._canvas.delete(cbox.conf_dot_id)
            cbox.conf_dot_id = 0
        for hid in cbox.handle_ids:
            self._canvas.delete(hid)
        cbox.handle_ids.clear()

        eff = self._effective_scale()
        x0, y0, x1, y1 = cbox.pdf_bbox
        cx0 = x0 * eff
        cy0 = y0 * eff
        cx1 = x1 * eff
        cy1 = y1 * eff

        color = self.LABEL_COLORS.get(cbox.element_type, "#888888")
        in_multi = cbox in self._multi_selected
        lw = 3 if (cbox.selected or in_multi) else 2

        # Fill for selected box (translucent approximation via stipple)
        fill_kw: dict = {}
        if cbox.selected or in_multi:
            fill_kw = {"fill": color, "stipple": "gray25"}

        # Render as polygon (merged) or rectangle (normal)
        if cbox.polygon:
            flat_coords: list[float] = []
            for px, py in cbox.polygon:
                flat_coords.append(px * eff)
                flat_coords.append(py * eff)
            if "fill" not in fill_kw:
                fill_kw["fill"] = ""  # transparent (create_polygon defaults to black)
            cbox.rect_id = self._canvas.create_polygon(
                *flat_coords,
                outline=color,
                width=lw,
                tags="det_box",
                **fill_kw,
            )
        else:
            cbox.rect_id = self._canvas.create_rectangle(
                cx0,
                cy0,
                cx1,
                cy1,
                outline=color,
                width=lw,
                tags="det_box",
                **fill_kw,
            )

        # Confidence badge background
        conf_str = f" ({cbox.confidence:.0%})" if cbox.confidence is not None else ""
        check = " ✓" if cbox.corrected else ""
        label_text = f"{cbox.element_type}{conf_str}{check}"

        # Draw confidence colour dot
        if cbox.confidence is not None:
            conf_color = self._confidence_color(cbox.confidence)
            dot_r = 4
            cbox.conf_dot_id = self._canvas.create_oval(
                cx0 - dot_r - 2,
                cy0 - 12 - dot_r,
                cx0 - 2 + dot_r,
                cy0 - 12 + dot_r,
                fill=conf_color,
                outline=conf_color,
                tags="det_label",
            )

        cbox.label_id = self._canvas.create_text(
            cx0 + 2,
            cy0 - 2,
            anchor="sw",
            text=label_text,
            fill=color,
            font=("TkDefaultFont", 8, "bold"),
            tags="det_label",
        )

        # Draw handles if selected
        if cbox.selected:
            self._draw_handles(cbox)

    def _draw_handles(self, cbox: CanvasBox) -> None:
        """Draw 8 resize handles around the selected box."""
        eff = self._effective_scale()
        x0, y0, x1, y1 = cbox.pdf_bbox
        cx0 = x0 * eff
        cy0 = y0 * eff
        cx1 = x1 * eff
        cy1 = y1 * eff
        mx = (cx0 + cx1) / 2
        my = (cy0 + cy1) / 2
        hs = HANDLE_SIZE

        positions = {
            "nw": (cx0, cy0),
            "n": (mx, cy0),
            "ne": (cx1, cy0),
            "e": (cx1, my),
            "se": (cx1, cy1),
            "s": (mx, cy1),
            "sw": (cx0, cy1),
            "w": (cx0, my),
        }

        color = self.LABEL_COLORS.get(cbox.element_type, "#888888")
        for pos_name, (hx, hy) in positions.items():
            hid = self._canvas.create_rectangle(
                hx - hs,
                hy - hs,
                hx + hs,
                hy + hs,
                fill=color,
                outline="white",
                tags="det_handle",
            )
            cbox.handle_ids.append(hid)

    # ── Selection ──────────────────────────────────────────────────

    def _on_word_click(self, event: tk.Event) -> str | None:
        """Handle Ctrl+Click for word overlay selection / lasso."""
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)

        if not (self._word_overlay_on and self._word_overlay_items):
            # Word overlay not active — fall through to normal click
            self._on_canvas_click(event)
            return None

        # Hit-test using PDF coordinates
        eff = self._effective_scale()
        pdf_x = cx / eff
        pdf_y = cy / eff
        for rid, winfo in self._word_overlay_items.items():
            if (
                winfo["x0"] <= pdf_x <= winfo["x1"]
                and winfo["top"] <= pdf_y <= winfo["bottom"]
            ):
                self._toggle_word_selected(rid)
                return "break"

        # No word hit — start a word-lasso on empty space
        self._lasso_start = (cx, cy)
        self._lasso_word = True
        self._deselect()
        return "break"

    def _on_canvas_click(self, event: tk.Event) -> None:
        """Handle click on canvas — select a box or start drawing."""
        # Get canvas coordinates (accounting for scroll)
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)

        if self._mode == "add":
            self._draw_start = (cx, cy)
            return

        # Shift+click for multi-select
        shift_held = bool(event.state & 0x0001)

        # Check if clicking on a handle of the selected box
        if self._selected_box and self._selected_box.handle_ids:
            for i, hid in enumerate(self._selected_box.handle_ids):
                coords = self._canvas.coords(hid)
                if coords and len(coords) == 4:
                    hx0, hy0, hx1, hy1 = coords
                    if hx0 <= cx <= hx1 and hy0 <= cy <= hy1:
                        self._drag_handle = HANDLE_POSITIONS[i]
                        self._drag_orig_bbox = self._selected_box.pdf_bbox
                        self._drag_orig_polygon = (
                            list(self._selected_box.polygon)
                            if self._selected_box.polygon
                            else None
                        )
                        return

        # Find which box was clicked
        eff = self._effective_scale()
        pdf_x = cx / eff
        pdf_y = cy / eff

        clicked = None
        # Iterate in reverse so top-drawn boxes are selected first
        for cbox in reversed(self._canvas_boxes):
            if (
                cbox.rect_id
                and self._canvas.itemcget(cbox.rect_id, "state") == "hidden"
            ):
                continue
            if cbox.polygon:
                # Point-in-polygon test for merged boxes
                if point_in_polygon(pdf_x, pdf_y, cbox.polygon):
                    clicked = cbox
                    break
            else:
                bx0, by0, bx1, by1 = cbox.pdf_bbox
                if bx0 <= pdf_x <= bx1 and by0 <= pdf_y <= by1:
                    clicked = cbox
                    break

        if clicked:
            if shift_held:
                self._toggle_multi_select(clicked)
            else:
                # Select (or keep selected) and prepare for move-drag.
                # If the user drags, we move the box; if they just
                # click and release, it's a plain select.
                if clicked is not self._selected_box:
                    self._clear_multi_select()
                    self._select_box(clicked)
                self._move_dragging = True
                self._move_start_pdf = (pdf_x, pdf_y)
                self._move_orig_bbox = clicked.pdf_bbox
                self._move_orig_polygon = (
                    list(clicked.polygon) if clicked.polygon else None
                )
        else:
            if not shift_held:
                self._clear_multi_select()
                self._clear_word_selection()
            # Start lasso drag if on empty space in select mode
            self._lasso_start = (cx, cy)
            self._lasso_word = False
            self._deselect()

    # ── Right-click copy / paste ───────────────────────────────────

    def _on_canvas_right_click(self, event: tk.Event) -> None:
        """Show a context menu — word actions if words selected, else box actions."""
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)

        eff = self._effective_scale()
        pdf_x = cx / eff
        pdf_y = cy / eff

        # ── If word overlay is active, check for right-click on a word ──
        if self._word_overlay_on and self._word_overlay_items:
            for rid, winfo in self._word_overlay_items.items():
                if (
                    winfo["x0"] <= pdf_x <= winfo["x1"]
                    and winfo["top"] <= pdf_y <= winfo["bottom"]
                ):
                    # Auto-select the word under cursor if not already
                    if rid not in self._selected_word_rids:
                        self._set_word_selected(rid, True)
                    break

        # ── Word context menu ──────────────────────────────────────
        if self._selected_word_rids:
            self._show_word_context_menu(event, pdf_x, pdf_y)
            return

        # ── Detection box context menu (original) ─────────────────
        clicked: CanvasBox | None = None
        for cbox in reversed(self._canvas_boxes):
            if (
                cbox.rect_id
                and self._canvas.itemcget(cbox.rect_id, "state") == "hidden"
            ):
                continue
            if cbox.polygon:
                if point_in_polygon(pdf_x, pdf_y, cbox.polygon):
                    clicked = cbox
                    break
            else:
                bx0, by0, bx1, by1 = cbox.pdf_bbox
                if bx0 <= pdf_x <= bx1 and by0 <= pdf_y <= by1:
                    clicked = cbox
                    break

        menu = tk.Menu(self._canvas, tearoff=0)
        if clicked:
            menu.add_command(
                label=f"Copy Box ({clicked.element_type})",
                command=lambda: self._copy_box(clicked),
            )
        if self._copied_box_template:
            t = self._copied_box_template["element_type"]
            menu.add_command(
                label=f"Paste Box ({t})",
                command=lambda: self._paste_box(pdf_x, pdf_y),
            )

        # ── Link to notes column ──────────────────────────────────
        n_multi = len(self._multi_selected)
        if clicked and clicked not in self._multi_selected:
            n_multi += 1
        if n_multi >= 2:
            # Check if selection contains linkable types
            link_targets = list(self._multi_selected)
            if clicked and clicked not in link_targets:
                link_targets.append(clicked)
            linkable_types = {"header", "notes_block"}
            if any(cb.element_type in linkable_types for cb in link_targets):
                menu.add_separator()
                menu.add_command(
                    label="Create Notes Column from Selection (L)",
                    command=self._on_link_column,
                )

        # ── Group actions ──────────────────────────────────────────
        has_group_items = False
        if clicked:
            if not clicked.group_id:
                menu.add_separator()
                menu.add_command(
                    label="Create Group (Set as Parent)",
                    command=lambda: self._create_group(clicked),
                )
                has_group_items = True
                if (
                    self._selected_box
                    and self._selected_box.is_group_root
                    and self._selected_box is not clicked
                ):
                    g_label = self._groups.get(
                        self._selected_box.group_id or "", {}
                    ).get("label", "?")
                    menu.add_command(
                        label=f"Add to Group \u2039{g_label}\u203a",
                        command=lambda: self._add_children_to_group([clicked]),
                    )
            else:
                if not has_group_items:
                    menu.add_separator()
                menu.add_command(
                    label="Remove from Group",
                    command=lambda: self._remove_from_group(clicked),
                )

        if menu.index("end") is not None:
            menu.tk_popup(event.x_root, event.y_root)
        else:
            menu.destroy()

    def _show_word_context_menu(
        self, event: tk.Event, pdf_x: float, pdf_y: float
    ) -> None:
        """Show context menu with all actions for selected words."""
        n = len(self._selected_word_rids)
        # Gather selected word texts
        texts: list[str] = []
        for rid in self._selected_word_rids:
            winfo = self._word_overlay_items.get(rid)
            if winfo and winfo.get("text"):
                texts.append(winfo["text"])
        preview = " ".join(texts)
        if len(preview) > 50:
            preview = preview[:47] + "..."

        menu = tk.Menu(self._canvas, tearoff=0)

        # ── Header ─────────────────────────────────────────────────
        menu.add_command(
            label=f"{n} word{'s' if n != 1 else ''} selected",
            state="disabled",
        )
        if preview:
            menu.add_command(
                label=f'"{preview}"',
                state="disabled",
            )
        menu.add_separator()

        # ── Copy text ──────────────────────────────────────────────
        def _copy_word_text():
            if texts:
                self.root.clipboard_clear()
                self.root.clipboard_append(" ".join(texts))
                self._status.configure(text=f"Copied text from {len(texts)} words")

        menu.add_command(label="Copy Text  (Ctrl+C)", command=_copy_word_text)

        # ── Create New Type ──────────────────────────────────────
        menu.add_separator()
        menu.add_command(
            label="Create New Type…",
            command=self._on_create_new_type_from_words,
        )

        # ── Merge / Create Detection ──────────────────────────────
        if n >= 2:
            menu.add_separator()
            if self._selected_box:
                menu.add_command(
                    label=f"Reshape \u2039{self._selected_box.element_type}\u203a to Words  (M)",
                    command=self._merge_words_into_detection,
                )
            else:
                menu.add_command(
                    label="Create Detection from Words  (M)",
                    command=self._merge_words_into_detection,
                )

            menu.add_separator()

            # ── Create as specific type ─────────────────────────
            type_menu = tk.Menu(menu, tearoff=0)
            for etype in self.ELEMENT_TYPES:
                color = self.LABEL_COLORS.get(etype, "#888")
                type_menu.add_command(
                    label=etype,
                    command=lambda t=etype: self._create_words_as_type(t),
                )
            menu.add_cascade(label="Create as Type \u25b6", menu=type_menu)

        # ── Group from words ──────────────────────────────────────
        if n >= 2:
            menu.add_separator()
            menu.add_command(
                label="Group Words  (G)",
                command=lambda: self._key_group(
                    type("E", (), {"widget": self._canvas, "state": 0})()
                ),
            )

        # ── Select controls ──────────────────────────────────────
        menu.add_separator()
        menu.add_command(
            label="Select All Words",
            command=self._select_all_words,
        )
        menu.add_command(
            label="Clear Selection",
            command=self._clear_word_selection,
        )

        menu.tk_popup(event.x_root, event.y_root)

    def _on_create_new_type_from_words(self) -> None:
        """Prompt for a new type name + color, persist it, then create from selected words."""
        if not self._selected_word_rids:
            return

        raw = simpledialog.askstring(
            "Create New Type",
            "Enter new type name:",
            parent=self.root,
        )
        if not raw:
            return

        label = self._normalize_element_type_name(raw)
        if not label:
            return

        if label in self.LABEL_COLORS:
            messagebox.showwarning(
                "Type Exists",
                f'Type "{label}" already exists.',
                parent=self.root,
            )
            return

        _rgb, hex_color = colorchooser.askcolor(
            title="Choose Type Color",
            parent=self.root,
        )
        if not hex_color:
            return

        self._register_element_type(label, color=hex_color)
        try:
            self._persist_element_type_to_registry(
                label=label,
                display_name=raw.strip(),
                color=hex_color,
            )
        except Exception:
            self._status.configure(text="Warning: failed to write label_registry.json")

        # Create detection from words using this type (force create, no classifier override)
        self._type_var.set(label)
        self._merge_words_into_detection(forced_type=label, force_create=True)

    def _create_words_as_type(self, element_type: str) -> None:
        """Create a new detection from selected words with a specific type."""
        old_type = self._type_var.get()
        self._type_var.set(element_type)
        self._merge_words_into_detection(forced_type=element_type, force_create=True)
        self._type_var.set(old_type)

    def _select_all_words(self) -> None:
        """Select every word in the overlay."""
        if not self._word_overlay_items:
            return
        for rid in self._word_overlay_items:
            if rid not in self._selected_word_rids:
                self._set_word_selected(rid, True)
        self._status.configure(
            text=f"Selected all {len(self._selected_word_rids)} words"
        )

    def _copy_box(self, cbox: CanvasBox) -> None:
        """Copy box dimensions and type to the internal clipboard."""
        x0, y0, x1, y1 = cbox.pdf_bbox
        self._copied_box_template = {
            "element_type": cbox.element_type,
            "width": x1 - x0,
            "height": y1 - y0,
        }
        self._status.configure(
            text=f"Copied {cbox.element_type} box ({x1 - x0:.0f}×{y1 - y0:.0f} pt)"
        )

    def _paste_box(self, pdf_x: float, pdf_y: float) -> None:
        """Paste a copied box centred at a PDF-space location."""
        if not self._copied_box_template or not self._doc_id:
            return

        w = self._copied_box_template["width"]
        h = self._copied_box_template["height"]
        chosen_type = self._copied_box_template["element_type"]

        x0 = pdf_x - w / 2
        y0 = pdf_y - h / 2
        x1 = pdf_x + w / 2
        y1 = pdf_y + h / 2

        # Clamp to non-negative coordinates
        if x0 < 0:
            x1 -= x0
            x0 = 0
        if y0 < 0:
            y1 -= y0
            y0 = 0

        pdf_bbox = (x0, y0, x1, y1)

        # Extract text from PDF under the pasted box
        text_content = ""
        if self._pdf_path:
            text_content = extract_text_in_bbox(self._pdf_path, self._page, pdf_bbox)

        # Compute features and save
        features = featurize_region(chosen_type, pdf_bbox, None, 2448.0, 1584.0)
        det_id = self._store.save_detection(
            doc_id=self._doc_id,
            page=self._page,
            run_id=self._run_id or "manual",
            element_type=chosen_type,
            bbox=pdf_bbox,
            text_content=text_content,
            features=features,
        )
        self._store.save_correction(
            doc_id=self._doc_id,
            page=self._page,
            correction_type="add",
            corrected_label=chosen_type,
            corrected_bbox=pdf_bbox,
            detection_id=det_id,
            session_id=self._session_id,
        )

        cbox = CanvasBox(
            detection_id=det_id,
            element_type=chosen_type,
            confidence=None,
            text_content=text_content,
            features=features,
            pdf_bbox=pdf_bbox,
            corrected=True,
        )
        self._canvas_boxes.append(cbox)
        self._draw_box(cbox)
        self._select_box(cbox)
        self._session_count += 1
        self._update_session_label()
        self._update_page_summary()
        n_chars = len(text_content)
        self._status.configure(
            text=f"Pasted {chosen_type} detection ({n_chars} chars extracted)"
        )

    def _key_copy_box(self, event: tk.Event) -> None:
        """Ctrl+C — copy selected words' text, or the selected box."""
        if isinstance(event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)):
            return  # let native copy handle it
        if not self._is_active_tab():
            return
        # If words are selected, copy their text to clipboard
        if self._selected_word_rids and self._word_overlay_items:
            texts = []
            for rid in self._selected_word_rids:
                winfo = self._word_overlay_items.get(rid)
                if winfo and winfo.get("text"):
                    texts.append(winfo["text"])
            if texts:
                self.root.clipboard_clear()
                self.root.clipboard_append(" ".join(texts))
                self._status.configure(
                    text=f"Copied text from {len(texts)} words to clipboard"
                )
                return
        if self._selected_box:
            self._copy_box(self._selected_box)

    def _key_paste_box(self, event: tk.Event) -> None:
        """Ctrl+V — paste a copied box at the centre of the current view."""
        if isinstance(event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)):
            return  # let native paste handle it
        if not self._is_active_tab() or not self._copied_box_template:
            return
        # Place the box at the centre of the visible canvas area
        eff = self._effective_scale()
        # Visible region in canvas coords
        try:
            vx0 = self._canvas.canvasx(0)
            vy0 = self._canvas.canvasy(0)
            vx1 = self._canvas.canvasx(self._canvas.winfo_width())
            vy1 = self._canvas.canvasy(self._canvas.winfo_height())
            pdf_cx = ((vx0 + vx1) / 2) / eff
            pdf_cy = ((vy0 + vy1) / 2) / eff
        except Exception:
            pdf_cx, pdf_cy = 300.0, 300.0
        self._paste_box(pdf_cx, pdf_cy)

    # ── Group actions ──────────────────────────────────────────────

    def _create_group(self, cbox: CanvasBox) -> None:
        """Create a new group with *cbox* as the root (parent)."""
        if not self._doc_id:
            return

        # Prompt for group name
        name_win = tk.Toplevel(self.root)
        name_win.title("New Group")
        name_win.geometry("280x110")
        name_win.transient(self.root)
        name_win.grab_set()

        ttk.Label(name_win, text="Group name:").pack(padx=10, pady=(10, 4))
        name_var = tk.StringVar(value=cbox.element_type)
        entry = ttk.Entry(name_win, textvariable=name_var, width=28)
        entry.pack(padx=10)
        entry.selection_range(0, "end")
        entry.focus_set()

        result: list[str | None] = [None]

        def on_ok(_event: tk.Event | None = None) -> None:
            result[0] = name_var.get().strip()
            name_win.destroy()

        def on_cancel() -> None:
            name_win.destroy()

        entry.bind("<Return>", on_ok)
        btn_f = ttk.Frame(name_win)
        btn_f.pack(pady=8)
        ttk.Button(btn_f, text="OK", command=on_ok).pack(side="left", padx=4)
        ttk.Button(btn_f, text="ABORT", command=on_cancel).pack(side="left", padx=4)

        name_win.wait_window()

        label = result[0]
        if not label:
            return

        group_id = self._store.create_group(
            doc_id=self._doc_id,
            page=self._page,
            group_label=label,
            root_detection_id=cbox.detection_id,
        )
        cbox.group_id = group_id
        cbox.is_group_root = True
        self._groups[group_id] = {
            "label": label,
            "root_detection_id": cbox.detection_id,
            "members": [cbox.detection_id],
        }
        self._draw_box(cbox)
        self._select_box(cbox)
        self._status.configure(text=f"Created group \u2039{label}\u203a")

    def _on_create_group(self) -> None:
        """Inspector button: create group from selected box."""
        if self._selected_box and not self._selected_box.group_id:
            self._create_group(self._selected_box)

    def _on_add_to_group(self) -> None:
        """Inspector button: add multi-selected boxes to the active group."""
        if not self._selected_box or not self._selected_box.is_group_root:
            self._status.configure(text="Select the group parent first")
            return
        # Gather targets — multi-selected minus the root itself
        targets = [
            cb
            for cb in self._multi_selected
            if cb is not self._selected_box and not cb.group_id
        ]
        if not targets:
            self._status.configure(
                text="Shift+click boxes to add, then click 'Add to Group'"
            )
            return
        self._add_children_to_group(targets)

    def _add_children_to_group(self, targets: list["CanvasBox"]) -> None:
        """Add a list of CanvasBoxes as children to the selected group."""
        if not self._selected_box or not self._selected_box.group_id:
            return
        gid = self._selected_box.group_id
        grp = self._groups.get(gid)
        if not grp:
            return

        count = 0
        next_order = len(grp["members"])
        for cb in targets:
            if cb.group_id:
                continue  # already in a group
            self._store.add_to_group(gid, cb.detection_id, sort_order=next_order)
            cb.group_id = gid
            cb.is_group_root = False
            grp["members"].append(cb.detection_id)
            next_order += 1
            count += 1
            self._draw_box(cb)

        if count:
            self._draw_group_links(gid)
            self._status.configure(
                text=f"Added {count} box(es) to group \u2039{grp['label']}\u203a"
            )

    def _on_remove_from_group(self) -> None:
        """Inspector button: remove selected box from its group."""
        if self._selected_box and self._selected_box.group_id:
            self._remove_from_group(self._selected_box)

    def _remove_from_group(self, cbox: CanvasBox) -> None:
        """Remove *cbox* from its group. Deletes group if root."""
        gid = cbox.group_id
        if not gid:
            return
        grp = self._groups.get(gid, {})
        label = grp.get("label", "?")
        is_root = cbox.is_group_root

        self._store.remove_from_group(gid, cbox.detection_id)

        if is_root:
            # Remove group from all canvas boxes
            for cb in self._canvas_boxes:
                if cb.group_id == gid:
                    cb.group_id = None
                    cb.is_group_root = False
                    self._draw_box(cb)
            self._groups.pop(gid, None)
            self._clear_group_links()
            self._status.configure(
                text=f"Deleted group \u2039{label}\u203a (parent removed)"
            )
        else:
            cbox.group_id = None
            cbox.is_group_root = False
            if gid in self._groups:
                members = self._groups[gid]["members"]
                if cbox.detection_id in members:
                    members.remove(cbox.detection_id)
            self._draw_box(cbox)
            self._draw_group_links(gid)
            self._status.configure(text=f"Removed from group \u2039{label}\u203a")

        # Refresh inspector
        if self._selected_box:
            self._select_box(self._selected_box)

    def _draw_group_links(self, group_id: str) -> None:
        """Draw dashed lines from child boxes to the group root."""
        self._clear_group_links()
        grp = self._groups.get(group_id)
        if not grp:
            return

        root_id = grp["root_detection_id"]
        root_box: CanvasBox | None = None
        children: list[CanvasBox] = []
        for cb in self._canvas_boxes:
            if cb.detection_id == root_id:
                root_box = cb
            elif cb.group_id == group_id:
                children.append(cb)

        if not root_box:
            return

        eff = self._effective_scale()
        rx = (root_box.pdf_bbox[0] + root_box.pdf_bbox[2]) / 2 * eff
        ry = (root_box.pdf_bbox[1] + root_box.pdf_bbox[3]) / 2 * eff
        color = self.LABEL_COLORS.get(root_box.element_type, "#888888")

        for child in children:
            cx = (child.pdf_bbox[0] + child.pdf_bbox[2]) / 2 * eff
            cy = (child.pdf_bbox[1] + child.pdf_bbox[3]) / 2 * eff
            lid = self._canvas.create_line(
                rx,
                ry,
                cx,
                cy,
                dash=(6, 4),
                fill=color,
                width=1,
                tags="group_link",
            )
            self._group_link_ids.append(lid)

    def _clear_group_links(self) -> None:
        """Remove all group connector lines from the canvas."""
        for lid in self._group_link_ids:
            self._canvas.delete(lid)
        self._group_link_ids.clear()

    def _update_group_inspector(self, cbox: CanvasBox | None) -> None:
        """Populate the group section of the inspector."""
        self._clear_group_links()
        # Hide both buttons by default
        self._btn_create_group.pack_forget()
        self._btn_add_to_group.pack_forget()
        self._btn_remove_group.pack_forget()

        if cbox is None or not cbox.group_id:
            self._insp_group_label.configure(text="—")
            if cbox is not None:
                self._btn_create_group.pack(side="left", padx=(0, 3))
            return

        grp = self._groups.get(cbox.group_id, {})
        label = grp.get("label", "?")
        if cbox.is_group_root:
            n_members = len(grp.get("members", [])) - 1  # exclude root
            self._insp_group_label.configure(
                text=f"\u25cf {label} (parent \u2014 {n_members} children)"
            )
            self._btn_add_to_group.pack(side="left", padx=(0, 3))
        else:
            self._insp_group_label.configure(text=f"\u2192 {label}")
        self._btn_remove_group.pack(side="left", padx=(0, 3))
        self._draw_group_links(cbox.group_id)

    def _select_box(self, cbox: CanvasBox) -> None:
        """Select a box and populate the inspector."""
        # Deselect previous
        if self._selected_box and self._selected_box is not cbox:
            self._selected_box.selected = False
            self._draw_box(self._selected_box)

        cbox.selected = True
        self._selected_box = cbox
        self._draw_box(cbox)

        # Populate inspector
        self._insp_id.configure(text=cbox.detection_id)
        self._type_var.set(cbox.element_type)
        conf_text = f"{cbox.confidence:.2%}" if cbox.confidence is not None else "—"
        self._insp_conf.configure(text=conf_text)

        self._insp_text.config(state="normal")
        self._insp_text.delete("1.0", "end")
        self._insp_text.insert("1.0", cbox.text_content)
        self._insp_text.config(state="disabled")
        self._model_suggestion = None
        self._suggest_label.configure(text="")
        self._suggest_btn.pack_forget()
        if cbox.features and self._classifier.model_exists():
            try:
                pred_label, pred_conf = self._classifier.predict(cbox.features)
                if pred_label != cbox.element_type:
                    self._model_suggestion = pred_label
                    self._suggest_label.configure(
                        text=f"Model suggests: {pred_label} ({pred_conf:.0%})"
                    )
                    self._suggest_btn.pack(side="left", padx=4)
            except Exception:
                pass

        # Update multi-select count
        self._update_multi_label()

        # Update group section
        self._update_group_inspector(cbox)

    def _deselect(self) -> None:
        """Deselect the current box."""
        if self._selected_box:
            self._selected_box.selected = False
            self._draw_box(self._selected_box)
            self._selected_box = None

        self._insp_id.configure(text="—")
        self._type_var.set("")
        self._insp_conf.configure(text="—")
        self._insp_text.config(state="normal")
        self._insp_text.delete("1.0", "end")
        self._insp_text.config(state="disabled")

        # Clear suggestion
        self._model_suggestion = None
        self._suggest_label.configure(text="")
        self._suggest_btn.pack_forget()

        # Clear group section
        self._update_group_inspector(None)

    # ── Drag (reshape + move + add mode) ────────────────────────────

    def _on_canvas_drag(self, event: tk.Event) -> None:
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)

        # Handle reshape drag
        if self._drag_handle and self._selected_box:
            self._do_handle_drag(cx, cy)
            return

        # Handle move drag (second click on selected box)
        if self._move_dragging and self._selected_box and self._move_start_pdf:
            self._do_move_drag(cx, cy)
            return

        # Handle add-mode drag
        if self._mode == "add" and self._draw_start:
            if self._draw_rect_id:
                self._canvas.delete(self._draw_rect_id)
            sx, sy = self._draw_start
            self._draw_rect_id = self._canvas.create_rectangle(
                sx,
                sy,
                cx,
                cy,
                outline="#cccccc",
                width=2,
                dash=(4, 4),
            )
            return

        # Lasso selection drag
        if self._lasso_start and self._mode == "select":
            if self._lasso_rect_id:
                self._canvas.delete(self._lasso_rect_id)
            sx, sy = self._lasso_start
            self._lasso_rect_id = self._canvas.create_rectangle(
                sx,
                sy,
                cx,
                cy,
                outline="#00bfff",
                width=1,
                dash=(3, 3),
            )

    def _do_move_drag(self, cx: float, cy: float) -> None:
        """Update the box position during a move drag."""
        if (
            not self._selected_box
            or not self._move_start_pdf
            or not self._move_orig_bbox
        ):
            return

        eff = self._effective_scale()
        pdf_x = cx / eff
        pdf_y = cy / eff

        dx = pdf_x - self._move_start_pdf[0]
        dy = pdf_y - self._move_start_pdf[1]

        ox0, oy0, ox1, oy1 = self._move_orig_bbox
        nx0 = ox0 + dx
        ny0 = oy0 + dy
        nx1 = ox1 + dx
        ny1 = oy1 + dy

        # Clamp to non-negative
        if nx0 < 0:
            nx1 -= nx0
            nx0 = 0
        if ny0 < 0:
            ny1 -= ny0
            ny0 = 0

        self._selected_box.pdf_bbox = (nx0, ny0, nx1, ny1)

        # Move polygon too if merged box
        if self._move_orig_polygon:
            self._selected_box.polygon = [
                (px + dx, py + dy) for px, py in self._move_orig_polygon
            ]

        self._draw_box(self._selected_box)

    def _finalize_move(self) -> None:
        """Save a reshape correction after move drag completes."""
        cbox = self._selected_box
        if not cbox or not self._move_orig_bbox or not self._doc_id:
            return

        orig_bbox = self._move_orig_bbox
        new_bbox = cbox.pdf_bbox

        if orig_bbox == new_bbox:
            return  # no change

        self._push_undo("reshape", cbox, extra={"orig_bbox": orig_bbox})
        self._store.save_correction(
            doc_id=self._doc_id,
            page=self._page,
            correction_type="reshape",
            corrected_label=cbox.element_type,
            corrected_bbox=new_bbox,
            detection_id=cbox.detection_id,
            original_label=cbox.element_type,
            original_bbox=orig_bbox,
            session_id=self._session_id,
        )
        cbox.corrected = True
        self._session_count += 1
        self._update_session_label()
        # Persist updated polygon to DB if present
        if cbox.polygon:
            self._store.update_detection_polygon(
                cbox.detection_id, cbox.polygon, new_bbox
            )
        self._draw_box(cbox)
        self._status.configure(text="Moved box to new position")

    def _do_handle_drag(self, cx: float, cy: float) -> None:
        """Update the box rectangle during a handle drag."""
        if not self._selected_box or not self._drag_orig_bbox:
            return

        eff = self._effective_scale()
        ox0, oy0, ox1, oy1 = self._drag_orig_bbox
        # Convert current mouse to PDF coordinates
        px = cx / eff
        py = cy / eff

        # Compute new bbox based on which handle is being dragged
        if not self._drag_handle:
            return
        nx0, ny0, nx1, ny1 = _reshape_bbox_from_handle(
            self._drag_orig_bbox, self._drag_handle, px, py, min_size=1.0
        )

        # If this is a merged box rendered as a polygon, reshape the polygon too.
        if self._selected_box.polygon and self._drag_orig_polygon:
            self._selected_box.polygon = _scale_polygon_to_bbox(
                self._drag_orig_bbox,
                self._drag_orig_polygon,
                (nx0, ny0, nx1, ny1),
            )

        # Live update on canvas
        self._selected_box.pdf_bbox = (nx0, ny0, nx1, ny1)
        self._draw_box(self._selected_box)

    def _on_canvas_release(self, event: tk.Event) -> None:
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)

        # Finalize move drag
        if self._move_dragging and self._selected_box:
            self._finalize_move()
            self._move_dragging = False
            self._move_start_pdf = None
            self._move_orig_bbox = None
            self._move_orig_polygon = None
            return

        # Finalize reshape
        if self._drag_handle and self._selected_box:
            self._finalize_reshape()
            self._drag_handle = None
            self._drag_orig_bbox = None
            self._drag_orig_polygon = None
            return

        # Finalize add-mode draw
        if self._mode == "add" and self._draw_start:
            self._finalize_add(cx, cy)
            return

        # Finalize lasso selection
        if self._lasso_start and self._lasso_rect_id:
            self._finalize_lasso(cx, cy)
            return
        self._lasso_start = None

    def _finalize_reshape(self) -> None:
        """Save a reshape correction after handle drag completes."""
        cbox = self._selected_box
        if not cbox or not self._drag_orig_bbox or not self._doc_id:
            return

        orig_bbox = self._drag_orig_bbox
        new_bbox = cbox.pdf_bbox

        if orig_bbox == new_bbox:
            return  # no change

        extra: dict = {"orig_bbox": orig_bbox}
        if self._drag_orig_polygon is not None:
            extra["orig_polygon"] = copy.deepcopy(self._drag_orig_polygon)
            extra["polygon"] = copy.deepcopy(cbox.polygon)

        self._push_undo("reshape", cbox, extra=extra)
        self._store.save_correction(
            doc_id=self._doc_id,
            page=self._page,
            correction_type="reshape",
            corrected_label=cbox.element_type,
            corrected_bbox=new_bbox,
            detection_id=cbox.detection_id,
            original_label=cbox.element_type,
            original_bbox=orig_bbox,
            session_id=self._session_id,
        )
        cbox.corrected = True
        self._session_count += 1
        self._update_session_label()
        if cbox.polygon:
            self._store.update_detection_polygon(cbox.detection_id, cbox.polygon, new_bbox)
        self._draw_box(cbox)

    def _finalize_add(self, cx: float, cy: float) -> None:
        """Complete adding a new detection after drawing a rectangle."""
        if self._draw_rect_id:
            self._canvas.delete(self._draw_rect_id)
            self._draw_rect_id = None

        if not self._draw_start or not self._doc_id:
            self._draw_start = None
            return

        sx, sy = self._draw_start
        self._draw_start = None

        eff = self._effective_scale()
        x0 = min(sx, cx) / eff
        y0 = min(sy, cy) / eff
        x1 = max(sx, cx) / eff
        y1 = max(sy, cy) / eff

        # Minimum size check
        if (x1 - x0) < 10 or (y1 - y0) < 10:
            self._status.configure(text="Box too small (min 10pt in each dimension)")
            return

        pdf_bbox = (x0, y0, x1, y1)

        # Ask for element type
        type_win = tk.Toplevel(self.root)
        type_win.title("New Element Type")
        type_win.geometry("260x120")
        type_win.transient(self.root)
        type_win.grab_set()

        ttk.Label(type_win, text="Element type:").pack(padx=10, pady=(10, 4))
        type_var = tk.StringVar(value=self.ELEMENT_TYPES[0])
        combo = ttk.Combobox(
            type_win,
            textvariable=type_var,
            values=self.ELEMENT_TYPES,
            width=20,
        )
        combo.pack(padx=10)

        result: list[str | None] = [None]

        def on_ok():
            result[0] = type_var.get()
            type_win.destroy()

        def on_cancel():
            type_win.destroy()

        btn_f = ttk.Frame(type_win)
        btn_f.pack(pady=10)
        ttk.Button(btn_f, text="OK", command=on_ok).pack(side="left", padx=4)
        ttk.Button(btn_f, text="ABORT", command=on_cancel).pack(side="left", padx=4)

        type_win.wait_window()

        chosen_type = result[0]
        if not chosen_type:
            return

        # Extract text from PDF under the drawn box
        text_content = ""
        if self._pdf_path:
            text_content = extract_text_in_bbox(self._pdf_path, self._page, pdf_bbox)

        # Compute features and save
        features = featurize_region(
            chosen_type, pdf_bbox, None, 2448.0, 1584.0  # fallback dims
        )
        det_id = self._store.save_detection(
            doc_id=self._doc_id,
            page=self._page,
            run_id=self._run_id or "manual",
            element_type=chosen_type,
            bbox=pdf_bbox,
            text_content=text_content,
            features=features,
        )
        # Also save as an "add" correction
        self._store.save_correction(
            doc_id=self._doc_id,
            page=self._page,
            correction_type="add",
            corrected_label=chosen_type,
            corrected_bbox=pdf_bbox,
            detection_id=det_id,
            session_id=self._session_id,
        )

        cbox = CanvasBox(
            detection_id=det_id,
            element_type=chosen_type,
            confidence=None,
            text_content=text_content,
            features=features,
            pdf_bbox=pdf_bbox,
            corrected=True,
        )
        self._canvas_boxes.append(cbox)
        self._draw_box(cbox)
        self._select_box(cbox)
        self._session_count += 1
        self._update_session_label()
        self._update_page_summary()
        n_chars = len(text_content)
        self._status.configure(
            text=f"Added {chosen_type} detection ({n_chars} chars extracted)"
        )

    # ── Inspector actions ──────────────────────────────────────────

    def _on_rescan_text(self) -> None:
        """Re-extract text from the PDF under the selected box's current bbox."""
        if not self._selected_box:
            self._status.configure(text="No box selected")
            return
        if not self._pdf_path:
            self._status.configure(text="No PDF loaded")
            return

        cbox = self._selected_box
        try:
            new_text = self._extract_text_for_box(cbox)
        except Exception as exc:
            self._status.configure(text=f"Rescan failed: {exc}")
            return

        cbox.text_content = new_text

        # Update the inspector text widget
        self._insp_text.config(state="normal")
        self._insp_text.delete("1.0", "end")
        self._insp_text.insert("1.0", new_text)
        self._insp_text.config(state="disabled")

        mode = "polygon" if cbox.polygon else "rect"
        n_chars = len(new_text)
        self._status.configure(
            text=f"Rescanned text for {cbox.element_type} ({mode}) — {n_chars} chars"
        )

    def _on_accept(self) -> None:
        # Batch-aware: apply to all multi-selected + selected box
        targets = list(self._multi_selected)
        if self._selected_box and self._selected_box not in targets:
            targets.append(self._selected_box)
        if not targets or not self._doc_id:
            self._status.configure(text="No box selected")
            return

        for cbox in targets:
            self._push_undo("accept", cbox)
            self._store.accept_detection(cbox.detection_id, self._doc_id, self._page)
            cbox.corrected = True
            self._session_count += 1
            self._draw_box(cbox)

        self._update_session_label()
        self._clear_multi_select()
        self._status.configure(text=f"Accepted {len(targets)} box(es)")

    def _on_relabel(self) -> None:
        # Batch-aware: apply to all multi-selected + selected box
        targets = list(self._multi_selected)
        if self._selected_box and self._selected_box not in targets:
            targets.append(self._selected_box)
        if not targets or not self._doc_id:
            self._status.configure(text="No box selected")
            return

        new_label = self._type_var.get()
        if not new_label:
            return

        count = 0
        for cbox in targets:
            if new_label == cbox.element_type:
                # No change — treat as accept for this box
                self._push_undo("accept", cbox)
                self._store.accept_detection(
                    cbox.detection_id, self._doc_id, self._page
                )
                cbox.corrected = True
                self._session_count += 1
                self._draw_box(cbox)
                continue

            self._push_undo("relabel", cbox, extra={"old_label": cbox.element_type})
            self._store.save_correction(
                doc_id=self._doc_id,
                page=self._page,
                correction_type="relabel",
                corrected_label=new_label,
                corrected_bbox=cbox.pdf_bbox,
                detection_id=cbox.detection_id,
                original_label=cbox.element_type,
                original_bbox=cbox.pdf_bbox,
                session_id=self._session_id,
            )
            cbox.element_type = new_label
            cbox.corrected = True
            self._session_count += 1
            self._draw_box(cbox)
            count += 1

        self._update_session_label()
        self._update_page_summary()
        self._clear_multi_select()
        self._status.configure(text=f"Relabelled {count} box(es) → {new_label}")

    def _on_delete(self) -> None:
        # Batch-aware: apply to all multi-selected + selected box
        targets = list(self._multi_selected)
        if self._selected_box and self._selected_box not in targets:
            targets.append(self._selected_box)
        if not targets or not self._doc_id:
            self._status.configure(text="No box selected")
            return

        n = len(targets)
        msg = (
            "Mark this detection as a false positive?"
            if n == 1
            else f"Reject {n} selected detections?"
        )
        if not messagebox.askyesno("Reject Detection", msg):
            return

        for cbox in targets:
            self._push_undo("delete", cbox)
            self._store.save_correction(
                doc_id=self._doc_id,
                page=self._page,
                correction_type="delete",
                corrected_label=cbox.element_type,
                corrected_bbox=cbox.pdf_bbox,
                detection_id=cbox.detection_id,
                original_label=cbox.element_type,
                original_bbox=cbox.pdf_bbox,
                session_id=self._session_id,
            )

            # Remove from canvas
            if cbox.rect_id:
                self._canvas.delete(cbox.rect_id)
            if cbox.label_id:
                self._canvas.delete(cbox.label_id)
            if cbox.conf_dot_id:
                self._canvas.delete(cbox.conf_dot_id)
            for hid in cbox.handle_ids:
                self._canvas.delete(hid)
            if cbox in self._canvas_boxes:
                self._canvas_boxes.remove(cbox)
            self._session_count += 1

        self._selected_box = None
        self._multi_selected.clear()
        self._deselect()
        self._update_multi_label()
        self._update_session_label()
        self._update_page_summary()
        self._status.configure(text=f"Rejected {n} detection(s)")

    def _update_session_label(self) -> None:
        self._session_label.configure(text=f"Session: {self._session_count} saved")

    # ── Zoom ───────────────────────────────────────────────────────

    def _on_mousewheel(self, event: tk.Event) -> None:
        """Scroll vertically with the mouse wheel."""
        self._canvas.yview_scroll(-1 * (event.delta // 120), "units")

    def _on_shift_mousewheel(self, event: tk.Event) -> None:
        """Scroll horizontally with Shift + mouse wheel."""
        self._canvas.xview_scroll(-1 * (event.delta // 120), "units")

    def _apply_zoom(self, factor: float) -> None:
        new_zoom = self._zoom * factor
        new_zoom = max(0.25, min(new_zoom, 5.0))
        if new_zoom == self._zoom:
            return
        self._zoom = new_zoom
        self._render_background()
        self._draw_all_boxes()
        self._status.configure(text=f"Zoom: {self._zoom:.0%}")

    def _fit_to_window(self) -> None:
        """Set zoom so the full page fits inside the visible canvas."""
        if self._bg_image is None:
            return
        # Force geometry update so winfo returns real sizes
        self._canvas.update_idletasks()
        canvas_w = self._canvas.winfo_width()
        canvas_h = self._canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            return
        img_w = self._bg_image.width  # rendered at current DPI (zoom 1.0)
        img_h = self._bg_image.height
        zoom_w = canvas_w / img_w
        zoom_h = canvas_h / img_h
        new_zoom = min(zoom_w, zoom_h)
        new_zoom = max(0.25, min(new_zoom, 5.0))
        self._zoom = new_zoom
        self._render_background()
        self._draw_all_boxes()
        self._status.configure(text=f"Fit: {self._zoom:.0%}")

    # ── Pan (middle mouse button) ──────────────────────────────────

    def _on_pan_start(self, event: tk.Event) -> None:
        self._pan_start = (event.x, event.y)
        self._canvas.config(cursor="fleur")

    def _on_pan_motion(self, event: tk.Event) -> None:
        if self._pan_start:
            dx = event.x - self._pan_start[0]
            dy = event.y - self._pan_start[1]
            self._canvas.xview_scroll(-dx, "units")
            self._canvas.yview_scroll(-dy, "units")
            self._pan_start = (event.x, event.y)

    def _on_pan_end(self, event: tk.Event) -> None:
        self._pan_start = None
        if self._mode == "add":
            self._canvas.config(cursor="crosshair")
        else:
            self._canvas.config(cursor="")

    # ── Polygon-aware text extraction ────────────────────────────────

    def _extract_text_for_box(self, cbox: CanvasBox) -> str:
        """Extract text using the polygon boundary when available,
        falling back to the rectangular bbox otherwise."""
        if not self._pdf_path:
            return ""
        if cbox.polygon:
            return extract_text_in_polygon(self._pdf_path, self._page, cbox.polygon)
        return extract_text_in_bbox(self._pdf_path, self._page, cbox.pdf_bbox)

    # ── Word overlay ───────────────────────────────────────────────

    def _toggle_word_overlay(self) -> None:
        """Toggle the pdfplumber word-boxes overlay on or off."""
        self._word_overlay_on = self._word_overlay_var.get()
        if self._word_overlay_on:
            self._draw_word_overlay()
        else:
            self._clear_word_overlay()

    def _draw_word_overlay(self) -> None:
        """Draw light-gray rectangles around every word on the current page."""
        self._clear_word_overlay()
        if not self._pdf_path:
            return
        try:
            from plancheck.ingest.ingest import extract_page_words

            words = extract_page_words(self._pdf_path, self._page)
        except Exception as exc:
            self._status.configure(text=f"Word overlay failed: {exc}")
            return

        eff = self._effective_scale()
        for w in words:
            cx0 = w["x0"] * eff
            cy0 = w["top"] * eff
            cx1 = w["x1"] * eff
            cy1 = w["bottom"] * eff
            rid = self._canvas.create_rectangle(
                cx0,
                cy0,
                cx1,
                cy1,
                outline="#b0b0b0",
                width=1,
                tags="word_overlay",
            )
            self._word_overlay_ids.append(rid)
            # Store word metadata keyed by canvas item id for hit-testing
            self._word_overlay_items[rid] = {
                "x0": w["x0"],
                "top": w["top"],
                "x1": w["x1"],
                "bottom": w["bottom"],
                "text": w.get("text", ""),
            }

        n = len(words)
        self._status.configure(text=f"Word overlay: {n} words on page {self._page}")

    def _clear_word_overlay(self) -> None:
        """Remove all word overlay rectangles from the canvas."""
        self._canvas.delete("word_overlay")
        self._word_overlay_ids.clear()
        self._word_overlay_items.clear()
        self._selected_word_rids.clear()

    # ── Keyboard shortcuts ─────────────────────────────────────────

    def _is_active_tab(self) -> bool:
        """Check if the annotation tab is currently visible."""
        try:
            return self.notebook.index("current") == self.notebook.index(self.frame)
        except Exception:
            return False

    def _key_accept(self, event: tk.Event) -> None:
        if self._is_active_tab() and not isinstance(
            event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)
        ):
            self._on_accept()

    def _key_delete(self, event: tk.Event) -> None:
        if self._is_active_tab() and not isinstance(
            event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)
        ):
            self._on_delete()

    def _key_relabel(self, event: tk.Event) -> None:
        if self._is_active_tab() and not isinstance(
            event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)
        ):
            self._type_combo.focus_set()

    def _key_deselect(self, event: tk.Event) -> None:
        if self._is_active_tab():
            self._deselect()

    def _key_prev_box(self, event: tk.Event) -> None:
        if not self._is_active_tab() or not self._canvas_boxes:
            return
        if isinstance(event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)):
            return
        if self._selected_box is None:
            self._select_box(self._canvas_boxes[-1])
        else:
            idx = self._canvas_boxes.index(self._selected_box)
            prev_idx = (idx - 1) % len(self._canvas_boxes)
            self._select_box(self._canvas_boxes[prev_idx])

    def _key_next_box(self, event: tk.Event) -> None:
        if not self._is_active_tab() or not self._canvas_boxes:
            return
        if isinstance(event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)):
            return
        if self._selected_box is None:
            self._select_box(self._canvas_boxes[0])
        else:
            idx = self._canvas_boxes.index(self._selected_box)
            next_idx = (idx + 1) % len(self._canvas_boxes)
            self._select_box(self._canvas_boxes[next_idx])

    def _key_zoom_in(self, event: tk.Event) -> None:
        if self._is_active_tab():
            self._apply_zoom(1.2)

    def _key_zoom_out(self, event: tk.Event) -> None:
        if self._is_active_tab():
            self._apply_zoom(1 / 1.2)

    def _key_fit(self, event: tk.Event) -> None:
        if isinstance(event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)):
            return
        if self._is_active_tab():
            self._fit_to_window()

    # ── Undo / Redo ────────────────────────────────────────────────

    def _push_undo(
        self,
        action: str,
        cbox: CanvasBox,
        *,
        extra: dict | None = None,
    ) -> None:
        """Push an undo record and clear the redo stack."""
        record: dict = {
            "action": action,
            "detection_id": cbox.detection_id,
            "element_type": cbox.element_type,
            "pdf_bbox": cbox.pdf_bbox,
            "confidence": cbox.confidence,
            "corrected": cbox.corrected,
        }
        if extra:
            record.update(extra)
        self._undo_stack.append(record)
        self._redo_stack.clear()

    def _undo(self) -> None:
        """Undo the last correction (visual only — DB is append-only)."""
        if not self._undo_stack:
            self._status.configure(text="Nothing to undo")
            return

        rec = self._undo_stack.pop()
        self._redo_stack.append(rec)
        action = rec["action"]

        # Find the affected canvas box
        target = None
        for cb in self._canvas_boxes:
            if cb.detection_id == rec["detection_id"]:
                target = cb
                break

        if action == "relabel" and target:
            old_label = rec.get("old_label", target.element_type)
            target.element_type = old_label
            target.corrected = rec.get("corrected", False)
            self._draw_box(target)
            self._status.configure(text=f"Undo relabel → {old_label}")
        elif action == "reshape" and target:
            orig = rec.get("orig_bbox")
            if orig:
                target.pdf_bbox = orig
                target.corrected = rec.get("corrected", False)
                # Restore polygon if the undo record saved one
                if "orig_polygon" in rec:
                    target.polygon = copy.deepcopy(rec["orig_polygon"])
                self._draw_box(target)
            self._status.configure(text="Undo reshape")
        elif action == "delete":
            # Re-add the box visually
            cbox = CanvasBox(
                detection_id=rec["detection_id"],
                element_type=rec["element_type"],
                confidence=rec.get("confidence"),
                text_content="",
                features={},
                pdf_bbox=rec["pdf_bbox"],
                corrected=rec.get("corrected", False),
            )
            self._canvas_boxes.append(cbox)
            self._draw_box(cbox)
            self._status.configure(text="Undo reject")
        elif action == "accept" and target:
            target.corrected = rec.get("corrected", False)
            self._draw_box(target)
            self._status.configure(text="Undo accept")
        elif action == "merge" and target:
            # Restore all original boxes from the merge record
            merged_boxes = rec.get("merged_boxes", [])
            # Remove the merged survivor from canvas
            if target.rect_id:
                self._canvas.delete(target.rect_id)
            if target.label_id:
                self._canvas.delete(target.label_id)
            for hid in target.handle_ids:
                self._canvas.delete(hid)
            self._canvas_boxes.remove(target)
            # Re-add all original boxes
            for mb in merged_boxes:
                cbox = CanvasBox(
                    detection_id=mb["detection_id"],
                    element_type=mb["element_type"],
                    confidence=mb.get("confidence"),
                    text_content=mb.get("text_content", ""),
                    features={},
                    pdf_bbox=mb["pdf_bbox"],
                    polygon=mb.get("polygon"),
                    corrected=mb.get("corrected", False),
                )
                self._canvas_boxes.append(cbox)
                self._draw_box(cbox)
            self._status.configure(
                text=f"Undo merge → restored {len(merged_boxes)} boxes"
            )
        else:
            self._status.configure(text="Undo (no visual change)")
        self._update_page_summary()

    def _redo(self) -> None:
        """Redo the last undone action."""
        if not self._redo_stack:
            self._status.configure(text="Nothing to redo")
            return

        rec = self._redo_stack.pop()
        self._undo_stack.append(rec)
        action = rec["action"]

        target = None
        for cb in self._canvas_boxes:
            if cb.detection_id == rec["detection_id"]:
                target = cb
                break

        if action == "relabel" and target:
            # rec stores the old_label; the "new" label was the type at the
            # time _push_undo was called, which is stored in element_type.
            target.element_type = rec["element_type"]
            target.corrected = True
            self._draw_box(target)
            self._status.configure(text=f"Redo relabel → {rec['element_type']}")
        elif action == "reshape" and target:
            target.pdf_bbox = rec["pdf_bbox"]
            target.corrected = True
            # If undo record includes a polygon, restore it. Otherwise keep legacy
            # behavior (some reshape flows intentionally clear the polygon).
            if "polygon" in rec:
                target.polygon = copy.deepcopy(rec["polygon"])
            elif "orig_polygon" in rec:
                target.polygon = None
            self._draw_box(target)
            self._status.configure(text="Redo reshape")
        elif action == "delete" and target:
            if target.rect_id:
                self._canvas.delete(target.rect_id)
            if target.label_id:
                self._canvas.delete(target.label_id)
            for hid in target.handle_ids:
                self._canvas.delete(hid)
            self._canvas_boxes.remove(target)
            if self._selected_box is target:
                self._deselect()
            self._status.configure(text="Redo reject")
        elif action == "accept" and target:
            target.corrected = True
            self._draw_box(target)
            self._status.configure(text="Redo accept")
        else:
            self._status.configure(text="Redo (no visual change)")
        self._update_page_summary()

    def _key_undo(self, event: tk.Event) -> None:
        if self._is_active_tab():
            self._undo()

    def _key_redo(self, event: tk.Event) -> None:
        if self._is_active_tab():
            self._redo()

    # ── Multi-select ───────────────────────────────────────────────

    def _toggle_multi_select(self, cbox: CanvasBox) -> None:
        """Toggle a box in the multi-selection set."""
        if cbox in self._multi_selected:
            self._multi_selected.remove(cbox)
        else:
            self._multi_selected.append(cbox)
        self._draw_box(cbox)
        self._update_multi_label()

    def _clear_multi_select(self) -> None:
        """Clear all multi-selected boxes."""
        prev = list(self._multi_selected)
        self._multi_selected.clear()
        for cb in prev:
            self._draw_box(cb)
        self._update_multi_label()

    def _update_multi_label(self) -> None:
        """Update the '3 selected' label in the inspector."""
        n = len(self._multi_selected)
        if self._selected_box and self._selected_box not in self._multi_selected:
            n += 1
        if n > 1:
            self._multi_label.configure(text=f"{n} selected")
        else:
            self._multi_label.configure(text="")

    # ── Word-box selection helpers ──────────────────────────────────

    def _set_word_selected(self, rid: int, selected: bool) -> None:
        """Highlight or un-highlight a single word overlay rectangle."""
        if selected:
            self._selected_word_rids.add(rid)
            self._canvas.itemconfigure(rid, outline="#00bfff", width=2)
        else:
            self._selected_word_rids.discard(rid)
            self._canvas.itemconfigure(rid, outline="#b0b0b0", width=1)

    def _toggle_word_selected(self, rid: int) -> None:
        """Toggle a word rectangle's selection state."""
        if rid in self._selected_word_rids:
            self._set_word_selected(rid, False)
        else:
            self._set_word_selected(rid, True)
        self._update_word_selection_label()

    def _clear_word_selection(self) -> None:
        """Deselect all word overlay rectangles."""
        for rid in list(self._selected_word_rids):
            self._set_word_selected(rid, False)
        self._selected_word_rids.clear()
        self._update_word_selection_label()

    def _update_word_selection_label(self) -> None:
        """Update the status bar with word selection count."""
        n = len(self._selected_word_rids)
        if n > 0:
            self._status.configure(text=f"{n} word(s) selected")

    def _key_select_all(self, event: tk.Event) -> None:
        """Ctrl+A — select all visible boxes (and all words if overlay is on)."""
        if not self._is_active_tab():
            return
        if isinstance(event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)):
            return
        self._multi_selected = [
            cb
            for cb in self._canvas_boxes
            if self._canvas.itemcget(cb.rect_id, "state") != "hidden"
        ]
        self._draw_all_boxes()
        self._update_multi_label()
        self._status.configure(text=f"Selected {len(self._multi_selected)} boxes")

    # ── Merge ──────────────────────────────────────────────────────

    def _on_merge(self) -> None:
        """Merge multi-selected overlapping boxes, or reshape a detection to
        the enclosing bbox of selected word-overlay rectangles."""

        # ── Word-merge path: reshape selected detection to word bbox ──
        if len(self._selected_word_rids) >= 2:
            self._merge_words_into_detection()
            return

        # ── Original detection-merge path ─────────────────────────────
        targets = list(self._multi_selected)
        if self._selected_box and self._selected_box not in targets:
            targets.append(self._selected_box)

        if len(targets) < 2:
            self._status.configure(
                text="Select at least 2 boxes to merge (Shift+Click)"
            )
            return
        if not self._doc_id:
            self._status.configure(text="No document loaded")
            return

        # Determine element type: use the largest box's type, or prompt if mixed
        types_seen = {cb.element_type for cb in targets}
        if len(types_seen) == 1:
            merged_type = types_seen.pop()
        else:
            # Find largest by area
            largest = max(
                targets,
                key=lambda cb: (
                    (cb.pdf_bbox[2] - cb.pdf_bbox[0])
                    * (cb.pdf_bbox[3] - cb.pdf_bbox[1])
                ),
            )
            merged_type = largest.element_type
            # Let user override
            answer = simpledialog.askstring(
                "Merge Type",
                f"Boxes have mixed types: {', '.join(sorted(types_seen))}.\n"
                f"Enter the merged element type (default: {merged_type}):",
                initialvalue=merged_type,
                parent=self.root,
            )
            if answer is None:
                return  # cancelled
            merged_type = answer.strip() or merged_type

        # Compute merged polygon
        bboxes = [cb.pdf_bbox for cb in targets]
        merged_poly = merge_boxes(bboxes)
        merged_bbox = polygon_bbox(merged_poly)

        # Re-extract text using the polygon boundary for precision
        if self._pdf_path:
            merged_text = extract_text_in_polygon(
                self._pdf_path, self._page, merged_poly
            )
        else:
            merged_text = "\n".join(
                cb.text_content for cb in targets if cb.text_content
            )

        # Aggregate features from the largest box
        largest = max(
            targets,
            key=lambda cb: (
                (cb.pdf_bbox[2] - cb.pdf_bbox[0]) * (cb.pdf_bbox[3] - cb.pdf_bbox[1])
            ),
        )
        merged_features = largest.features

        # Pick survivor (the first box) — will be reshaped
        survivor = targets[0]
        consumed = targets[1:]

        # Push undo record for the whole merge
        self._push_undo(
            "merge",
            survivor,
            extra={
                "merged_boxes": [
                    {
                        "detection_id": cb.detection_id,
                        "element_type": cb.element_type,
                        "pdf_bbox": cb.pdf_bbox,
                        "polygon": cb.polygon,
                        "confidence": cb.confidence,
                        "text_content": cb.text_content,
                        "corrected": cb.corrected,
                    }
                    for cb in targets
                ],
            },
        )

        # Persist: delete corrections for consumed boxes
        for cb in consumed:
            self._store.save_correction(
                doc_id=self._doc_id,
                page=self._page,
                correction_type="delete",
                corrected_label=cb.element_type,
                corrected_bbox=cb.pdf_bbox,
                detection_id=cb.detection_id,
                original_label=cb.element_type,
                original_bbox=cb.pdf_bbox,
                session_id=self._session_id,
            )
            # Remove from canvas
            if cb.rect_id:
                self._canvas.delete(cb.rect_id)
            if cb.label_id:
                self._canvas.delete(cb.label_id)
            for hid in cb.handle_ids:
                self._canvas.delete(hid)
            self._canvas_boxes.remove(cb)
            self._session_count += 1

        # Persist: reshape survivor to the merged bbox
        self._store.save_correction(
            doc_id=self._doc_id,
            page=self._page,
            correction_type="reshape",
            corrected_label=merged_type,
            corrected_bbox=merged_bbox,
            detection_id=survivor.detection_id,
            original_label=survivor.element_type,
            original_bbox=survivor.pdf_bbox,
            session_id=self._session_id,
        )

        # Update survivor in-place
        survivor.element_type = merged_type
        survivor.pdf_bbox = merged_bbox
        survivor.polygon = merged_poly
        survivor.text_content = merged_text
        survivor.features = merged_features
        survivor.corrected = True
        survivor.merged_from = [cb.detection_id for cb in targets]
        self._session_count += 1

        # Persist polygon to DB so it roundtrips on reload
        self._store.update_detection_polygon(
            survivor.detection_id, merged_poly, merged_bbox
        )

        # Redraw
        self._selected_box = None
        self._clear_multi_select()
        self._draw_box(survivor)

        self._update_session_label()
        self._update_page_summary()
        self._status.configure(
            text=f"Merged {len(targets)} boxes → {merged_type} (polygon with {len(merged_poly)} vertices)"
        )

    def _merge_words_into_detection(
        self, *, forced_type: str | None = None, force_create: bool = False
    ) -> None:
        """Reshape the currently selected detection to the enclosing bbox
        of the selected word-overlay rectangles, or create a new detection
        if no detection box is selected.  When multiple words are selected
        the result is a *union polygon* (via Shapely) so the outline hugs
        the words tightly — important when nearby elements are close."""
        if not self._doc_id:
            self._status.configure(text="No document loaded")
            return

        # Collect per-word bboxes (PDF coords)
        word_bboxes: list[tuple[float, float, float, float]] = []
        texts: list[str] = []
        for rid in self._selected_word_rids:
            winfo = self._word_overlay_items.get(rid)
            if not winfo:
                continue
            word_bboxes.append(
                (winfo["x0"], winfo["top"], winfo["x1"], winfo["bottom"])
            )
            if winfo.get("text"):
                texts.append(winfo["text"])

        if not word_bboxes:
            self._status.configure(text="No valid words selected")
            return

        # Compute union polygon from the word bboxes
        merged_poly: list[tuple[float, float]] | None = None
        if len(word_bboxes) >= 2:
            try:
                merged_poly = merge_boxes(word_bboxes)
            except Exception:
                merged_poly = None
        new_bbox = (
            polygon_bbox(merged_poly)
            if merged_poly
            else (
                min(b[0] for b in word_bboxes),
                min(b[1] for b in word_bboxes),
                max(b[2] for b in word_bboxes),
                max(b[3] for b in word_bboxes),
            )
        )

        # Re-extract text using the polygon or bbox for precision
        if self._pdf_path and merged_poly:
            merged_text = extract_text_in_polygon(
                self._pdf_path, self._page, merged_poly
            )
        elif self._pdf_path:
            merged_text = extract_text_in_bbox(self._pdf_path, self._page, new_bbox)
        else:
            merged_text = " ".join(texts)

        n_words = len(word_bboxes)

        if self._selected_box and not force_create and forced_type is None:
            # ── Reshape existing detection ───────────────────────
            cbox = self._selected_box
            orig_bbox = cbox.pdf_bbox
            orig_polygon = list(cbox.polygon) if cbox.polygon else None

            self._push_undo(
                "reshape",
                cbox,
                extra={"orig_bbox": orig_bbox, "orig_polygon": orig_polygon},
            )

            self._store.save_correction(
                doc_id=self._doc_id,
                page=self._page,
                correction_type="reshape",
                corrected_label=cbox.element_type,
                corrected_bbox=new_bbox,
                detection_id=cbox.detection_id,
                original_label=cbox.element_type,
                original_bbox=orig_bbox,
                corrected_text=merged_text,
                session_id=self._session_id,
            )

            cbox.pdf_bbox = new_bbox
            cbox.polygon = merged_poly
            cbox.text_content = merged_text
            cbox.corrected = True
            self._session_count += 1

            self._store.update_detection_polygon(
                cbox.detection_id, merged_poly, new_bbox
            )

            self._draw_box(cbox)
            self._clear_word_selection()
            self._update_session_label()
            self._update_page_summary()
            poly_info = f" (polygon {len(merged_poly)} pts)" if merged_poly else ""
            self._status.configure(
                text=f"Reshaped {cbox.element_type} to enclose {n_words} words{poly_info}"
            )
        else:
            # ── Create new detection from words ──────────────────
            # Auto-label via classifier if possible, unless user forced a type
            features = featurize_region("misc_title", new_bbox, None, 2448.0, 1584.0)
            chosen_type = forced_type or (self._type_var.get() or "misc_title")
            if forced_type is None and features and self._classifier.model_exists():
                try:
                    pred_label, pred_conf = self._classifier.predict(features)
                    if pred_conf and pred_conf > 0.5:
                        chosen_type = pred_label
                except Exception:
                    pass

            det_id = self._store.save_detection(
                doc_id=self._doc_id,
                page=self._page,
                run_id=self._run_id or "manual",
                element_type=chosen_type,
                bbox=new_bbox,
                text_content=merged_text,
                features=features,
            )
            self._store.save_correction(
                doc_id=self._doc_id,
                page=self._page,
                correction_type="add",
                corrected_label=chosen_type,
                corrected_bbox=new_bbox,
                detection_id=det_id,
                session_id=self._session_id,
            )

            cbox = CanvasBox(
                detection_id=det_id,
                element_type=chosen_type,
                confidence=None,
                text_content=merged_text,
                features=features,
                pdf_bbox=new_bbox,
                polygon=merged_poly,
                corrected=True,
            )
            # Persist polygon so it roundtrips on reload
            if merged_poly:
                self._store.update_detection_polygon(det_id, merged_poly, new_bbox)
            self._canvas_boxes.append(cbox)
            self._draw_box(cbox)
            self._select_box(cbox)
            self._session_count += 1
            self._update_session_label()
            self._update_page_summary()
            self._clear_word_selection()
            poly_info = f" (polygon {len(merged_poly)} pts)" if merged_poly else ""
            self._status.configure(
                text=f"Created {chosen_type} from {n_words} words{poly_info}"
            )

    def _key_merge(self, event: tk.Event) -> None:
        """Keyboard shortcut M for merge."""
        self._on_merge()

    def _key_group(self, event: tk.Event) -> None:
        """Keyboard shortcut G for group creation.  Works with a selected
        detection box *or* with selected word boxes (creates a detection first)."""
        if isinstance(event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)):
            return
        if not self._is_active_tab():
            return

        # If words are selected, create a detection from them first
        if len(self._selected_word_rids) >= 2 and not self._selected_box:
            self._merge_words_into_detection()
            # _merge_words_into_detection selects the new box if created

        if not self._selected_box:
            self._status.configure(text="Select a box or Alt-click words first")
            return
        if self._selected_box.group_id:
            grp = self._groups.get(self._selected_box.group_id, {})
            label = grp.get("label", "?")
            self._status.configure(text=f"Already in group \u2039{label}\u203a")
        else:
            self._create_group(self._selected_box)

    def _key_link_column(self, event: tk.Event) -> None:
        """Keyboard shortcut L — create a notes_column from selected boxes."""
        if isinstance(event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)):
            return
        if not self._is_active_tab():
            return
        self._on_link_column()

    def _on_link_column(self) -> None:
        """Create a notes_column detection that encloses the selected
        header / notes_block boxes, then group them under it."""
        if not self._doc_id:
            return

        # ── Collect targets ────────────────────────────────────────
        targets = list(self._multi_selected)
        if self._selected_box and self._selected_box not in targets:
            targets.append(self._selected_box)

        if len(targets) < 2:
            self._status.configure(
                text="Shift+click ≥2 headers / notes blocks, then press L"
            )
            return

        # ── Validate types ─────────────────────────────────────────
        linkable_types = {"header", "notes_block"}
        non_linkable = [cb for cb in targets if cb.element_type not in linkable_types]
        if non_linkable:
            bad = ", ".join(sorted({cb.element_type for cb in non_linkable}))
            messagebox.showwarning(
                "Invalid Selection",
                f"Only header and notes_block boxes can be linked "
                f"into a notes column.\n\nFound: {bad}",
            )
            return

        # ── Reject boxes already in a group ────────────────────────
        already_grouped = [cb for cb in targets if cb.group_id]
        if already_grouped:
            self._status.configure(
                text=f"{len(already_grouped)} box(es) already in a group"
            )
            return

        # ── Compute tight enclosing bbox ───────────────────────────
        x0 = min(cb.pdf_bbox[0] for cb in targets)
        y0 = min(cb.pdf_bbox[1] for cb in targets)
        x1 = max(cb.pdf_bbox[2] for cb in targets)
        y1 = max(cb.pdf_bbox[3] for cb in targets)
        col_bbox = (x0, y0, x1, y1)

        # ── Extract text from the region ───────────────────────────
        text_content = ""
        if self._pdf_path:
            text_content = extract_text_in_bbox(self._pdf_path, self._page, col_bbox)

        # ── Create the notes_column detection ──────────────────────
        features = featurize_region("notes_column", col_bbox, None, 2448.0, 1584.0)
        det_id = self._store.save_detection(
            doc_id=self._doc_id,
            page=self._page,
            run_id=self._run_id or "manual",
            element_type="notes_column",
            bbox=col_bbox,
            text_content=text_content,
            features=features,
        )
        self._store.save_correction(
            doc_id=self._doc_id,
            page=self._page,
            correction_type="add",
            corrected_label="notes_column",
            corrected_bbox=col_bbox,
            detection_id=det_id,
            session_id=self._session_id,
        )

        col_box = CanvasBox(
            detection_id=det_id,
            element_type="notes_column",
            confidence=None,
            text_content=text_content,
            features=features,
            pdf_bbox=col_bbox,
            corrected=True,
        )
        self._canvas_boxes.append(col_box)
        self._draw_box(col_box)

        # Push notes_column rect behind children so they stay clickable
        if col_box.rect_id:
            self._canvas.tag_lower(col_box.rect_id, "det_box")

        # ── Group: notes_column = root, children = members ─────────
        group_id = self._store.create_group(
            doc_id=self._doc_id,
            page=self._page,
            group_label="notes_column",
            root_detection_id=det_id,
        )
        col_box.group_id = group_id
        col_box.is_group_root = True
        self._groups[group_id] = {
            "label": "notes_column",
            "root_detection_id": det_id,
            "members": [det_id],
        }

        # Sort children top-to-bottom
        targets.sort(key=lambda cb: cb.pdf_bbox[1])
        grp = self._groups[group_id]
        for i, cb in enumerate(targets, start=1):
            self._store.add_to_group(group_id, cb.detection_id, sort_order=i)
            cb.group_id = group_id
            cb.is_group_root = False
            grp["members"].append(cb.detection_id)
            self._draw_box(cb)

        self._draw_group_links(group_id)

        # ── Bookkeeping ────────────────────────────────────────────
        self._session_count += 1
        self._update_session_label()
        self._update_page_summary()
        self._clear_multi_select()
        self._select_box(col_box)
        n_children = len(targets)
        self._status.configure(text=f"Created notes_column from {n_children} boxes")

    def _key_toggle_words(self, event: tk.Event) -> None:
        """Keyboard shortcut W for toggling word overlay."""
        if isinstance(event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)):
            return
        if not self._is_active_tab():
            return
        self._word_overlay_var.set(not self._word_overlay_var.get())
        self._toggle_word_overlay()

    # ── Lasso selection ────────────────────────────────────────────

    def _finalize_lasso(self, cx: float, cy: float) -> None:
        """Select all boxes intersecting the lasso rectangle."""
        if self._lasso_rect_id:
            self._canvas.delete(self._lasso_rect_id)
            self._lasso_rect_id = None

        if not self._lasso_start:
            return

        sx, sy = self._lasso_start
        self._lasso_start = None

        eff = self._effective_scale()
        lx0 = min(sx, cx) / eff
        ly0 = min(sy, cy) / eff
        lx1 = max(sx, cx) / eff
        ly1 = max(sy, cy) / eff

        # Minimum drag distance to count as lasso
        if (lx1 - lx0) < 5 and (ly1 - ly0) < 5:
            return

        for cb in self._canvas_boxes:
            if self._canvas.itemcget(cb.rect_id, "state") == "hidden":
                continue
            bx0, by0, bx1, by1 = cb.pdf_bbox
            # Check intersection
            if bx0 < lx1 and bx1 > lx0 and by0 < ly1 and by1 > ly0:
                if cb not in self._multi_selected:
                    self._multi_selected.append(cb)
                    self._draw_box(cb)

        self._update_multi_label()

        # ── Alt-lasso also selects word overlay boxes ───────────
        word_count = 0
        if self._lasso_word and self._word_overlay_on and self._word_overlay_items:
            for rid, winfo in self._word_overlay_items.items():
                wx0, wy0 = winfo["x0"], winfo["top"]
                wx1, wy1 = winfo["x1"], winfo["bottom"]
                if wx0 < lx1 and wx1 > lx0 and wy0 < ly1 and wy1 > ly0:
                    if rid not in self._selected_word_rids:
                        self._set_word_selected(rid, True)
                    word_count += 1

        self._lasso_word = False

        parts = []
        if self._multi_selected:
            parts.append(f"{len(self._multi_selected)} boxes")
        if word_count or self._selected_word_rids:
            parts.append(f"{len(self._selected_word_rids)} words")
        self._status.configure(
            text=f"Lasso selected {', '.join(parts) if parts else '0 items'}"
        )

    # ── Filters ────────────────────────────────────────────────────

    def _apply_filters(self) -> None:
        """Show/hide boxes based on the current filter settings."""
        min_conf = self._filter_conf_min.get()
        uncorrected_only = self._filter_uncorrected_only.get()

        visible = 0
        for cb in self._canvas_boxes:
            show = True
            # Label type filter
            var = self._filter_label_vars.get(cb.element_type)
            if var and not var.get():
                show = False
            # Confidence filter
            if show and cb.confidence is not None and cb.confidence < min_conf:
                show = False
            # Uncorrected only
            if show and uncorrected_only and cb.corrected:
                show = False

            state = "normal" if show else "hidden"
            if cb.rect_id:
                self._canvas.itemconfigure(cb.rect_id, state=state)
            if cb.label_id:
                self._canvas.itemconfigure(cb.label_id, state=state)
            for hid in cb.handle_ids:
                self._canvas.itemconfigure(hid, state=state)

            if show:
                visible += 1

        self._status.configure(
            text=f"Showing {visible}/{len(self._canvas_boxes)} detections"
        )

    # ── Model suggestion ──────────────────────────────────────────

    def _apply_suggestion(self) -> None:
        """Apply the model's suggested label to the selected box."""
        if not self._selected_box or not self._model_suggestion or not self._doc_id:
            return

        cbox = self._selected_box
        new_label = self._model_suggestion

        self._push_undo("relabel", cbox, extra={"old_label": cbox.element_type})
        self._store.save_correction(
            doc_id=self._doc_id,
            page=self._page,
            correction_type="relabel",
            corrected_label=new_label,
            corrected_bbox=cbox.pdf_bbox,
            detection_id=cbox.detection_id,
            original_label=cbox.element_type,
            original_bbox=cbox.pdf_bbox,
            session_id=self._session_id,
        )
        old_label = cbox.element_type
        cbox.element_type = new_label
        cbox.corrected = True
        self._session_count += 1
        self._update_session_label()
        self._draw_box(cbox)

        self._model_suggestion = None
        self._suggest_label.configure(text="")
        self._suggest_btn.pack_forget()
        self._type_var.set(new_label)
        self._status.configure(text=f"Applied suggestion: {old_label} → {new_label}")

    # ── Page navigation ───────────────────────────────────────────

    def _on_prev_page(self) -> None:
        """Navigate to the previous page."""
        cur = self._page_var.get()
        if cur > 0:
            self._page_var.set(cur - 1)
            self._navigate_to_page()

    def _on_next_page(self) -> None:
        """Navigate to the next page."""
        upper = self._page_count - 1 if self._page_count > 0 else 0
        cur = self._page_var.get()
        if cur < upper:
            self._page_var.set(cur + 1)
            self._navigate_to_page()

    def _on_page_spin_enter(self, _event: Any = None) -> None:
        """Handle Enter key or manual spinbox changes."""
        self._navigate_to_page()

    def _navigate_to_page(self) -> None:
        """Render the current page preview and load any existing detections.

        Shows a clear status message indicating whether the pipeline
        has been run for this page.
        """
        if not self._pdf_path:
            return

        self._page = self._page_var.get()
        # Clamp to valid range
        if self._page_count > 0 and self._page >= self._page_count:
            self._page = self._page_count - 1
            self._page_var.set(self._page)
        if self._page < 0:
            self._page = 0
            self._page_var.set(0)

        self._resolution = self._dpi_var.get()
        self._scale = self._resolution / 72.0

        self._status.configure(text=f"Loading page {self._page}…")
        self.root.update_idletasks()

        # Render page preview
        try:
            from plancheck.ingest.ingest import render_page_image

            self._bg_image = render_page_image(
                self._pdf_path, self._page, resolution=self._resolution
            )
            self._render_background()
        except Exception as exc:
            self._status.configure(text=f"Error rendering page: {exc}")
            return

        # Load detections from latest pipeline run only
        self._doc_id = self._store.register_document(self._pdf_path)
        dets = self._store.get_latest_detections_for_page(self._doc_id, self._page)
        self._canvas_boxes.clear()
        self._selected_box = None
        self._multi_selected.clear()

        for d in dets:
            self._canvas_boxes.append(
                CanvasBox(
                    detection_id=d["detection_id"],
                    element_type=d["element_type"],
                    confidence=d["confidence"],
                    text_content=d["text_content"],
                    features=d["features"],
                    pdf_bbox=d["bbox"],
                    polygon=d.get("polygon"),
                )
            )

        # Deduplicate overlapping same-type boxes
        self._deduplicate_boxes()

        # Load any saved groups for this page
        self._load_groups_for_page()

        self._draw_all_boxes()
        n = len(self._canvas_boxes)
        page_label = f"Page {self._page}"
        if self._page_count > 0:
            page_label += f" of {self._page_count}"
        if n > 0:
            self._status.configure(text=f"{page_label} — {n} detections")
        else:
            self._status.configure(text=f"{page_label} — ready for annotation")
        self._update_page_summary()

    # ── Page element summary ───────────────────────────────────────

    def _update_page_summary(self) -> None:
        """Refresh the per-page element type summary in the sidebar."""
        if not self._canvas_boxes:
            self._page_elements_label.configure(
                text="(no detections)", foreground="gray"
            )
            return
        counts = Counter(cb.element_type for cb in self._canvas_boxes)
        total = sum(counts.values())
        lines = [f"Total: {total}"]
        for etype, n in counts.most_common():
            color = self.LABEL_COLORS.get(etype, "#888888")
            lines.append(f"  {etype}: {n}")
        self._page_elements_label.configure(
            text="\n".join(lines), foreground="#222222"
        )

    # ── Model training ─────────────────────────────────────────────

    def _on_train_model(self) -> None:
        """Train the classifier in a background thread."""
        self._model_status_label.configure(text="Training…", foreground="orange")
        self.root.update_idletasks()

        # Cancel any previous training run
        self._train_cancel_event.clear()
        self._train_gen += 1
        my_gen = self._train_gen

        def _train():
            try:
                if (
                    self._train_cancel_event.is_set()
                    or self._closing
                    or my_gen != self._train_gen
                ):
                    return
                from plancheck.corrections.store import CorrectionStore as _CS

                store = _CS()
                n_examples = store.build_training_set()
                if n_examples < 10:
                    self._safe_after(
                        0,
                        lambda: self._model_status_label.configure(
                            text=f"Need more data ({n_examples} examples)",
                            foreground="red",
                        ),
                    )
                    return

                import tempfile

                tmp = Path(tempfile.mkdtemp()) / "train.jsonl"
                store.export_training_jsonl(tmp)

                if (
                    self._train_cancel_event.is_set()
                    or self._closing
                    or my_gen != self._train_gen
                ):
                    return

                clf = ElementClassifier()
                metrics = clf.train(str(tmp))
                self._last_metrics = metrics

                if (
                    self._train_cancel_event.is_set()
                    or self._closing
                    or my_gen != self._train_gen
                ):
                    return

                # Record training run in the database
                try:
                    store.save_training_run(
                        metrics, model_path=str(clf.model_path), notes="GUI train"
                    )
                except Exception:
                    pass  # non-critical

                acc = metrics.get("accuracy", 0)
                f1m = metrics.get("f1_macro", 0)
                self._safe_after(
                    0,
                    lambda: self._model_status_label.configure(
                        text=f"Model trained — acc {acc:.1%}  F1 {f1m:.1%}",
                        foreground="green",
                    ),
                )
                # Reload classifier
                self._classifier = ElementClassifier()
            except Exception as exc:
                self._safe_after(
                    0,
                    lambda: self._model_status_label.configure(
                        text=f"Train failed: {exc}",
                        foreground="red",
                    ),
                )

        threading.Thread(target=_train, daemon=True).start()

    def _safe_after(self, delay_ms: int, callback) -> None:
        """Schedule callback on the UI thread if the window still exists."""
        if getattr(self, "_closing", False):
            return
        try:
            if hasattr(self.root, "winfo_exists") and not self.root.winfo_exists():
                return
        except Exception:
            pass
        try:
            self.root.after(delay_ms, callback)
        except Exception:
            pass

    def _on_show_metrics(self) -> None:
        """Show the last training metrics in a popup."""
        if not self._last_metrics:
            messagebox.showinfo("Metrics", "No metrics available. Train a model first.")
            return

        from plancheck.corrections.metrics import format_metrics_table

        text = format_metrics_table(self._last_metrics)

        win = tk.Toplevel(self.root)
        win.title("Model Metrics")
        win.geometry("500x400")
        win.transient(self.root)

        txt = tk.Text(win, wrap="none", font=("Courier", 9))
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        txt.insert("1.0", text)
        self._make_text_readonly(txt)
        self._add_copy_menu(txt)

    def _on_show_training_history(self) -> None:
        """Show a popup with all past training runs."""
        try:
            history = self._store.get_training_history()
        except Exception:
            history = []

        if not history:
            messagebox.showinfo("Training History", "No training runs recorded yet.")
            return

        lines: list[str] = [
            f"{'Run ID':<14s} {'Date':<22s} {'#Train':>6s} {'#Val':>5s} "
            f"{'Acc':>7s} {'F1mac':>7s} {'F1wt':>7s} {'Notes':<12s}",
            "-" * 88,
        ]
        for r in history:
            ts = r["trained_at"][:19].replace("T", " ")
            lines.append(
                f"{r['run_id']:<14s} {ts:<22s} {r['n_train']:>6d} {r['n_val']:>5d} "
                f"{r['accuracy']:>6.1%} {r['f1_macro']:>6.1%} {r['f1_weighted']:>6.1%} "
                f"{r.get('notes', ''):<12s}"
            )

        win = tk.Toplevel(self.root)
        win.title("Training History")
        win.geometry("700x400")
        win.transient(self.root)

        txt = tk.Text(win, wrap="none", font=("Courier", 9))
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        txt.insert("1.0", "\n".join(lines))
        self._make_text_readonly(txt)
        self._add_copy_menu(txt)

    def _on_show_feature_importance(self) -> None:
        """Show feature importance from the trained model."""
        if not self._classifier.model_exists():
            messagebox.showinfo("Feature Importance", "No model trained yet.")
            return

        importance = self._classifier.get_feature_importance()
        if not importance:
            messagebox.showinfo(
                "Feature Importance", "Could not extract feature importance."
            )
            return

        lines: list[str] = [
            f"{'Feature':<30s} {'Importance':>12s}",
            "-" * 44,
        ]
        for name, imp in importance.items():
            bar = "█" * max(1, int(imp * 200))
            lines.append(f"{name:<30s} {imp:>12.6f}  {bar}")

        win = tk.Toplevel(self.root)
        win.title("Feature Importance")
        win.geometry("600x500")
        win.transient(self.root)

        txt = tk.Text(win, wrap="none", font=("Courier", 9))
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        txt.insert("1.0", "\n".join(lines))
        self._make_text_readonly(txt)
        self._add_copy_menu(txt)

    def _update_model_status(self) -> None:
        """Check if a trained model file exists and update the label."""
        if self._classifier.model_exists():
            self._model_status_label.configure(
                text="Model loaded ✓", foreground="green"
            )
        else:
            self._model_status_label.configure(
                text="No model trained", foreground="gray"
            )

    # ── Annotation stats ──────────────────────────────────────────

    def _refresh_stats(self) -> None:
        """Query the DB for annotation statistics and display them."""
        try:
            cur = self._store._conn.cursor()
            n_docs = cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            n_dets = cur.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
            n_corr = cur.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
            n_train = cur.execute("SELECT COUNT(*) FROM training_examples").fetchone()[
                0
            ]

            # Per-type breakdown
            rows = cur.execute(
                "SELECT element_type, COUNT(*) FROM detections GROUP BY element_type"
            ).fetchall()
            breakdown = "\n".join(f"  {r[0]}: {r[1]}" for r in rows)

            text = (
                f"Docs: {n_docs}  Dets: {n_dets}\n"
                f"Corrections: {n_corr}  Training: {n_train}\n"
                f"{breakdown}"
            )
            self._stats_label.configure(text=text)
        except Exception as exc:
            self._stats_label.configure(text=f"Error: {exc}")

    def _on_clear_old_runs(self) -> None:
        """Remove detection data from old pipeline runs, keeping only the latest."""
        ok = messagebox.askyesno(
            "Clear Old Run Data",
            "This will remove detection data from old pipeline runs.\n"
            "Corrections and ML training data will be preserved.\n\n"
            "Continue?",
            parent=self.root,
        )
        if not ok:
            return
        try:
            n = self._store.purge_all_stale_detections()
            self._refresh_stats()
            if self._pdf_path:
                self._navigate_to_page()
            self._status.configure(text=f"Purged {n} old detection(s)")
        except Exception as exc:
            messagebox.showerror("Error", str(exc), parent=self.root)

    # ── Active learning ────────────────────────────────────────────

    def _on_suggest_next(self) -> None:
        """Use active learning to suggest the next page to annotate."""
        if not self._classifier.model_exists():
            messagebox.showinfo(
                "No Model",
                "Train a model first to enable active learning suggestions.",
            )
            return

        try:
            from plancheck.corrections.active_learning import suggest_next_page

            result = suggest_next_page(self._store, self._classifier._model_path)
            if result is None:
                messagebox.showinfo(
                    "Active Learning",
                    "No unannotated pages with high uncertainty found.",
                )
                return

            doc_id, page = result
            # Look up PDF path from doc_id
            row = self._store._conn.execute(
                "SELECT pdf_path, filename FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if row:
                pdf_path_str = row["pdf_path"] or row["filename"]
                pdf_path = Path(pdf_path_str)
                if not pdf_path.exists():
                    # Try filename in current directory as fallback
                    pdf_path = Path(row["filename"])
                if pdf_path.exists():
                    self.state.set_pdf(pdf_path)
                    self._page_var.set(page)
                    self._navigate_to_page()
                    self._status.configure(
                        text=f"Active learning: page {page} (highest uncertainty)"
                    )
                else:
                    messagebox.showwarning(
                        "File not found",
                        f"PDF not found: {pdf_path_str}",
                    )
            else:
                messagebox.showwarning("Not Found", f"Document {doc_id} not in DB.")
        except Exception as exc:
            messagebox.showerror("Active Learning Error", str(exc))

    # ── Snapshots ──────────────────────────────────────────────────

    def _on_snapshot(self) -> None:
        """Create a snapshot of the corrections database."""
        tag = simpledialog.askstring(
            "Snapshot",
            "Enter a tag for this snapshot (optional):",
            parent=self.root,
        )
        if tag is None:
            return  # cancelled
        tag = tag.strip() or "manual"
        try:
            path = self._store.snapshot(tag)
            self._status.configure(text=f"Snapshot saved: {path.name}")
        except Exception as exc:
            messagebox.showerror("Snapshot Error", str(exc))

    def _on_restore_snapshot(self) -> None:
        """Show a list of snapshots and restore the chosen one."""
        snaps = self._store.list_snapshots()
        if not snaps:
            messagebox.showinfo("No Snapshots", "No snapshots available.")
            return

        # Build a selection dialog
        win = tk.Toplevel(self.root)
        win.title("Restore Snapshot")
        win.geometry("420x300")
        win.transient(self.root)
        win.grab_set()

        ttk.Label(win, text="Choose a snapshot to restore:").pack(padx=10, pady=(10, 4))
        listbox = tk.Listbox(win, width=55, height=10)
        listbox.pack(padx=10, fill="both", expand=True)

        for s in snaps:
            ts = s.get("timestamp", "?")
            tag = s.get("tag", "")
            size = s.get("size_kb", 0)
            listbox.insert("end", f"{ts}  [{tag}]  ({size:.0f} KB)")

        def on_restore():
            sel = listbox.curselection()
            if not sel:
                return
            idx = sel[0]
            snap_path = snaps[idx]["path"]
            if messagebox.askyesno(
                "Restore",
                "This will overwrite the current database. Continue?",
                parent=win,
            ):
                try:
                    self._store.restore_snapshot(snap_path)
                    self._status.configure(text="Snapshot restored")
                    win.destroy()
                    # Reload current view
                    if self._pdf_path:
                        self._navigate_to_page()
                except Exception as exc:
                    messagebox.showerror("Restore Error", str(exc), parent=win)

        ttk.Button(win, text="Restore", command=on_restore).pack(padx=10, pady=8)
