"""Tab 6 – Annotation: interactive detection correction UI.

Renders the PDF page with clickable, editable detection boxes.
Every correction is persisted to ``CorrectionStore`` immediately.
"""

from __future__ import annotations

import copy
import threading
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any, List, Optional
from uuid import uuid4

_project = Path(__file__).resolve().parent.parent.parent

from PIL import Image, ImageTk
from widgets import LogPanel, StatusBar
from worker import PipelineWorker

from plancheck.analysis.box_merge import merge_boxes, polygon_bbox
from plancheck.config import GroupingConfig
from plancheck.corrections.classifier import ElementClassifier
from plancheck.corrections.features import featurize, featurize_region
from plancheck.corrections.store import CorrectionStore
from plancheck.ingest.ingest import extract_text_in_bbox, extract_text_in_polygon

# ── Geometry helpers ───────────────────────────────────────────────────


def _point_in_polygon(px: float, py: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test.

    *polygon* is a list of (x, y) vertices forming a closed ring
    (first == last vertex).  Returns True if (px, py) lies inside.
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


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
    click-to-select, relabel, reshape, add, and delete.  Every action
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
        notebook.add(self.frame, text="Annotation")

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
        self._load_mode: str = "pipeline"  # or "database"
        self._draw_start: tuple[float, float] | None = None
        self._draw_rect_id: int | None = None
        self._lasso_start: tuple[float, float] | None = None
        self._lasso_rect_id: int | None = None
        self._session_id: str = uuid4().hex[:8]
        self._session_count: int = 0
        self._store = CorrectionStore()
        self._worker: PipelineWorker | None = None
        self._classifier = ElementClassifier()

        # Track which pages have had the pipeline run
        self._pipeline_pages: set[int] = set()

        # Undo / redo stacks
        self._undo_stack: list[dict] = []
        self._redo_stack: list[dict] = []

        # Drag-handle state
        self._drag_handle: str | None = None
        self._drag_orig_bbox: tuple[float, float, float, float] | None = None

        # Move-drag state (click inside already-selected box to drag)
        self._move_dragging: bool = False
        self._move_start_pdf: tuple[float, float] | None = None
        self._move_orig_bbox: tuple[float, float, float, float] | None = None
        self._move_orig_polygon: list[tuple[float, float]] | None = None

        # Pan state
        self._pan_start: tuple[int, int] | None = None

        # Clipboard for box copy/paste
        self._copied_box_template: dict | None = None

        # Box groups (WBS hierarchy)
        self._groups: dict[str, dict] = {}
        self._group_link_ids: list[int] = []

        # Word overlay state
        self._word_overlay_on: bool = False
        self._word_overlay_ids: list[int] = []

        # Filter state
        self._filter_label_vars: dict[str, tk.BooleanVar] = {}
        self._filter_conf_min: tk.DoubleVar = tk.DoubleVar(value=0.0)
        self._filter_uncorrected_only: tk.BooleanVar = tk.BooleanVar(value=False)

        # Model metrics cache
        self._last_metrics: dict | None = None

        self._build_ui()

        # Subscribe to GuiState events
        self.state.subscribe("pdf_changed", self._on_pdf_changed)

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

        _btn_run = ttk.Button(top, text="Run Pipeline", command=self._on_run_pipeline)
        _btn_run.pack(side="left", padx=2)
        self._tooltip(_btn_run, "Run the detection pipeline on the current page")
        _btn_run_all = ttk.Button(
            top, text="Run All Pages", command=self._on_run_all_pages
        )
        _btn_run_all.pack(side="left", padx=2)
        self._tooltip(_btn_run_all, "Run the pipeline on every page of the PDF")
        _btn_load = ttk.Button(
            top, text="Load Detections", command=self._on_load_detections
        )
        _btn_load.pack(side="left", padx=2)
        self._tooltip(_btn_load, "Load previously saved detections from the database")

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
        def _insp_enter(event: tk.Event) -> None:
            self._insp_canvas.bind_all(
                "<MouseWheel>",
                lambda ev: self._insp_canvas.yview_scroll(
                    int(-1 * (ev.delta / 120)), "units"
                ),
            )

        def _insp_leave(event: tk.Event) -> None:
            self._insp_canvas.unbind_all("<MouseWheel>")

        self._insp_canvas.bind("<Enter>", _insp_enter)
        self._insp_canvas.bind("<Leave>", _insp_leave)

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
            text_frame, width=24, height=4, wrap="word", state="disabled"
        )
        self._insp_text.grid(row=0, column=0, sticky="ew")
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
        _btn_delete = ttk.Button(btn_frame, text="Delete ✗", command=self._on_delete)
        _btn_delete.pack(side="left", padx=3)
        self._tooltip(_btn_delete, "Mark this detection as a false positive (D)")

        # ── Batch buttons ─────────────────────────────────────────
        row += 1
        batch_frame = ttk.Frame(inspector)
        batch_frame.grid(row=row, column=0, columnspan=2, pady=2)
        _btn_batch_acc = ttk.Button(
            batch_frame, text="Batch Accept", command=self._on_batch_accept
        )
        _btn_batch_acc.pack(side="left", padx=3)
        self._tooltip(_btn_batch_acc, "Accept all selected boxes at once")
        _btn_batch_rel = ttk.Button(
            batch_frame, text="Batch Relabel", command=self._on_batch_relabel
        )
        _btn_batch_rel.pack(side="left", padx=3)
        self._tooltip(_btn_batch_rel, "Relabel all selected boxes to the chosen type")
        _btn_merge = ttk.Button(batch_frame, text="Merge ⊞", command=self._on_merge)
        _btn_merge.pack(side="left", padx=3)
        self._tooltip(_btn_merge, "Merge selected boxes into one polygon (M)")
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
        filter_frame = ttk.Frame(inspector)
        filter_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=8)
        for i, etype in enumerate(self.ELEMENT_TYPES):
            var = tk.BooleanVar(value=True)
            self._filter_label_vars[etype] = var
            color = self.LABEL_COLORS.get(etype, "#888")
            cb = ttk.Checkbutton(
                filter_frame,
                text=etype,
                variable=var,
                command=self._apply_filters,
            )
            cb.grid(row=i // 2, column=i % 2, sticky="w")

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

        row += 1
        _btn_save = ttk.Button(
            inspector, text="Save Session", command=self._on_save_session
        )
        _btn_save.grid(row=row, column=0, columnspan=2, padx=4, pady=4)
        self._tooltip(_btn_save, "Persist all session corrections to the database")

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
        _btn_stats.grid(row=row, column=0, columnspan=2, padx=4, pady=2)
        self._tooltip(_btn_stats, "Recalculate annotation statistics from the database")

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
            "Shortcuts: A=Accept  D=Delete\n"
            "R=Relabel  M=Merge\n"
            "Esc=Deselect  ←→ Cycle\n"
            "F=Fit  +/- Zoom  Scroll=Pan\n"
            "Ctrl+Z/Y Undo/Redo\n"
            "Shift+Click Multi-select\n"
            "Ctrl+A Select all\n"
            "Ctrl+C Copy box  Ctrl+V Paste\n"
            "G=Group  Right-click: menu"
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
        self.root.bind("<Key-w>", self._key_toggle_words)

        # Initialize model status
        self._update_model_status()

    # ── PDF selection ──────────────────────────────────────────────

    def _browse_pdf(self) -> None:
        f = filedialog.askopenfilename(
            title="Select PDF",
            filetypes=[("PDF", "*.pdf"), ("All", "*.*")],
            initialdir=str(_project / "input"),
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
            self._pipeline_pages.clear()
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

    # ── Mode ───────────────────────────────────────────────────────

    def _on_mode_change(self) -> None:
        self._mode = self._mode_var.get()
        if self._mode == "add":
            self._canvas.config(cursor="crosshair")
            self._deselect()
        else:
            self._canvas.config(cursor="")

    # ── Dynamic element types ──────────────────────────────────────

    def _register_element_type(self, name: str) -> None:
        """Register a new element type with auto-assigned color.

        Updates LABEL_COLORS, ELEMENT_TYPES, the type combo boxes,
        and the filter checkboxes.
        """
        name = name.strip().lower().replace(" ", "_")
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
        idx = len(self.LABEL_COLORS) % len(_palette)
        color = _palette[idx]

        self.LABEL_COLORS[name] = color
        if name not in self.ELEMENT_TYPES:
            self.ELEMENT_TYPES.append(name)

        # Update combo boxes
        self._type_combo.configure(values=self.ELEMENT_TYPES)

        # Add filter checkbox
        if name not in self._filter_label_vars:
            var = tk.BooleanVar(value=True)
            self._filter_label_vars[name] = var
            # Find the filter frame and add to it
            for child in self.frame.winfo_children():
                for sub in child.winfo_children():
                    if isinstance(sub, ttk.Frame):
                        for w in sub.winfo_children():
                            if isinstance(w, ttk.Checkbutton):
                                parent = sub
                                i = len(
                                    [
                                        c
                                        for c in parent.winfo_children()
                                        if isinstance(c, ttk.Checkbutton)
                                    ]
                                )
                                cb = ttk.Checkbutton(
                                    parent,
                                    text=name,
                                    variable=var,
                                    command=self._apply_filters,
                                )
                                cb.grid(row=i // 2, column=i % 2, sticky="w")
                                return

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

    # ── Pipeline / load ────────────────────────────────────────────

    def _on_run_pipeline(self) -> None:
        """Run the pipeline and load detections onto the canvas."""
        if not self._pdf_path:
            messagebox.showwarning("No PDF", "Select a PDF file first.")
            return

        self._page = self._page_var.get()
        self._resolution = self._dpi_var.get()
        self._scale = self._resolution / 72.0
        self._status.configure(text="Running pipeline…")
        self._progress_var.set(0)
        self._progress.grid()
        self.root.update_idletasks()

        try:
            from plancheck.pipeline import run_pipeline

            cfg = self.state.config if self.state.config else GroupingConfig()
            pr = run_pipeline(
                self._pdf_path, self._page, cfg, resolution=self._resolution
            )
        except Exception as exc:
            messagebox.showerror("Pipeline Error", str(exc))
            self._status.configure(text="Pipeline failed")
            self._progress.grid_remove()
            return

        # Register document
        self._doc_id = self._store.register_document(self._pdf_path)
        self._run_id = f"run_{uuid4().hex[:12]}"

        # Set background image from pipeline result
        self._bg_image = pr.background_image
        self._render_background()

        # Save detections
        self._canvas_boxes.clear()

        # Notes-column detections (one box per column: header + all blocks)
        for nc in pr.notes_columns:
            bbox = nc.bbox()
            if bbox == (0, 0, 0, 0):
                continue
            header_block = nc.header
            features = featurize_region(
                "notes_column",
                bbox,
                header_block,
                pr.page_width,
                pr.page_height,
                entry_count=len(nc.notes_blocks),
            )
            try:
                text_content = nc.header_text()
            except Exception:
                text_content = ""
            det_id = self._store.save_detection(
                doc_id=self._doc_id,
                page=self._page,
                run_id=self._run_id,
                element_type="notes_column",
                bbox=bbox,
                text_content=text_content or "",
                features=features,
                confidence=None,
            )
            self._canvas_boxes.append(
                CanvasBox(
                    detection_id=det_id,
                    element_type="notes_column",
                    confidence=None,
                    text_content=text_content or "",
                    features=features,
                    pdf_bbox=bbox,
                )
            )

        # Block-level detections
        for block in pr.blocks:
            lbl = getattr(block, "label", None)
            if lbl in ("note_column_header", "note_column_subheader"):
                etype = "header"
            elif lbl == "notes_block" or getattr(block, "is_notes", False):
                etype = "notes_block"
            elif getattr(block, "is_header", False):
                etype = "header"
            else:
                continue  # skip unclassified blocks

            features = featurize(block, pr.page_width, pr.page_height)
            text = " ".join(b.text for b in block.get_all_boxes())[:500]
            bbox = block.bbox()
            if bbox == (0, 0, 0, 0):
                continue

            det_id = self._store.save_detection(
                doc_id=self._doc_id,
                page=self._page,
                run_id=self._run_id,
                element_type=etype,
                bbox=bbox,
                text_content=text,
                features=features,
                confidence=None,
            )
            self._canvas_boxes.append(
                CanvasBox(
                    detection_id=det_id,
                    element_type=etype,
                    confidence=None,
                    text_content=text,
                    features=features,
                    pdf_bbox=bbox,
                )
            )

        # Region-level detections
        region_lists: list[tuple[list, str]] = [
            (pr.abbreviation_regions, "abbreviations"),
            (pr.legend_regions, "legend"),
            (pr.revision_regions, "revision"),
            (pr.standard_detail_regions, "standard_detail"),
            (pr.misc_title_regions, "misc_title"),
        ]
        for region_list, etype in region_lists:
            for region in region_list:
                bbox = region.bbox()
                if bbox == (0, 0, 0, 0):
                    continue

                header_block = getattr(region, "header", None)
                entry_count = len(getattr(region, "entries", []))
                features = featurize_region(
                    etype,
                    bbox,
                    header_block,
                    pr.page_width,
                    pr.page_height,
                    entry_count=entry_count,
                )

                # Safe text extraction
                try:
                    text_content = region.header_text()
                except Exception:
                    text_content = getattr(region, "text", "")

                confidence = getattr(region, "confidence", None)

                det_id = self._store.save_detection(
                    doc_id=self._doc_id,
                    page=self._page,
                    run_id=self._run_id,
                    element_type=etype,
                    bbox=bbox,
                    text_content=text_content or "",
                    features=features,
                    confidence=confidence,
                )
                self._canvas_boxes.append(
                    CanvasBox(
                        detection_id=det_id,
                        element_type=etype,
                        confidence=confidence,
                        text_content=text_content or "",
                        features=features,
                        pdf_bbox=bbox,
                    )
                )

        # Title blocks (bbox is a field, not a method)
        for tb in pr.title_blocks:
            bbox = tb.bbox  # TitleBlockInfo.bbox is a tuple field
            if bbox == (0, 0, 0, 0):
                continue
            features = featurize_region(
                "title_block",
                bbox,
                None,
                pr.page_width,
                pr.page_height,
            )
            det_id = self._store.save_detection(
                doc_id=self._doc_id,
                page=self._page,
                run_id=self._run_id,
                element_type="title_block",
                bbox=bbox,
                text_content=tb.raw_text[:500] if tb.raw_text else "",
                features=features,
                confidence=tb.confidence,
            )
            self._canvas_boxes.append(
                CanvasBox(
                    detection_id=det_id,
                    element_type="title_block",
                    confidence=tb.confidence,
                    text_content=tb.raw_text[:500] if tb.raw_text else "",
                    features=features,
                    pdf_bbox=bbox,
                )
            )

        # Apply prior corrections + ML feedback
        self._apply_prior_corrections()

        # Mark this page as processed
        self._pipeline_pages.add(self._page)

        # Load any saved groups for this page
        self._load_groups_for_page()

        # Draw all boxes
        self._draw_all_boxes()
        n = len(self._canvas_boxes)
        corrected = sum(1 for cb in self._canvas_boxes if cb.corrected)
        page_label = f"page {self._page}"
        if self._page_count > 0:
            page_label += f" of {self._page_count}"
        status = f"Pipeline: {n} detections on {page_label}"
        if corrected:
            status += f" ({corrected} with prior corrections applied)"
        self._status.configure(text=status)
        self._progress_var.set(100)
        self._progress.grid_remove()

    def _on_run_all_pages(self) -> None:
        """Run the pipeline on every page of the current PDF."""
        if not self._pdf_path:
            messagebox.showwarning("No PDF", "Select a PDF file first.")
            return
        if self._page_count <= 0:
            messagebox.showwarning("No Pages", "Could not determine page count.")
            return

        total_detections = 0
        self._resolution = self._dpi_var.get()
        self._scale = self._resolution / 72.0
        self._progress_var.set(0)
        self._progress.grid()

        for pg in range(self._page_count):
            pct = (pg / self._page_count) * 100
            self._progress_var.set(pct)
            self._status.configure(
                text=f"Running pipeline on page {pg + 1} of {self._page_count}…"
            )
            self.root.update_idletasks()

            try:
                from plancheck.pipeline import run_pipeline

                cfg = self.state.config if self.state.config else GroupingConfig()
                pr = run_pipeline(self._pdf_path, pg, cfg, resolution=self._resolution)
            except Exception as exc:
                self._status.configure(text=f"Pipeline failed on page {pg}: {exc}")
                continue

            self._doc_id = self._store.register_document(self._pdf_path)
            run_id = f"run_{uuid4().hex[:12]}"

            page_count = 0

            # Notes-column detections (one box per column)
            for nc in pr.notes_columns:
                bbox = nc.bbox()
                if bbox == (0, 0, 0, 0):
                    continue
                header_block = nc.header
                features = featurize_region(
                    "notes_column",
                    bbox,
                    header_block,
                    pr.page_width,
                    pr.page_height,
                    entry_count=len(nc.notes_blocks),
                )
                try:
                    text_content = nc.header_text()
                except Exception:
                    text_content = ""
                self._store.save_detection(
                    doc_id=self._doc_id,
                    page=pg,
                    run_id=run_id,
                    element_type="notes_column",
                    bbox=bbox,
                    text_content=text_content or "",
                    features=features,
                    confidence=None,
                )
                page_count += 1

            # Block-level detections
            for block in pr.blocks:
                lbl = getattr(block, "label", None)
                if lbl in ("note_column_header", "note_column_subheader"):
                    etype = "header"
                elif lbl == "notes_block" or getattr(block, "is_notes", False):
                    etype = "notes_block"
                elif getattr(block, "is_header", False):
                    etype = "header"
                else:
                    continue

                features = featurize(block, pr.page_width, pr.page_height)
                text = " ".join(b.text for b in block.get_all_boxes())[:500]
                bbox = block.bbox()
                if bbox == (0, 0, 0, 0):
                    continue

                self._store.save_detection(
                    doc_id=self._doc_id,
                    page=pg,
                    run_id=run_id,
                    element_type=etype,
                    bbox=bbox,
                    text_content=text,
                    features=features,
                    confidence=None,
                )
                page_count += 1

            # Region-level detections
            region_lists: list[tuple[list, str]] = [
                (pr.abbreviation_regions, "abbreviations"),
                (pr.legend_regions, "legend"),
                (pr.revision_regions, "revision"),
                (pr.standard_detail_regions, "standard_detail"),
                (pr.misc_title_regions, "misc_title"),
            ]
            for region_list, etype in region_lists:
                for region in region_list:
                    bbox = region.bbox()
                    if bbox == (0, 0, 0, 0):
                        continue
                    header_block = getattr(region, "header", None)
                    entry_count = len(getattr(region, "entries", []))
                    features = featurize_region(
                        etype,
                        bbox,
                        header_block,
                        pr.page_width,
                        pr.page_height,
                        entry_count=entry_count,
                    )
                    try:
                        text_content = region.header_text()
                    except Exception:
                        text_content = getattr(region, "text", "")
                    confidence = getattr(region, "confidence", None)
                    self._store.save_detection(
                        doc_id=self._doc_id,
                        page=pg,
                        run_id=run_id,
                        element_type=etype,
                        bbox=bbox,
                        text_content=text_content or "",
                        features=features,
                        confidence=confidence,
                    )
                    page_count += 1

            # Title blocks
            for tb in pr.title_blocks:
                bbox = tb.bbox
                if bbox == (0, 0, 0, 0):
                    continue
                features = featurize_region(
                    "title_block",
                    bbox,
                    None,
                    pr.page_width,
                    pr.page_height,
                )
                self._store.save_detection(
                    doc_id=self._doc_id,
                    page=pg,
                    run_id=run_id,
                    element_type="title_block",
                    bbox=bbox,
                    text_content=tb.raw_text[:500] if tb.raw_text else "",
                    features=features,
                    confidence=tb.confidence,
                )
                page_count += 1

            self._pipeline_pages.add(pg)
            total_detections += page_count

        # Navigate to current page to show results
        self._navigate_to_page()
        self._progress_var.set(100)
        self._progress.grid_remove()
        self._status.configure(
            text=f"Pipeline complete: {total_detections} detections "
            f"across {self._page_count} pages"
        )

    def _on_load_detections(self) -> None:
        """Reload detections from the database for the current page."""
        if not self._pdf_path:
            messagebox.showwarning("No PDF", "Select a PDF file first.")
            return

        self._page = self._page_var.get()
        self._resolution = self._dpi_var.get()
        self._scale = self._resolution / 72.0
        self._doc_id = self._store.register_document(self._pdf_path)

        # Render background
        from plancheck.ingest.ingest import render_page_image

        self._bg_image = render_page_image(
            self._pdf_path, self._page, resolution=self._resolution
        )
        self._render_background()

        # Load from DB
        dets = self._store.get_detections_for_page(self._doc_id, self._page)
        self._canvas_boxes.clear()
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

        # Apply any corrections to mark boxes
        self._apply_prior_corrections()

        # Load any saved groups for this page
        self._load_groups_for_page()

        self._draw_all_boxes()
        n = len(self._canvas_boxes)
        corrected = sum(1 for cb in self._canvas_boxes if cb.corrected)
        status = f"Loaded {n} detections from DB for page {self._page}"
        if corrected:
            status += f" ({corrected} previously corrected)"
        self._status.configure(text=status)

    def _apply_prior_corrections(self) -> None:
        """Apply prior corrections from the DB to the current canvas boxes.

        For each prior correction (from earlier annotation sessions):
        - **relabel/accept**: update the CanvasBox element_type and mark corrected
        - **reshape**: update element_type + bbox and mark corrected
        - **delete**: remove the box from the canvas

        Uses IoU-based spatial matching so corrections carry forward
        even when detection IDs change between pipeline re-runs.
        """
        if not self._doc_id:
            return

        prior = self._store.get_prior_corrections_by_bbox(self._doc_id, self._page)
        if not prior:
            return

        def _iou(a, b):
            x0 = max(a[0], b[0])
            y0 = max(a[1], b[1])
            x1 = min(a[2], b[2])
            y1 = min(a[3], b[3])
            inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
            area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0.0

        to_remove: list[CanvasBox] = []

        for corr in prior:
            orig = corr.get("original_bbox")
            if not orig or orig == (None, None, None, None):
                continue

            # Find best matching canvas box
            best_iou = 0.0
            best_cb: CanvasBox | None = None
            for cb in self._canvas_boxes:
                iou = _iou(orig, cb.pdf_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_cb = cb

            if best_iou < 0.5 or best_cb is None:
                continue

            ctype = corr["correction_type"]
            if ctype == "delete":
                to_remove.append(best_cb)
            elif ctype in ("relabel", "accept"):
                best_cb.element_type = corr["corrected_label"]
                best_cb.corrected = True
            elif ctype == "reshape":
                best_cb.element_type = corr["corrected_label"]
                best_cb.pdf_bbox = corr["corrected_bbox"]
                best_cb.corrected = True

        for cb in to_remove:
            if cb in self._canvas_boxes:
                self._canvas_boxes.remove(cb)

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
            self._canvas.create_oval(
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
                if _point_in_polygon(pdf_x, pdf_y, cbox.polygon):
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
            # Start lasso drag if on empty space in select mode
            self._lasso_start = (cx, cy)
            self._deselect()

    # ── Right-click copy / paste ───────────────────────────────────

    def _on_canvas_right_click(self, event: tk.Event) -> None:
        """Show a context menu for Copy Box / Paste Box."""
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)

        # Hit-test to see if a box is under the cursor
        eff = self._effective_scale()
        pdf_x = cx / eff
        pdf_y = cy / eff

        clicked: CanvasBox | None = None
        for cbox in reversed(self._canvas_boxes):
            if (
                cbox.rect_id
                and self._canvas.itemcget(cbox.rect_id, "state") == "hidden"
            ):
                continue
            if cbox.polygon:
                if _point_in_polygon(pdf_x, pdf_y, cbox.polygon):
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
                # If the currently selected box is a group root and
                # this clicked box is different and ungrouped, offer
                # "Add to Group"
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
        n_chars = len(text_content)
        self._status.configure(
            text=f"Pasted {chosen_type} detection ({n_chars} chars extracted)"
        )

    def _key_copy_box(self, event: tk.Event) -> None:
        """Ctrl+C — copy the selected box (only when focus is not in a text widget)."""
        if isinstance(event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)):
            return  # let native copy handle it
        if self._is_active_tab() and self._selected_box:
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
        ttk.Button(btn_f, text="Cancel", command=on_cancel).pack(side="left", padx=4)

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
        nx0, ny0, nx1, ny1 = ox0, oy0, ox1, oy1
        h = self._drag_handle
        if "w" in h:
            nx0 = min(px, ox1 - 1)
        if "e" in h:
            nx1 = max(px, ox0 + 1)
        if "n" in h:
            ny0 = min(py, oy1 - 1)
        if "s" in h:
            ny1 = max(py, oy0 + 1)

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
        ttk.Button(btn_f, text="Cancel", command=on_cancel).pack(side="left", padx=4)

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
        if not self._selected_box or not self._doc_id:
            self._status.configure(text="No box selected")
            return

        cbox = self._selected_box
        self._push_undo("accept", cbox)
        self._store.accept_detection(cbox.detection_id, self._doc_id, self._page)
        cbox.corrected = True
        self._session_count += 1
        self._update_session_label()
        self._draw_box(cbox)
        self._status.configure(
            text=f"Accepted {cbox.element_type} ({cbox.detection_id})"
        )

    def _on_relabel(self) -> None:
        if not self._selected_box or not self._doc_id:
            self._status.configure(text="No box selected")
            return

        cbox = self._selected_box
        new_label = self._type_var.get()
        if not new_label:
            return

        if new_label == cbox.element_type:
            # No change — treat as accept
            self._on_accept()
            return

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
        self._status.configure(text=f"Relabelled {old_label} → {new_label}")

    def _on_delete(self) -> None:
        if not self._selected_box or not self._doc_id:
            self._status.configure(text="No box selected")
            return

        if not messagebox.askyesno(
            "Delete Detection",
            "Mark this detection as a false positive?",
        ):
            return

        cbox = self._selected_box
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
        for hid in cbox.handle_ids:
            self._canvas.delete(hid)
        self._canvas_boxes.remove(cbox)
        self._selected_box = None
        self._deselect()

        self._session_count += 1
        self._update_session_label()
        self._status.configure(text=f"Deleted {cbox.element_type}")

    def _on_save_session(self) -> None:
        messagebox.showinfo(
            "Session Saved",
            f"Session complete. {self._session_count} corrections saved.",
        )
        self._session_id = uuid4().hex[:8]
        self._session_count = 0
        self._update_session_label()

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

        n = len(words)
        self._status.configure(text=f"Word overlay: {n} words on page {self._page}")

    def _clear_word_overlay(self) -> None:
        """Remove all word overlay rectangles from the canvas."""
        self._canvas.delete("word_overlay")
        self._word_overlay_ids.clear()

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
            self._status.configure(text="Undo delete")
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
            self._status.configure(text="Redo delete")
        elif action == "accept" and target:
            target.corrected = True
            self._draw_box(target)
            self._status.configure(text="Redo accept")
        else:
            self._status.configure(text="Redo (no visual change)")

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

    def _key_select_all(self, event: tk.Event) -> None:
        """Ctrl+A — select all visible boxes."""
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

    # ── Batch operations ───────────────────────────────────────────

    def _on_batch_accept(self) -> None:
        """Accept all multi-selected boxes at once."""
        targets = list(self._multi_selected)
        if self._selected_box and self._selected_box not in targets:
            targets.append(self._selected_box)
        if not targets or not self._doc_id:
            self._status.configure(text="No boxes selected for batch accept")
            return

        for cb in targets:
            self._store.accept_detection(cb.detection_id, self._doc_id, self._page)
            cb.corrected = True
            self._session_count += 1
            self._draw_box(cb)

        self._update_session_label()
        self._clear_multi_select()
        self._status.configure(text=f"Batch accepted {len(targets)} boxes")

    def _on_batch_relabel(self) -> None:
        """Relabel all multi-selected boxes to the type chosen in the combo."""
        targets = list(self._multi_selected)
        if self._selected_box and self._selected_box not in targets:
            targets.append(self._selected_box)
        if not targets or not self._doc_id:
            self._status.configure(text="No boxes selected for batch relabel")
            return

        new_label = self._type_var.get()
        if not new_label:
            messagebox.showwarning("No Type", "Select a target element type first.")
            return

        for cb in targets:
            if cb.element_type == new_label:
                continue
            self._store.save_correction(
                doc_id=self._doc_id,
                page=self._page,
                correction_type="relabel",
                corrected_label=new_label,
                corrected_bbox=cb.pdf_bbox,
                detection_id=cb.detection_id,
                original_label=cb.element_type,
                original_bbox=cb.pdf_bbox,
                session_id=self._session_id,
            )
            cb.element_type = new_label
            cb.corrected = True
            self._session_count += 1
            self._draw_box(cb)

        self._update_session_label()
        self._clear_multi_select()
        self._status.configure(
            text=f"Batch relabelled {len(targets)} boxes → {new_label}"
        )

    # ── Merge ──────────────────────────────────────────────────────

    def _on_merge(self) -> None:
        """Merge multi-selected overlapping boxes into a single polygon shape."""
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
        self._status.configure(
            text=f"Merged {len(targets)} boxes → {merged_type} (polygon with {len(merged_poly)} vertices)"
        )

    def _key_merge(self, event: tk.Event) -> None:
        """Keyboard shortcut M for merge."""
        self._on_merge()

    def _key_group(self, event: tk.Event) -> None:
        """Keyboard shortcut G for group creation."""
        if isinstance(event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)):
            return
        if not self._is_active_tab() or not self._selected_box:
            return
        if self._selected_box.group_id:
            grp = self._groups.get(self._selected_box.group_id, {})
            label = grp.get("label", "?")
            self._status.configure(text=f"Already in group \u2039{label}\u203a")
        else:
            self._create_group(self._selected_box)

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
        self._status.configure(text=f"Lasso selected {len(self._multi_selected)} boxes")

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

        # Load existing detections from DB (if any)
        self._doc_id = self._store.register_document(self._pdf_path)
        dets = self._store.get_detections_for_page(self._doc_id, self._page)
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

        # Apply any corrections
        self._apply_prior_corrections()
        self._draw_all_boxes()

        # Build informative status
        n = len(self._canvas_boxes)
        corrected = sum(1 for cb in self._canvas_boxes if cb.corrected)
        page_label = f"Page {self._page}"
        if self._page_count > 0:
            page_label += f" of {self._page_count}"

        if n > 0:
            status = f"{page_label} — {n} detections"
            if corrected:
                status += f" ({corrected} corrected)"
        elif self._page in self._pipeline_pages:
            status = f"{page_label} — pipeline ran, no detections found"
        else:
            status = f"{page_label} — click 'Run Pipeline' to detect boxes"

        self._status.configure(text=status)

    # ── Model training ─────────────────────────────────────────────

    def _on_train_model(self) -> None:
        """Train the classifier in a background thread."""
        self._model_status_label.configure(text="Training…", foreground="orange")
        self.root.update_idletasks()

        def _train():
            try:
                from plancheck.corrections.store import CorrectionStore as _CS

                store = _CS()
                n_examples = store.build_training_set()
                if n_examples < 5:
                    self.root.after(
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

                clf = ElementClassifier()
                metrics = clf.train(str(tmp))
                self._last_metrics = metrics

                acc = metrics.get("accuracy", 0)
                self.root.after(
                    0,
                    lambda: self._model_status_label.configure(
                        text=f"Model trained — acc {acc:.1%}",
                        foreground="green",
                    ),
                )
                # Reload classifier
                self._classifier = ElementClassifier()
            except Exception as exc:
                self.root.after(
                    0,
                    lambda: self._model_status_label.configure(
                        text=f"Train failed: {exc}",
                        foreground="red",
                    ),
                )

        threading.Thread(target=_train, daemon=True).start()

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
                "SELECT pdf_path FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if row:
                pdf_path = Path(row[0])
                if pdf_path.exists():
                    self.state.set_pdf(pdf_path)
                    self._page_var.set(page)
                    self._on_load_detections()
                    self._status.configure(
                        text=f"Active learning: page {page} (highest uncertainty)"
                    )
                else:
                    messagebox.showwarning(
                        "File not found",
                        f"PDF not found: {pdf_path}",
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
                        self._on_load_detections()
                except Exception as exc:
                    messagebox.showerror("Restore Error", str(exc), parent=win)

        ttk.Button(win, text="Restore", command=on_restore).pack(padx=10, pady=8)
