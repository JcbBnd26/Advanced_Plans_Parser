"""Tab 6 – Annotation: interactive detection correction UI.

Renders the PDF page with clickable, editable detection boxes.
Every correction is persisted to ``CorrectionStore`` immediately.

This module is a thin coordinator.  The implementation is split across:
- annotation_state  – CanvasBox dataclass, handle constants, pure helpers
- canvas_renderer   – CanvasRendererMixin (all drawing methods)
- event_handler     – EventHandlerMixin (all event/interaction methods)
- pdf_loader        – PdfLoaderMixin (PDF loading, navigation, zoom)
- annotation_store  – AnnotationStoreMixin (label registry, store, model)
- context_menu      – ContextMenuMixin (right-click context menus)
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any
from uuid import uuid4

from PIL import Image, ImageTk

from plancheck.corrections.classifier import ElementClassifier
from plancheck.corrections.store import CorrectionStore

# ── Re-exports for backward compatibility ──────────────────────────────
from .annotation_state import HANDLE_SIZE  # noqa: F401
from .annotation_state import (
    HANDLE_POSITIONS,
    CanvasBox,
    _reshape_bbox_from_handle,
    _scale_polygon_to_bbox,
)
from .annotation_store import AnnotationStoreMixin
from .canvas_renderer import CanvasRendererMixin
from .context_menu import ContextMenuMixin
from .event_handler import EventHandlerMixin
from .pdf_loader import PdfLoaderMixin
from .widgets import LogPanel, StatusBar
from .worker import PipelineWorker


class AnnotationTab(
    CanvasRendererMixin,
    EventHandlerMixin,
    PdfLoaderMixin,
    AnnotationStoreMixin,
    ContextMenuMixin,
):
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
        self._pipeline_ran_for_doc: bool = False
        self._active_drift_text: str = ""
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
        self._filter_row_widgets: dict[
            str, tuple[tk.Frame, tk.Checkbutton, tk.Label]
        ] = {}

        # Model metrics cache
        self._last_metrics: dict | None = None

        self._build_ui()

        try:
            self.root.bind(
                "<Destroy>", lambda e: setattr(self, "_closing", True), add="+"
            )
        except Exception:  # noqa: BLE001 — binding is best-effort
            pass

        # Subscribe to GuiState events
        self.state.subscribe("pdf_changed", self._on_pdf_changed)
        self.state.subscribe("run_completed", self._on_run_completed)
        self.state.subscribe("pipeline_starting", self._close_store)

    def _close_store(self) -> None:
        """Close the database connection while the pipeline runs."""
        if self._store is not None:
            try:
                self._store.close()
            except Exception:  # noqa: BLE001 — best effort
                pass
            self._store = None

    def _reopen_store(self) -> None:
        """Reopen the database connection after pipeline finishes."""
        if self._store is None:
            from plancheck.corrections.store import CorrectionStore

            self._store = CorrectionStore()

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
                # Guard: widget may have been destroyed before callback fires
                if tip_win is not None or not widget.winfo_exists():
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
            if not self._insp_wheel_active or not self._insp_canvas.winfo_ismapped():
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
        self._type_var.trace_add("write", self._on_type_selection_changed)
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
        ttk.Label(inspector, text="Subtype:").grid(
            row=row, column=0, sticky="w", padx=4, pady=2
        )
        self._subtype_var = tk.StringVar()
        self._subtype_combo = ttk.Combobox(
            inspector,
            textvariable=self._subtype_var,
            values=self._title_subtypes(),
            width=18,
            state="disabled",
        )
        self._subtype_combo.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        self._subtype_combo.bind("<<ComboboxSelected>>", self._on_subtype_selected)
        self._tooltip(
            self._subtype_combo,
            "Optional Stage-2 title subtype for title-block annotations.",
        )

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
        self._suggest_label.pack(anchor="w")
        self._suggest_detail_label = ttk.Label(
            self._suggest_frame,
            text="",
            foreground="#8a4b00",
            wraplength=220,
            justify="left",
        )
        self._suggest_detail_label.pack(anchor="w")
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
        filter_btns.grid(
            row=row, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 0)
        )
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
            inspector,
            text="(no page loaded)",
            foreground="gray",
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
        _btn_bootstrap = ttk.Button(
            model_btns, text="Bootstrap", command=self._on_bootstrap_training
        )
        _btn_bootstrap.pack(side="left", padx=3)
        self._tooltip(
            _btn_bootstrap,
            "Generate training data from high-confidence rule-based detections "
            "(use for cold-start when no corrections exist)",
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

        # ── Drift Warning Indicator ───────────────────────────────
        row += 1
        self._drift_indicator = ttk.Label(inspector, text="", foreground="orange")
        self._drift_indicator.grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4, pady=1
        )
        self._tooltip(
            self._drift_indicator,
            "Drift detected: this page looks different from training data",
        )

        row += 1
        self._runtime_summary_label = ttk.Label(
            inspector,
            text="",
            foreground="gray",
            wraplength=220,
            justify="left",
            font=("TkDefaultFont", 8),
        )
        self._runtime_summary_label.grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4, pady=1
        )
        self._add_copy_menu(self._runtime_summary_label)
        self._tooltip(
            self._runtime_summary_label,
            "Current routing mode, drift posture, and retrain readiness",
        )

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
        self._sync_title_subtype_controls(self._type_var.get())

        # Initialize model status
        self._update_model_status()

    # ── PDF selection ──────────────────────────────────────────────

    def _on_run_completed(self) -> None:
        """Re-load the current page after a pipeline run finishes.

        The pipeline now persists detections to the correction store,
        so refreshing the page will pick up the fresh results.
        """
        # Reopen store if it was closed for the pipeline
        self._reopen_store()
        # Ensure our connection sees the latest data from the pipeline's commits
        self._store.refresh()
        self._pipeline_ran_for_doc = True
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
