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
from tkinter import simpledialog, ttk
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
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)
        notebook.add(self.frame, text="ML Trainer")

        # ── State ──────────────────────────────────────────────────
        self._pdf_path: Path | None = None
        self._doc_id: str | None = None
        self._run_id: str = ""
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

        self._draw_start: tuple[float, float] | None = None
        self._draw_rect_id: int | None = None
        self._lasso_start: tuple[float, float] | None = None
        self._lasso_rect_id: int | None = None
        self._lasso_shift: bool = False
        self._lasso_word: bool = False
        self._word_click_candidate_rid: int | None = None
        self._session_id: str = uuid4().hex[:8]
        self._session_count: int = 0
        self._store: CorrectionStore | None = None
        self._worker: PipelineWorker | None = None
        self._classifier = ElementClassifier()

        # Training session state (page-by-page micro-retrain)
        self._training_session_active: bool = False
        self._session_page_has_corrections: bool = False
        self._session_baseline_metrics: dict = {}
        self._session_baseline_f1: float = 0.0
        self._session_corrections_total: int = 0
        self._session_pages_reviewed: int = 0

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

        # ── Performance: drag throttle & caches ────────────────────
        self._drag_after_id: str | None = None  # pending after() for drag
        self._drag_pending_coords: tuple[float, float] | None = None
        self._zoom_image_cache: dict[float, Image.Image] = {}
        self._zoom_cache_max: int = 5
        self._word_cache: dict[tuple[str, int], list[dict]] = {}

        self._build_ui()

        try:
            self.root.bind(
                "<Destroy>", lambda e: setattr(self, "_closing", True), add="+"
            )
        except Exception:  # noqa: BLE001 — binding is best-effort
            pass

        # Lazy-populate the Project dropdown on first tab visit
        self._needs_project_refresh = True
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed, add="+")

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

    def _on_tab_changed(self, _event: tk.Event) -> None:
        """Populate the Project dropdown on first visit to the ML Trainer tab."""
        if not self._needs_project_refresh:
            return
        try:
            selected = self.notebook.index(self.notebook.select())
            my_index = self.notebook.index(self.frame)
        except Exception:  # noqa: BLE001 — notebook may be in bad state
            return
        if selected == my_index:
            self._refresh_project_dropdown()
            if self._doc_id:
                self._refresh_run_dropdown(self._doc_id)
            self._needs_project_refresh = False

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

    # ── Collapsible section helper ────────────────────────────────

    def _make_collapsible_section(
        self,
        parent: ttk.Frame,
        title: str,
        *,
        collapsed: bool = True,
    ) -> tuple[ttk.Frame, ttk.Frame, tk.BooleanVar]:
        """Create a collapsible section with a toggle button.

        Returns ``(outer_frame, content_frame, expanded_var)``.
        *content_frame* is hidden via ``grid_remove()`` when collapsed.
        """
        expanded_var = tk.BooleanVar(value=not collapsed)
        outer = ttk.Frame(parent)
        outer.columnconfigure(0, weight=1)

        def _toggle() -> None:
            if expanded_var.get():
                expanded_var.set(False)
                content.grid_remove()
                toggle_btn.configure(text=f"\u25b6 {title}")
            else:
                expanded_var.set(True)
                content.grid(row=1, column=0, sticky="ew")
                toggle_btn.configure(text=f"\u25bc {title}")
            # Update inspector scroll region after collapse/expand
            if hasattr(self, "_insp_canvas"):
                self._insp_canvas.configure(scrollregion=self._insp_canvas.bbox("all"))

        arrow = "\u25bc" if not collapsed else "\u25b6"
        toggle_btn = ttk.Button(
            outer,
            text=f"{arrow} {title}",
            command=_toggle,
            style="Toolbutton",
        )
        toggle_btn.grid(row=0, column=0, sticky="ew")

        content = ttk.Frame(outer)
        content.columnconfigure(1, weight=1)
        if not collapsed:
            content.grid(row=1, column=0, sticky="ew")
        # else: left un-gridded (collapsed)

        return outer, content, expanded_var

    # ── Filter summary helper ─────────────────────────────────────

    def _update_filter_summary(self) -> None:
        """Update the collapsed-state filter summary line."""
        if not hasattr(self, "_filter_summary_label"):
            return
        active = sum(1 for v in self._filter_label_vars.values() if v.get())
        total = len(self._filter_label_vars)
        parts: list[str] = [f"{active}/{total} types"]
        conf_min = self._filter_conf_min.get()
        if conf_min > 0:
            parts.append(f"conf \u2265 {conf_min:.2f}")
        if self._filter_uncorrected_only.get():
            parts.append("uncorrected only")
        self._filter_summary_label.configure(text=" | ".join(parts))

    # ── Auto-expand training section ──────────────────────────────

    def _ensure_training_section_visible(self) -> None:
        """Expand the Train & Evaluate section if it is collapsed."""
        if hasattr(self, "_sec2_expanded") and not self._sec2_expanded.get():
            self._sec2_expanded.set(True)
            if hasattr(self, "_sec2_content"):
                self._sec2_content.grid(row=1, column=0, sticky="ew")
            if hasattr(self, "_sec2_toggle_btn"):
                self._sec2_toggle_btn.configure(text="\u25bc Train & Evaluate")
            if hasattr(self, "_insp_canvas"):
                self._insp_canvas.configure(scrollregion=self._insp_canvas.bbox("all"))

    # ── UI construction ────────────────────────────────────────────

    def _build_ui(self) -> None:
        pad = {"padx": 6, "pady": 3}

        # ── Top bar ───────────────────────────────────────────────
        top = ttk.Frame(self.frame)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", **pad)

        # ── Project / Run dropdowns ───────────────────────────────
        ttk.Label(top, text="Project:").pack(side="left")
        self._project_var = tk.StringVar()
        self._project_combo = ttk.Combobox(
            top,
            textvariable=self._project_var,
            state="readonly",
            width=30,
        )
        self._project_combo.pack(side="left", padx=2)
        self._project_combo.bind("<<ComboboxSelected>>", self._on_project_selected)
        self._tooltip(
            self._project_combo,
            "Select a previously processed document. Allows reviewing"
            " detections even when the original PDF is unavailable.",
        )
        # Internal mapping: combo index → doc_id
        self._project_doc_ids: list[str] = []

        _btn_tag = ttk.Button(top, text="✎", width=2, command=self._edit_project_tag)
        _btn_tag.pack(side="left", padx=1)
        self._tooltip(_btn_tag, "Set or change the project tag for this document.")

        ttk.Label(top, text="Run:").pack(side="left", padx=(4, 0))
        self._run_var = tk.StringVar()
        self._run_combo = ttk.Combobox(
            top,
            textvariable=self._run_var,
            state="readonly",
            width=24,
        )
        self._run_combo.pack(side="left", padx=2)
        self._run_combo.bind("<<ComboboxSelected>>", self._on_run_selected)
        self._tooltip(
            self._run_combo,
            "Switch between processing runs for the selected document."
            " Each run represents a separate pipeline execution.",
        )
        # Internal mapping: combo index → run_id
        self._run_ids: list[str] = []
        self._run_pages: dict[str, list[int]] = {}

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
        self._tooltip(
            self._page_spin,
            "Current page number. Press Enter after changing it to jump to that page.",
        )

        self._page_count_label = ttk.Label(top, text="/ ?")
        self._page_count_label.pack(side="left", padx=2)
        self._tooltip(
            self._page_count_label, "Total number of pages in the loaded PDF."
        )

        _btn_prev = ttk.Button(top, text="◀", width=2, command=self._on_prev_page)
        _btn_prev.pack(side="left", padx=1)
        self._tooltip(_btn_prev, "Go to the previous page. Shortcut: Ctrl+Left Arrow.")
        _btn_next = ttk.Button(top, text="▶", width=2, command=self._on_next_page)
        _btn_next.pack(side="left", padx=1)
        self._tooltip(_btn_next, "Go to the next page. Shortcut: Ctrl+Right Arrow.")

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
        self._tooltip(
            self._dpi_spin,
            "Rendering resolution for the page view. Higher DPI improves visual detail but may load more slowly.",
        )

        _btn_zoom_out = ttk.Button(
            top, text="\u2212", width=2, command=lambda: self._apply_zoom(1 / 1.2)
        )
        _btn_zoom_out.pack(side="left", padx=1)
        self._tooltip(
            _btn_zoom_out, "Zoom out to see more of the page at once. Shortcut: -"
        )
        _btn_zoom_in = ttk.Button(
            top, text="+", width=2, command=lambda: self._apply_zoom(1.2)
        )
        _btn_zoom_in.pack(side="left", padx=1)
        self._tooltip(
            _btn_zoom_in,
            "Zoom in for closer inspection of boxes, text, and page details. Shortcut: +",
        )
        _btn_fit = ttk.Button(top, text="Fit", width=3, command=self._fit_to_window)
        _btn_fit.pack(side="left", padx=2)
        self._tooltip(_btn_fit, "Fit the full page into the current view. Shortcut: F.")

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
            "Show or hide word-level boxes detected from the PDF text layer. Use this when checking text extraction or selecting words for merges.",
        )

        self._offline_mode_var = tk.BooleanVar(value=False)
        self._btn_offline = ttk.Checkbutton(
            top,
            text="Offline",
            variable=self._offline_mode_var,
            command=self._on_offline_mode_toggle,
        )
        self._btn_offline.pack(side="left", padx=2)
        self._tooltip(
            self._btn_offline,
            "Force PNG backdrop mode — uses the stored page image instead of the"
            " live PDF. Disables the word overlay. Turns on automatically when the"
            " original PDF is not on disk.",
        )

        # ── Canvas + Inspector split (VS Code-style draggable sash) ─────
        self._h_paned = ttk.PanedWindow(self.frame, orient="horizontal")
        self._h_paned.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)

        # ── Canvas (left pane) ────────────────────────────────────
        canvas_frame = ttk.Frame(self._h_paned)
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        self._h_paned.add(canvas_frame, weight=3)

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
        # Hover tooltip
        self._canvas.bind("<Motion>", self._on_canvas_motion)
        self._canvas.bind("<Leave>", lambda e: self._hide_hover_tooltip())

        # Mode banner: an overlay label placed at the top of the canvas in Add mode
        self._mode_banner = tk.Label(
            self._canvas,
            text="",
            background="#c0392b",
            foreground="white",
            font=("TkDefaultFont", 9, "bold"),
            padx=8,
            pady=4,
        )

        # ── Inspector (right) ─────────────────────────────────────
        # ── Inspector (right pane) ────────────────────────────────
        inspector_outer = ttk.LabelFrame(self._h_paned, text="Inspector")
        inspector_outer.columnconfigure(0, weight=1)
        inspector_outer.rowconfigure(0, weight=1)
        self._h_paned.add(inspector_outer, weight=1)

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

        # ═══════════════════════════════════════════════════════════
        # SECTION 1: Review & Correct (always visible)
        # ═══════════════════════════════════════════════════════════
        sec1 = ttk.LabelFrame(inspector, text="Review & Correct")
        sec1.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=(4, 2))
        sec1.columnconfigure(1, weight=1)

        s1r = 0  # section 1 row counter

        # ── Detection ID ──────────────────────────────────────────
        ttk.Label(sec1, text="ID:").grid(row=s1r, column=0, sticky="w", padx=4, pady=2)
        self._insp_id = ttk.Label(sec1, text="—")
        self._insp_id.grid(row=s1r, column=1, sticky="w", padx=4, pady=2)
        self._add_copy_menu(self._insp_id)
        self._tooltip(
            self._insp_id,
            "Unique identifier for the selected detection. Useful when cross-checking saved corrections or debugging a specific item.",
        )

        # ── Element type ──────────────────────────────────────────
        s1r += 1
        ttk.Label(sec1, text="Type:").grid(
            row=s1r, column=0, sticky="w", padx=4, pady=2
        )
        self._type_var = tk.StringVar()
        self._type_var.trace_add("write", self._on_type_selection_changed)
        type_row_frame = ttk.Frame(sec1)
        type_row_frame.grid(row=s1r, column=1, sticky="ew", padx=4, pady=2)
        self._type_combo = ttk.Combobox(
            type_row_frame,
            textvariable=self._type_var,
            values=self.ELEMENT_TYPES,
            width=14,
        )
        self._type_combo.pack(side="left")
        self._type_combo.bind("<Return>", self._on_type_entered)
        self._tooltip(
            self._type_combo,
            "Element type for the selected detection. Change this when the current label is wrong or when adding a new detection.",
        )
        _btn_add_type = ttk.Button(
            type_row_frame, text="+", width=2, command=self._on_add_element_type
        )
        _btn_add_type.pack(side="left", padx=2)
        self._tooltip(
            _btn_add_type,
            "Register a new element type so it can be used in this session and future reviews.",
        )

        # ── Subtype ───────────────────────────────────────────────
        s1r += 1
        ttk.Label(sec1, text="Subtype:").grid(
            row=s1r, column=0, sticky="w", padx=4, pady=2
        )
        self._subtype_var = tk.StringVar()
        self._subtype_combo = ttk.Combobox(
            sec1,
            textvariable=self._subtype_var,
            values=self._title_subtypes(),
            width=18,
            state="disabled",
        )
        self._subtype_combo.grid(row=s1r, column=1, sticky="ew", padx=4, pady=2)
        self._subtype_combo.bind("<<ComboboxSelected>>", self._on_subtype_selected)
        self._tooltip(
            self._subtype_combo,
            "Optional Stage 2 title subtype. Use this for title-related elements when a more specific label is needed.",
        )

        # ── Confidence ────────────────────────────────────────────
        s1r += 1
        ttk.Label(sec1, text="Conf:").grid(
            row=s1r, column=0, sticky="w", padx=4, pady=2
        )
        self._insp_conf = ttk.Label(sec1, text="—")
        self._insp_conf.grid(row=s1r, column=1, sticky="w", padx=4, pady=2)
        self._add_copy_menu(self._insp_conf)
        self._tooltip(
            self._insp_conf,
            "Confidence score for the selected detection. Lower values usually deserve closer review.",
        )

        # ── Text + Rescan ─────────────────────────────────────────
        s1r += 1
        ttk.Label(sec1, text="Text:").grid(
            row=s1r, column=0, sticky="nw", padx=4, pady=2
        )
        text_frame = ttk.Frame(sec1)
        text_frame.grid(row=s1r, column=1, sticky="ew", padx=4, pady=2)
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
        self._tooltip(
            self._insp_text,
            "Extracted text for the selected detection. Review this when checking OCR quality or deciding how to relabel an item.",
        )
        _btn_rescan = ttk.Button(
            text_frame, text="Rescan \u21bb", width=10, command=self._on_rescan_text
        )
        _btn_rescan.grid(row=1, column=0, sticky="w", pady=(2, 0))
        self._tooltip(
            _btn_rescan,
            "Re-extract text from the current PDF region. Use this after moving or resizing a box.",
        )

        # ── Model suggestion (moved before action buttons) ────────
        s1r += 1
        self._suggest_frame = ttk.Frame(sec1)
        self._suggest_frame.grid(
            row=s1r, column=0, columnspan=2, sticky="ew", padx=4, pady=2
        )
        self._suggest_label = ttk.Label(
            self._suggest_frame, text="", foreground="#0060c0"
        )
        self._suggest_label.pack(anchor="w")
        self._tooltip(
            self._suggest_label,
            "Model suggestion for the selected detection, including confidence and whether it came from Stage 1 or Stage 2.",
        )
        self._suggest_detail_label = ttk.Label(
            self._suggest_frame,
            text="",
            foreground="#8a4b00",
            wraplength=220,
            justify="left",
        )
        self._suggest_detail_label.pack(anchor="w")
        self._tooltip(
            self._suggest_detail_label,
            "Extra routing detail for the model suggestion. This may explain Stage 1 to Stage 2 refinement, low-confidence subtype results, or why Stage 2 was skipped.",
        )
        self._suggest_btn = ttk.Button(
            self._suggest_frame, text="Apply", command=self._apply_suggestion
        )
        self._tooltip(
            self._suggest_btn,
            "Apply the model's suggested label to the selected detection. Use this when the suggested type is correct and you want to save time.",
        )
        self._model_suggestion: str | None = None

        # ── Action buttons ────────────────────────────────────────
        s1r += 1
        btn_frame = ttk.Frame(sec1)
        btn_frame.grid(row=s1r, column=0, columnspan=2, pady=6)

        _btn_accept = ttk.Button(
            btn_frame, text="Accept \u2713", command=self._on_accept
        )
        _btn_accept.pack(side="left", padx=3)
        self._tooltip(
            _btn_accept, "Confirm that the selected detection is correct. Shortcut: A."
        )
        _btn_relabel = ttk.Button(btn_frame, text="Relabel", command=self._on_relabel)
        _btn_relabel.pack(side="left", padx=3)
        self._tooltip(
            _btn_relabel,
            "Save the selected type as a correction for this detection. Use this when the box is correct but the label is wrong. Shortcut: R.",
        )
        _btn_delete = ttk.Button(
            btn_frame, text="Reject \u2717", command=self._on_delete
        )
        _btn_delete.pack(side="left", padx=3)
        self._tooltip(
            _btn_delete, "Mark the selected detection as a false positive. Shortcut: D."
        )
        _btn_dismiss = ttk.Button(
            btn_frame, text="Skip \u2298", command=self._on_dismiss
        )
        _btn_dismiss.pack(side="left", padx=3)
        self._tooltip(
            _btn_dismiss,
            "Dismiss this detection without affecting training data. "
            "Use when you're unsure of the label or want to skip for now. "
            "Shortcut: X.",
        )

        # ── Merge + Multi-select ──────────────────────────────────
        s1r += 1
        batch_frame = ttk.Frame(sec1)
        batch_frame.grid(row=s1r, column=0, columnspan=2, pady=2)
        _btn_merge = ttk.Button(
            batch_frame, text="Merge \u229e", command=self._on_merge
        )
        _btn_merge.pack(side="left", padx=3)
        self._tooltip(
            _btn_merge,
            "Combine selected boxes or words into one detection. Use this when the pipeline split a single logical item into multiple pieces. Shortcut: M.",
        )
        s1r += 1
        self._multi_label = ttk.Label(sec1, text="", foreground="blue")
        self._multi_label.grid(row=s1r, column=0, columnspan=2, sticky="w", padx=4)
        self._tooltip(
            self._multi_label,
            "Shows how many boxes are selected together. Use multi-selection for merges, grouping, and batch review actions.",
        )

        # ── Group membership ──────────────────────────────────────
        s1r += 1
        ttk.Label(sec1, text="Group:").grid(
            row=s1r, column=0, sticky="nw", padx=4, pady=2
        )
        group_frame = ttk.Frame(sec1)
        group_frame.grid(row=s1r, column=1, sticky="ew", padx=4, pady=2)
        self._insp_group_label = ttk.Label(group_frame, text="\u2014", wraplength=180)
        self._insp_group_label.pack(side="top", anchor="w")
        self._add_copy_menu(self._insp_group_label)
        self._tooltip(
            self._insp_group_label,
            "Shows whether the selected detection belongs to a group and how it fits into that group.",
        )

        group_btn_frame = ttk.Frame(group_frame)
        group_btn_frame.pack(side="top", anchor="w", pady=(2, 0))
        self._btn_create_group = ttk.Button(
            group_btn_frame, text="Create Group", command=self._on_create_group
        )
        self._btn_create_group.pack(side="left", padx=(0, 3))
        self._tooltip(
            self._btn_create_group,
            "Create a new group using the selected box as the parent item. Shortcut: G.",
        )

        self._btn_add_to_group = ttk.Button(
            group_btn_frame, text="Add to Group", command=self._on_add_to_group
        )
        self._btn_add_to_group.pack(side="left", padx=(0, 3))
        self._tooltip(
            self._btn_add_to_group,
            "Add the current selection to the active group. Use this for related boxes that belong under one parent item.",
        )
        self._btn_add_to_group.pack_forget()

        self._btn_remove_group = ttk.Button(
            group_btn_frame, text="Remove", command=self._on_remove_from_group
        )
        self._btn_remove_group.pack(side="left", padx=(0, 3))
        self._tooltip(
            self._btn_remove_group,
            "Remove the selected box from its group. If the selected box is the parent, the group may be removed.",
        )
        self._btn_remove_group.pack_forget()

        # ── Filters (collapsible sub-section) ─────────────────────
        s1r += 1
        flt_outer, flt_content, self._filters_expanded = self._make_collapsible_section(
            sec1, "Filters", collapsed=True
        )
        flt_outer.grid(row=s1r, column=0, columnspan=2, sticky="ew", padx=4, pady=2)

        # Summary label visible when filters are collapsed
        self._filter_summary_label = ttk.Label(
            flt_outer, text="", foreground="gray", font=("TkDefaultFont", 8)
        )
        self._filter_summary_label.grid(row=2, column=0, sticky="w", padx=20)

        fr = 0  # filter content row counter
        filter_btns = ttk.Frame(flt_content)
        filter_btns.grid(
            row=fr, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 0)
        )
        _btn_show_all = ttk.Button(
            filter_btns,
            text="Show All",
            command=self._select_all_filter_types,
            width=10,
        )
        _btn_show_all.pack(side="left", padx=(0, 4))
        self._tooltip(
            _btn_show_all,
            "Turn all element-type filters back on so every detection type becomes visible again.",
        )
        _btn_hide_all = ttk.Button(
            filter_btns,
            text="Hide All",
            command=self._deselect_all_filter_types,
            width=10,
        )
        _btn_hide_all.pack(side="left")
        self._tooltip(
            _btn_hide_all,
            "Turn all element-type filters off. Use this when you want to re-enable only one or two types for focused review.",
        )
        self._filter_color_btn = ttk.Button(
            filter_btns,
            text="Pick Color",
            command=self._choose_active_filter_color,
            width=18,
        )
        self._filter_color_btn.pack(side="left", padx=(8, 0))
        self._tooltip(
            self._filter_color_btn,
            "Choose the display color for the currently selected filter type.",
        )

        fr += 1
        self._filter_frame = ttk.Frame(flt_content)
        self._filter_frame.grid(row=fr, column=0, columnspan=2, sticky="ew", padx=8)
        if self.ELEMENT_TYPES:
            self._active_filter_color_type = self.ELEMENT_TYPES[0]
        self._rebuild_filter_controls()

        fr += 1
        conf_frame = ttk.Frame(flt_content)
        conf_frame.grid(row=fr, column=0, columnspan=2, sticky="ew", padx=4, pady=2)
        ttk.Label(conf_frame, text="Min conf:").pack(side="left")
        self._conf_value_label = ttk.Label(conf_frame, text="0.00", width=4)
        self._conf_value_label.pack(side="right", padx=(4, 0))
        _conf_scale = ttk.Scale(
            conf_frame,
            from_=0.0,
            to=1.0,
            variable=self._filter_conf_min,
            orient="horizontal",
            command=lambda v: (
                self._conf_value_label.configure(text=f"{float(v):.2f}"),
                self._apply_filters(),
                self._update_filter_summary(),
            ),
        )
        _conf_scale.pack(side="left", fill="x", expand=True)
        self._tooltip(
            _conf_scale,
            "Hide detections below the selected confidence threshold. Use this to focus on stronger predictions or isolate weaker ones for review.",
        )

        fr += 1
        _cb_uncorrected = ttk.Checkbutton(
            flt_content,
            text="Uncorrected only",
            variable=self._filter_uncorrected_only,
            command=lambda: (self._apply_filters(), self._update_filter_summary()),
        )
        _cb_uncorrected.grid(row=fr, column=0, columnspan=2, sticky="w", padx=8)
        self._tooltip(
            _cb_uncorrected,
            "Show only detections that have not been corrected yet. Use this to focus on unfinished review work.",
        )

        self._update_filter_summary()

        # ═══════════════════════════════════════════════════════════
        # SECTION 2: Train & Evaluate (collapsible, collapsed)
        # ═══════════════════════════════════════════════════════════
        sec2_outer, sec2, self._sec2_expanded = self._make_collapsible_section(
            inspector, "Train & Evaluate", collapsed=True
        )
        sec2_outer.grid(row=1, column=0, columnspan=2, sticky="ew", padx=4, pady=2)
        # Store references for auto-expand helper
        self._sec2_content = sec2
        # Find the toggle button (first child of sec2_outer)
        for child in sec2_outer.winfo_children():
            if isinstance(child, ttk.Button):
                self._sec2_toggle_btn = child
                break

        s2r = 0  # section 2 row counter

        # ── Model status ──────────────────────────────────────────
        self._model_status_label = ttk.Label(
            sec2, text="No model trained", foreground="gray"
        )
        self._model_status_label.grid(
            row=s2r, column=0, columnspan=2, sticky="w", padx=4, pady=2
        )
        self._add_copy_menu(self._model_status_label)
        self._tooltip(
            self._model_status_label,
            "Current model status, including whether a model is loaded and whether retraining is approaching or past the threshold.",
        )

        # ── Runtime summary ───────────────────────────────────────
        s2r += 1
        self._runtime_summary_label = ttk.Label(
            sec2,
            text="",
            foreground="gray",
            wraplength=220,
            justify="left",
            font=("TkDefaultFont", 8),
        )
        self._runtime_summary_label.grid(
            row=s2r, column=0, columnspan=2, sticky="w", padx=4, pady=1
        )
        self._add_copy_menu(self._runtime_summary_label)
        self._tooltip(
            self._runtime_summary_label,
            "Quick summary of the current ML session: routing mode, drift posture, and retrain readiness.",
        )

        # ── Drift indicator ───────────────────────────────────────
        s2r += 1
        self._drift_indicator = ttk.Label(sec2, text="", foreground="orange")
        self._drift_indicator.grid(
            row=s2r, column=0, columnspan=2, sticky="w", padx=4, pady=1
        )
        self._tooltip(
            self._drift_indicator,
            "Warns when the current page looks different from the data used to build the drift reference. Treat predictions more cautiously when drift is active.",
        )

        # ── Model buttons ────────────────────────────────────────
        s2r += 1
        model_btns = ttk.Frame(sec2)
        model_btns.grid(row=s2r, column=0, columnspan=2, padx=4, pady=2)
        _btn_train = ttk.Button(
            model_btns, text="Train Model", command=self._on_train_model
        )
        _btn_train.pack(side="left", padx=3)
        self._tooltip(
            _btn_train,
            "Retrain the main model from accepted and relabeled corrections. Use this after enough new corrections have accumulated.",
        )
        _btn_bootstrap = ttk.Button(
            model_btns, text="Bootstrap", command=self._on_bootstrap_training
        )
        _btn_bootstrap.pack(side="left", padx=3)
        self._tooltip(
            _btn_bootstrap,
            "Create starter training data from stronger existing detections. Use this when you do not yet have enough manual corrections for a normal retrain.",
        )
        self._training_session_btn = ttk.Button(
            model_btns,
            text="Start Training Session",
            command=self._toggle_training_session,
        )
        self._training_session_btn.pack(side="left", padx=3)
        self._tooltip(
            self._training_session_btn,
            "Page-by-page training mode: after reviewing each page, the model "
            "micro-retrains on all corrections and re-predicts the next page. "
            "Construction plans are repetitive — fix it once, the model "
            "propagates it forward.",
        )

        s2r += 1
        model_btns2 = ttk.Frame(sec2)
        model_btns2.grid(row=s2r, column=0, columnspan=2, padx=4, pady=2)
        _btn_metrics = ttk.Button(
            model_btns2, text="Metrics", command=self._on_show_metrics
        )
        _btn_metrics.pack(side="left", padx=3)
        self._tooltip(
            _btn_metrics,
            "Open the latest training metrics, including accuracy and class-level performance.",
        )
        _btn_history = ttk.Button(
            model_btns2, text="History", command=self._on_show_training_history
        )
        _btn_history.pack(side="left", padx=3)
        self._tooltip(
            _btn_history,
            "Show past training runs so you can compare how the model has changed over time.",
        )
        _btn_importance = ttk.Button(
            model_btns2, text="Importance", command=self._on_show_feature_importance
        )
        _btn_importance.pack(side="left", padx=3)
        self._tooltip(
            _btn_importance,
            "Show which input features influenced the trained model most strongly.",
        )

        # ═══════════════════════════════════════════════════════════
        # SECTION 3: Manage & Maintain (collapsible, collapsed)
        # ═══════════════════════════════════════════════════════════
        sec3_outer, sec3, self._sec3_expanded = self._make_collapsible_section(
            inspector, "Manage & Maintain", collapsed=True
        )
        sec3_outer.grid(row=2, column=0, columnspan=2, sticky="ew", padx=4, pady=2)

        s3r = 0  # section 3 row counter

        # ── Stats ─────────────────────────────────────────────────
        ttk.Label(sec3, text="Stats:", font=("TkDefaultFont", 9, "bold")).grid(
            row=s3r, column=0, columnspan=2, sticky="w", padx=4
        )
        s3r += 1
        self._stats_label = ttk.Label(
            sec3, text="", foreground="gray", font=("TkDefaultFont", 8)
        )
        self._stats_label.grid(
            row=s3r, column=0, columnspan=2, sticky="w", padx=8, pady=2
        )
        self._add_copy_menu(self._stats_label)
        self._tooltip(
            self._stats_label,
            "Summary of documents, detections, corrections, training examples, and retrain readiness.",
        )

        # ── Page Elements ─────────────────────────────────────────
        s3r += 1
        ttk.Label(sec3, text="Page Elements:", font=("TkDefaultFont", 9, "bold")).grid(
            row=s3r, column=0, columnspan=2, sticky="w", padx=4
        )
        s3r += 1
        self._page_elements_label = ttk.Label(
            sec3,
            text="(no page loaded)",
            foreground="gray",
            font=("TkDefaultFont", 8),
        )
        self._page_elements_label.grid(
            row=s3r, column=0, columnspan=2, sticky="w", padx=8, pady=2
        )
        self._add_copy_menu(self._page_elements_label)
        self._tooltip(
            self._page_elements_label,
            "Shows how many detections of each type are on the current page. Use this for a quick sense of page content and coverage.",
        )

        # ── Refresh / Clear buttons ───────────────────────────────
        s3r += 1
        _btn_stats = ttk.Button(sec3, text="Refresh Stats", command=self._refresh_stats)
        _btn_stats.grid(row=s3r, column=0, padx=4, pady=2)
        self._tooltip(
            _btn_stats,
            "Recalculate annotation and training statistics from the correction database.",
        )
        _btn_clear_runs = ttk.Button(
            sec3, text="Clear Old Runs", command=self._on_clear_old_runs
        )
        _btn_clear_runs.grid(row=s3r, column=1, padx=4, pady=2)
        self._tooltip(
            _btn_clear_runs,
            "Remove saved detection data from older pipeline runs while keeping corrections and training data.",
        )

        # ── Suggest Next Page ─────────────────────────────────────
        s3r += 1
        _btn_suggest = ttk.Button(
            sec3, text="Suggest Next Page", command=self._on_suggest_next
        )
        _btn_suggest.grid(row=s3r, column=0, columnspan=2, padx=4, pady=2)
        self._tooltip(
            _btn_suggest,
            "Jump to the page that is most likely to benefit from more annotation work.",
        )

        # ── Snapshots ─────────────────────────────────────────────
        s3r += 1
        snap_btns = ttk.Frame(sec3)
        snap_btns.grid(row=s3r, column=0, columnspan=2, padx=4, pady=2)
        _btn_snap = ttk.Button(snap_btns, text="Snapshot", command=self._on_snapshot)
        _btn_snap.pack(side="left", padx=3)
        self._tooltip(
            _btn_snap,
            "Save a timestamped backup of the correction database before major edits or retraining.",
        )
        _btn_restore = ttk.Button(
            snap_btns, text="Restore\u2026", command=self._on_restore_snapshot
        )
        _btn_restore.pack(side="left", padx=3)
        self._tooltip(
            _btn_restore,
            "Restore corrections from a previous snapshot if you need to recover older annotation work.",
        )

        # ═══════════════════════════════════════════════════════════
        # PINNED FOOTER (always visible, outside sections)
        # ═══════════════════════════════════════════════════════════

        # ── Session counter ───────────────────────────────────────
        self._session_label = ttk.Label(inspector, text="Session: 0 saved")
        self._session_label.grid(
            row=3, column=0, columnspan=2, sticky="w", padx=4, pady=2
        )
        self._add_copy_menu(self._session_label)
        self._tooltip(
            self._session_label,
            "Number of corrections saved during the current session.",
        )

        # ── Keyboard legend ───────────────────────────────────────
        ttk.Separator(inspector).grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=4
        )
        kb_text = (
            "Shortcuts: A=Accept  D=Reject\n"
            "R=Relabel  M=Merge\n"
            "Esc=Deselect  \u2190\u2192 Cycle\n"
            "F=Fit  +/- Zoom  Scroll=Pan\n"
            "Ctrl+Z/Y Undo/Redo\n"
            "Shift+Click Multi-select boxes\n"
            "Ctrl+Click Select words\n"
            "Ctrl+A Select all\n"
            "Ctrl+C Copy box/word text\n"
            "Ctrl+V Paste box\n"
            "Ctrl+Drag Draw new box\n"
            "G=Group  L=Link Column\n"
            "W=Words  X=Skip/Dismiss\n"
            "Right-click: menu"
        )
        ttk.Label(
            inspector, text=kb_text, foreground="gray", font=("TkDefaultFont", 8)
        ).grid(row=5, column=0, columnspan=2, sticky="w", padx=4)

        # ── Status bar ────────────────────────────────────────────
        self._status = ttk.Label(self.frame, text="Ready", relief="sunken", anchor="w")
        self._status.grid(row=2, column=0, sticky="ew", padx=2)
        self._add_copy_menu(self._status)
        self._tooltip(self._status, "Current status and last operation result")

        # ── Progress bar ──────────────────────────────────────────
        self._progress_var = tk.DoubleVar(value=0)
        self._progress = ttk.Progressbar(
            self.frame, variable=self._progress_var, maximum=100
        )
        self._progress.grid(row=3, column=0, sticky="ew", padx=2, pady=(0, 2))
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
        self.root.bind("<Key-x>", self._key_dismiss)

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
        # Refresh the project and run dropdowns to include new runs
        self._refresh_project_dropdown()
        if self._doc_id:
            self._refresh_run_dropdown(self._doc_id)
        if self._pdf_path:
            self._navigate_to_page()

    # ── Project / Run browsing ─────────────────────────────────────

    def _refresh_project_dropdown(self) -> None:
        """Populate the Project combobox from the correction store."""
        self._reopen_store()
        docs = self._store.get_all_documents()
        self._project_doc_ids = [d["doc_id"] for d in docs]
        labels: list[str] = []
        for d in docs:
            tag = d.get("project_tag") or ""
            runs = d.get("run_count", 0)
            name = d.get("filename", "?")
            if tag:
                labels.append(f"{name} ({tag}) — {runs} run(s)")
            else:
                labels.append(f"{name} — {runs} run(s)")
        self._project_combo["values"] = labels
        # Auto-select current doc if loaded
        if self._doc_id and self._doc_id in self._project_doc_ids:
            idx = self._project_doc_ids.index(self._doc_id)
            self._project_combo.current(idx)

    def _sync_dropdowns(self) -> None:
        """Refresh Project and Run dropdowns to match the current document."""
        self._refresh_project_dropdown()
        if self._doc_id:
            self._refresh_run_dropdown(self._doc_id)

    def _on_project_selected(self, _event: Any = None) -> None:
        """Handle selection of a document from the Project dropdown."""
        idx = self._project_combo.current()
        if idx < 0 or idx >= len(self._project_doc_ids):
            return
        doc_id = self._project_doc_ids[idx]
        self._reopen_store()
        docs = self._store.get_all_documents()
        doc_info = next((d for d in docs if d["doc_id"] == doc_id), None)
        if not doc_info:
            return
        # Resolve PDF path
        pdf_path_str = doc_info.get("pdf_path", "")
        pdf_path = (
            Path(pdf_path_str) if pdf_path_str and Path(pdf_path_str).exists() else None
        )
        self._load_document(
            pdf_path,
            doc_id=doc_id,
            page_count=doc_info.get("page_count", 0),
        )

    def _refresh_run_dropdown(self, doc_id: str) -> None:
        """Populate the Run combobox for a given document."""
        self._reopen_store()
        runs = self._store.get_runs_for_doc(doc_id)
        self._run_ids = [r["run_id"] for r in runs]
        self._run_pages: dict[str, list[int]] = {}
        labels: list[str] = []
        for r in runs:
            tag = r.get("tag") or ""
            run_id = r["run_id"]
            pages = r.get("pages_processed", [])
            self._run_pages[run_id] = sorted(pages)
            suffix = f" [{len(pages)}p]"
            if tag:
                labels.append(f"{run_id} ({tag}){suffix}")
            else:
                labels.append(f"{run_id}{suffix}")
        self._run_combo["values"] = labels
        if labels:
            self._run_combo.current(0)  # select latest run
        self._selected_run_id: str | None = self._run_ids[0] if self._run_ids else None
        # Navigate to the first processed page of the selected run
        self._jump_to_run_page()

    def _on_run_selected(self, _event: Any = None) -> None:
        """Handle selection of a run from the Run dropdown."""
        idx = self._run_combo.current()
        if idx < 0 or idx >= len(self._run_ids):
            return
        self._selected_run_id = self._run_ids[idx]
        self._jump_to_run_page()

    def _jump_to_run_page(self) -> None:
        """Navigate to the first processed page of the currently selected run."""
        run_id = self._selected_run_id
        pages = self._run_pages.get(run_id or "", []) if run_id else []
        if pages:
            self._page_var.set(pages[0])
        self._navigate_to_page()

    def _on_offline_mode_toggle(self) -> None:
        """Re-render the current page when the Offline toggle changes."""
        if self._offline_mode_var.get():
            # Turning offline on disables the word overlay
            self._word_overlay_var.set(False)
            self._word_overlay_on = False
            self._clear_word_overlay()
        self._navigate_to_page()

    def _edit_project_tag(self) -> None:
        """Prompt the user to set/change the project tag."""
        if not self._doc_id:
            self._status.configure(text="Load a document first")
            return
        self._reopen_store()
        docs = self._store.get_all_documents()
        doc = next((d for d in docs if d["doc_id"] == self._doc_id), None)
        current_tag = doc.get("project_tag", "") if doc else ""
        new_tag = simpledialog.askstring(
            "Project Tag",
            "Enter a project tag for this document:",
            initialvalue=current_tag,
            parent=self.root,
        )
        if new_tag is not None:
            self._store.update_project_tag(self._doc_id, new_tag.strip())
            self._refresh_project_dropdown()
