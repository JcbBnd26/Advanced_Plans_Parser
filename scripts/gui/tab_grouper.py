"""Grouper tab — interactive GlyphBox grouping and pattern capture.

Provides three modes accessible via a toolbar:

- **Learn Session** — clean canvas, build groups from scratch by
  Shift+clicking word boxes and confirming them as groups.
- **Edit Mode** — machine groupings shown; tweak boundaries with
  add/remove/split/merge gestures.
- **Inspect Mode** — single-click reveals group membership (read-only).

Every confirmed group writes an IsGroup fingerprint to the
``group_snapshots`` table and a correction signal to the
``group_corrections`` table via ``CorrectionStore``.

Architecture
------------
``GrouperTab`` composes three mixins:

- :mod:`.grouper_canvas`  — ``GrouperCanvasMixin`` (all drawing)
- :mod:`.grouper_events`  — ``GrouperEventsMixin`` (all mouse events)

Session state is held in a :class:`.grouper_state.GrouperSessionState`
instance (``self._gsession``) shared across mixins.
"""

from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Optional
from uuid import uuid4

from plancheck.config import GroupingConfig
from plancheck.corrections.store import CorrectionStore

from .grouper_canvas import GrouperCanvasMixin
from .grouper_events import GrouperEventsMixin
from .grouper_state import GROUPER_MODES, GrouperSessionState

log = logging.getLogger(__name__)

# ── Palette ───────────────────────────────────────────────────────────────────
_COLOR_GLYPH_OUTLINE = "#888888"
_COLOR_SELECTION = "#FFD700"
_COLOR_CONFIRMED = "#22AA44"
_COLOR_MACHINE = "#4488CC"
_COLOR_INSPECT = "#00CCCC"


class GrouperTab(GrouperCanvasMixin, GrouperEventsMixin):
    """Grouper tab: interactive GlyphBox grouping and IsGroup capture.

    Parameters
    ----------
    notebook:
        Parent ``ttk.Notebook`` that hosts the tab.
    gui_state:
        Shared ``GuiState`` instance from the master ``AppGui``.
    """

    def __init__(self, notebook: ttk.Notebook, gui_state: Any) -> None:
        self.notebook = notebook
        self.state = gui_state
        self.root = notebook.winfo_toplevel()

        # ── Session state ─────────────────────────────────────────────
        self._gsession = GrouperSessionState()
        self._cfg: GroupingConfig = GroupingConfig()

        # ── CorrectionStore ───────────────────────────────────────────
        db_path: Optional[Path] = (
            gui_state.db_path() if hasattr(gui_state, "db_path") else None
        )
        self._store = CorrectionStore(db_path=db_path)

        # ── Canvas rendering bookkeeping ──────────────────────────────
        self._scale: float = 1.0  # PDF-point → canvas-pixel multiplier
        self._zoom: float = 1.0
        self._photo: Any = None  # ImageTk.PhotoImage — must stay alive

        # ── Main frame ────────────────────────────────────────────────
        self.frame = ttk.Frame(notebook)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(2, weight=1)  # row 0=toolbar, 1=controls, 2=canvas
        notebook.add(self.frame, text="Grouper")

        self._build_toolbar()
        self._build_control_bar()
        self._build_canvas_area()
        self._build_status_bar()

        # React to the active Grouper mode
        self._mode_var.trace_add("write", lambda *_: self._on_mode_changed())

    # ── Toolbar ───────────────────────────────────────────────────────────────

    def _build_toolbar(self) -> None:
        """Top toolbar: PDF selector, page counter, and mode radio buttons."""
        bar = ttk.Frame(self.frame, padding=(4, 4))
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(5, weight=1)  # spacer

        # PDF selector
        ttk.Button(bar, text="Open PDF…", command=self._browse_pdf).grid(
            row=0, column=0, padx=(0, 6)
        )
        self._pdf_label = ttk.Label(bar, text="No PDF loaded", foreground="#888888")
        self._pdf_label.grid(row=0, column=1, padx=(0, 12))

        # Page navigation
        ttk.Label(bar, text="Page:").grid(row=0, column=2)
        self._page_var = tk.IntVar(value=0)
        self._page_spin = ttk.Spinbox(
            bar,
            from_=0,
            to=0,
            width=5,
            textvariable=self._page_var,
            command=self._on_page_spin,
        )
        self._page_spin.grid(row=0, column=3, padx=2)
        self._page_count_label = ttk.Label(bar, text="/ 0")
        self._page_count_label.grid(row=0, column=4, padx=(0, 12))

        # Mode selector
        self._mode_var = tk.StringVar(value="learn")
        ttk.Label(bar, text="Mode:").grid(row=0, column=6, padx=(0, 4))
        for col_offset, (mode_key, mode_label) in enumerate(
            [("learn", "Learn"), ("edit", "Edit")]
        ):
            ttk.Radiobutton(
                bar,
                text=mode_label,
                variable=self._mode_var,
                value=mode_key,
            ).grid(row=0, column=7 + col_offset, padx=2)

    # ── Control bar ───────────────────────────────────────────────────────────

    def _build_control_bar(self) -> None:
        """Context-sensitive control bar below the toolbar."""
        self._ctrl_bar = ttk.Frame(self.frame, padding=(4, 2))
        self._ctrl_bar.grid(row=1, column=0, sticky="ew")

        # Show Machine Groups toggle (visible in all modes)
        self._show_machine_var = tk.BooleanVar(value=False)
        self._machine_toggle = ttk.Checkbutton(
            self._ctrl_bar,
            text="Show Machine Groups",
            variable=self._show_machine_var,
            command=self._on_toggle_machine_groups,
        )
        self._machine_toggle.grid(row=0, column=0, padx=(0, 16))

        # Learn/Edit session controls
        self._btn_save_next = ttk.Button(
            self._ctrl_bar, text="Save & Next", command=self._save_and_next
        )
        self._btn_save_next.grid(row=0, column=1, padx=4)

        self._btn_clear = ttk.Button(
            self._ctrl_bar, text="Clear Groupings", command=self._clear_groupings
        )
        self._btn_clear.grid(row=0, column=2, padx=4)

        self._btn_abort = ttk.Button(
            self._ctrl_bar, text="Abort Session", command=self._abort_session
        )
        self._btn_abort.grid(row=0, column=3, padx=4)

        self._set_session_controls_enabled(False)

    # ── Canvas area ───────────────────────────────────────────────────────────

    def _build_canvas_area(self) -> None:
        """Canvas + scrollbars in the main body area."""
        container = ttk.Frame(self.frame)
        container.grid(row=2, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        self._canvas = tk.Canvas(container, background="#2b2b2b", cursor="crosshair")
        self._canvas.grid(row=0, column=0, sticky="nsew")

        vscroll = ttk.Scrollbar(
            container, orient="vertical", command=self._canvas.yview
        )
        vscroll.grid(row=0, column=1, sticky="ns")
        hscroll = ttk.Scrollbar(
            container, orient="horizontal", command=self._canvas.xview
        )
        hscroll.grid(row=1, column=0, sticky="ew")

        self._canvas.configure(
            yscrollcommand=vscroll.set,
            xscrollcommand=hscroll.set,
        )

        # Placeholder prompt shown before any PDF is loaded
        self._canvas.create_text(
            10,
            10,
            text="Open a PDF to start a Grouper session.",
            anchor="nw",
            fill="#666666",
            font=("Segoe UI", 12),
            tags="prompt",
        )

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_status_bar(self) -> None:
        status_frame = ttk.Frame(self.frame, relief="sunken", padding=(4, 2))
        status_frame.grid(row=3, column=0, sticky="ew")
        self._status_label = ttk.Label(status_frame, text="Ready", anchor="w")
        self._status_label.pack(fill="x")

    def _set_status(self, text: str) -> None:
        self._status_label.configure(text=text)

    # ── PDF loading ───────────────────────────────────────────────────────────

    def _browse_pdf(self) -> None:
        f = filedialog.askopenfilename(
            title="Select PDF for Grouper session",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")],
            initialdir=str(Path("input")),
        )
        if not f:
            return
        self._load_pdf(Path(f))

    def _load_pdf(self, pdf_path: Path) -> None:
        """Initialise a new session for the given PDF."""
        import pdfplumber

        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("PDF Error", f"Cannot open PDF:\n{exc}")
            return

        # Register doc in CorrectionStore to get doc_id
        try:
            doc_id = self._store.register_document(pdf_path)
        except Exception as exc:  # noqa: BLE001
            log.error("grouper: failed to register document", exc_info=True)
            doc_id = ""

        # Initialise session state
        self._gsession = GrouperSessionState(
            session_id=uuid4().hex,
            pdf_path=pdf_path,
            current_page=0,
            page_count=page_count,
            mode=self._mode_var.get(),
            doc_id=doc_id,
            active=True,
        )

        # Update toolbar
        self._pdf_label.configure(text=pdf_path.name, foreground="#dddddd")
        self._page_spin.configure(to=max(0, page_count - 1))
        self._page_var.set(0)
        self._page_count_label.configure(text=f"/ {page_count}")
        self._set_session_controls_enabled(True)

        log.info("grouper: session started — %s (%d pages)", pdf_path.name, page_count)
        self._load_current_page()

    def _load_current_page(self) -> None:
        """Run grouper pipeline for the current page and refresh the canvas."""
        if self._gsession.pdf_path is None:
            return

        page = self._gsession.current_page
        self._set_status(f"Loading page {page}…")
        self.root.update_idletasks()

        try:
            from plancheck.grouping.grouper_pipeline import run_grouper_pipeline

            data = run_grouper_pipeline(self._gsession.pdf_path, page, self._cfg)
        except Exception as exc:  # noqa: BLE001
            log.error("grouper: pipeline failed for page %d", page, exc_info=True)
            self._set_status(f"Error loading page {page}: {exc}")
            return

        self._gsession.page_data = data
        self._scale = 1.0  # canvas renderer will set real scale in Phase E
        self._render_canvas()
        n_groups = len(self._gsession.confirmed_groups)
        self._set_status(
            f"Page {page} — {len(data.boxes)} tokens, "
            f"{len(data.machine_groups)} machine groups, "
            f"{n_groups} confirmed"
        )

    # ── Canvas rendering ──────────────────────────────────────────────────────

    def _render_canvas(self) -> None:
        """Full canvas redraw.  Delegated to GrouperCanvasMixin in Phase E."""
        # Phase D stub: just clear and show token count as text
        self._canvas.delete("all")
        if self._gsession.page_data is None:
            return
        n = len(self._gsession.page_data.boxes)
        self._canvas.create_text(
            10,
            10,
            text=f"Page {self._gsession.current_page} — {n} tokens loaded.\n"
            "Canvas rendering implemented in Phase E.",
            anchor="nw",
            fill="#aaaaaa",
            font=("Segoe UI", 11),
        )

    # ── Session controls ──────────────────────────────────────────────────────

    def _on_page_spin(self) -> None:
        """Spinbox command: navigate to the selected page number."""
        if not self._gsession.active:
            return
        new_page = self._page_var.get()
        if new_page == self._gsession.current_page:
            return
        self._gsession.reset_page_state()
        self._gsession.current_page = new_page
        self._load_current_page()

    def _on_mode_changed(self) -> None:
        new_mode = self._mode_var.get()
        if new_mode not in GROUPER_MODES:
            return
        self._gsession.mode = new_mode
        log.debug("grouper: mode → %s", new_mode)
        self._render_canvas()

    def _on_toggle_machine_groups(self) -> None:
        self._gsession.show_machine_groups = self._show_machine_var.get()
        self._render_canvas()

    def _save_and_next(self) -> None:
        """Persist current page groups and advance to next page."""
        if not self._gsession.active:
            return
        self._gsession.commit_page()
        n_saved = len(self._gsession.saved_pages.get(self._gsession.current_page, []))
        log.info(
            "grouper: saved page %d (%d groups)",
            self._gsession.current_page,
            n_saved,
        )
        next_page = self._gsession.current_page + 1
        if next_page >= self._gsession.page_count:
            self._set_status("Last page reached — all pages saved.")
            self._set_session_controls_enabled(False)
            return
        self._gsession.reset_page_state()
        self._gsession.current_page = next_page
        self._page_var.set(next_page)
        self._load_current_page()

    def _clear_groupings(self) -> None:
        """Clear confirmed groups on the current page (with confirmation)."""
        if not self._gsession.active:
            return
        if not self._gsession.confirmed_groups:
            return
        if not messagebox.askyesno(
            "Clear Groupings",
            f"Clear all {len(self._gsession.confirmed_groups)} group(s) on "
            f"page {self._gsession.current_page}?\n\nThis cannot be undone.",
        ):
            return
        self._gsession.confirmed_groups.clear()
        self._gsession.selected_indices.clear()
        self._gsession.inspected_group_idx = None
        self._render_canvas()
        self._set_status(f"Cleared groupings on page {self._gsession.current_page}.")

    def _abort_session(self) -> None:
        """End session, preserving all previously saved pages."""
        if not self._gsession.active:
            return
        total_pages = len(self._gsession.saved_pages)
        total_groups = sum(len(gs) for gs in self._gsession.saved_pages.values())
        self._gsession.active = False
        self._set_session_controls_enabled(False)
        log.info(
            "grouper: session aborted — %d pages saved, %d groups total",
            total_pages,
            total_groups,
        )
        self._set_status(
            f"Session aborted. {total_pages} page(s) / {total_groups} group(s) kept."
        )

    def _set_session_controls_enabled(self, enabled: bool) -> None:
        """Enable or disable session-specific control bar buttons."""
        state = "normal" if enabled else "disabled"
        for widget in (
            self._btn_save_next,
            self._btn_clear,
            self._btn_abort,
            self._page_spin,
        ):
            widget.configure(state=state)
