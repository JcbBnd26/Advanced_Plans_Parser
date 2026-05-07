"""
Advanced Plan Parser – Master GUI Controller.

Slim orchestrator that wires up seven tabs via a shared GuiState.

Tabs
----
1. Pipeline         – PDF selection, OCR toggles, config knobs, run with embedded log
2. Runs & Reports   – Browse runs/, inspect manifests, open reports/artifacts
3. Database         – Browse CorrectionStore data across all documents and runs
4. Diagnostics      – Font metrics, benchmark, VOCRPP, tuning, grouping playground
5. Sheet Recreation – Generate text-only PDFs from runs
6. ML Trainer       – Interactive detection correction UI for ML training
7. Query            – Natural-language query interface for plan data

Architecture
------------
GuiState holds shared state (pdf_path, last_run_dir, config) and a simple
pub/sub mechanism so tabs can react to events (pdf_changed, run_completed,
load_config) without direct coupling.
"""

from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import TYPE_CHECKING, Any

from plancheck.config import GroupingConfig
from plancheck.config.constants import DEFAULT_CORRECTIONS_DB

if TYPE_CHECKING:
    from plancheck.corrections.experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)


def _format_startup_retrain_status(result: object) -> str:
    """Format startup retrain status for the GUI status bar."""
    if result is None:
        return "Startup retrain skipped — no corrections database found"

    if getattr(result, "error", ""):
        return f"Startup retrain failed: {result.error}"

    if getattr(result, "rolled_back", False):
        return "Startup retrain rolled back — F1 regressed"

    if not getattr(result, "retrained", False):
        new_corrections = getattr(result, "new_corrections", 0)
        threshold = getattr(result, "threshold", 0)
        return (
            "Startup retrain not needed " f"({new_corrections}/{threshold} corrections)"
        )

    stage1_f1 = getattr(result, "metrics", {}).get("f1_weighted", 0.0)
    msg = f"Auto-retrained on startup (S1 F1: {stage1_f1:.1%})"

    if getattr(result, "stage2_trained", False):
        stage2_f1 = getattr(result, "stage2_metrics", {}).get("f1_weighted", 0.0)
        msg += f"; S2 F1: {stage2_f1:.1%}"
    elif getattr(result, "stage2_error", ""):
        msg += f"; S2 failed: {result.stage2_error}"
    elif getattr(result, "stage2_skipped_reason", ""):
        msg += f"; S2 skipped: {result.stage2_skipped_reason}"

    return msg


# ---------------------------------------------------------------------------
# GuiState – shared state + pub/sub
# ---------------------------------------------------------------------------


class GuiState:
    """Lightweight shared state object passed to every tab.

    Attributes
    ----------
    pdf_path : Path | None
        Currently selected PDF file.
    last_run_dir : Path | None
        Most recently completed run directory.
    config : GroupingConfig
        Working configuration (loaded/saved by Pipeline tab).

    Events
    ------
    * ``pdf_changed``   – fired when a new PDF is selected
    * ``run_completed`` – fired after a pipeline run finishes
    * ``load_config``   – fired when a config file is loaded
    """

    def __init__(self) -> None:
        self.pdf_path: Path | None = None
        self.last_run_dir: Path | None = None
        self.config: GroupingConfig = GroupingConfig()
        self.config_file_path: Path | None = None
        self.pending_config: dict | None = None
        self._subscribers: dict[str, list] = {}
        self._error_display_cb: Any = None
        self.experiment_tracker: ExperimentTracker | None = None
        self.project_dir: Path | None = None
        self.project_meta: dict | None = None

        # Tab visibility state (persisted to app config)
        self.tab_visibility: dict[str, bool] = {
            "database": True,
            "diagnostics": True,
        }

    def subscribe(self, event: str, callback) -> None:
        self._subscribers.setdefault(event, []).append(callback)

    def set_error_display(self, callback) -> None:
        """Register a callback to surface internal errors in the GUI."""
        self._error_display_cb = callback

    def notify(self, event: str) -> None:
        for cb in self._subscribers.get(event, []):
            try:
                cb()
            except (
                Exception
            ) as exc:  # noqa: BLE001 — subscribers must not break event loop
                logger.error(
                    "GuiState subscriber failed for event=%s callback=%r",
                    event,
                    cb,
                    exc_info=True,
                )
                if self._error_display_cb:
                    try:
                        self._error_display_cb(
                            f"Internal error in {event} handler: {exc}"
                        )
                    except Exception:  # noqa: BLE001
                        pass

    def set_pdf(self, path: Path | None) -> None:
        self.pdf_path = path
        self.notify("pdf_changed")

    def set_last_run(self, run_dir: Path) -> None:
        self.last_run_dir = run_dir
        self.notify("run_completed")

    def set_config(
        self,
        config: GroupingConfig,
        *,
        config_file_path: Path | None = None,
    ) -> None:
        """Replace the working config and remember its source path."""
        self.config = GroupingConfig.from_dict(config.to_dict())
        self.config_file_path = config_file_path

    def queue_config_load(
        self,
        config_dict: dict,
        *,
        config_file_path: Path | None = None,
    ) -> None:
        """Queue a config snapshot for the Pipeline tab to consume."""
        self.pending_config = dict(config_dict)
        self.config_file_path = config_file_path
        self.notify("load_config")

    def set_project(self, project_dir: Path) -> None:
        """Activate a project: load metadata, build config, fire event.

        Parameters
        ----------
        project_dir
            Directory containing ``project.json``.
        """
        from plancheck.config.project import (
            add_recent_project,
            build_project_config,
            load_project,
        )

        meta = load_project(project_dir)
        cfg = build_project_config(project_dir)

        self.project_dir = Path(project_dir)
        self.project_meta = meta
        self.config = cfg

        add_recent_project(project_dir, meta["name"])
        logger.info("Activated project %r at %s", meta["name"], project_dir)
        self.notify("project_changed")

    def db_path(self) -> Path:
        """Resolve the corrections database path for the active context."""
        if self.project_dir:
            return self.project_dir / "corrections.db"
        return DEFAULT_CORRECTIONS_DB


# ---------------------------------------------------------------------------
# Main GUI class
# ---------------------------------------------------------------------------


class PlanParserGUI:
    """Master controller: window setup, tab assembly, keyboard shortcuts."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Advanced Plans Parser")
        self.root.geometry("1100x850")
        self.root.minsize(900, 700)
        self.root.resizable(True, True)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.state = GuiState()

        self._build_ui()
        self._bind_shortcuts()

        # Run startup check for auto-retrain in background
        self.root.after(500, self._run_startup_check)

        # Shutdown safety: cancel background workers before destroying the root.
        try:
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to set WM_DELETE_WINDOW protocol", exc_info=True)

    def _run_startup_check(self) -> None:
        """Check if auto-retrain is needed on startup."""
        import threading

        startup_retrain_enabled = getattr(
            self.state.config,
            "ml_retrain_on_startup",
            False,
        )

        if startup_retrain_enabled:
            self._status_bar.set_status("Startup retrain check running...")

        def _check():
            try:
                from plancheck.corrections.retrain_trigger import startup_check

                result = startup_check(self.state.config)
                if startup_retrain_enabled:
                    msg = _format_startup_retrain_status(result)
                    self.root.after(0, lambda: self._status_bar.set_status(msg))
            except Exception:  # noqa: BLE001 — startup check is best-effort
                logger.debug("Startup check failed", exc_info=True)
                if startup_retrain_enabled:
                    self.root.after(
                        0,
                        lambda: self._status_bar.set_status(
                            "Startup retrain check failed"
                        ),
                    )

        threading.Thread(target=_check, daemon=True).start()

    def _on_close(self) -> None:
        """Best-effort shutdown: cancel workers, then close the window."""
        for tab in (
            getattr(self, "_pipeline_tab", None),
            getattr(self, "_diagnostics_tab", None),
            getattr(self, "_recreation_tab", None),
            getattr(self, "_annotation_tab", None),
            getattr(self, "_query_tab", None),
        ):
            if not tab:
                continue

            request_cancel = getattr(tab, "request_cancel", None)
            if callable(request_cancel):
                try:
                    request_cancel()
                except Exception:  # noqa: BLE001
                    logger.debug("Tab cancel request failed", exc_info=True)

            worker = getattr(tab, "_worker", None)
            if worker and hasattr(worker, "cancel"):
                try:
                    worker.cancel()
                except Exception:  # noqa: BLE001
                    logger.debug("Worker cancel failed", exc_info=True)
        try:
            self.root.destroy()
        except Exception:  # noqa: BLE001
            logger.debug("Root destroy failed", exc_info=True)

    def _setup_theme(self) -> None:
        """Apply consistent ttk styling across the application."""
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass  # fall back to default if clam unavailable

        # -- Colours ------------------------------------------------
        bg = "#1e1e1e"
        surface = "#252526"
        surface_alt = "#2d2d30"
        border = "#3c3c3c"
        fg = "#d4d4d4"
        fg_dim = "#808080"
        accent = "#0078d4"

        self.root.configure(bg=bg)

        style.configure(
            ".",
            background=surface,
            foreground=fg,
            bordercolor=border,
            focuscolor=accent,
            font=("TkDefaultFont", 9),
        )
        style.configure("TFrame", background=surface)
        style.configure("TLabel", background=surface, foreground=fg)
        style.configure("TLabelframe", background=surface, foreground=fg)
        style.configure("TLabelframe.Label", background=surface, foreground=fg)
        style.configure("TButton", padding=(8, 4))
        style.configure("TNotebook", background=bg, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(10, 4), font=("Segoe UI", 10))
        style.map(
            "TNotebook.Tab",
            background=[("selected", surface), ("!selected", surface_alt)],
            foreground=[("selected", fg), ("!selected", fg_dim)],
        )
        style.configure("TEntry", fieldbackground=surface_alt, foreground=fg)
        style.configure("TCombobox", fieldbackground=surface_alt, foreground=fg)
        style.configure("TCheckbutton", background=surface, foreground=fg)
        style.configure(
            "Treeview",
            background=surface_alt,
            foreground=fg,
            fieldbackground=surface_alt,
        )
        style.configure("Treeview.Heading", background=surface, foreground=fg)
        style.map(
            "Treeview",
            background=[("selected", accent)],
            foreground=[("selected", "#ffffff")],
        )
        # Run button (already referenced in tab_pipeline.py)
        style.configure(
            "Run.TButton", font=("TkDefaultFont", 12, "bold"), padding=(20, 10)
        )

    def _build_ui(self) -> None:
        self._setup_theme()

        # ── Menu bar ──────────────────────────────────────────────────
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # ── File menu (preserved from project system work) ────────────
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(
            label="New Project…",
            command=self._new_project,
            accelerator="Ctrl+N",
        )
        file_menu.add_command(
            label="Open Project…",
            command=self._open_project,
            accelerator="Ctrl+O",
        )

        # Recent Projects submenu (populated dynamically)
        self._recent_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Recent Projects", menu=self._recent_menu)

        file_menu.add_separator()
        file_menu.add_command(
            label="Export Project…",
            command=self._export_project,
        )
        file_menu.add_command(
            label="Import Project…",
            command=self._import_project,
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Exit",
            command=self._on_close,
            accelerator="Alt+F4",
        )

        # ── Edit menu ────────────────────────────────────────────────
        self._edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=self._edit_menu)
        self._edit_menu.add_command(
            label="Undo",
            command=self._edit_undo,
            accelerator="Ctrl+Z",
            state="disabled",
        )
        self._edit_menu.add_command(
            label="Redo",
            command=self._edit_redo,
            accelerator="Ctrl+Y",
            state="disabled",
        )

        # ── View menu ───────────────────────────────────────────────
        self._view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=self._view_menu)

        self._view_menu.add_command(
            label="Pipeline",
            command=lambda: self._focus_tab(0),
        )
        self._view_menu.add_command(
            label="ML Trainer",
            command=lambda: self._focus_tab_by_key("annotation"),
        )
        self._view_menu.add_separator()

        self._db_visible_var = tk.BooleanVar(
            value=self.state.tab_visibility.get("database", True)
        )
        self._view_menu.add_checkbutton(
            label="Database Inspector",
            variable=self._db_visible_var,
            command=lambda: self._toggle_tab_visibility(
                "database", self._db_visible_var
            ),
        )
        self._diag_visible_var = tk.BooleanVar(
            value=self.state.tab_visibility.get("diagnostics", True)
        )
        self._view_menu.add_checkbutton(
            label="Diagnostics Panel",
            variable=self._diag_visible_var,
            command=lambda: self._toggle_tab_visibility(
                "diagnostics", self._diag_visible_var
            ),
        )

        # ── Settings menu ───────────────────────────────────────────
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(
            label="General…",
            command=self._open_general_settings,
        )
        settings_menu.add_separator()
        settings_menu.add_command(
            label="ML Runtime…",
            command=self._open_ml_runtime_settings,
        )
        settings_menu.add_command(
            label="ML Features…",
            command=self._open_ml_features_settings,
        )
        settings_menu.add_command(
            label="LLM Configuration…",
            command=self._open_llm_config_settings,
        )
        settings_menu.add_separator()
        settings_menu.add_command(
            label="Pipeline Defaults…",
            command=self._open_pipeline_defaults,
        )

        # ── Help menu ───────────────────────────────────────────────
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(
            label="About PlanCheck",
            command=self._show_about,
        )
        help_menu.add_command(
            label="Documentation",
            command=lambda: self._status_bar.set_status(
                "Documentation — not yet available"
            ),
        )

        # ── Tab container ─────────────────────────────────────────────
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        from .tab_pipeline import PipelineTab
        from .widgets import StatusBar

        # Build PipelineTab eagerly — it is the first visible tab and
        # its construction time is covered by the splash screen.
        self._pipeline_tab = PipelineTab(self.notebook, self.state)

        # Track original tab positions and widgets for hide/show
        # Maps key → (original_index, tab_widget_or_placeholder, tab_text)
        self._tab_registry: dict[str, dict] = {}
        self._tab_key_by_index: dict[int, str] = {}

        # All other tabs are lazy: add lightweight placeholders now and
        # import + build the real tab on first selection.  This avoids
        # heavy imports (PIL, shapely, sklearn, pdfplumber) and large
        # widget trees until the user actually needs them.
        self._lazy_tab_defs: dict[int, tuple[str, str]] = {}  # idx → (attr, key)
        self._lazy_placeholders: dict[int, ttk.Frame] = {}

        _lazy_specs: list[tuple[str, str, str]] = [
            ("Runs & Reports", "_runs_tab", "runs"),
            ("Database", "_database_tab", "database"),
            ("Diagnostics", "_diagnostics_tab", "diagnostics"),
            ("Sheet Recreation", "_recreation_tab", "recreation"),
            ("ML Trainer", "_annotation_tab", "annotation"),
            ("Grouper", "_grouper_tab", "grouper"),
            ("Query", "_query_tab", "query"),
        ]

        for tab_text, attr_name, key in _lazy_specs:
            placeholder = ttk.Frame(self.notebook)
            ttk.Label(
                placeholder,
                text="Loading…",
                font=("Segoe UI", 10),
                foreground="gray",
            ).place(relx=0.5, rely=0.5, anchor="center")
            self.notebook.add(placeholder, text=tab_text)
            idx = self.notebook.index("end") - 1
            self._lazy_tab_defs[idx] = (attr_name, key)
            self._lazy_placeholders[idx] = placeholder
            self._tab_registry[key] = {
                "original_index": idx,
                "widget": placeholder,
                "text": tab_text,
                "visible": True,
            }
            self._tab_key_by_index[idx] = key

        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed, add="+")

        # ── Status bar ────────────────────────────────────────────────
        self._status_bar = StatusBar(self.root)
        self._status_bar.grid(row=1, column=0, sticky="ew")

        # Surface internal event-handler errors in the status bar
        self.state.set_error_display(lambda msg: self._status_bar.set_status(msg))

        # Update status when state changes
        self.state.subscribe("pdf_changed", self._update_status)
        self.state.subscribe("run_completed", self._update_status)
        self.state.subscribe("project_changed", self._on_project_changed)

    # ------------------------------------------------------------------
    # Tab changed handler (lazy loading + Edit menu state)
    # ------------------------------------------------------------------

    def _on_tab_changed(self, event: Any) -> None:
        """Build lazy tabs and update Edit menu state on tab switch."""
        self._on_lazy_tab(event)
        self._update_edit_menu_state()

    def _on_lazy_tab(self, _event: Any) -> None:
        """Build a tab on first selection, replacing its placeholder."""
        idx = self.notebook.index("current")
        if idx not in self._lazy_tab_defs:
            return

        attr_name, key = self._lazy_tab_defs.pop(idx)
        placeholder = self._lazy_placeholders.pop(idx)

        tab = self._materialise_tab(key)
        setattr(self, attr_name, tab)

        # The tab's __init__ already appended its frame to the notebook
        # (via notebook.add).  Swap it into the placeholder's position.
        self.notebook.forget(placeholder)
        placeholder.destroy()
        self.notebook.insert(idx, tab.frame)
        self.notebook.select(idx)

        # Update tab registry after materialisation
        if key in self._tab_registry:
            self._tab_registry[key]["widget"] = tab.frame

    def _materialise_tab(self, key: str) -> Any:
        """Import and construct a single lazy tab."""
        if key == "runs":
            from .tab_runs import RunsTab

            return RunsTab(self.notebook, self.state)
        if key == "database":
            from .tab_database import DatabaseTab

            return DatabaseTab(self.notebook, gui_state=self.state)
        if key == "diagnostics":
            from .tab_diagnostics import DiagnosticsTab

            return DiagnosticsTab(self.notebook, self.state)
        if key == "recreation":
            from .tab_recreation import RecreationTab

            return RecreationTab(self.notebook, self.state)
        if key == "annotation":
            from .tab_annotation import AnnotationTab

            return AnnotationTab(self.notebook, gui_state=self.state)
        if key == "grouper":
            from .tab_grouper import GrouperTab

            return GrouperTab(self.notebook, gui_state=self.state)
        if key == "query":
            from .tab_query import QueryTab

            return QueryTab(self.notebook, self.state)
        msg = f"Unknown lazy tab key: {key!r}"
        raise ValueError(msg)

    def _bind_shortcuts(self) -> None:
        """Global keyboard shortcuts."""
        self.root.bind("<Control-o>", lambda e: self._quick_open_pdf())
        self.root.bind("<Control-n>", lambda e: self._new_project())
        self.root.bind("<Control-z>", lambda e: self._edit_undo())
        self.root.bind("<Control-y>", lambda e: self._edit_redo())
        self.root.bind("<Control-s>", lambda e: self._context_save())
        self.root.bind("<F5>", lambda e: self._run_pipeline_shortcut())
        self.root.bind("<Control-comma>", lambda e: self._open_general_settings())
        self.root.bind("<Control-Key-1>", lambda e: self.notebook.select(0))
        self.root.bind("<Control-Key-2>", lambda e: self.notebook.select(1))
        self.root.bind("<Control-Key-3>", lambda e: self.notebook.select(2))
        self.root.bind("<Control-Key-4>", lambda e: self.notebook.select(3))
        self.root.bind("<Control-Key-5>", lambda e: self.notebook.select(4))
        self.root.bind("<Control-Key-6>", lambda e: self.notebook.select(5))
        self.root.bind("<Control-Key-7>", lambda e: self.notebook.select(6))

    def _quick_open_pdf(self) -> None:
        f = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")],
            initialdir=str(Path("input")),
        )
        if f:
            self.state.set_pdf(Path(f))

    def _context_save(self) -> None:
        """Ctrl+S: save context-dependently based on active tab."""
        # If we have a config file path, save config
        if self.state.config_file_path and self.state.config:
            try:
                self.state.config.save(self.state.config_file_path)
                self._status_bar.set_center("Config saved")
            except Exception:  # noqa: BLE001 — save is best-effort
                self._status_bar.set_center("Save failed")
            return
        self._status_bar.set_center("Nothing to save")

    def _update_status(self) -> None:
        left_parts = []
        if self.state.project_meta:
            left_parts.append(f"Project: {self.state.project_meta['name']}")
        if self.state.pdf_path:
            left_parts.append(f"PDF: {self.state.pdf_path.name}")
        self._status_bar.set_status(" | ".join(left_parts) if left_parts else "Ready")
        if self.state.last_run_dir:
            self._status_bar.set_center(f"Last run: {self.state.last_run_dir.name}")

    # ------------------------------------------------------------------
    # Edit menu actions
    # ------------------------------------------------------------------

    def _update_edit_menu_state(self) -> None:
        """Enable Undo/Redo only when the ML Trainer tab is active."""
        annotation = getattr(self, "_annotation_tab", None)
        is_annotation_active = False
        if annotation is not None:
            try:
                current = self.notebook.select()
                if current and str(annotation.frame) == str(current):
                    is_annotation_active = True
            except Exception:  # noqa: BLE001
                pass

        state = "normal" if is_annotation_active else "disabled"
        try:
            self._edit_menu.entryconfig("Undo", state=state)
            self._edit_menu.entryconfig("Redo", state=state)
        except Exception:  # noqa: BLE001
            pass

    def _edit_undo(self) -> None:
        """Invoke undo on the annotation tab if active."""
        annotation = getattr(self, "_annotation_tab", None)
        if annotation is None:
            return
        undo_mgr = getattr(annotation, "_undo_manager", None)
        if undo_mgr and hasattr(undo_mgr, "undo"):
            undo_mgr.undo()

    def _edit_redo(self) -> None:
        """Invoke redo on the annotation tab if active."""
        annotation = getattr(self, "_annotation_tab", None)
        if annotation is None:
            return
        undo_mgr = getattr(annotation, "_undo_manager", None)
        if undo_mgr and hasattr(undo_mgr, "redo"):
            undo_mgr.redo()

    # ------------------------------------------------------------------
    # View menu actions — tab visibility
    # ------------------------------------------------------------------

    def _focus_tab(self, index: int) -> None:
        """Select a tab by index."""
        try:
            self.notebook.select(index)
        except Exception:  # noqa: BLE001
            pass

    def _focus_tab_by_key(self, key: str) -> None:
        """Select a tab by its registry key, showing it first if hidden."""
        info = self._tab_registry.get(key)
        if info is None:
            return
        if not info["visible"]:
            # Re-show the tab first
            if key == "database":
                self._db_visible_var.set(True)
            elif key == "diagnostics":
                self._diag_visible_var.set(True)
            self._toggle_tab_visibility(key, tk.BooleanVar(value=True))

        widget = info["widget"]
        try:
            self.notebook.select(widget)
        except Exception:  # noqa: BLE001
            pass

    def _toggle_tab_visibility(self, key: str, var: tk.BooleanVar) -> None:
        """Show or hide a tab based on the View menu checkbutton."""
        info = self._tab_registry.get(key)
        if info is None:
            return

        widget = info["widget"]
        should_show = var.get()

        if should_show and not info["visible"]:
            # Re-insert at the correct position
            target_idx = info["original_index"]
            # Count how many tabs before this one are currently visible
            actual_idx = 0
            for other_key, other_info in sorted(
                self._tab_registry.items(),
                key=lambda x: x[1]["original_index"],
            ):
                if other_info["original_index"] < target_idx and other_info["visible"]:
                    actual_idx += 1
            # +1 for the Pipeline tab which is always at index 0
            actual_idx += 1
            try:
                self.notebook.insert(actual_idx, widget, text=info["text"])
            except Exception:  # noqa: BLE001
                self.notebook.add(widget, text=info["text"])
            info["visible"] = True
        elif not should_show and info["visible"]:
            try:
                self.notebook.forget(widget)
            except Exception:  # noqa: BLE001
                pass
            info["visible"] = False

        self.state.tab_visibility[key] = should_show

    # ------------------------------------------------------------------
    # Settings menu actions
    # ------------------------------------------------------------------

    def _open_general_settings(self) -> None:
        from .settings_dialogs import GeneralSettingsDialog

        GeneralSettingsDialog(self.root, self.state)

    def _open_ml_runtime_settings(self) -> None:
        from .settings_dialogs import MLRuntimeDialog

        dlg = MLRuntimeDialog(self.root, self.state)
        self.root.wait_window(dlg)
        if dlg.applied:
            self.state.notify("ml_config_changed")

    def _open_ml_features_settings(self) -> None:
        from .settings_dialogs import MLFeaturesDialog

        dlg = MLFeaturesDialog(self.root, self.state)
        self.root.wait_window(dlg)
        if dlg.applied:
            self.state.notify("ml_features_changed")

    def _open_llm_config_settings(self) -> None:
        from .settings_dialogs import LLMConfigDialog

        dlg = LLMConfigDialog(self.root, self.state)
        self.root.wait_window(dlg)
        if dlg.applied:
            self.state.notify("llm_config_changed")

    def _open_pipeline_defaults(self) -> None:
        self._status_bar.set_status(
            "Pipeline Defaults — configure via Load/Save Config in the Pipeline tab"
        )

    # ------------------------------------------------------------------
    # Help menu actions
    # ------------------------------------------------------------------

    def _show_about(self) -> None:
        from tkinter import messagebox

        messagebox.showinfo(
            "About PlanCheck",
            "PlanCheck — Architectural/Engineering Plan Parser\n\n"
            "ML-powered construction drawing analysis\n"
            "with OCR reconciliation and LLM integration.\n\n"
            "Python 3.10+",
            parent=self.root,
        )

    # ------------------------------------------------------------------
    # Pipeline shortcut
    # ------------------------------------------------------------------

    def _run_pipeline_shortcut(self) -> None:
        """F5 shortcut: trigger pipeline run via the Pipeline tab."""
        tab = getattr(self, "_pipeline_tab", None)
        if tab and hasattr(tab, "_run_processing"):
            tab._run_processing()

    # ------------------------------------------------------------------
    # Project menu actions
    # ------------------------------------------------------------------

    def _new_project(self) -> None:
        from .project_dialog import NewProjectDialog

        dlg = NewProjectDialog(self.root, self.state)
        self.root.wait_window(dlg)

    def _open_project(self) -> None:
        from .project_dialog import open_project_dialog

        open_project_dialog(self.root, self.state)

    def _export_project(self) -> None:
        from .project_dialog import export_project_dialog

        export_project_dialog(self.root, self.state)

    def _import_project(self) -> None:
        from .project_dialog import import_project_dialog

        import_project_dialog(self.root, self.state)

    def _on_project_changed(self) -> None:
        """React to a project being activated or switched."""
        meta = self.state.project_meta
        if meta:
            self.root.title(f"Advanced Plans Parser — {meta['name']}")
            self._status_bar.set_status(f"Project: {meta['name']}")
        else:
            self.root.title("Advanced Plans Parser")
        self._refresh_recent_menu()

    def _refresh_recent_menu(self) -> None:
        """Rebuild the Recent Projects submenu."""
        self._recent_menu.delete(0, "end")

        from plancheck.config.project import clear_recent_projects, get_recent_projects

        recents = get_recent_projects()
        for entry in recents:
            path = entry["path"]
            name = entry.get("name", Path(path).name)
            self._recent_menu.add_command(
                label=f"{name}  ({path})",
                command=lambda p=path: self._open_recent(p),
            )

        if recents:
            self._recent_menu.add_separator()
            self._recent_menu.add_command(
                label="Clear Recent",
                command=self._clear_recent,
            )

    def _open_recent(self, project_path: str) -> None:
        try:
            self.state.set_project(Path(project_path))
        except Exception as exc:  # noqa: BLE001 — surface to user
            logger.error("Failed to open recent project: %s", exc, exc_info=True)
            from tkinter import messagebox

            messagebox.showerror(
                "Error",
                f"Failed to open project:\n{exc}",
                parent=self.root,
            )

    def _clear_recent(self) -> None:
        from plancheck.config.project import clear_recent_projects

        clear_recent_projects()
        self._refresh_recent_menu()


def main() -> None:
    """Launch the GUI application."""
    _setup_logging()
    try:
        root = tk.Tk()
        _app = PlanParserGUI(root)
        root.mainloop()
    except Exception:
        logger.critical("GUI crashed on startup", exc_info=True)
        _write_crash_log()
        raise


def _setup_logging() -> None:
    """Configure file logging so crashes are visible even under pythonw."""
    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)
    handler = logging.FileHandler(log_path / "gui.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    )
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)


def _write_crash_log() -> None:
    """Write a crash report next to the executable for easy discovery."""
    import traceback

    crash_path = Path("logs") / "gui_crash.txt"
    crash_path.parent.mkdir(exist_ok=True)
    crash_path.write_text(traceback.format_exc(), encoding="utf-8")


if __name__ == "__main__":
    main()
