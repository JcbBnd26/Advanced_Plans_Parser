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
from typing import TYPE_CHECKING

from plancheck.config import GroupingConfig

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

    def _build_ui(self) -> None:
        # ── Tab container ─────────────────────────────────────────────
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        # ── Import and create each tab ────────────────────────────────
        from .tab_annotation import AnnotationTab
        from .tab_database import DatabaseTab
        from .tab_diagnostics import DiagnosticsTab
        from .tab_pipeline import PipelineTab
        from .tab_query import QueryTab
        from .tab_recreation import RecreationTab
        from .tab_runs import RunsTab
        from .widgets import StatusBar

        self._pipeline_tab = PipelineTab(self.notebook, self.state)
        self._runs_tab = RunsTab(self.notebook, self.state)
        self._database_tab = DatabaseTab(self.notebook, gui_state=self.state)
        self._diagnostics_tab = DiagnosticsTab(self.notebook, self.state)
        self._recreation_tab = RecreationTab(self.notebook, self.state)
        self._annotation_tab = AnnotationTab(self.notebook, gui_state=self.state)
        self._query_tab = QueryTab(self.notebook, self.state)

        # ── Status bar ────────────────────────────────────────────────
        self._status_bar = StatusBar(self.root)
        self._status_bar.grid(row=1, column=0, sticky="ew")

        # Surface internal event-handler errors in the status bar
        self.state.set_error_display(lambda msg: self._status_bar.set_status(msg))

        # Update status when state changes
        self.state.subscribe("pdf_changed", self._update_status)
        self.state.subscribe("run_completed", self._update_status)

    def _bind_shortcuts(self) -> None:
        """Global keyboard shortcuts."""
        self.root.bind("<Control-o>", lambda e: self._quick_open_pdf())
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

    def _update_status(self) -> None:
        parts = []
        if self.state.pdf_path:
            parts.append(f"PDF: {self.state.pdf_path.name}")
        if self.state.last_run_dir:
            parts.append(f"Last run: {self.state.last_run_dir.name}")
        self._status_bar.set_status(" | ".join(parts) if parts else "Ready")


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
