"""
Advanced Plan Parser – Master GUI Controller.

Slim orchestrator that wires up five tabs via a shared GuiState.

Tabs
----
1. Pipeline       – PDF selection, OCR toggles, config knobs, run with embedded log
2. Runs & Reports – Browse runs/, inspect manifests, open reports/artifacts
3. Visual Debug   – Overlay viewer with 11 layer types + knobs
4. Diagnostics    – Font metrics, benchmark, VOCRPP, tuning, grouping playground
5. Sheet Recreation – Generate text-only PDFs from runs

Architecture
------------
GuiState holds shared state (pdf_path, last_run_dir, config) and a simple
pub/sub mechanism so tabs can react to events (pdf_changed, run_completed,
load_config) without direct coupling.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src and scripts to path for imports
_project = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project / "src"))
sys.path.insert(0, str(_project / "scripts" / "runners"))
sys.path.insert(0, str(_project / "scripts" / "utils"))
sys.path.insert(0, str(_project / "scripts" / "gui"))

import tkinter as tk
from tkinter import filedialog, ttk

from plancheck.config import GroupingConfig

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
        self._subscribers: dict[str, list] = {}

    def subscribe(self, event: str, callback) -> None:
        self._subscribers.setdefault(event, []).append(callback)

    def notify(self, event: str) -> None:
        for cb in self._subscribers.get(event, []):
            try:
                cb()
            except Exception:
                pass  # Don't let one subscriber crash others

    def set_pdf(self, path: Path | None) -> None:
        self.pdf_path = path
        self.notify("pdf_changed")

    def set_last_run(self, run_dir: Path) -> None:
        self.last_run_dir = run_dir
        self.notify("run_completed")


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

    def _build_ui(self) -> None:
        # ── Tab container ─────────────────────────────────────────────
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        # ── Import and create each tab ────────────────────────────────
        from overlay_viewer import OverlayViewerTab
        from tab_diagnostics import DiagnosticsTab
        from tab_pipeline import PipelineTab
        from tab_recreation import RecreationTab
        from tab_runs import RunsTab
        from widgets import StatusBar

        self._pipeline_tab = PipelineTab(self.notebook, self.state)
        self._runs_tab = RunsTab(self.notebook, self.state)
        self._overlay_tab = OverlayViewerTab(self.notebook, gui_state=self.state)
        self._diagnostics_tab = DiagnosticsTab(self.notebook, self.state)
        self._recreation_tab = RecreationTab(self.notebook, self.state)

        # ── Status bar ────────────────────────────────────────────────
        self._status_bar = StatusBar(self.root)
        self._status_bar.grid(row=1, column=0, sticky="ew")

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

    def _quick_open_pdf(self) -> None:
        f = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")],
            initialdir=str(_project / "input"),
        )
        if f:
            self.state.set_pdf(Path(f))

    def _update_status(self) -> None:
        parts = []
        if self.state.pdf_path:
            parts.append(f"PDF: {self.state.pdf_path.name}")
        if self.state.last_run_dir:
            parts.append(f"Last run: {self.state.last_run_dir.name}")
        if parts:
            self._status_bar.set_status(" | ".join(parts))


def main() -> None:
    """Launch the GUI application."""
    root = tk.Tk()
    _app = PlanParserGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
