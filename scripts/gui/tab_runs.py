"""Tab 2 – Runs & Reports: browse past runs, inspect manifests, view reports.

Coordinator that composes:
- RunBrowserWidget  (run_browser.py)  – Treeview listing, selection, filtering
- ReportViewerWidget (report_viewer.py) – manifest detail panel, artifact export
"""

from __future__ import annotations

import os
import shutil
import sys
import webbrowser
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from .report_viewer import ReportViewerWidget
from .run_browser import RunBrowserWidget


class RunsTab:
    """Tab 2: Run browser and report viewer."""

    def __init__(self, notebook: ttk.Notebook, gui_state: Any) -> None:
        self.notebook = notebook
        self.state = gui_state
        self.root = notebook.winfo_toplevel()

        self.frame = ttk.Frame(notebook)
        self.frame.columnconfigure(0, weight=1)  # browser
        self.frame.columnconfigure(1, weight=2)  # viewer
        self.frame.rowconfigure(1, weight=1)
        notebook.add(self.frame, text="Runs & Reports")

        self._runs_root = Path(__file__).resolve().parent.parent.parent / "runs"
        self._current_manifest: dict | None = None
        self._current_run_dir: Path | None = None

        self._build_ui()

        self.state.subscribe("run_completed", lambda: self.refresh_runs())

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        pad = {"padx": 6, "pady": 4}

        # ── Toolbar ──────────────────────────────────────────────────
        toolbar = ttk.Frame(self.frame)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", **pad)
        ttk.Button(toolbar, text="Refresh", command=self.refresh_runs).pack(
            side="left", padx=2
        )
        ttk.Button(toolbar, text="Open Folder", command=self._open_folder).pack(
            side="left", padx=2
        )
        ttk.Button(
            toolbar, text="Open HTML Report", command=self._open_html_report
        ).pack(side="left", padx=2)
        ttk.Button(
            toolbar, text="Open JSON Report", command=self._open_json_report
        ).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Re-run Config", command=self._rerun_config).pack(
            side="left", padx=2
        )
        ttk.Button(toolbar, text="Delete Run", command=self._delete_run).pack(
            side="right", padx=2
        )

        # ── Run browser (left) ────────────────────────────────────────
        self._browser = RunBrowserWidget(
            self.frame,
            self._runs_root,
            on_select=self._on_run_selected,
        )
        self._browser.frame.grid(row=1, column=0, sticky="nsew", **pad)

        # ── Report viewer (right) ─────────────────────────────────────
        self._viewer = ReportViewerWidget(self.frame, self.root)
        self._viewer.frame.grid(row=1, column=1, sticky="nsew", **pad)

    # ------------------------------------------------------------------
    # Wiring between browser and viewer
    # ------------------------------------------------------------------

    def _on_run_selected(self, run_dir: Path, manifest: dict | None) -> None:
        self._current_run_dir = run_dir
        self._current_manifest = manifest
        self._viewer.show_run(run_dir, manifest)

    # ------------------------------------------------------------------
    # Public API (used by gui.py via state subscription)
    # ------------------------------------------------------------------

    def refresh_runs(self) -> None:
        """Refresh the run list."""
        self._browser.refresh_runs()

    # ------------------------------------------------------------------
    # Toolbar actions
    # ------------------------------------------------------------------

    def _open_folder(self) -> None:
        if self._current_run_dir and self._current_run_dir.is_dir():
            if sys.platform == "win32":
                os.startfile(self._current_run_dir)
        else:
            if self._runs_root.is_dir() and sys.platform == "win32":
                os.startfile(self._runs_root)

    def _open_html_report(self) -> None:
        if not self._current_run_dir:
            messagebox.showinfo("No Run", "Select a run first.")
            return
        report = self._current_run_dir / "report.html"
        if report.exists():
            webbrowser.open(str(report))
        else:
            messagebox.showinfo(
                "No Report", f"No report.html in {self._current_run_dir.name}"
            )

    def _open_json_report(self) -> None:
        if not self._current_run_dir:
            messagebox.showinfo("No Run", "Select a run first.")
            return
        report = self._current_run_dir / "report.json"
        if report.exists():
            self._viewer._open_artifact(report)
        else:
            messagebox.showinfo(
                "No Report", f"No report.json in {self._current_run_dir.name}"
            )

    def _rerun_config(self) -> None:
        """Load the config snapshot from the selected run back into the Pipeline tab."""
        if not self._current_manifest:
            messagebox.showinfo("No Run", "Select a run with a manifest first.")
            return
        config = self._current_manifest.get("config_snapshot", {})
        if not config:
            messagebox.showinfo("No Config", "No config_snapshot in this manifest.")
            return
        self.state.pending_config = config
        self.state.notify("load_config")
        messagebox.showinfo(
            "Config Loaded",
            "Configuration loaded into Pipeline tab.\nSwitch to Pipeline tab to review.",
        )

    def _delete_run(self) -> None:
        if not self._current_run_dir or not self._current_run_dir.is_dir():
            return
        result = messagebox.askyesno(
            "Delete Run",
            f"Delete run '{self._current_run_dir.name}' and all its artifacts?\n\nThis cannot be undone.",
        )
        if result:
            shutil.rmtree(self._current_run_dir, ignore_errors=True)
            self._current_run_dir = None
            self._current_manifest = None
            self._viewer.clear()
            self.refresh_runs()
