"""Run browser widget – Treeview listing of past runs with selection callback."""

from __future__ import annotations

import json
from pathlib import Path
from tkinter import ttk
from typing import Callable


class RunBrowserWidget:
    """Left-panel widget: lists runs/ directories in a sortable Treeview.

    Parameters
    ----------
    parent :
        Parent tkinter widget.
    runs_root :
        Path to the ``runs/`` directory.
    on_select :
        Callback invoked with ``(run_dir: Path, manifest: dict | None)``
        whenever the user selects a run.
    """

    def __init__(
        self,
        parent,
        runs_root: Path,
        on_select: Callable[[Path, dict | None], None] | None = None,
    ) -> None:
        self._runs_root = runs_root
        self._on_select_callback = on_select

        self.frame = ttk.LabelFrame(parent, text="Runs", padding=4)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)

        cols = ("timestamp", "pdf", "pages", "status")
        self._tree = ttk.Treeview(
            self.frame, columns=cols, show="headings", selectmode="browse"
        )
        self._tree.heading("timestamp", text="Timestamp", anchor="w")
        self._tree.heading("pdf", text="PDF", anchor="w")
        self._tree.heading("pages", text="Pages", anchor="center")
        self._tree.heading("status", text="Status", anchor="center")
        self._tree.column("timestamp", width=160, minwidth=120)
        self._tree.column("pdf", width=140, minwidth=100)
        self._tree.column("pages", width=50, minwidth=40)
        self._tree.column("status", width=60, minwidth=50)

        tree_sb = ttk.Scrollbar(self.frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=tree_sb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        tree_sb.grid(row=0, column=1, sticky="ns")

        self._tree.bind("<<TreeviewSelect>>", self._on_run_selected)

        self.refresh_runs()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh_runs(self) -> None:
        """Scan runs/ directory and repopulate the Treeview."""
        self._tree.delete(*self._tree.get_children())
        if not self._runs_root.is_dir():
            return

        run_dirs = sorted(
            [
                d
                for d in self._runs_root.iterdir()
                if d.is_dir() and d.name.startswith("run_")
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for run_dir in run_dirs:
            manifest_path = run_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    timestamp = manifest.get("created_at", "")[:19].replace("T", " ")
                    pdf_name = manifest.get("pdf_name", "?")
                    pages = manifest.get("pages_processed", [])
                    page_str = f"{len(pages)}" if pages else "?"
                    status_str = self._compute_run_status(manifest)
                except Exception:
                    timestamp = run_dir.name[4:19]
                    pdf_name = "?"
                    page_str = "?"
                    status_str = "?"
            else:
                timestamp = run_dir.name[4:19]
                pdf_name = "(no manifest)"
                page_str = "?"
                status_str = "?"

            self._tree.insert(
                "",
                "end",
                iid=str(run_dir),
                values=(timestamp, pdf_name, page_str, status_str),
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_run_selected(self, _event=None) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        run_dir = Path(sel[0])
        manifest_path = run_dir / "manifest.json"

        manifest: dict | None = None
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = None

        if self._on_select_callback is not None:
            self._on_select_callback(run_dir, manifest)

    @staticmethod
    def _compute_run_status(manifest: dict) -> str:
        """Derive a run-level status string from page entries."""
        pages = manifest.get("pages", [])
        if not pages:
            return "?"
        errors = sum(
            1
            for p in pages
            if isinstance(p, dict) and "error" in p and "stages" not in p
        )
        if errors == len(pages):
            return "\u2718 Failed"
        if errors > 0:
            return "\u26a0 Partial"
        return "\u2714 OK"
