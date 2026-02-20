"""Tab 2 – Runs & Reports: browse past runs, inspect manifests, view reports.

Features:
- Run browser (Treeview listing all runs/ subdirectories)
- Manifest inspector with collapsible per-page details
- Report viewer (HTML / JSON)
- Artifact quick view (overlay PNGs, JSON data, CSV tables)
- Run management (delete, open folder, re-run with same config)
"""

from __future__ import annotations

import json
import os
import shutil
import tkinter as tk
import webbrowser
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from widgets import CollapsibleFrame, LogPanel


class RunsTab:
    """Tab 2: Run browser and report viewer."""

    def __init__(self, notebook: ttk.Notebook, gui_state: Any) -> None:
        self.notebook = notebook
        self.state = gui_state
        self.root = notebook.winfo_toplevel()

        self.frame = ttk.Frame(notebook)
        self.frame.columnconfigure(0, weight=1)  # tree
        self.frame.columnconfigure(1, weight=2)  # detail
        self.frame.rowconfigure(1, weight=1)
        notebook.add(self.frame, text="Runs & Reports")

        self._runs_root = Path(__file__).resolve().parent.parent.parent / "runs"
        self._current_manifest: dict | None = None
        self._current_run_dir: Path | None = None

        self._build_ui()

        # Listen for run completion
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

        # ── Run list (left panel) ────────────────────────────────────
        tree_frame = ttk.LabelFrame(self.frame, text="Runs", padding=4)
        tree_frame.grid(row=1, column=0, sticky="nsew", **pad)
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)

        cols = ("timestamp", "pdf", "pages")
        self._tree = ttk.Treeview(
            tree_frame, columns=cols, show="headings", selectmode="browse"
        )
        self._tree.heading("timestamp", text="Timestamp", anchor="w")
        self._tree.heading("pdf", text="PDF", anchor="w")
        self._tree.heading("pages", text="Pages", anchor="center")
        self._tree.column("timestamp", width=160, minwidth=120)
        self._tree.column("pdf", width=140, minwidth=100)
        self._tree.column("pages", width=50, minwidth=40)

        tree_sb = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=tree_sb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        tree_sb.grid(row=0, column=1, sticky="ns")

        self._tree.bind("<<TreeviewSelect>>", self._on_run_selected)

        # ── Detail panel (right) ─────────────────────────────────────
        detail_frame = ttk.LabelFrame(self.frame, text="Run Details", padding=4)
        detail_frame.grid(row=1, column=1, sticky="nsew", **pad)
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(0, weight=1)

        # Scrollable detail area
        self._detail_canvas = tk.Canvas(detail_frame, highlightthickness=0)
        detail_sb = ttk.Scrollbar(
            detail_frame, orient="vertical", command=self._detail_canvas.yview
        )
        self._detail_inner = ttk.Frame(self._detail_canvas)
        self._detail_inner.columnconfigure(0, weight=1)
        self._detail_inner.bind(
            "<Configure>",
            lambda e: self._detail_canvas.configure(
                scrollregion=self._detail_canvas.bbox("all")
            ),
        )
        self._detail_canvas_window = self._detail_canvas.create_window(
            (0, 0), window=self._detail_inner, anchor="nw"
        )
        self._detail_canvas.configure(yscrollcommand=detail_sb.set)
        self._detail_canvas.grid(row=0, column=0, sticky="nsew")
        detail_sb.grid(row=0, column=1, sticky="ns")

        def _resize_detail(event):
            self._detail_canvas.itemconfig(
                self._detail_canvas_window, width=event.width
            )

        self._detail_canvas.bind("<Configure>", _resize_detail)
        self._detail_canvas.bind(
            "<Enter>",
            lambda e: self._detail_canvas.bind_all("<MouseWheel>", self._detail_scroll),
        )
        self._detail_canvas.bind(
            "<Leave>", lambda e: self._detail_canvas.unbind_all("<MouseWheel>")
        )

        # Initial load
        self.refresh_runs()

    def _detail_scroll(self, event) -> None:
        self._detail_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ------------------------------------------------------------------
    # Run list management
    # ------------------------------------------------------------------

    def refresh_runs(self) -> None:
        """Scan runs/ directory and populate the tree."""
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
                except Exception:
                    timestamp = run_dir.name[4:19]  # from run_YYYYMMDD_HHMMSS
                    pdf_name = "?"
                    page_str = "?"
            else:
                timestamp = run_dir.name[4:19]
                pdf_name = "(no manifest)"
                page_str = "?"

            self._tree.insert(
                "",
                "end",
                iid=str(run_dir),
                values=(timestamp, pdf_name, page_str),
            )

    def _on_run_selected(self, _event=None) -> None:
        """Load and display the manifest for the selected run."""
        sel = self._tree.selection()
        if not sel:
            return
        run_dir = Path(sel[0])
        self._current_run_dir = run_dir
        manifest_path = run_dir / "manifest.json"

        # Clear detail panel
        for w in self._detail_inner.winfo_children():
            w.destroy()

        if not manifest_path.exists():
            ttk.Label(self._detail_inner, text="No manifest.json found.").grid(
                row=0, column=0, sticky="w", padx=6, pady=4
            )
            self._current_manifest = None
            return

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self._current_manifest = manifest
        except Exception as e:
            ttk.Label(self._detail_inner, text=f"Error reading manifest: {e}").grid(
                row=0, column=0, sticky="w", padx=6, pady=4
            )
            return

        self._populate_detail(manifest)

    def _populate_detail(self, manifest: dict) -> None:
        """Build the detail panel from a manifest dict."""
        row = 0

        # ── Summary ─────────────────────────────────────────────────
        summary_frame = CollapsibleFrame(
            self._detail_inner, "Summary", initially_open=True
        )
        summary_frame.grid(row=row, column=0, sticky="ew", pady=2, padx=4)
        row += 1

        info_pairs = [
            ("Run ID", manifest.get("run_id", "?")),
            ("Created", manifest.get("created_at", "?")),
            ("Source PDF", manifest.get("pdf_name", "?")),
            ("Fingerprint", manifest.get("input_fingerprint", "?")[:16] + "..."),
            ("Resolution", f"{manifest.get('render_resolution_dpi', '?')} DPI"),
            ("Pages", str(manifest.get("pages_processed", []))),
        ]
        for i, (label, val) in enumerate(info_pairs):
            ttk.Label(
                summary_frame.content,
                text=f"{label}:",
                font=("TkDefaultFont", 9, "bold"),
            ).grid(row=i, column=0, sticky="w", padx=(4, 8), pady=1)
            ttk.Label(summary_frame.content, text=str(val), wraplength=300).grid(
                row=i, column=1, sticky="w", pady=1
            )

        # ── Cross-page findings ──────────────────────────────────────
        cross_findings = manifest.get("cross_page_findings", [])
        if cross_findings:
            findings_frame = CollapsibleFrame(
                self._detail_inner, f"Cross-Page Findings ({len(cross_findings)})"
            )
            findings_frame.grid(row=row, column=0, sticky="ew", pady=2, padx=4)
            row += 1
            for i, finding in enumerate(cross_findings):
                sev = finding.get("severity", "info")
                msg = finding.get("message", str(finding))
                fg = (
                    "#e06c75"
                    if sev in ("error", "ERROR")
                    else "#e5c07b" if sev in ("warning", "WARNING") else "#d4d4d4"
                )
                ttk.Label(
                    findings_frame.content,
                    text=f"[{sev.upper()}] {msg}",
                    foreground=fg,
                    wraplength=400,
                ).grid(row=i, column=0, sticky="w", padx=4, pady=1)

        # ── Per-page details ─────────────────────────────────────────
        pages = manifest.get("pages", [])
        for page_data in pages:
            if not isinstance(page_data, dict):
                continue
            page_num = page_data.get("page", "?")
            quality = page_data.get("page_quality", "?")
            findings_count = page_data.get("semantic_findings_count", 0)

            page_title = f"Page {page_num}"
            if quality != "?":
                page_title += (
                    f" (quality: {quality:.2f})"
                    if isinstance(quality, float)
                    else f" (quality: {quality})"
                )
            if findings_count:
                page_title += f" [{findings_count} findings]"

            page_frame = CollapsibleFrame(self._detail_inner, page_title)
            page_frame.grid(row=row, column=0, sticky="ew", pady=2, padx=4)
            row += 1

            prow = 0

            # Stage timings
            stages = page_data.get("stages", [])
            if stages:
                for stage_info in stages:
                    s_name = stage_info.get("stage", "?")
                    s_status = stage_info.get("status", "?")
                    s_dur = stage_info.get("duration_ms", 0)
                    s_text = f"{s_name}: {s_status}"
                    if s_dur:
                        s_text += f" ({s_dur:.0f}ms)"
                    fg = (
                        "#98c379"
                        if s_status == "success"
                        else "#7f848e" if s_status == "skipped" else "#e06c75"
                    )
                    ttk.Label(page_frame.content, text=s_text, foreground=fg).grid(
                        row=prow, column=0, sticky="w", padx=4, pady=1
                    )
                    prow += 1

            # Counts
            counts = page_data.get("counts", {})
            if counts:
                ttk.Separator(page_frame.content, orient="horizontal").grid(
                    row=prow, column=0, sticky="ew", pady=4
                )
                prow += 1
                count_text = "  ".join(
                    f"{k}={v}" for k, v in counts.items() if isinstance(v, (int, float))
                )
                ttk.Label(
                    page_frame.content,
                    text=count_text,
                    wraplength=400,
                    foreground="gray",
                ).grid(row=prow, column=0, sticky="w", padx=4)
                prow += 1

            # Findings
            findings = page_data.get("semantic_findings", [])
            if findings:
                ttk.Separator(page_frame.content, orient="horizontal").grid(
                    row=prow, column=0, sticky="ew", pady=4
                )
                prow += 1
                for finding in findings:
                    if isinstance(finding, dict):
                        sev = finding.get("severity", "info")
                        msg = finding.get("message", str(finding))
                    else:
                        sev, msg = "info", str(finding)
                    fg = (
                        "#e06c75"
                        if "error" in sev.lower()
                        else "#e5c07b" if "warn" in sev.lower() else "#d4d4d4"
                    )
                    ttk.Label(
                        page_frame.content,
                        text=f"  [{sev}] {msg}",
                        foreground=fg,
                        wraplength=380,
                    ).grid(row=prow, column=0, sticky="w", padx=4, pady=1)
                    prow += 1

            # Artifacts
            artifacts = page_data.get("artifacts", {})
            if artifacts:
                ttk.Separator(page_frame.content, orient="horizontal").grid(
                    row=prow, column=0, sticky="ew", pady=4
                )
                prow += 1
                ttk.Label(
                    page_frame.content,
                    text="Artifacts:",
                    font=("TkDefaultFont", 9, "bold"),
                ).grid(row=prow, column=0, sticky="w", padx=4)
                prow += 1
                for art_name, art_path in artifacts.items():
                    link = ttk.Label(
                        page_frame.content,
                        text=f"  {art_name}: {art_path}",
                        foreground="#61afef",
                        cursor="hand2",
                    )
                    link.grid(row=prow, column=0, sticky="w", padx=4, pady=1)
                    full_path = (
                        self._current_run_dir / art_path
                        if self._current_run_dir
                        else Path(art_path)
                    )
                    link.bind(
                        "<Button-1>", lambda e, p=full_path: self._open_artifact(p)
                    )
                    prow += 1

        # ── Config snapshot ──────────────────────────────────────────
        config = manifest.get("config_snapshot", {})
        if config:
            cfg_frame = CollapsibleFrame(self._detail_inner, "Config Snapshot")
            cfg_frame.grid(row=row, column=0, sticky="ew", pady=2, padx=4)
            row += 1

            cfg_text = tk.Text(
                cfg_frame.content,
                height=8,
                wrap="word",
                bg="#1e1e1e",
                fg="#d4d4d4",
                font=("Consolas", 8),
                state="normal",
                relief="sunken",
            )
            cfg_text.pack(fill="both", expand=True, padx=2, pady=2)
            cfg_text.insert("1.0", json.dumps(config, indent=2))
            cfg_text.config(state="disabled")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _open_folder(self) -> None:
        if self._current_run_dir and self._current_run_dir.is_dir():
            os.startfile(self._current_run_dir)
        else:
            if self._runs_root.is_dir():
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
            self._open_artifact(report)
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
        # Push config back to Pipeline tab via shared state
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
            self.refresh_runs()
            # Clear detail
            for w in self._detail_inner.winfo_children():
                w.destroy()

    def _open_artifact(self, path: Path) -> None:
        """Open an artifact file based on its extension."""
        if not path.exists():
            messagebox.showinfo("Not Found", f"File not found: {path}")
            return

        suffix = path.suffix.lower()
        if suffix in (".png", ".jpg", ".jpeg", ".bmp"):
            # Open in default image viewer
            os.startfile(path)
        elif suffix == ".html":
            webbrowser.open(str(path))
        elif suffix in (".json", ".csv", ".txt"):
            # Show in a popup text viewer
            self._show_text_viewer(path)
        else:
            os.startfile(path)

    def _show_text_viewer(self, path: Path) -> None:
        """Show a file in a popup text window."""
        win = tk.Toplevel(self.root)
        win.title(f"Artifact: {path.name}")
        win.geometry("700x500")
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)

        text = tk.Text(
            win,
            wrap="none",
            bg="#1e1e1e",
            fg="#d4d4d4",
            font=("Consolas", 9),
            state="normal",
        )
        v_sb = ttk.Scrollbar(win, orient="vertical", command=text.yview)
        h_sb = ttk.Scrollbar(win, orient="horizontal", command=text.xview)
        text.configure(yscrollcommand=v_sb.set, xscrollcommand=h_sb.set)
        text.grid(row=0, column=0, sticky="nsew")
        v_sb.grid(row=0, column=1, sticky="ns")
        h_sb.grid(row=1, column=0, sticky="ew")

        try:
            content = path.read_text(encoding="utf-8")
            if path.suffix == ".json":
                try:
                    parsed = json.loads(content)
                    content = json.dumps(parsed, indent=2)
                except Exception:
                    pass
            text.insert("1.0", content)
        except Exception as e:
            text.insert("1.0", f"Error reading file: {e}")
        text.config(state="disabled")

        ttk.Button(win, text="Close", command=win.destroy).grid(row=2, column=0, pady=4)
