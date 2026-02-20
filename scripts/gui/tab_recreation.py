"""Tab 5 – Sheet Recreation: generate text-only recreation PDFs from existing runs.

Exposes all options from ``recreate_sheet()`` and ``run_sheet_recreation.py``
in a GUI panel with embedded log output.
"""

from __future__ import annotations

import json
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

_project = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project / "src"))
sys.path.insert(0, str(_project / "scripts" / "runners"))

from widgets import CollapsibleFrame, LogPanel
from worker import PipelineWorker


class RecreationTab:
    """Tab 5: Sheet recreation from pipeline runs."""

    def __init__(self, notebook: ttk.Notebook, gui_state: Any) -> None:
        self.notebook = notebook
        self.state = gui_state
        self.root = notebook.winfo_toplevel()

        self.frame = ttk.Frame(notebook)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=3)
        self.frame.rowconfigure(1, weight=1)
        notebook.add(self.frame, text="Sheet Recreation")

        self._run_dir: Path | None = None
        self._source_pdf: Path | None = None
        self._worker: PipelineWorker | None = None

        self._build_ui()

        # Listen for run completion – auto-select latest run
        self.state.subscribe("run_completed", self._on_run_completed)

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 4}

        # ── Top settings area ────────────────────────────────────────
        settings = ttk.Frame(self.frame)
        settings.grid(row=0, column=0, sticky="nsew", **pad)
        settings.columnconfigure(1, weight=1)

        row = 0

        # Run directory selection
        ttk.Label(
            settings, text="Run Directory:", font=("TkDefaultFont", 9, "bold")
        ).grid(row=row, column=0, sticky="w", pady=4)
        dir_frame = ttk.Frame(settings)
        dir_frame.grid(row=row, column=1, sticky="ew", pady=4)
        dir_frame.columnconfigure(1, weight=1)

        ttk.Button(dir_frame, text="Browse...", command=self._select_run_dir).grid(
            row=0, column=0, padx=(0, 4)
        )
        self._dir_label_var = tk.StringVar(value="No run selected")
        ttk.Label(dir_frame, textvariable=self._dir_label_var, foreground="gray").grid(
            row=0, column=1, sticky="w"
        )

        # Quick-pick from recent runs
        ttk.Button(dir_frame, text="Latest Run", command=self._pick_latest_run).grid(
            row=0, column=2, padx=(8, 0)
        )
        row += 1

        # ── Options ──────────────────────────────────────────────────
        opts_frame = CollapsibleFrame(
            settings, "Recreation Options", initially_open=True
        )
        opts_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1

        oc = opts_frame.content
        oc.columnconfigure(1, weight=1)

        # Color mode
        orow = 0
        ttk.Label(oc, text="Color Mode:").grid(row=orow, column=0, sticky="w", pady=2)
        self._color_mode = tk.StringVar(value="plain")
        mode_frame = ttk.Frame(oc)
        mode_frame.grid(row=orow, column=1, sticky="w", pady=2)
        ttk.Radiobutton(
            mode_frame,
            text="Plain (black text)",
            variable=self._color_mode,
            value="plain",
        ).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(
            mode_frame,
            text="Origin (color by source)",
            variable=self._color_mode,
            value="origin",
        ).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(
            mode_frame, text="PDF Layers", variable=self._color_mode, value="layers"
        ).pack(side="left")
        orow += 1

        # Block boundaries
        self._draw_blocks_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            oc, text="Draw block boundaries & labels", variable=self._draw_blocks_var
        ).grid(row=orow, column=0, columnspan=2, sticky="w", pady=2)
        orow += 1

        # Source PDF watermark
        ttk.Label(oc, text="Watermark PDF:").grid(
            row=orow, column=0, sticky="w", pady=2
        )
        wm_frame = ttk.Frame(oc)
        wm_frame.grid(row=orow, column=1, sticky="ew", pady=2)
        ttk.Button(wm_frame, text="Select...", command=self._select_source_pdf).pack(
            side="left", padx=(0, 4)
        )
        self._source_pdf_label = tk.StringVar(value="None (no watermark)")
        ttk.Label(
            wm_frame, textvariable=self._source_pdf_label, foreground="gray"
        ).pack(side="left")
        ttk.Button(wm_frame, text="Clear", command=self._clear_source_pdf).pack(
            side="left", padx=(8, 0)
        )
        orow += 1

        # Watermark opacity
        ttk.Label(oc, text="Watermark Opacity:").grid(
            row=orow, column=0, sticky="w", pady=2
        )
        self._opacity_var = tk.StringVar(value="0.15")
        opacity_frame = ttk.Frame(oc)
        opacity_frame.grid(row=orow, column=1, sticky="w", pady=2)
        ttk.Scale(
            opacity_frame,
            from_=0.05,
            to=0.5,
            orient="horizontal",
            variable=self._opacity_var,
            length=200,
        ).pack(side="left")
        ttk.Label(opacity_frame, textvariable=self._opacity_var, width=6).pack(
            side="left", padx=(4, 0)
        )
        orow += 1

        # Pages
        ttk.Label(oc, text="Pages:").grid(row=orow, column=0, sticky="w", pady=2)
        pages_frame = ttk.Frame(oc)
        pages_frame.grid(row=orow, column=1, sticky="w", pady=2)
        self._pages_var = tk.StringVar(value="")
        ttk.Entry(pages_frame, textvariable=self._pages_var, width=20).pack(side="left")
        ttk.Label(
            pages_frame,
            text="(blank = all, or comma-separated 1-based)",
            foreground="gray",
        ).pack(side="left", padx=(6, 0))
        orow += 1

        # ── Generate button ──────────────────────────────────────────
        btn_frame = ttk.Frame(settings)
        btn_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        btn_frame.columnconfigure(0, weight=1)

        gen_style = ttk.Style()
        gen_style.configure(
            "Gen.TButton", font=("TkDefaultFont", 11, "bold"), padding=(16, 8)
        )

        self._gen_button = ttk.Button(
            btn_frame,
            text="Generate Recreation PDF",
            command=self._generate,
            style="Gen.TButton",
        )
        self._gen_button.grid(row=0, column=0, sticky="ew", padx=20)

        # ── Log panel ────────────────────────────────────────────────
        self.log_panel = LogPanel(self.frame, height=8)
        self.log_panel.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 6))

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def _select_run_dir(self) -> None:
        d = filedialog.askdirectory(
            title="Select Run Directory",
            initialdir=str(_project / "runs"),
        )
        if d:
            self._run_dir = Path(d)
            self._dir_label_var.set(self._run_dir.name)

    def _pick_latest_run(self) -> None:
        runs_root = _project / "runs"
        if not runs_root.is_dir():
            messagebox.showinfo("No Runs", "No runs directory found.")
            return
        run_dirs = sorted(
            [
                d
                for d in runs_root.iterdir()
                if d.is_dir() and d.name.startswith("run_")
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not run_dirs:
            messagebox.showinfo("No Runs", "No run directories found.")
            return
        self._run_dir = run_dirs[0]
        self._dir_label_var.set(self._run_dir.name)

    def _select_source_pdf(self) -> None:
        f = filedialog.askopenfilename(
            title="Select Source PDF for Watermark",
            filetypes=[("PDF", "*.pdf"), ("All", "*.*")],
            initialdir=str(_project / "input"),
        )
        if f:
            self._source_pdf = Path(f)
            self._source_pdf_label.set(self._source_pdf.name)

    def _clear_source_pdf(self) -> None:
        self._source_pdf = None
        self._source_pdf_label.set("None (no watermark)")

    def _on_run_completed(self) -> None:
        last = getattr(self.state, "last_run_dir", None)
        if last and Path(last).is_dir():
            self._run_dir = Path(last)
            self._dir_label_var.set(self._run_dir.name)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def _generate(self) -> None:
        if not self._run_dir or not self._run_dir.is_dir():
            messagebox.showwarning("No Run", "Select a run directory first.")
            return

        artifacts_dir = self._run_dir / "artifacts"
        if not artifacts_dir.is_dir():
            messagebox.showwarning(
                "No Artifacts", f"No artifacts/ folder in {self._run_dir.name}."
            )
            return

        self.log_panel.clear()
        self._gen_button.config(state="disabled")

        # Gather options
        color_mode = self._color_mode.get()
        draw_blocks = self._draw_blocks_var.get()
        try:
            opacity = float(self._opacity_var.get())
        except ValueError:
            opacity = 0.15
        opacity = max(0.05, min(0.5, opacity))

        pages_str = self._pages_var.get().strip()
        pages = None
        if pages_str:
            try:
                pages = [int(p.strip()) for p in pages_str.split(",")]
            except ValueError:
                messagebox.showerror(
                    "Invalid Pages", "Pages must be comma-separated integers."
                )
                self._gen_button.config(state="normal")
                return

        run_dir = self._run_dir
        source_pdf = self._source_pdf

        self._worker = PipelineWorker(self.root, self.log_panel)

        def target():
            from plancheck.export.sheet_recreation import ORIGIN_COLORS, recreate_sheet

            # Determine color_map / use_layers
            use_layers = color_mode == "layers"
            color_map = None
            if color_mode == "origin":
                color_map = ORIGIN_COLORS

            out = recreate_sheet(
                run_dir=run_dir,
                pages=pages,
                color_map=color_map,
                draw_blocks=draw_blocks,
                use_layers=use_layers,
                source_pdf=source_pdf,
                watermark_opacity=opacity,
            )
            print(f"Sheet recreation saved -> {out}")
            return out

        def on_done(result, error, elapsed):
            self._gen_button.config(state="normal")
            if result and not error:
                self.log_panel.write(f"PDF saved: {result}", "SUCCESS")
                # Offer to open
                if messagebox.askyesno(
                    "Open PDF?", f"Open the recreation PDF?\n{result}"
                ):
                    os.startfile(result)

        self._worker.run(target, on_done=on_done)
