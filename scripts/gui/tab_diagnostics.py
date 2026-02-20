"""Tab 4 – Diagnostics: font metrics, benchmark runner, tuning harness, VOCRPP preview.

Collapsible sections for each diagnostic tool with embedded output.
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

_project = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project / "src"))
sys.path.insert(0, str(_project / "scripts" / "runners"))
sys.path.insert(0, str(_project / "scripts" / "diagnostics"))

from widgets import CollapsibleFrame, LogPanel
from worker import PipelineWorker


class DiagnosticsTab:
    """Tab 4: Diagnostics tools."""

    def __init__(self, notebook: ttk.Notebook, gui_state: Any) -> None:
        self.notebook = notebook
        self.state = gui_state
        self.root = notebook.winfo_toplevel()

        self.frame = ttk.Frame(notebook)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=3)
        self.frame.rowconfigure(1, weight=1)
        notebook.add(self.frame, text="Diagnostics")

        self._worker: PipelineWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 4}

        # ── Scrollable top ───────────────────────────────────────────
        self._canvas = tk.Canvas(self.frame, highlightthickness=0)
        sb = ttk.Scrollbar(self.frame, orient="vertical", command=self._canvas.yview)
        self._inner = ttk.Frame(self._canvas)
        self._inner.columnconfigure(0, weight=1)
        self._inner.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        cw = self._canvas.create_window((0, 0), window=self._inner, anchor="nw")
        self._canvas.configure(yscrollcommand=sb.set)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        def _resize(event):
            self._canvas.itemconfig(cw, width=event.width)

        self._canvas.bind("<Configure>", _resize)
        self._canvas.bind(
            "<Enter>",
            lambda e: self._canvas.bind_all(
                "<MouseWheel>",
                lambda ev: self._canvas.yview_scroll(
                    int(-1 * (ev.delta / 120)), "units"
                ),
            ),
        )
        self._canvas.bind("<Leave>", lambda e: self._canvas.unbind_all("<MouseWheel>"))

        row = 0

        # ── 1. Font Diagnostics ──────────────────────────────────────
        font_section = CollapsibleFrame(
            self._inner, "Font Metrics Diagnostics", initially_open=True
        )
        font_section.grid(row=row, column=0, sticky="ew", **pad)
        row += 1

        fc = font_section.content
        fc.columnconfigure(0, weight=1)

        self.font_metrics_var = tk.BooleanVar(value=False)
        self.visual_metrics_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(
            fc,
            text="FontMetricsAnalyzer (Heuristic font width analysis)",
            variable=self.font_metrics_var,
        ).grid(row=0, column=0, sticky="w", pady=2)

        ttk.Checkbutton(
            fc,
            text="VisualMetricsAnalyzer (Pixel-accurate font width analysis)",
            variable=self.visual_metrics_var,
        ).grid(row=1, column=0, sticky="w", pady=2)

        ttk.Label(
            fc,
            text="Runs as standalone tests and writes JSON reports.",
            foreground="gray",
        ).grid(row=2, column=0, sticky="w", pady=2)

        ttk.Button(
            fc,
            text="Run Font Diagnostics",
            command=self._run_font_diagnostics,
        ).grid(row=3, column=0, sticky="w", pady=(4, 2))

        # ── 2. Benchmark Runner ──────────────────────────────────────
        bench_section = CollapsibleFrame(self._inner, "A/B/C/D Benchmark")
        bench_section.grid(row=row, column=0, sticky="ew", **pad)
        row += 1

        bc = bench_section.content
        bc.columnconfigure(1, weight=1)

        ttk.Label(bc, text="Compare pipeline conditions on the same page(s):").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=2
        )

        self._bench_a = tk.BooleanVar(value=True)
        self._bench_b = tk.BooleanVar(value=True)
        self._bench_c = tk.BooleanVar(value=True)
        self._bench_d = tk.BooleanVar(value=False)

        conds_frame = ttk.Frame(bc)
        conds_frame.grid(row=1, column=0, columnspan=3, sticky="w", pady=2)
        ttk.Checkbutton(conds_frame, text="A: TOCR only", variable=self._bench_a).pack(
            side="left", padx=(0, 8)
        )
        ttk.Checkbutton(conds_frame, text="B: +VOCR", variable=self._bench_b).pack(
            side="left", padx=(0, 8)
        )
        ttk.Checkbutton(conds_frame, text="C: +Reconcile", variable=self._bench_c).pack(
            side="left", padx=(0, 8)
        )
        ttk.Checkbutton(conds_frame, text="D: +VOCRPP", variable=self._bench_d).pack(
            side="left"
        )

        ttk.Label(bc, text="OCR DPI:").grid(row=2, column=0, sticky="w", pady=2)
        self._bench_ocr_dpi = tk.StringVar(value="180")
        ttk.Spinbox(
            bc,
            textvariable=self._bench_ocr_dpi,
            values=(120, 150, 180, 200, 300),
            width=8,
            state="readonly",
        ).grid(row=2, column=1, sticky="w", padx=(8, 0))

        ttk.Button(bc, text="Run Benchmark", command=self._run_benchmark).grid(
            row=3, column=0, sticky="w", pady=(4, 2)
        )

        ttk.Label(
            bc,
            text="Uses selected PDF and page range from Pipeline tab.",
            foreground="gray",
        ).grid(row=4, column=0, columnspan=3, sticky="w")

        # ── 3. VOCRPP Preview ────────────────────────────────────────
        vocrpp_section = CollapsibleFrame(self._inner, "VOCRPP Preprocessing Preview")
        vocrpp_section.grid(row=row, column=0, sticky="ew", **pad)
        row += 1

        vc = vocrpp_section.content
        vc.columnconfigure(0, weight=1)

        ttk.Label(
            vc,
            text="Run OCR image preprocessing and view before/after.",
            foreground="gray",
        ).grid(row=0, column=0, sticky="w", pady=2)

        vocrpp_opts = ttk.Frame(vc)
        vocrpp_opts.grid(row=1, column=0, sticky="ew", pady=2)

        self._pp_grayscale = tk.BooleanVar(value=True)
        self._pp_clahe = tk.BooleanVar(value=True)
        self._pp_denoise = tk.BooleanVar(value=False)
        self._pp_binarize = tk.BooleanVar(value=False)
        self._pp_sharpen = tk.BooleanVar(value=False)

        ttk.Checkbutton(
            vocrpp_opts, text="Grayscale", variable=self._pp_grayscale
        ).pack(side="left", padx=(0, 6))
        ttk.Checkbutton(vocrpp_opts, text="CLAHE", variable=self._pp_clahe).pack(
            side="left", padx=(0, 6)
        )
        ttk.Checkbutton(vocrpp_opts, text="Denoise", variable=self._pp_denoise).pack(
            side="left", padx=(0, 6)
        )
        ttk.Checkbutton(vocrpp_opts, text="Binarize", variable=self._pp_binarize).pack(
            side="left", padx=(0, 6)
        )
        ttk.Checkbutton(vocrpp_opts, text="Sharpen", variable=self._pp_sharpen).pack(
            side="left"
        )

        ttk.Label(vc, text="DPI:").grid(row=2, column=0, sticky="w")
        self._pp_dpi = tk.StringVar(value="300")
        pp_dpi_frame = ttk.Frame(vc)
        pp_dpi_frame.grid(row=2, column=0, sticky="w")
        ttk.Label(pp_dpi_frame, text="DPI:").pack(side="left")
        ttk.Spinbox(
            pp_dpi_frame,
            textvariable=self._pp_dpi,
            values=(150, 200, 300, 400),
            width=6,
            state="readonly",
        ).pack(side="left", padx=4)

        ttk.Button(
            vc, text="Run Preprocessing", command=self._run_preprocess_preview
        ).grid(row=3, column=0, sticky="w", pady=(4, 2))

        # ── 4. Tuning Harness ────────────────────────────────────────
        tuning_section = CollapsibleFrame(self._inner, "Config Tuning Harness")
        tuning_section.grid(row=row, column=0, sticky="ew", **pad)
        row += 1

        tc = tuning_section.content
        tc.columnconfigure(0, weight=1)

        ttk.Label(
            tc,
            text="Sweep config parameters and compare quality scores.",
            foreground="gray",
        ).grid(row=0, column=0, sticky="w", pady=2)
        ttk.Label(
            tc, text="Uses the default recipe unless a recipe file is loaded."
        ).grid(row=1, column=0, sticky="w", pady=1)

        tune_btns = ttk.Frame(tc)
        tune_btns.grid(row=2, column=0, sticky="w", pady=4)
        ttk.Button(tune_btns, text="Run Sweep", command=self._run_tuning).pack(
            side="left", padx=2
        )
        ttk.Button(tune_btns, text="Load Recipe...", command=self._load_recipe).pack(
            side="left", padx=2
        )
        ttk.Button(tune_btns, text="List Recipes", command=self._list_recipes).pack(
            side="left", padx=2
        )

        # ── 5. Grouping Playground ───────────────────────────────────
        grouping_section = CollapsibleFrame(self._inner, "Grouping Playground")
        grouping_section.grid(row=row, column=0, sticky="ew", **pad)
        row += 1

        gc = grouping_section.content
        gc.columnconfigure(0, weight=1)

        ttk.Label(
            gc,
            text="Load a boxes.json and re-run grouping with custom knobs.",
            foreground="gray",
        ).grid(row=0, column=0, sticky="w", pady=2)
        gp_btns = ttk.Frame(gc)
        gp_btns.grid(row=1, column=0, sticky="w", pady=4)
        ttk.Button(
            gp_btns, text="Load boxes.json...", command=self._load_boxes_json
        ).pack(side="left", padx=2)
        ttk.Button(gp_btns, text="Run Grouping", command=self._run_grouping).pack(
            side="left", padx=2
        )

        self._boxes_label_var = tk.StringVar(value="No file loaded")
        ttk.Label(gc, textvariable=self._boxes_label_var, foreground="gray").grid(
            row=2, column=0, sticky="w"
        )

        self._boxes_path: Path | None = None

        # ── Log panel ────────────────────────────────────────────────
        self.log_panel = LogPanel(self.frame, height=8)
        self.log_panel.grid(
            row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 6)
        )

    # ------------------------------------------------------------------
    # PDF helper (uses shared state)
    # ------------------------------------------------------------------

    def _get_pdf_and_pages(self) -> tuple[Path | None, int, int | None]:
        """Try to get the PDF and page range from the Pipeline tab state."""
        pdf = self.state.pdf_path
        if pdf is None:
            messagebox.showwarning("No PDF", "Select a PDF in the Pipeline tab first.")
            return None, 0, None
        return pdf, 0, None

    # ------------------------------------------------------------------
    # Font Diagnostics
    # ------------------------------------------------------------------

    def _run_font_diagnostics(self) -> None:
        pdf = self.state.pdf_path
        if not pdf:
            messagebox.showwarning("No PDF", "Select a PDF in the Pipeline tab first.")
            return
        if not (self.font_metrics_var.get() or self.visual_metrics_var.get()):
            messagebox.showinfo("No Tools", "Enable at least one diagnostics tool.")
            return

        runs_root = _project / "runs"
        self.log_panel.clear()

        self._worker = PipelineWorker(self.root, self.log_panel)

        def target():
            from run_font_metrics_diagnostics import _make_run_dir, run_diagnostics

            run_dir = _make_run_dir(runs_root, pdf.stem[:15])
            out = run_diagnostics(
                pdf=pdf,
                out_dir=run_dir,
                start=0,
                end=None,
                run_heuristic=self.font_metrics_var.get(),
                run_visual=self.visual_metrics_var.get(),
                visual_resolution=300,
            )
            return out

        def on_done(result, error, elapsed):
            if result and not error:
                self.log_panel.write(f"Report saved: {result}", "SUCCESS")

        self._worker.run(target, on_done=on_done)

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def _run_benchmark(self) -> None:
        pdf = self.state.pdf_path
        if not pdf:
            messagebox.showwarning("No PDF", "Select a PDF in the Pipeline tab first.")
            return

        conditions = []
        if self._bench_a.get():
            conditions.append("A")
        if self._bench_b.get():
            conditions.append("B")
        if self._bench_c.get():
            conditions.append("C")
        if self._bench_d.get():
            conditions.append("D")

        if not conditions:
            messagebox.showinfo(
                "No Conditions", "Select at least one benchmark condition."
            )
            return

        runs_root = _project / "runs"
        ocr_dpi = int(self._bench_ocr_dpi.get())
        self.log_panel.clear()

        self._worker = PipelineWorker(self.root, self.log_panel)

        def target():
            from run_benchmark import (
                _CONDITIONS,
                _build_comparison,
                _print_table,
                _read_manifest,
            )
            from run_pdf_batch import run_pdf

            from plancheck.config import GroupingConfig

            manifests = {}
            for cond in conditions:
                if cond not in _CONDITIONS:
                    continue
                cond_def = _CONDITIONS[cond]
                cfg = GroupingConfig(**vars(cond_def["cfg"]))
                cfg.ocr_reconcile_resolution = ocr_dpi
                prefix = f"{pdf.stem[:15]}_bench{cond}".replace(" ", "_")
                print(f"\n--- Condition {cond}: {cond_def['label']} ---")
                rd = run_pdf(
                    pdf=pdf,
                    start=0,
                    end=None,
                    resolution=200,
                    run_root=runs_root,
                    run_prefix=prefix,
                    cfg=cfg,
                )
                manifests[cond] = _read_manifest(rd)

            comparison = _build_comparison(manifests)
            _print_table(comparison)
            return comparison

        def on_done(result, error, elapsed):
            if not error:
                self.log_panel.write("Benchmark complete.", "SUCCESS")

        self._worker.run(target, on_done=on_done)

    # ------------------------------------------------------------------
    # VOCRPP Preview
    # ------------------------------------------------------------------

    def _run_preprocess_preview(self) -> None:
        pdf = self.state.pdf_path
        if not pdf:
            messagebox.showwarning("No PDF", "Select a PDF in the Pipeline tab first.")
            return

        dpi = int(self._pp_dpi.get())
        self.log_panel.clear()

        self._worker = PipelineWorker(self.root, self.log_panel)

        def target():
            import subprocess

            args = [
                f'"{str(pdf)}"',
                f"--render-dpi {dpi}",
                "--start 0",
                f'--run-root "{str(_project / "runs")}"',
            ]
            # Build flags
            cmd = f'python scripts/diagnostics/run_ocr_preprocess.py {" ".join(args)}'
            print(f"Running: {cmd}")
            result = subprocess.run(
                [
                    "python",
                    str(_project / "scripts" / "diagnostics" / "run_ocr_preprocess.py"),
                ]
                + [
                    str(pdf),
                    "--render-dpi",
                    str(dpi),
                    "--start",
                    "0",
                    "--run-root",
                    str(_project / "runs"),
                ],
                capture_output=True,
                text=True,
                cwd=str(_project),
            )
            if result.stdout:
                print(result.stdout)
            if result.returncode != 0 and result.stderr:
                print(f"STDERR: {result.stderr}")
            return result.returncode

        def on_done(result, error, elapsed):
            if not error:
                self.log_panel.write("Preprocessing complete.", "SUCCESS")

        self._worker.run(target, on_done=on_done)

    # ------------------------------------------------------------------
    # Tuning Harness
    # ------------------------------------------------------------------

    def _run_tuning(self) -> None:
        pdf = self.state.pdf_path
        if not pdf:
            messagebox.showwarning("No PDF", "Select a PDF in the Pipeline tab first.")
            return

        self.log_panel.clear()
        self._worker = PipelineWorker(self.root, self.log_panel)

        def target():
            from run_tuning_harness import _default_recipe, print_comparison, run_trial

            from plancheck.config import GroupingConfig

            recipe = _default_recipe()
            trials = []
            for knob_set in recipe:
                trial = run_trial(
                    pdf=pdf,
                    start=0,
                    end=1,
                    resolution=200,
                    run_root=_project / "runs",
                    knobs=knob_set,
                )
                trials.append(trial)
            print_comparison(trials)
            return trials

        def on_done(result, error, elapsed):
            if not error:
                self.log_panel.write("Tuning sweep complete.", "SUCCESS")

        self._worker.run(target, on_done=on_done)

    def _load_recipe(self) -> None:
        path = filedialog.askopenfilename(
            title="Load Tuning Recipe",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
        )
        if path:
            self.log_panel.write(f"Recipe loaded: {Path(path).name}", "INFO")

    def _list_recipes(self) -> None:
        self.log_panel.clear()
        try:
            from run_tuning_harness import list_recipes

            recipes = list_recipes()
            if not recipes:
                self.log_panel.write("No saved recipes found.", "INFO")
            else:
                for r in recipes:
                    self.log_panel.write(
                        f"  {r.get('name', '?')}: {r.get('knob_count', '?')} knobs, {r.get('created', '?')}",
                        "INFO",
                    )
        except Exception as e:
            self.log_panel.write(f"Error: {e}", "ERROR")

    # ------------------------------------------------------------------
    # Grouping Playground
    # ------------------------------------------------------------------

    def _load_boxes_json(self) -> None:
        path = filedialog.askopenfilename(
            title="Load boxes.json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
            initialdir=str(_project / "runs"),
        )
        if path:
            self._boxes_path = Path(path)
            self._boxes_label_var.set(self._boxes_path.name)

    def _run_grouping(self) -> None:
        if not self._boxes_path or not self._boxes_path.exists():
            messagebox.showwarning("No File", "Load a boxes.json file first.")
            return

        self.log_panel.clear()
        self._worker = PipelineWorker(self.root, self.log_panel)

        def target():
            import json as json_mod

            from plancheck import GlyphBox, GroupingConfig, build_clusters_v2, nms_prune
            from plancheck.grouping import group_notes_columns, link_continued_columns

            data = json_mod.loads(self._boxes_path.read_text(encoding="utf-8"))
            tokens = [GlyphBox.from_dict(d) for d in data]
            print(f"Loaded {len(tokens)} tokens")

            cfg = GroupingConfig()
            tokens = nms_prune(tokens, cfg.iou_prune)
            # Use a reasonable page height
            page_h = max(b.y1 for b in tokens) if tokens else 792
            blocks = build_clusters_v2(tokens, page_h, cfg)
            cols = group_notes_columns(blocks, cfg=cfg)
            link_continued_columns(cols, blocks=blocks, cfg=cfg)

            print(f"Blocks: {len(blocks)}")
            print(f"Notes columns: {len(cols)}")
            for i, blk in enumerate(blocks):
                bbox = blk.bbox()
                items = len(blk.lines) if blk.lines else len(blk.rows)
                lbl = blk.label or ""
                print(
                    f"  Block {i}: items={items} table={blk.is_table} notes={blk.is_notes} label={lbl} bbox=({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f})"
                )
            return blocks

        def on_done(result, error, elapsed):
            if not error:
                self.log_panel.write("Grouping complete.", "SUCCESS")

        self._worker.run(target, on_done=on_done)
