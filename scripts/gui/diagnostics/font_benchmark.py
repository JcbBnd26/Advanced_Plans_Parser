"""Font metrics diagnostics and A/B/C/D benchmark sections."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from ..widgets import CollapsibleFrame, LogPanel
from ..worker import PipelineWorker

# ---------------------------------------------------------------------------
# Section 1 – Font Diagnostics
# ---------------------------------------------------------------------------


class FontDiagnosticsSection(CollapsibleFrame):
    """Collapsible section for font metrics diagnostic tools."""

    def __init__(
        self,
        parent: tk.Widget,
        log_panel: LogPanel,
        state: Any,
        root: tk.Tk,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            parent, "Font Metrics Diagnostics", initially_open=True, **kwargs
        )
        self._log = log_panel
        self._state = state
        self._root = root
        self._worker: PipelineWorker | None = None
        self._build()

    def _build(self) -> None:
        fc = self.content
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

    def _run_font_diagnostics(self) -> None:
        pdf = self._state.pdf_path
        if not pdf:
            messagebox.showwarning("No PDF", "Select a PDF in the Pipeline tab first.")
            return
        if not (self.font_metrics_var.get() or self.visual_metrics_var.get()):
            messagebox.showinfo("No Tools", "Enable at least one diagnostics tool.")
            return

        runs_root = Path("runs")
        self._log.clear()
        self._worker = PipelineWorker(self._root, self._log)

        def target():
            from ...diagnostics.run_font_metrics_diagnostics import run_diagnostics
            from ...utils.run_utils import make_run_dir

            run_dir = make_run_dir(
                runs_root=runs_root,
                label=f"fontdiag_{pdf.stem.replace(' ', '_')[:15]}",
            )
            return run_diagnostics(
                pdf=pdf,
                out_dir=run_dir,
                start=0,
                end=None,
                run_heuristic=self.font_metrics_var.get(),
                run_visual=self.visual_metrics_var.get(),
                visual_resolution=300,
            )

        def on_done(result, error, elapsed):
            if result and not error:
                self._log.write(f"Report saved: {result}", "SUCCESS")

        self._worker.run(target, on_done=on_done)


# ---------------------------------------------------------------------------
# Section 2 – A/B/C/D Benchmark
# ---------------------------------------------------------------------------


class BenchmarkSection(CollapsibleFrame):
    """Collapsible section for the A/B/C/D pipeline benchmark runner."""

    def __init__(
        self,
        parent: tk.Widget,
        log_panel: LogPanel,
        state: Any,
        root: tk.Tk,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, "A/B/C/D Benchmark", **kwargs)
        self._log = log_panel
        self._state = state
        self._root = root
        self._worker: PipelineWorker | None = None
        self._build()

    def _build(self) -> None:
        bc = self.content
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

    def _run_benchmark(self) -> None:
        pdf = self._state.pdf_path
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

        runs_root = Path("runs")
        ocr_dpi = int(self._bench_ocr_dpi.get())
        self._log.clear()
        self._worker = PipelineWorker(self._root, self._log)

        def target():
            from plancheck.config import GroupingConfig

            from ...diagnostics.run_benchmark import (
                _CONDITIONS,
                _build_comparison,
                _print_table,
                _read_manifest,
            )
            from ...runners.run_pdf_batch import run_pdf

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
                self._log.write("Benchmark complete.", "SUCCESS")

        self._worker.run(target, on_done=on_done)
