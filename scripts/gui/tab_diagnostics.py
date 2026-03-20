"""Tab 4 – Diagnostics: font metrics, benchmark runner, and ML tools.

Collapsible sections for each diagnostic tool with embedded output.

Note: VOCRPP preview, config tuning harness, and grouping playground have
been removed from the GUI – those functions will be managed by the LLM
layer in a future release.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from .widgets import CollapsibleFrame, LogPanel
from .worker import PipelineWorker


def _resolve_runtime_path(raw_path: str) -> Path | None:
    """Resolve a potentially relative model path against the workspace."""
    path_text = raw_path.strip()
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _build_ml_runtime_summary(
    cfg: Any,
    *,
    pending_corrections: int | None = None,
    db_present: bool = False,
) -> dict[str, str]:
    """Build a lightweight runtime summary for ML-related diagnostics."""
    stage1_path = _resolve_runtime_path(getattr(cfg, "ml_model_path", ""))
    stage2_path = _resolve_runtime_path(getattr(cfg, "ml_stage2_model_path", ""))
    drift_path = _resolve_runtime_path(getattr(cfg, "ml_drift_stats_path", ""))
    candidate_path = _resolve_runtime_path(
        getattr(cfg, "vocr_cand_classifier_path", "")
    )

    threshold = getattr(cfg, "ml_retrain_threshold", 50)
    if db_present and pending_corrections is not None:
        retrain_summary = f"{pending_corrections}/{threshold} pending"
        if pending_corrections >= threshold:
            retrain_summary += " — recommended"
    else:
        retrain_summary = "No corrections database"

    return {
        "Routing": (
            "Hierarchical title refinement"
            if getattr(cfg, "ml_hierarchical_enabled", False)
            else "Stage 1 only"
        ),
        "Stage 1 model": (
            f"Ready ({stage1_path.name})"
            if stage1_path and stage1_path.exists()
            else "Missing"
        ),
        "Stage 2 model": (
            "Inactive"
            if not getattr(cfg, "ml_hierarchical_enabled", False)
            else (
                f"Ready ({stage2_path.name})"
                if stage2_path and stage2_path.exists()
                else "Missing"
            )
        ),
        "Drift detection": (
            "Disabled"
            if not getattr(cfg, "ml_drift_enabled", False)
            else (
                f"Enabled (threshold {getattr(cfg, 'ml_drift_threshold', 0.0):.2f}, "
                f"stats {'ready' if drift_path and drift_path.exists() else 'missing'})"
            )
        ),
        "Retrain readiness": retrain_summary,
        "Candidate ML": (
            "Disabled"
            if not getattr(cfg, "vocr_cand_ml_enabled", False)
            else (
                f"Enabled ({candidate_path.name})"
                if candidate_path and candidate_path.exists()
                else "Enabled, model missing"
            )
        ),
        "Layout runtime": (
            "Disabled"
            if not getattr(cfg, "ml_layout_enabled", False)
            else getattr(cfg, "ml_layout_model_path", "").strip() or "Model missing"
        ),
    }


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
            from ..diagnostics.run_font_metrics_diagnostics import run_diagnostics
            from ..utils.run_utils import make_run_dir

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

            from ..diagnostics.run_benchmark import (
                _CONDITIONS,
                _build_comparison,
                _print_table,
                _read_manifest,
            )
            from ..runners.run_pdf_batch import run_pdf

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


# ---------------------------------------------------------------------------
# Section 3 – ML Calibration
# ---------------------------------------------------------------------------


class MLCalibrationSection(CollapsibleFrame):
    """Collapsible section for ML model calibration diagnostics."""

    def __init__(
        self,
        parent: tk.Widget,
        log_panel: LogPanel,
        state: Any,
        root: tk.Tk,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, "ML Calibration", **kwargs)
        self._log = log_panel
        self._state = state
        self._root = root
        self._worker: PipelineWorker | None = None
        self._build()

    def _build(self) -> None:
        cc = self.content
        cc.columnconfigure(0, weight=1)

        self._calibration_target_var = tk.StringVar(value="stage1")

        ttk.Label(
            cc,
            text="Generate a reliability diagram and ECE for Stage 1 or Stage 2.",
            foreground="gray",
        ).grid(row=0, column=0, sticky="w", pady=2)

        target_row = ttk.Frame(cc)
        target_row.grid(row=1, column=0, sticky="w", pady=(2, 4))
        ttk.Label(target_row, text="Target:").pack(side="left")
        ttk.Radiobutton(
            target_row,
            text="Stage 1",
            variable=self._calibration_target_var,
            value="stage1",
        ).pack(side="left", padx=(8, 4))
        ttk.Radiobutton(
            target_row,
            text="Stage 2",
            variable=self._calibration_target_var,
            value="stage2",
        ).pack(side="left", padx=4)

        cal_btns = ttk.Frame(cc)
        cal_btns.grid(row=2, column=0, sticky="w", pady=4)
        ttk.Button(
            cal_btns,
            text="Generate Reliability Diagram",
            command=self._run_calibration_diagram,
        ).pack(side="left", padx=2)

        self._cal_canvas_frame = ttk.Frame(cc)
        self._cal_canvas_frame.grid(row=3, column=0, sticky="ew", pady=2)

    def _resolve_calibration_target(self) -> tuple[str, Path, Path]:
        """Resolve the selected calibration target to its model and JSONL paths."""
        cfg = self._state.config
        if self._calibration_target_var.get() == "stage2":
            return (
                "Stage 2",
                Path(cfg.ml_stage2_model_path),
                Path("data") / "training_data_stage2.jsonl",
            )
        return (
            "Stage 1",
            Path(cfg.ml_model_path),
            Path("data") / "training_data.jsonl",
        )

    def _run_calibration_diagram(self) -> None:
        target_name, model_path, jsonl_path = self._resolve_calibration_target()

        if not model_path.exists():
            messagebox.showwarning(
                "No Model",
                f"No trained {target_name} model found. Run training first.",
            )
            return
        if not jsonl_path.exists():
            messagebox.showwarning(
                "No Data",
                f"No calibration data found for {target_name}. Run training first to generate it.",
            )
            return

        self._log.clear()
        self._worker = PipelineWorker(self._root, self._log)

        def target():
            if target_name == "Stage 2":
                from plancheck.corrections.subtype_classifier import (
                    TitleSubtypeClassifier,
                )

                clf = TitleSubtypeClassifier(model_path=model_path)
            else:
                from plancheck.corrections.classifier import ElementClassifier

                clf = ElementClassifier(model_path=model_path)
            return clf.calibration_curve(jsonl_path)

        def on_done(result, error, elapsed):
            if error or not result:
                return
            ece = result.get("ece", 0.0)
            curves = result.get("curves", {})
            self._log.write(
                f"{target_name} Expected Calibration Error (ECE): {ece:.4f}",
                "INFO",
            )
            for cls_name, data in curves.items():
                n_bins = len(data["mean_predicted"])
                self._log.write(f"  {cls_name}: {n_bins} bins", "INFO")
            self._draw_reliability_diagram(curves, ece)

        self._worker.run(target, on_done=on_done)

    def _draw_reliability_diagram(self, curves: dict, ece: float) -> None:
        """Render a reliability diagram into the calibration canvas frame."""
        for w in self._cal_canvas_frame.winfo_children():
            w.destroy()

        try:
            import matplotlib

            matplotlib.use("Agg")
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
        except ImportError:
            self._log.write(
                "matplotlib not installed — cannot render diagram.", "WARNING"
            )
            return

        fig = Figure(figsize=(5, 4), dpi=96)
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")

        for cls_name, data in curves.items():
            mp = data["mean_predicted"]
            fp = data["fraction_positive"]
            ax.plot(mp, fp, "o-", markersize=4, label=cls_name)

        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Reliability Diagram  (ECE = {ece:.4f})")
        ax.legend(loc="lower right", fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self._cal_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", expand=True)


class MLRuntimeSummarySection(CollapsibleFrame):
    """Collapsible section summarizing ML runtime posture."""

    def __init__(
        self,
        parent: tk.Widget,
        log_panel: LogPanel,
        state: Any,
        root: tk.Tk,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, "ML Runtime Summary", **kwargs)
        self._log = log_panel
        self._state = state
        self._root = root
        self._value_labels: dict[str, ttk.Label] = {}
        self._build()

    def _build(self) -> None:
        section = self.content
        section.columnconfigure(1, weight=1)

        ttk.Label(
            section,
            text="Quick summary of routing, retrain readiness, drift, and runtime model availability.",
            foreground="gray",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=2)

        keys = [
            "Routing",
            "Stage 1 model",
            "Stage 2 model",
            "Drift detection",
            "Retrain readiness",
            "Candidate ML",
            "Layout runtime",
        ]
        for index, key in enumerate(keys, start=1):
            ttk.Label(section, text=f"{key}:", font=("TkDefaultFont", 9, "bold")).grid(
                row=index,
                column=0,
                sticky="nw",
                padx=(0, 8),
                pady=1,
            )
            value_label = ttk.Label(section, text="", wraplength=420)
            value_label.grid(row=index, column=1, sticky="w", pady=1)
            self._value_labels[key] = value_label

        ttk.Button(section, text="Refresh Summary", command=self._refresh_summary).grid(
            row=len(keys) + 1,
            column=0,
            sticky="w",
            pady=(6, 2),
        )
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        pending_corrections = None
        db_present = False
        db_path = Path("data") / "corrections.db"
        if db_path.exists():
            db_present = True
            try:
                from plancheck.corrections.store import CorrectionStore

                store = CorrectionStore(db_path)
                try:
                    pending_corrections = store.count_corrections_since_last_train()
                finally:
                    store.close()
            except Exception as exc:  # noqa: BLE001
                self._log.write(f"Runtime summary refresh failed: {exc}", "WARNING")

        summary = _build_ml_runtime_summary(
            self._state.config,
            pending_corrections=pending_corrections,
            db_present=db_present,
        )
        for key, label in self._value_labels.items():
            label.configure(text=summary.get(key, "—"))


# ---------------------------------------------------------------------------
# Section 4 – Training Progress Charts
# ---------------------------------------------------------------------------


class TrainingProgressSection(CollapsibleFrame):
    """Collapsible section with training progress visualization charts."""

    def __init__(
        self,
        parent: tk.Widget,
        log_panel: LogPanel,
        state: Any,
        root: tk.Tk,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, "Training Progress (ML Charts)", **kwargs)
        self._log = log_panel
        self._state = state
        self._root = root
        self._charts_loaded = False
        self._build()

    def expand(self) -> None:
        """Override expand to auto-refresh charts on first expand."""
        super().expand()
        if not self._charts_loaded:
            self._charts_loaded = True
            # Schedule refresh after UI settles
            self._root.after(100, self._refresh_all)

    def _build(self) -> None:
        cc = self.content
        cc.columnconfigure(0, weight=1)
        cc.columnconfigure(1, weight=1)

        # Chart frames - 2x2 grid
        self._f1_canvas = ttk.Frame(cc)
        self._f1_canvas.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        self._heatmap_canvas = ttk.Frame(cc)
        self._heatmap_canvas.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)

        self._corrections_canvas = ttk.Frame(cc)
        self._corrections_canvas.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)

        self._confidence_canvas = ttk.Frame(cc)
        self._confidence_canvas.grid(row=1, column=1, sticky="nsew", padx=2, pady=2)

        # Refresh button
        btn_row = ttk.Frame(cc)
        btn_row.grid(row=2, column=0, columnspan=2, pady=4)
        ttk.Button(btn_row, text="Refresh Charts", command=self._refresh_all).pack()

    def _refresh_all(self) -> None:
        """Refresh all training progress charts."""
        db_path = Path("data") / "corrections.db"
        if not db_path.exists():
            self._log.write("No corrections.db found.", "WARNING")
            return

        from plancheck.corrections.store import CorrectionStore

        store = CorrectionStore(db_path)

        try:
            from plancheck.corrections.experiment_tracker import ExperimentTracker

            tracker = ExperimentTracker(store)
            # Fetch experiments with per_class and holdout_predictions included
            experiments = tracker.list_experiments(
                limit=20, sort_by="trained_at", ascending=True
            )

            if experiments:
                self._draw_f1_chart(experiments)
                self._log.write(f"F1 chart: {len(experiments)} runs", "INFO")

                # Per-class heatmap from experiments (most recent first for heatmap)
                heatmap_exps = tracker.list_experiments(
                    limit=10, sort_by="trained_at", ascending=False
                )
                self._draw_perclass_heatmap(heatmap_exps)
                self._log.write("Per-class heatmap rendered.", "INFO")

                # Get holdout predictions from most recent run
                latest = heatmap_exps[0] if heatmap_exps else None
                if latest and latest.holdout_predictions:
                    self._draw_confidence_dist(latest.holdout_predictions)
                    self._log.write("Confidence distribution rendered.", "INFO")

            # Corrections trend
            self._draw_corrections_trend(store)

        except Exception as exc:
            self._log.write(f"Chart refresh failed: {exc}", "ERROR")
        finally:
            store.close()

    def _draw_f1_chart(self, experiments: list) -> None:
        """Draw F1 Over Time line chart."""
        for w in self._f1_canvas.winfo_children():
            w.destroy()

        try:
            import matplotlib

            matplotlib.use("Agg")
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
        except ImportError:
            return

        fig = Figure(figsize=(4, 3), dpi=80)
        ax = fig.add_subplot(111)

        dates = list(range(len(experiments)))
        f1_weighted = [e.f1_weighted for e in experiments]
        f1_macro = [e.f1_macro for e in experiments]

        ax.plot(dates, f1_weighted, "b-o", markersize=4, label="F1 Weighted")
        ax.plot(dates, f1_macro, "g--s", markersize=4, label="F1 Macro")

        ax.set_xlabel("Training Run")
        ax.set_ylabel("F1 Score")
        ax.set_title("F1 Over Time")
        ax.legend(loc="lower right", fontsize=7)
        ax.set_ylim(0, 1)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self._f1_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _draw_perclass_heatmap(self, experiments: list) -> None:
        """Draw per-class F1 heatmap.

        Parameters
        ----------
        experiments : list[ExperimentSummary]
            List of experiment summaries with per_class data.
        """
        for w in self._heatmap_canvas.winfo_children():
            w.destroy()

        try:
            import matplotlib

            matplotlib.use("Agg")
            import numpy as np
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
        except ImportError:
            return

        # Collect all classes across runs
        all_classes = set()
        for exp in experiments:
            pc = exp.per_class if hasattr(exp, "per_class") else {}
            all_classes.update(pc.keys())

        if not all_classes:
            return

        classes = sorted(all_classes)
        # Build matrix: rows = runs (most recent at top), cols = classes
        matrix = []
        run_labels = []
        for exp in experiments:
            pc = exp.per_class if hasattr(exp, "per_class") else {}
            row = [pc.get(c, {}).get("f1", 0.0) for c in classes]
            matrix.append(row)
            run_id = exp.run_id if hasattr(exp, "run_id") else ""
            run_labels.append(run_id[:8])

        matrix = np.array(matrix)

        fig = Figure(figsize=(4, 3), dpi=80)
        ax = fig.add_subplot(111)

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(len(run_labels)))
        ax.set_yticklabels(run_labels, fontsize=6)
        ax.set_title("Per-Class F1 by Run", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self._heatmap_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _draw_corrections_trend(self, store: Any) -> None:
        """Draw corrections per document trend chart."""
        for w in self._corrections_canvas.winfo_children():
            w.destroy()

        try:
            import matplotlib

            matplotlib.use("Agg")
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
        except ImportError:
            return

        # Get recent corrections grouped by date
        try:
            recent = store.get_recent_corrections(limit=100)
        except Exception:  # noqa: BLE001 — chart is optional
            return

        if not recent:
            return

        # Group by date
        from collections import Counter

        date_counts = Counter()
        for corr in recent:
            ts = corr.get("corrected_at", "")[:10]  # YYYY-MM-DD
            if ts:
                date_counts[ts] += 1

        if not date_counts:
            return

        dates = sorted(date_counts.keys())
        counts = [date_counts[d] for d in dates]

        fig = Figure(figsize=(4, 3), dpi=80)
        ax = fig.add_subplot(111)

        ax.bar(range(len(dates)), counts, color="steelblue")
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels([d[5:] for d in dates], rotation=45, ha="right", fontsize=6)
        ax.set_xlabel("Date")
        ax.set_ylabel("Corrections")
        ax.set_title("Corrections per Day", fontsize=9)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self._corrections_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _draw_confidence_dist(self, holdout_predictions: list) -> None:
        """Draw confidence distribution histogram."""
        for w in self._confidence_canvas.winfo_children():
            w.destroy()

        try:
            import matplotlib

            matplotlib.use("Agg")
            import numpy as np
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
        except ImportError:
            return

        if not holdout_predictions:
            return

        correct = [
            p.get("confidence", 0)
            for p in holdout_predictions
            if p.get("label_true") == p.get("label_pred")
        ]
        incorrect = [
            p.get("confidence", 0)
            for p in holdout_predictions
            if p.get("label_true") != p.get("label_pred")
        ]

        fig = Figure(figsize=(4, 3), dpi=80)
        ax = fig.add_subplot(111)

        bins = np.linspace(0, 1, 11)
        if correct:
            ax.hist(correct, bins, alpha=0.7, label="Correct", color="green")
        if incorrect:
            ax.hist(incorrect, bins, alpha=0.7, label="Incorrect", color="red")

        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        ax.set_title("Confidence Distribution", fontsize=9)
        ax.legend(loc="upper left", fontsize=7)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self._confidence_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


# ---------------------------------------------------------------------------
# Section 5 – Model Comparison
# ---------------------------------------------------------------------------


class ModelComparisonSection(CollapsibleFrame):
    """Collapsible section for comparing two training runs side-by-side."""

    def __init__(
        self,
        parent: tk.Widget,
        log_panel: LogPanel,
        state: Any,
        root: tk.Tk,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, "Model Comparison", **kwargs)
        self._log = log_panel
        self._state = state
        self._root = root
        self._build()

    def _build(self) -> None:
        mc = self.content
        mc.columnconfigure(1, weight=1)

        ttk.Label(
            mc,
            text="Compare two training runs side-by-side.",
            foreground="gray",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=2)

        ttk.Label(mc, text="Run A:").grid(row=1, column=0, sticky="w", pady=2)
        self._cmp_run_a = tk.StringVar()
        self._cmp_combo_a = ttk.Combobox(
            mc, textvariable=self._cmp_run_a, state="readonly", width=40
        )
        self._cmp_combo_a.grid(row=1, column=1, sticky="w", padx=4)

        ttk.Label(mc, text="Run B:").grid(row=2, column=0, sticky="w", pady=2)
        self._cmp_run_b = tk.StringVar()
        self._cmp_combo_b = ttk.Combobox(
            mc, textvariable=self._cmp_run_b, state="readonly", width=40
        )
        self._cmp_combo_b.grid(row=2, column=1, sticky="w", padx=4)

        cmp_btns = ttk.Frame(mc)
        cmp_btns.grid(row=3, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Button(
            cmp_btns, text="Refresh Runs", command=self._refresh_training_runs
        ).pack(side="left", padx=2)
        ttk.Button(cmp_btns, text="Compare", command=self._compare_runs).pack(
            side="left", padx=2
        )

    def _refresh_training_runs(self) -> None:
        db_path = Path("data") / "corrections.db"
        if not db_path.exists():
            messagebox.showwarning("No Database", "corrections.db not found.")
            return

        from plancheck.corrections.experiment_tracker import ExperimentTracker
        from plancheck.corrections.store import CorrectionStore

        store = CorrectionStore(db_path)
        tracker = ExperimentTracker(store)
        experiments = tracker.list_experiments(limit=50, sort_by="trained_at")
        store.close()

        if not experiments:
            messagebox.showinfo("No Runs", "No training runs found.")
            return

        display_values = [
            f"{exp.run_id} | F1={exp.f1_weighted:.3f} | {exp.trained_at[:16]}"
            for exp in experiments
        ]

        self._cmp_combo_a["values"] = display_values
        self._cmp_combo_b["values"] = display_values
        if len(display_values) >= 2:
            self._cmp_combo_a.current(0)
            self._cmp_combo_b.current(1)
        elif len(display_values) == 1:
            self._cmp_combo_a.current(0)
            self._cmp_combo_b.current(0)

        self._log.write(f"Loaded {len(experiments)} training run(s).", "INFO")

    def _compare_runs(self) -> None:
        run_a_str = self._cmp_run_a.get()
        run_b_str = self._cmp_run_b.get()
        if not run_a_str or not run_b_str:
            messagebox.showwarning(
                "Select Runs", "Select two training runs to compare."
            )
            return

        db_path = Path("data") / "corrections.db"
        from plancheck.corrections.experiment_tracker import ExperimentTracker
        from plancheck.corrections.store import CorrectionStore

        store = CorrectionStore(db_path)
        tracker = ExperimentTracker(store)

        id_a = run_a_str.split(" | ")[0].strip()
        id_b = run_b_str.split(" | ")[0].strip()

        try:
            threshold = getattr(self._state.config, "ml_comparison_threshold", 0.005)
            comparison = tracker.compare_experiments(id_a, id_b, threshold=threshold)
        except Exception as exc:
            store.close()
            messagebox.showwarning("Comparison Error", str(exc))
            return
        store.close()

        run_a = comparison.run_a
        run_b = comparison.run_b

        self._log.clear()
        self._log.write(f"Comparing {id_a} vs {id_b}", "INFO")
        self._log.write(
            f"{'Metric':<25} {'Run A':>10} {'Run B':>10} {'Delta':>10}", "INFO"
        )
        self._log.write("-" * 57, "INFO")

        for metric in ("accuracy", "f1_macro", "f1_weighted"):
            va = getattr(run_a, metric, 0.0)
            vb = getattr(run_b, metric, 0.0)
            delta = vb - va
            sign = "+" if delta >= 0 else ""
            self._log.write(
                f"{metric:<25} {va:>10.4f} {vb:>10.4f} {sign}{delta:>9.4f}", "INFO"
            )

        # Show per-class deltas from comparison
        if comparison.per_class_deltas:
            self._log.write("", "INFO")
            self._log.write(f"{'Class':<20} {'Delta':>10}", "INFO")
            self._log.write("-" * 32, "INFO")
            for cls, delta in sorted(comparison.per_class_deltas.items()):
                sign = "+" if delta >= 0 else ""
                self._log.write(f"{cls:<20} {sign}{delta:>9.4f}", "INFO")

        if comparison.improved_classes:
            self._log.write("", "INFO")
            self._log.write(
                f"Improved: {', '.join(comparison.improved_classes)}", "INFO"
            )
        if comparison.regressed_classes:
            self._log.write(
                f"Regressed: {', '.join(comparison.regressed_classes)}", "WARNING"
            )


# ---------------------------------------------------------------------------
# Section 5 – Layout Model (LayoutLMv3)
# ---------------------------------------------------------------------------


class LayoutModelSection(CollapsibleFrame):
    """Collapsible section for LayoutLMv3 layout detection."""

    def __init__(
        self,
        parent: tk.Widget,
        log_panel: LogPanel,
        state: Any,
        root: tk.Tk,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, "Layout Model (LayoutLMv3)", **kwargs)
        self._log = log_panel
        self._state = state
        self._root = root
        self._worker: PipelineWorker | None = None
        self._build()

    def _build(self) -> None:
        lm = self.content
        lm.columnconfigure(1, weight=1)

        ttk.Label(
            lm,
            text="Run LayoutLMv3 layout detection on the current page with a fine-tuned checkpoint.",
            foreground="gray",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=2)

        ttk.Label(lm, text="Model:").grid(row=1, column=0, sticky="w", pady=2)
        self._layout_model_var = tk.StringVar(value=self._initial_layout_model_value())
        ttk.Entry(lm, textvariable=self._layout_model_var, width=50).grid(
            row=1, column=1, sticky="ew", padx=4
        )

        ttk.Label(
            lm,
            text="Leave blank until you have a fine-tuned checkpoint; the base model is not usable for layout inference.",
            foreground="gray",
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=2)

        layout_btns = ttk.Frame(lm)
        layout_btns.grid(row=3, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Button(
            layout_btns, text="Run Layout Detection", command=self._run_layout_detection
        ).pack(side="left", padx=2)
        ttk.Button(
            layout_btns, text="Check Availability", command=self._check_layout_avail
        ).pack(side="left", padx=2)

    def _initial_layout_model_value(self) -> str:
        """Prefer a configured fine-tuned layout model; otherwise start blank."""
        configured = getattr(self._state.config, "ml_layout_model_path", "").strip()
        if configured == "microsoft/layoutlmv3-base":
            return ""
        return configured

    def _is_invalid_layout_model(self, model_name: str) -> bool:
        """Return True when the selected model is the unfine-tuned base checkpoint."""
        return model_name.strip() == "microsoft/layoutlmv3-base"

    def _check_layout_avail(self) -> None:
        self._log.clear()
        try:
            from plancheck.analysis.layout_model import is_layout_available

            avail = is_layout_available()
            if avail:
                model_name = self._layout_model_var.get().strip()
                if self._is_invalid_layout_model(model_name) or not model_name:
                    self._log.write(
                        "LayoutLMv3 dependencies are available, but you still need a fine-tuned checkpoint path.",
                        "WARNING",
                    )
                else:
                    self._log.write(
                        "LayoutLMv3 is available and a checkpoint path is configured.",
                        "SUCCESS",
                    )
            else:
                self._log.write(
                    "LayoutLMv3 NOT available. Install with: pip install 'plancheck[layout]'",
                    "WARNING",
                )
        except Exception as exc:
            self._log.write(f"Error checking availability: {exc}", "ERROR")

    def _run_layout_detection(self) -> None:
        pdf = self._state.pdf_path
        if pdf is None:
            messagebox.showwarning("No PDF", "Select a PDF in the Pipeline tab first.")
            return
        model_name = self._layout_model_var.get().strip()
        if not model_name:
            messagebox.showwarning(
                "No Model",
                "Enter a fine-tuned LayoutLMv3 model name or checkpoint path.",
            )
            return
        if self._is_invalid_layout_model(model_name):
            messagebox.showwarning(
                "Invalid Model",
                "The LayoutLMv3 base checkpoint has a random classification head. Use a fine-tuned checkpoint instead.",
            )
            return

        self._log.clear()
        self._worker = PipelineWorker(self._root, self._log)

        def target():
            from plancheck import GlyphBox, GroupingConfig, extract_tokens
            from plancheck.analysis.layout_model import (
                is_layout_available,
                predict_layout,
            )
            from plancheck.ingest import render_page_image

            if not is_layout_available():
                raise RuntimeError(
                    "LayoutLMv3 not available. "
                    "Install with: pip install 'plancheck[layout]'"
                )

            print(f"Loading page from {pdf}...")
            cfg = GroupingConfig()
            tokens, pw, ph = extract_tokens(str(pdf), 0, cfg)
            image = render_page_image(pdf, 0, resolution=150)

            print(f"Running layout detection ({model_name})...")
            preds = predict_layout(image, tokens, pw, ph, model_name_or_path=model_name)

            print(f"\nLayout predictions: {len(preds)}")
            for p in preds:
                bbox = p.bbox
                print(
                    f"  {p.label:20s} conf={p.confidence:.3f} "
                    f"bbox=({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}) "
                    f"tokens={len(p.token_indices)}"
                )
            return preds

        def on_done(result, error, elapsed):
            if not error:
                n = len(result) if result else 0
                self._log.write(
                    f"Layout detection complete: {n} regions ({elapsed:.1f}s).",
                    "SUCCESS",
                )

        self._worker.run(target, on_done=on_done)


# ---------------------------------------------------------------------------
# Section 6 – Text Embeddings (Sentence-Transformer)
# ---------------------------------------------------------------------------


class TextEmbeddingsSection(CollapsibleFrame):
    """Collapsible section for sentence-transformer embedding tools."""

    def __init__(
        self,
        parent: tk.Widget,
        log_panel: LogPanel,
        state: Any,
        root: tk.Tk,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, "Text Embeddings (Sentence-Transformer)", **kwargs)
        self._log = log_panel
        self._state = state
        self._root = root
        self._worker: PipelineWorker | None = None
        self._build()

    def _build(self) -> None:
        em = self.content
        em.columnconfigure(1, weight=1)

        ttk.Label(
            em,
            text="Dense semantic embeddings for block text (supplements kw_* features).",
            foreground="gray",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=2)

        ttk.Label(em, text="Model:").grid(row=1, column=0, sticky="w", pady=2)
        self._emb_model_var = tk.StringVar(value="all-MiniLM-L6-v2")
        ttk.Entry(em, textvariable=self._emb_model_var, width=50).grid(
            row=1, column=1, sticky="ew", padx=4
        )

        emb_btns = ttk.Frame(em)
        emb_btns.grid(row=2, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Button(
            emb_btns, text="Check Availability", command=self._check_embeddings_avail
        ).pack(side="left", padx=2)
        ttk.Button(emb_btns, text="Test Embedding", command=self._test_embedding).pack(
            side="left", padx=2
        )

    def _check_embeddings_avail(self) -> None:
        self._log.clear()
        try:
            from plancheck.corrections.text_embeddings import is_embeddings_available

            avail = is_embeddings_available()
            if avail:
                self._log.write("sentence-transformers is available.", "SUCCESS")
            else:
                self._log.write(
                    "sentence-transformers NOT available. Install with: "
                    "pip install 'plancheck[embeddings]'",
                    "WARNING",
                )
        except Exception as exc:
            self._log.write(f"Error checking availability: {exc}", "ERROR")

    def _test_embedding(self) -> None:
        self._log.clear()
        self._worker = PipelineWorker(self._root, self._log)
        model_name = self._emb_model_var.get().strip() or "all-MiniLM-L6-v2"

        def target():
            from plancheck.corrections.text_embeddings import (
                TextEmbedder,
                is_embeddings_available,
            )

            if not is_embeddings_available():
                raise RuntimeError(
                    "sentence-transformers not available. "
                    "Install with: pip install 'plancheck[embeddings]'"
                )

            embedder = TextEmbedder(model_name=model_name)
            test_texts = [
                "GENERAL NOTES",
                "CONSTRUCTION NOTES",
                "LEGEND",
                "ABBREVIATIONS",
                "REVISION SCHEDULE",
            ]
            embeddings = embedder.embed_batch(test_texts)
            print(f"Model: {model_name}")
            print(f"Embedding dim: {embedder.embedding_dim}")
            print(f"\nSemantic similarity test:")
            import numpy as np

            for i in range(len(test_texts)):
                for j in range(i + 1, len(test_texts)):
                    sim = float(np.dot(embeddings[i], embeddings[j]))
                    print(f"  '{test_texts[i]}' vs '{test_texts[j]}': {sim:.3f}")
            return embeddings

        def on_done(result, error, elapsed):
            if not error:
                self._log.write(f"Embedding test complete ({elapsed:.1f}s).", "SUCCESS")

        self._worker.run(target, on_done=on_done)


# ---------------------------------------------------------------------------
# Section 7 – LLM Semantic Checks
# ---------------------------------------------------------------------------


class LLMSemanticChecksSection(CollapsibleFrame):
    """Collapsible section for optional LLM-assisted content analysis."""

    def __init__(
        self,
        parent: tk.Widget,
        log_panel: LogPanel,
        state: Any,
        root: tk.Tk,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, "LLM Semantic Checks", **kwargs)
        self._log = log_panel
        self._state = state
        self._root = root
        self._worker: PipelineWorker | None = None
        self._build()

    def _build(self) -> None:
        lc = self.content
        lc.columnconfigure(1, weight=1)

        ttk.Label(
            lc,
            text="Optional LLM-assisted content analysis (off by default).",
            foreground="gray",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=2)

        ttk.Label(lc, text="Provider:").grid(row=1, column=0, sticky="w", pady=2)
        self._llm_provider_var = tk.StringVar(value="ollama")
        ttk.Combobox(
            lc,
            textvariable=self._llm_provider_var,
            width=20,
            values=["ollama", "openai", "anthropic"],
            state="readonly",
        ).grid(row=1, column=1, sticky="w", padx=4)

        ttk.Label(lc, text="Model:").grid(row=2, column=0, sticky="w", pady=2)
        self._llm_model_var = tk.StringVar(value="llama3.1:8b")
        ttk.Entry(lc, textvariable=self._llm_model_var, width=50).grid(
            row=2, column=1, sticky="ew", padx=4
        )

        llm_btns = ttk.Frame(lc)
        llm_btns.grid(row=3, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Button(
            llm_btns, text="Check Availability", command=self._check_llm_avail
        ).pack(side="left", padx=2)
        ttk.Button(llm_btns, text="Run LLM Checks", command=self._run_llm_checks).pack(
            side="left", padx=2
        )

    def _check_llm_avail(self) -> None:
        self._log.clear()
        provider = self._llm_provider_var.get()
        try:
            from plancheck.checks.llm_checks import is_llm_available

            avail = is_llm_available(provider)
            if avail:
                self._log.write(f"LLM provider '{provider}' is available.", "SUCCESS")
            else:
                self._log.write(
                    f"LLM provider '{provider}' NOT available. "
                    f"Install with: pip install 'plancheck[llm]'",
                    "WARNING",
                )
        except Exception as exc:
            self._log.write(f"Error checking availability: {exc}", "ERROR")

    def _run_llm_checks(self) -> None:
        pdf = self._state.pdf_path
        if pdf is None:
            messagebox.showwarning("No PDF", "Select a PDF in the Pipeline tab first.")
            return

        provider = self._llm_provider_var.get()
        model = self._llm_model_var.get().strip()
        if not model:
            messagebox.showwarning("No Model", "Enter an LLM model name.")
            return

        self._log.clear()
        self._worker = PipelineWorker(self._root, self._log)

        def target():
            from plancheck import GroupingConfig
            from plancheck.checks.llm_checks import is_llm_available, run_llm_checks
            from plancheck.pipeline import run_pipeline

            if not is_llm_available(provider):
                raise RuntimeError(
                    f"LLM provider '{provider}' not available. "
                    f"Install with: pip install 'plancheck[llm]'"
                )

            print(f"Running pipeline on page 0...")
            cfg = GroupingConfig()
            pr = run_pipeline(pdf, 0, cfg=cfg)

            print(f"Running LLM checks ({provider}/{model})...")
            findings = run_llm_checks(
                notes_columns=pr.notes_columns,
                provider=provider,
                model=model,
            )
            print(f"\nLLM findings: {len(findings)}")
            for f in findings:
                print(f"  [{f.severity}] {f.check_id}: {f.message}")
            return findings

        def on_done(result, error, elapsed):
            if not error:
                n = len(result) if result else 0
                self._log.write(
                    f"LLM checks complete: {n} findings ({elapsed:.1f}s).", "SUCCESS"
                )

        self._worker.run(target, on_done=on_done)


# ---------------------------------------------------------------------------
# Section 8 – Cross-Page GNN
# ---------------------------------------------------------------------------


class CrossPageGNNSection(CollapsibleFrame):
    """Collapsible section for cross-page GNN inconsistency detection."""

    def __init__(
        self,
        parent: tk.Widget,
        log_panel: LogPanel,
        state: Any,
        root: tk.Tk,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, "Cross-Page GNN", **kwargs)
        self._log = log_panel
        self._state = state
        self._root = root
        self._build()

    def _build(self) -> None:
        gn = self.content
        gn.columnconfigure(1, weight=1)

        ttk.Label(
            gn,
            text="Graph neural network for cross-page inconsistency detection.",
            foreground="gray",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=2)

        ttk.Label(gn, text="Model:").grid(row=1, column=0, sticky="w", pady=2)
        self._gnn_model_var = tk.StringVar(value="data/document_gnn.pt")
        ttk.Entry(gn, textvariable=self._gnn_model_var, width=50).grid(
            row=1, column=1, sticky="ew", padx=4
        )

        gnn_btns = ttk.Frame(gn)
        gnn_btns.grid(row=2, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Button(
            gnn_btns, text="Check Availability", command=self._check_gnn_avail
        ).pack(side="left", padx=2)

    def _check_gnn_avail(self) -> None:
        self._log.clear()
        try:
            from plancheck.analysis.gnn import is_gnn_available

            avail = is_gnn_available()
            if avail:
                self._log.write(
                    "PyTorch Geometric is available (torch + torch_geometric).",
                    "SUCCESS",
                )
            else:
                self._log.write(
                    "PyTorch Geometric NOT available. Install with: "
                    "pip install 'plancheck[gnn]'",
                    "WARNING",
                )
        except Exception as exc:
            self._log.write(f"Error checking availability: {exc}", "ERROR")


# ---------------------------------------------------------------------------
# DiagnosticsTab – thin composer
# ---------------------------------------------------------------------------


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

        # Mousewheel scroll state (avoid unbind_all, which breaks other tabs)
        self._wheel_active: bool = False
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
        self._canvas.bind("<Enter>", lambda e: setattr(self, "_wheel_active", True))
        self._canvas.bind("<Leave>", lambda e: setattr(self, "_wheel_active", False))

        # Bind once globally; handler is gated by _wheel_active
        self.root.bind_all("<MouseWheel>", self._on_mousewheel, add="+")

        # ── Log panel ────────────────────────────────────────────────
        self.log_panel = LogPanel(self.frame, height=8)
        self.log_panel.grid(
            row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 6)
        )

        # ── Instantiate section widgets ──────────────────────────────
        section_kwargs = dict(
            log_panel=self.log_panel, state=self.state, root=self.root
        )

        sections = [
            FontDiagnosticsSection(self._inner, **section_kwargs),
            BenchmarkSection(self._inner, **section_kwargs),
            MLCalibrationSection(self._inner, **section_kwargs),
            MLRuntimeSummarySection(self._inner, **section_kwargs),
            TrainingProgressSection(self._inner, **section_kwargs),
            ModelComparisonSection(self._inner, **section_kwargs),
            LayoutModelSection(self._inner, **section_kwargs),
            TextEmbeddingsSection(self._inner, **section_kwargs),
            LLMSemanticChecksSection(self._inner, **section_kwargs),
            CrossPageGNNSection(self._inner, **section_kwargs),
        ]

        for row, section in enumerate(sections):
            section.grid(row=row, column=0, sticky="ew", **pad)

    def _on_mousewheel(self, event) -> None:
        if not self._wheel_active or not self._canvas.winfo_ismapped():
            return
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
