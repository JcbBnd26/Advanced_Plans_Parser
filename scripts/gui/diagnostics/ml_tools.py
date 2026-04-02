"""ML training and calibration diagnostic sections."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from ..widgets import CollapsibleFrame, LogPanel
from ..worker import PipelineWorker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ML Runtime Summary
# ---------------------------------------------------------------------------


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
        db_path = self._state.db_path()
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
        db_path = self._state.db_path()
        if not db_path.exists():
            self._log.write("No corrections.db found.", "WARNING")
            return

        from plancheck.corrections.store import CorrectionStore

        store = CorrectionStore(db_path)

        try:
            from plancheck.corrections.experiment_tracker import ExperimentTracker

            tracker = ExperimentTracker(store)
            experiments = tracker.list_experiments(
                limit=20, sort_by="trained_at", ascending=True
            )

            if experiments:
                self._draw_f1_chart(experiments)
                self._log.write(f"F1 chart: {len(experiments)} runs", "INFO")

                heatmap_exps = tracker.list_experiments(
                    limit=10, sort_by="trained_at", ascending=False
                )
                self._draw_perclass_heatmap(heatmap_exps)
                self._log.write("Per-class heatmap rendered.", "INFO")

                latest = heatmap_exps[0] if heatmap_exps else None
                if latest and latest.holdout_predictions:
                    self._draw_confidence_dist(latest.holdout_predictions)
                    self._log.write("Confidence distribution rendered.", "INFO")

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
        """Draw per-class F1 heatmap."""
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

        all_classes = set()
        for exp in experiments:
            pc = exp.per_class if hasattr(exp, "per_class") else {}
            all_classes.update(pc.keys())

        if not all_classes:
            return

        classes = sorted(all_classes)
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

        try:
            recent = store.get_recent_corrections(limit=100)
        except Exception:  # noqa: BLE001 — chart is optional
            return

        if not recent:
            return

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
        db_path = self._state.db_path()
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

        db_path = self._state.db_path()
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
