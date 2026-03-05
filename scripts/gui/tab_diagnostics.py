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

        ttk.Label(
            cc,
            text="Generate a reliability diagram and ECE for the trained model.",
            foreground="gray",
        ).grid(row=0, column=0, sticky="w", pady=2)

        cal_btns = ttk.Frame(cc)
        cal_btns.grid(row=1, column=0, sticky="w", pady=4)
        ttk.Button(
            cal_btns,
            text="Generate Reliability Diagram",
            command=self._run_calibration_diagram,
        ).pack(side="left", padx=2)

        self._cal_canvas_frame = ttk.Frame(cc)
        self._cal_canvas_frame.grid(row=2, column=0, sticky="ew", pady=2)

    def _run_calibration_diagram(self) -> None:
        model_path = Path("data") / "element_classifier.pkl"
        jsonl_path = Path("data") / "training_data.jsonl"

        if not model_path.exists():
            messagebox.showwarning(
                "No Model", "No trained model found. Run training first."
            )
            return
        if not jsonl_path.exists():
            messagebox.showwarning(
                "No Data",
                "No training_data.jsonl found. Run training first to generate it.",
            )
            return

        self._log.clear()
        self._worker = PipelineWorker(self._root, self._log)

        def target():
            from plancheck.corrections.classifier import ElementClassifier

            clf = ElementClassifier(model_path=model_path)
            return clf.calibration_curve(jsonl_path)

        def on_done(result, error, elapsed):
            if error or not result:
                return
            ece = result.get("ece", 0.0)
            curves = result.get("curves", {})
            self._log.write(f"Expected Calibration Error (ECE): {ece:.4f}", "INFO")
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
# Section 4 – Model Comparison
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

        from plancheck.corrections.store import CorrectionStore

        store = CorrectionStore(db_path)
        history = store.get_training_history()
        store.close()

        if not history:
            messagebox.showinfo("No Runs", "No training runs found.")
            return

        display_values = [
            f"{run['run_id']} | F1={run['f1_weighted']:.3f} | {run['trained_at'][:16]}"
            for run in history
        ]

        self._cmp_combo_a["values"] = display_values
        self._cmp_combo_b["values"] = display_values
        if len(display_values) >= 2:
            self._cmp_combo_a.current(0)
            self._cmp_combo_b.current(1)
        elif len(display_values) == 1:
            self._cmp_combo_a.current(0)
            self._cmp_combo_b.current(0)

        self._log.write(f"Loaded {len(history)} training run(s).", "INFO")

    def _compare_runs(self) -> None:
        run_a_str = self._cmp_run_a.get()
        run_b_str = self._cmp_run_b.get()
        if not run_a_str or not run_b_str:
            messagebox.showwarning(
                "Select Runs", "Select two training runs to compare."
            )
            return

        db_path = Path("data") / "corrections.db"
        from plancheck.corrections.store import CorrectionStore

        store = CorrectionStore(db_path)
        history = store.get_training_history()
        store.close()

        id_a = run_a_str.split(" | ")[0].strip()
        id_b = run_b_str.split(" | ")[0].strip()

        run_a = next((r for r in history if r["run_id"] == id_a), None)
        run_b = next((r for r in history if r["run_id"] == id_b), None)

        if not run_a or not run_b:
            messagebox.showwarning("Not Found", "Could not find selected runs.")
            return

        self._log.clear()
        self._log.write(f"Comparing {id_a} vs {id_b}", "INFO")
        self._log.write(
            f"{'Metric':<25} {'Run A':>10} {'Run B':>10} {'Delta':>10}", "INFO"
        )
        self._log.write("-" * 57, "INFO")

        for metric in ("accuracy", "f1_macro", "f1_weighted"):
            va = run_a.get(metric, 0.0)
            vb = run_b.get(metric, 0.0)
            delta = vb - va
            sign = "+" if delta >= 0 else ""
            self._log.write(
                f"{metric:<25} {va:>10.4f} {vb:>10.4f} {sign}{delta:>9.4f}", "INFO"
            )

        pc_a = run_a.get("per_class", {})
        pc_b = run_b.get("per_class", {})
        all_classes = sorted(set(pc_a.keys()) | set(pc_b.keys()))
        if all_classes:
            self._log.write("", "INFO")
            self._log.write(
                f"{'Class':<20} {'F1(A)':>8} {'F1(B)':>8} {'Delta':>8}", "INFO"
            )
            self._log.write("-" * 46, "INFO")
            for cls in all_classes:
                f1_a = pc_a.get(cls, {}).get("f1", 0.0)
                f1_b = pc_b.get(cls, {}).get("f1", 0.0)
                delta = f1_b - f1_a
                sign = "+" if delta >= 0 else ""
                self._log.write(
                    f"{cls:<20} {f1_a:>8.4f} {f1_b:>8.4f} {sign}{delta:>7.4f}", "INFO"
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
            text="Run LayoutLMv3 layout detection on the current page.",
            foreground="gray",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=2)

        ttk.Label(lm, text="Model:").grid(row=1, column=0, sticky="w", pady=2)
        self._layout_model_var = tk.StringVar(value="microsoft/layoutlmv3-base")
        ttk.Entry(lm, textvariable=self._layout_model_var, width=50).grid(
            row=1, column=1, sticky="ew", padx=4
        )

        layout_btns = ttk.Frame(lm)
        layout_btns.grid(row=2, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Button(
            layout_btns, text="Run Layout Detection", command=self._run_layout_detection
        ).pack(side="left", padx=2)
        ttk.Button(
            layout_btns, text="Check Availability", command=self._check_layout_avail
        ).pack(side="left", padx=2)

    def _check_layout_avail(self) -> None:
        self._log.clear()
        try:
            from plancheck.analysis.layout_model import is_layout_available

            avail = is_layout_available()
            if avail:
                self._log.write(
                    "LayoutLMv3 is available (transformers + torch installed).",
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
            messagebox.showwarning("No Model", "Enter a LayoutLMv3 model name or path.")
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
