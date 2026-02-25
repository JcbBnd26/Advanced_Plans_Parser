"""Tab 7 – MLOps: drift monitoring, retrain control, feature cache, experiments.

Four collapsible sections providing visibility into the production ML
infrastructure added in Phase 4.
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

_project = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project / "scripts" / "runners"))

from widgets import CollapsibleFrame, LogPanel
from worker import PipelineWorker


class MLOpsTab:
    """Tab 7: MLOps — drift, retrain, cache, experiments."""

    def __init__(self, notebook: ttk.Notebook, gui_state: Any) -> None:
        self.notebook = notebook
        self.state = gui_state
        self.root = notebook.winfo_toplevel()

        self.frame = ttk.Frame(notebook)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=3)
        self.frame.rowconfigure(1, weight=1)
        notebook.add(self.frame, text="MLOps")

        self._worker: PipelineWorker | None = None
        self._build_ui()

    # ── UI construction ────────────────────────────────────────────

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 4}

        # ── Scrollable top section ────────────────────────────────
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

        def _resize(event: Any) -> None:
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

        # ── 1. Drift Monitor ─────────────────────────────────────
        drift_sec = CollapsibleFrame(self._inner, "Drift Monitor", initially_open=True)
        drift_sec.grid(row=row, column=0, sticky="ew", **pad)
        row += 1

        dc = drift_sec.content
        dc.columnconfigure(1, weight=1)

        ttk.Label(dc, text="Drift stats path:").grid(
            row=0, column=0, sticky="w", padx=4, pady=2
        )
        self._drift_path_var = tk.StringVar(value="data/drift_stats.json")
        ttk.Entry(dc, textvariable=self._drift_path_var, width=50).grid(
            row=0, column=1, sticky="ew", padx=4, pady=2
        )

        ttk.Label(dc, text="Threshold:").grid(
            row=1, column=0, sticky="w", padx=4, pady=2
        )
        self._drift_thresh_var = tk.StringVar(value="0.3")
        ttk.Entry(dc, textvariable=self._drift_thresh_var, width=10).grid(
            row=1, column=1, sticky="w", padx=4, pady=2
        )

        btn_frame_d = ttk.Frame(dc)
        btn_frame_d.grid(row=2, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Button(
            btn_frame_d, text="Check Drift Stats", command=self._on_check_drift
        ).pack(side="left", padx=4)
        ttk.Button(
            btn_frame_d,
            text="Regenerate Stats",
            command=self._on_regen_drift,
        ).pack(side="left", padx=4)

        self._drift_info = tk.Text(dc, height=6, wrap="word", state="disabled")
        self._drift_info.grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=4, pady=4
        )

        # ── 2. Retrain Control ────────────────────────────────────
        retrain_sec = CollapsibleFrame(
            self._inner, "Retrain Control", initially_open=True
        )
        retrain_sec.grid(row=row, column=0, sticky="ew", **pad)
        row += 1

        rc = retrain_sec.content
        rc.columnconfigure(1, weight=1)

        ttk.Label(rc, text="DB path:").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self._db_path_var = tk.StringVar(value="data/corrections.db")
        ttk.Entry(rc, textvariable=self._db_path_var, width=50).grid(
            row=0, column=1, sticky="ew", padx=4, pady=2
        )

        ttk.Label(rc, text="Retrain threshold:").grid(
            row=1, column=0, sticky="w", padx=4, pady=2
        )
        self._retrain_thresh_var = tk.StringVar(value="50")
        ttk.Entry(rc, textvariable=self._retrain_thresh_var, width=10).grid(
            row=1, column=1, sticky="w", padx=4, pady=2
        )

        btn_frame_r = ttk.Frame(rc)
        btn_frame_r.grid(row=2, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Button(
            btn_frame_r, text="Check Status", command=self._on_check_retrain
        ).pack(side="left", padx=4)
        ttk.Button(btn_frame_r, text="Retrain Now", command=self._on_retrain_now).pack(
            side="left", padx=4
        )

        self._retrain_info = tk.Text(rc, height=6, wrap="word", state="disabled")
        self._retrain_info.grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=4, pady=4
        )

        # ── 3. Feature Cache ─────────────────────────────────────
        cache_sec = CollapsibleFrame(self._inner, "Feature Cache", initially_open=False)
        cache_sec.grid(row=row, column=0, sticky="ew", **pad)
        row += 1

        cc = cache_sec.content
        cc.columnconfigure(1, weight=1)

        btn_frame_c = ttk.Frame(cc)
        btn_frame_c.grid(row=0, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Button(btn_frame_c, text="Cache Stats", command=self._on_cache_stats).pack(
            side="left", padx=4
        )
        ttk.Button(
            btn_frame_c, text="Invalidate Cache", command=self._on_invalidate_cache
        ).pack(side="left", padx=4)

        self._cache_info = tk.Text(cc, height=4, wrap="word", state="disabled")
        self._cache_info.grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=4, pady=4
        )

        # ── 4. Experiment History ─────────────────────────────────
        exp_sec = CollapsibleFrame(
            self._inner, "Experiment History", initially_open=False
        )
        exp_sec.grid(row=row, column=0, sticky="ew", **pad)
        row += 1

        ec = exp_sec.content
        ec.columnconfigure(0, weight=1)

        btn_frame_e = ttk.Frame(ec)
        btn_frame_e.grid(row=0, column=0, sticky="w", pady=4)
        ttk.Button(
            btn_frame_e, text="Refresh", command=self._on_refresh_experiments
        ).pack(side="left", padx=4)
        ttk.Button(
            btn_frame_e, text="Export CSV", command=self._on_export_experiments
        ).pack(side="left", padx=4)

        cols = (
            "run_id",
            "trained_at",
            "n_train",
            "accuracy",
            "f1_weighted",
            "notes",
        )
        self._exp_tree = ttk.Treeview(ec, columns=cols, show="headings", height=8)
        for c in cols:
            self._exp_tree.heading(c, text=c)
            width = 100 if c not in ("run_id", "trained_at") else 160
            self._exp_tree.column(c, width=width, anchor="w")
        self._exp_tree.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        exp_sb = ttk.Scrollbar(ec, orient="vertical", command=self._exp_tree.yview)
        self._exp_tree.configure(yscrollcommand=exp_sb.set)
        exp_sb.grid(row=1, column=1, sticky="ns")

        # ── Log panel ─────────────────────────────────────────────
        self._log = LogPanel(self.frame)
        self._log.grid(row=1, column=0, columnspan=2, sticky="nsew")

    # ── Helpers ────────────────────────────────────────────────────

    def _set_text(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _get_store(self):
        """Open a CorrectionStore from the DB path entry."""
        from plancheck.corrections.store import CorrectionStore

        db = Path(self._db_path_var.get())
        if not db.exists():
            messagebox.showerror("Error", f"Database not found: {db}")
            return None
        return CorrectionStore(db)

    # ── Drift handlers ─────────────────────────────────────────────

    def _on_check_drift(self) -> None:
        try:
            from plancheck.corrections.drift_detection import DriftDetector

            p = Path(self._drift_path_var.get())
            if not p.exists():
                self._set_text(
                    self._drift_info,
                    f"Drift stats file not found: {p}\n"
                    "Train a model first to generate drift stats.",
                )
                return
            det = DriftDetector.load(p)
            info = (
                f"Drift stats loaded from {p}\n"
                f"  Reference samples: {det._n_samples}\n"
                f"  Feature dims:      {det._n_features}\n"
                f"  Threshold:         {det.threshold}"
            )
            self._set_text(self._drift_info, info)
        except Exception as exc:
            self._set_text(self._drift_info, f"Error: {exc}")

    def _on_regen_drift(self) -> None:
        try:
            from plancheck.corrections.drift_detection import DriftDetector

            store = self._get_store()
            if store is None:
                return
            try:
                jsonl = Path(self._db_path_var.get()).parent / "training_data.jsonl"
                if not jsonl.exists():
                    self._set_text(
                        self._drift_info,
                        "No training_data.jsonl found. Run training first.",
                    )
                    return
                threshold = float(self._drift_thresh_var.get())
                det = DriftDetector(threshold=threshold)
                det.fit(jsonl)
                out = Path(self._drift_path_var.get())
                det.save(out)
                self._set_text(
                    self._drift_info,
                    f"Drift stats regenerated → {out}\n"
                    f"  Samples: {det._n_samples}, dims: {det._n_features}",
                )
            finally:
                store.close()
        except Exception as exc:
            self._set_text(self._drift_info, f"Error: {exc}")

    # ── Retrain handlers ───────────────────────────────────────────

    def _on_check_retrain(self) -> None:
        try:
            store = self._get_store()
            if store is None:
                return
            try:
                threshold = int(self._retrain_thresh_var.get())
                n_new = store.count_corrections_since_last_train()
                last = store.last_train_date() or "never"
                needed = store.should_retrain(threshold=threshold)
                info = (
                    f"Last training:       {last}\n"
                    f"New corrections:     {n_new}\n"
                    f"Threshold:           {threshold}\n"
                    f"Retrain recommended: {'YES' if needed else 'no'}"
                )
                self._set_text(self._retrain_info, info)
            finally:
                store.close()
        except Exception as exc:
            self._set_text(self._retrain_info, f"Error: {exc}")

    def _on_retrain_now(self) -> None:
        try:
            from plancheck.corrections.retrain_trigger import auto_retrain

            store = self._get_store()
            if store is None:
                return
            try:
                threshold = int(self._retrain_thresh_var.get())
                self._set_text(self._retrain_info, "Retraining in progress...")
                self.root.update_idletasks()
                result = auto_retrain(
                    store,
                    model_path="data/element_classifier.pkl",
                    threshold=threshold,
                )
                lines = [
                    f"Retrained:     {result.retrained}",
                    f"Accepted:      {result.accepted}",
                    f"Rolled back:   {result.rolled_back}",
                ]
                if result.metrics:
                    lines.append(
                        f"F1 weighted:   " f"{result.metrics.get('f1_weighted', 0):.4f}"
                    )
                    lines.append(
                        f"Accuracy:      " f"{result.metrics.get('accuracy', 0):.4f}"
                    )
                if result.error:
                    lines.append(f"Error:         {result.error}")
                self._set_text(self._retrain_info, "\n".join(lines))
            finally:
                store.close()
        except Exception as exc:
            self._set_text(self._retrain_info, f"Error: {exc}")

    # ── Cache handlers ─────────────────────────────────────────────

    def _on_cache_stats(self) -> None:
        try:
            store = self._get_store()
            if store is None:
                return
            try:
                stats = store.cache_stats()
                info = (
                    f"Total entries:       {stats['total_entries']}\n"
                    f"Distinct detections: {stats['distinct_detections']}\n"
                    f"Distinct versions:   {stats['distinct_versions']}"
                )
                self._set_text(self._cache_info, info)
            finally:
                store.close()
        except Exception as exc:
            self._set_text(self._cache_info, f"Error: {exc}")

    def _on_invalidate_cache(self) -> None:
        if not messagebox.askyesno(
            "Confirm",
            "Delete all cached feature vectors?",
        ):
            return
        try:
            store = self._get_store()
            if store is None:
                return
            try:
                n = store.invalidate_cache()
                self._set_text(
                    self._cache_info, f"Cache invalidated: {n} entries deleted."
                )
            finally:
                store.close()
        except Exception as exc:
            self._set_text(self._cache_info, f"Error: {exc}")

    # ── Experiment handlers ────────────────────────────────────────

    def _on_refresh_experiments(self) -> None:
        try:
            from plancheck.corrections.experiment_tracker import ExperimentTracker

            store = self._get_store()
            if store is None:
                return
            try:
                tracker = ExperimentTracker(store)
                exps = tracker.list_experiments(limit=100)
                self._exp_tree.delete(*self._exp_tree.get_children())
                for exp in exps:
                    self._exp_tree.insert(
                        "",
                        "end",
                        values=(
                            exp.run_id,
                            exp.trained_at,
                            exp.n_train,
                            f"{exp.accuracy:.4f}",
                            f"{exp.f1_weighted:.4f}",
                            exp.notes,
                        ),
                    )
            finally:
                store.close()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _on_export_experiments(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Export experiments CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            from plancheck.corrections.experiment_tracker import ExperimentTracker

            store = self._get_store()
            if store is None:
                return
            try:
                tracker = ExperimentTracker(store)
                n = tracker.export_csv(Path(path))
                messagebox.showinfo("Export", f"Exported {n} experiments to {path}")
            finally:
                store.close()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
