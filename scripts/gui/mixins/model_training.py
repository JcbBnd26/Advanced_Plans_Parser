"""Model training mixin for GUI annotation tab.

Handles ML model training, metrics display, training history,
feature importance, and active learning suggestions.
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import TYPE_CHECKING

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tkinter import ttk

    from plancheck.corrections.classifier import ElementClassifier
    from plancheck.corrections.store import CorrectionStore


class ModelTrainingMixin:
    """Mixin providing ML model training and metrics functionality.

    Expected attributes on the host class:
        _classifier: ElementClassifier
        _store: CorrectionStore
        _model_status_label: ttk.Label
        _stats_label: ttk.Label
        _last_metrics: dict | None
        _train_cancel_event: threading.Event
        _train_gen: int
        _closing: bool
        root: tk.Tk
    """

    # ── Thread-safe UI update ────────────────────────────────────

    def _safe_after(self, delay_ms: int, callback) -> None:
        """Schedule callback on the UI thread if the window still exists."""
        if getattr(self, "_closing", False):
            return
        try:
            if hasattr(self.root, "winfo_exists") and not self.root.winfo_exists():
                return
        except Exception:  # noqa: BLE001
            log.debug("winfo_exists check failed", exc_info=True)
        try:
            self.root.after(delay_ms, callback)
        except Exception:  # noqa: BLE001
            log.debug("root.after scheduling failed", exc_info=True)

    def _reload_classifiers(self) -> None:
        """Clear hierarchical caches and reload the Stage-1 classifier."""
        from plancheck.corrections.classifier import ElementClassifier
        from plancheck.corrections.hierarchical_classifier import clear_classifier_cache

        clear_classifier_cache()
        model_path = Path(self.state.config.ml_model_path)
        self._classifier = ElementClassifier(model_path=model_path)

    def _format_retrain_status(
        self,
        result,
        *,
        action_label: str,
        bootstrap_examples: int | None = None,
    ) -> tuple[str, str]:
        """Convert a retrain result into a concise UI status string."""
        if result.error:
            return f"{action_label} failed: {result.error}", "red"

        if result.rolled_back:
            f1 = result.metrics.get("f1_weighted", 0)
            return (
                f"Model rolled back — F1 regressed ({f1:.1%})",
                "orange",
            )

        if not result.accepted:
            return f"{action_label} incomplete", "gray"

        prefix = action_label
        if bootstrap_examples is not None:
            prefix = f"{action_label} ({bootstrap_examples} examples)"

        stage1_f1 = result.metrics.get("f1_weighted", 0)
        text = f"{prefix} — S1 F1 {stage1_f1:.1%}"
        color = "green"

        if result.stage2_trained:
            stage2_f1 = result.stage2_metrics.get("f1_weighted", 0)
            text += f"; S2 F1 {stage2_f1:.1%}"
        elif result.stage2_error:
            text += f"; Stage 2 failed: {result.stage2_error}"
            color = "orange"
        elif result.stage2_skipped_reason:
            text += f"; Stage 2 skipped: {result.stage2_skipped_reason}"

        return text, color

    def _format_model_readiness_status(
        self,
        *,
        model_exists: bool,
        pending_corrections: int | None = None,
        threshold: int | None = None,
    ) -> tuple[str, str]:
        """Format model load status together with retrain readiness."""
        if model_exists:
            text = "Model loaded ✓"
            color = "green"
        else:
            text = "No model trained"
            color = "gray"

        if pending_corrections is None or threshold is None or threshold <= 0:
            return text, color

        text += f" | Pending corrections: {pending_corrections}/{threshold}"
        if pending_corrections >= threshold:
            text += " — retrain recommended"
            color = "orange" if model_exists else "red"
        elif pending_corrections >= max(1, int(threshold * 0.8)):
            text += " — nearing threshold"
            if model_exists:
                color = "#e5c07b"

        return text, color

    def _format_annotation_runtime_summary(
        self,
        *,
        pending_corrections: int | None = None,
        threshold: int | None = None,
        active_drift_text: str = "",
    ) -> str:
        """Format routing, drift, and retrain posture for the inspector."""
        routing_text = (
            "hierarchical"
            if getattr(self.state.config, "ml_hierarchical_enabled", False)
            else "Stage 1 only"
        )

        drift_segment = "Drift: disabled"
        if getattr(self.state.config, "ml_drift_enabled", False):
            if active_drift_text:
                drift_segment = active_drift_text
            else:
                drift_path = Path(getattr(self.state.config, "ml_drift_stats_path", ""))
                if not drift_path.is_absolute():
                    drift_path = Path.cwd() / drift_path
                drift_segment = (
                    "Drift: monitoring"
                    if drift_path.exists()
                    else "Drift: stats missing"
                )

        retrain_segment = "Retrain: unavailable"
        if pending_corrections is not None and threshold is not None:
            retrain_segment = f"Retrain: {pending_corrections}/{threshold} pending"
            if pending_corrections >= threshold:
                retrain_segment += " (recommended)"

        return f"Routing: {routing_text} | {drift_segment} | {retrain_segment}"

    def _update_annotation_runtime_summary(self) -> None:
        """Refresh the compact annotation runtime summary label when present."""
        if not hasattr(self, "_runtime_summary_label"):
            return

        try:
            threshold = getattr(self.state.config, "ml_retrain_threshold", 50)
            pending_corrections = self._store.count_corrections_since_last_train()
        except Exception:  # noqa: BLE001
            threshold = None
            pending_corrections = None

        summary = self._format_annotation_runtime_summary(
            pending_corrections=pending_corrections,
            threshold=threshold,
            active_drift_text=getattr(self, "_active_drift_text", ""),
        )
        self._runtime_summary_label.configure(text=summary)

    # ── Training ─────────────────────────────────────────────────

    def _on_train_model(self) -> None:
        """Train the classifier in a background thread using auto_retrain()."""
        self._model_status_label.configure(text="Training…", foreground="orange")
        self.root.update_idletasks()

        # Cancel any previous training run
        self._train_cancel_event.clear()
        self._train_gen += 1
        my_gen = self._train_gen

        def _train():
            try:
                if (
                    self._train_cancel_event.is_set()
                    or self._closing
                    or my_gen != self._train_gen
                ):
                    return

                from plancheck.corrections.retrain_trigger import auto_retrain
                from plancheck.corrections.store import CorrectionStore as _CS

                store = _CS()

                # Force retrain by setting threshold=0 when user explicitly requests
                result = auto_retrain(
                    store,
                    model_path=self.state.config.ml_model_path,
                    stage2_model_path=self.state.config.ml_stage2_model_path,
                    calibrate=True,
                    ensemble=False,
                    threshold=0,  # Always train when user clicks button
                )
                store.close()

                if (
                    self._train_cancel_event.is_set()
                    or self._closing
                    or my_gen != self._train_gen
                ):
                    return

                # Store metrics for display
                if result.metrics:
                    self._last_metrics = result.metrics
                self._last_stage2_metrics = result.stage2_metrics or None

                # Update status based on result
                status_text, status_color = self._format_retrain_status(
                    result,
                    action_label="Model trained",
                )
                self._safe_after(
                    0,
                    lambda: self._model_status_label.configure(
                        text=status_text,
                        foreground=status_color,
                    ),
                )

                self._reload_classifiers()

            except Exception as exc:
                self._safe_after(
                    0,
                    lambda: self._model_status_label.configure(
                        text=f"Train failed: {exc}",
                        foreground="red",
                    ),
                )

        threading.Thread(target=_train, daemon=True).start()

    # ── Bootstrap training ───────────────────────────────────────

    def _on_bootstrap_training(self) -> None:
        """Generate pseudo-labels from high-confidence detections and train.

        This bootstraps the classifier without requiring manual corrections,
        enabling cold-start learning for new users.
        """
        self._model_status_label.configure(text="Bootstrapping…", foreground="orange")
        self.root.update_idletasks()

        def _bootstrap():
            try:
                from plancheck.corrections.retrain_trigger import auto_retrain
                from plancheck.corrections.store import CorrectionStore as _CS

                store = _CS()

                # Generate pseudo-labels from high-confidence rule-based detections
                n_generated = store.generate_pseudo_labels(
                    confidence_threshold=0.95,
                    max_per_label=500,
                )
                log.info("Generated %d pseudo-labels for bootstrap", n_generated)

                if n_generated == 0:
                    store.close()
                    self._safe_after(
                        0,
                        lambda: self._model_status_label.configure(
                            text="No high-confidence detections to bootstrap from",
                            foreground="gray",
                        ),
                    )
                    return

                # Train on the pseudo-labels
                result = auto_retrain(
                    store,
                    model_path=self.state.config.ml_model_path,
                    stage2_model_path=self.state.config.ml_stage2_model_path,
                    calibrate=True,
                    ensemble=False,
                    threshold=0,  # Force training
                )
                store.close()

                # Store metrics for display
                if result.metrics:
                    self._last_metrics = result.metrics
                self._last_stage2_metrics = result.stage2_metrics or None

                # Update status
                status_text, status_color = self._format_retrain_status(
                    result,
                    action_label="Bootstrapped",
                    bootstrap_examples=n_generated,
                )
                self._safe_after(
                    0,
                    lambda: self._model_status_label.configure(
                        text=status_text,
                        foreground=status_color,
                    ),
                )

                self._reload_classifiers()

            except Exception as exc:
                log.exception("Bootstrap training failed")
                self._safe_after(
                    0,
                    lambda: self._model_status_label.configure(
                        text=f"Bootstrap failed: {exc}",
                        foreground="red",
                    ),
                )

        threading.Thread(target=_bootstrap, daemon=True).start()

    # ── Metrics display ──────────────────────────────────────────

    def _on_show_metrics(self) -> None:
        """Show the last training metrics in a popup."""
        if not self._last_metrics:
            messagebox.showinfo("Metrics", "No metrics available. Train a model first.")
            return

        from plancheck.corrections.metrics import format_metrics_table

        text = format_metrics_table(self._last_metrics)

        win = tk.Toplevel(self.root)
        win.title("Model Metrics")
        win.geometry("500x400")
        win.transient(self.root)

        txt = tk.Text(win, wrap="none", font=("Courier", 9))
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        txt.insert("1.0", text)
        self._make_text_readonly(txt)
        self._add_copy_menu(txt)

    def _on_show_training_history(self) -> None:
        """Show a popup with all past training runs."""
        try:
            from plancheck.corrections.experiment_tracker import ExperimentTracker

            tracker = ExperimentTracker(self._store)
            experiments = tracker.list_experiments(limit=50, sort_by="trained_at")
        except Exception:  # noqa: BLE001
            log.debug("Failed to get training history", exc_info=True)
            experiments = []

        if not experiments:
            messagebox.showinfo("Training History", "No training runs recorded yet.")
            return

        lines: list[str] = [
            f"{'Run ID':<14s} {'Date':<22s} {'#Train':>6s} {'#Val':>5s} "
            f"{'Acc':>7s} {'F1mac':>7s} {'F1wt':>7s} {'Notes':<12s}",
            "-" * 88,
        ]
        for exp in experiments:
            ts = exp.trained_at[:19].replace("T", " ")
            lines.append(
                f"{exp.run_id:<14s} {ts:<22s} {exp.n_train:>6d} {exp.n_val:>5d} "
                f"{exp.accuracy:>6.1%} {exp.f1_macro:>6.1%} {exp.f1_weighted:>6.1%} "
                f"{exp.notes:<12s}"
            )

        win = tk.Toplevel(self.root)
        win.title("Training History")
        win.geometry("700x400")
        win.transient(self.root)

        txt = tk.Text(win, wrap="none", font=("Courier", 9))
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        txt.insert("1.0", "\n".join(lines))
        self._make_text_readonly(txt)
        self._add_copy_menu(txt)

    def _on_show_feature_importance(self) -> None:
        """Show feature importance from the trained model."""
        if not self._classifier.model_exists():
            messagebox.showinfo("Feature Importance", "No model trained yet.")
            return

        importance = self._classifier.get_feature_importance()
        if not importance:
            messagebox.showinfo(
                "Feature Importance", "Could not extract feature importance."
            )
            return

        lines: list[str] = [
            f"{'Feature':<30s} {'Importance':>12s}",
            "-" * 44,
        ]
        for name, imp in importance.items():
            bar = "█" * max(1, int(imp * 200))
            lines.append(f"{name:<30s} {imp:>12.6f}  {bar}")

        win = tk.Toplevel(self.root)
        win.title("Feature Importance")
        win.geometry("600x500")
        win.transient(self.root)

        txt = tk.Text(win, wrap="none", font=("Courier", 9))
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        txt.insert("1.0", "\n".join(lines))
        self._make_text_readonly(txt)
        self._add_copy_menu(txt)

    def _update_model_status(self) -> None:
        """Check if a trained model file exists and update the label."""
        try:
            threshold = getattr(self.state.config, "ml_retrain_threshold", 50)
            pending_corrections = self._store.count_corrections_since_last_train()
        except Exception:  # noqa: BLE001
            threshold = None
            pending_corrections = None

        text, color = self._format_model_readiness_status(
            model_exists=self._classifier.model_exists(),
            pending_corrections=pending_corrections,
            threshold=threshold,
        )
        self._model_status_label.configure(text=text, foreground=color)
        self._update_annotation_runtime_summary()

    # ── Annotation stats ─────────────────────────────────────────

    def _refresh_stats(self) -> None:
        """Query the DB for annotation statistics and display them."""
        try:
            ov = self._store.get_db_overview()
            n_docs = ov["total_documents"]
            n_dets = ov["total_detections"]
            n_corr = ov["total_corrections"]
            n_train = ov["total_training_examples"]
            pending_retrain = self._store.count_corrections_since_last_train()
            retrain_threshold = getattr(self.state.config, "ml_retrain_threshold", 50)
            retrain_flag = self._store.should_retrain(threshold=retrain_threshold)

            # Per-type breakdown
            type_breakdown = self._store.get_detection_type_breakdown()
            breakdown = "\n".join(f"  {k}: {v}" for k, v in type_breakdown.items())

            text = (
                f"Docs: {n_docs}  Dets: {n_dets}\n"
                f"Corrections: {n_corr}  Training: {n_train}\n"
                f"Retrain: {pending_retrain}/{retrain_threshold}"
                f"{'  recommended' if retrain_flag else ''}\n"
                f"{breakdown}"
            )
            self._stats_label.configure(text=text)
            self._update_model_status()
            self._update_annotation_runtime_summary()
        except Exception as exc:
            self._stats_label.configure(text=f"Error: {exc}")

    def _on_clear_old_runs(self) -> None:
        """Remove detection data from old pipeline runs, keeping only the latest."""
        ok = messagebox.askyesno(
            "Clear Old Run Data",
            "This will remove detection data from old pipeline runs.\n"
            "Corrections and ML training data will be preserved.\n\n"
            "Continue?",
            parent=self.root,
        )
        if not ok:
            return
        try:
            n = self._store.purge_all_stale_detections()
            self._refresh_stats()
            if self._pdf_path:
                self._navigate_to_page()
            self._status.configure(text=f"Purged {n} old detection(s)")
        except Exception as exc:
            messagebox.showerror("Error", str(exc), parent=self.root)

    # ── Active learning ──────────────────────────────────────────

    def _on_suggest_next(self) -> None:
        """Use active learning to suggest the next page to annotate."""
        if not self._classifier.model_exists():
            messagebox.showinfo(
                "No Model",
                "Train a model first to enable active learning suggestions.",
            )
            return

        try:
            from plancheck.corrections.active_learning import suggest_next_page

            suggestion = suggest_next_page(self._store, self._classifier)
            if suggestion:
                messagebox.showinfo(
                    "Next Page to Annotate",
                    f"Document: {suggestion['doc_id']}\n"
                    f"Page: {suggestion['page']}\n"
                    f"Confidence: {suggestion['uncertainty']:.1%} uncertain",
                )
            else:
                messagebox.showinfo(
                    "Active Learning",
                    "No more pages to suggest. All pages have been reviewed.",
                )
        except Exception as exc:
            messagebox.showerror("Error", str(exc), parent=self.root)

    # ── Snapshots ────────────────────────────────────────────────

    def _on_snapshot(self) -> None:
        """Create a snapshot of the corrections database."""
        from tkinter import simpledialog

        tag = simpledialog.askstring(
            "Snapshot",
            "Enter a tag for this snapshot (optional):",
            parent=self.root,
        )
        if tag is None:
            return  # cancelled
        tag = tag.strip() or "manual"
        try:
            path = self._store.snapshot(tag)
            self._status.configure(text=f"Snapshot saved: {path.name}")
        except Exception as exc:
            messagebox.showerror("Snapshot Error", str(exc))

    def _on_restore_snapshot(self) -> None:
        """Show a list of snapshots and restore the chosen one."""
        from tkinter import ttk

        snaps = self._store.list_snapshots()
        if not snaps:
            messagebox.showinfo("No Snapshots", "No snapshots available.")
            return

        # Build a selection dialog
        win = tk.Toplevel(self.root)
        win.title("Restore Snapshot")
        win.geometry("420x300")
        win.transient(self.root)
        win.grab_set()

        ttk.Label(win, text="Choose a snapshot to restore:").pack(padx=10, pady=(10, 4))

        lb = tk.Listbox(win, font=("Courier", 9))
        lb.pack(fill="both", expand=True, padx=10, pady=4)
        for s in snaps:
            lb.insert("end", f"{s['tag']:12s}  {s['created'][:19]}  {s['size']:>8d}B")

        def _do_restore():
            sel = lb.curselection()
            if not sel:
                return
            snap = snaps[sel[0]]
            win.destroy()
            ok = messagebox.askyesno(
                "Confirm Restore",
                f"Restore snapshot '{snap['tag']}' from {snap['created'][:19]}?\n\n"
                "This will overwrite current database!",
                parent=self.root,
            )
            if ok:
                try:
                    self._store.restore(snap["path"])
                    self._refresh_stats()
                    if self._pdf_path:
                        self._navigate_to_page()
                    self._status.configure(text=f"Restored: {snap['tag']}")
                except Exception as exc:
                    messagebox.showerror("Restore Error", str(exc))

        ttk.Button(win, text="Restore", command=_do_restore).pack(pady=8)
