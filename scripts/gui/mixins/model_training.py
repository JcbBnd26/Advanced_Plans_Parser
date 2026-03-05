"""Model training mixin for GUI annotation tab.

Handles ML model training, metrics display, training history,
feature importance, and active learning suggestions.
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import TYPE_CHECKING

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
        except Exception:
            pass
        try:
            self.root.after(delay_ms, callback)
        except Exception:
            pass

    # ── Training ─────────────────────────────────────────────────

    def _on_train_model(self) -> None:
        """Train the classifier in a background thread."""
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
                from plancheck.corrections.store import CorrectionStore as _CS

                store = _CS()
                n_examples = store.build_training_set()
                if n_examples < 10:
                    self._safe_after(
                        0,
                        lambda: self._model_status_label.configure(
                            text=f"Need more data ({n_examples} examples)",
                            foreground="red",
                        ),
                    )
                    return

                import tempfile

                tmp = Path(tempfile.mkdtemp()) / "train.jsonl"
                store.export_training_jsonl(tmp)

                if (
                    self._train_cancel_event.is_set()
                    or self._closing
                    or my_gen != self._train_gen
                ):
                    return

                from plancheck.corrections.classifier import ElementClassifier

                clf = ElementClassifier()
                metrics = clf.train(str(tmp))
                self._last_metrics = metrics

                if (
                    self._train_cancel_event.is_set()
                    or self._closing
                    or my_gen != self._train_gen
                ):
                    return

                # Record training run in the database
                try:
                    store.save_training_run(
                        metrics, model_path=str(clf.model_path), notes="GUI train"
                    )
                except Exception:
                    pass  # non-critical

                acc = metrics.get("accuracy", 0)
                f1m = metrics.get("f1_macro", 0)
                self._safe_after(
                    0,
                    lambda: self._model_status_label.configure(
                        text=f"Model trained — acc {acc:.1%}  F1 {f1m:.1%}",
                        foreground="green",
                    ),
                )
                # Reload classifier
                from plancheck.corrections.classifier import ElementClassifier

                self._classifier = ElementClassifier()
            except Exception as exc:
                self._safe_after(
                    0,
                    lambda: self._model_status_label.configure(
                        text=f"Train failed: {exc}",
                        foreground="red",
                    ),
                )

        threading.Thread(target=_train, daemon=True).start()

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
            history = self._store.get_training_history()
        except Exception:
            history = []

        if not history:
            messagebox.showinfo("Training History", "No training runs recorded yet.")
            return

        lines: list[str] = [
            f"{'Run ID':<14s} {'Date':<22s} {'#Train':>6s} {'#Val':>5s} "
            f"{'Acc':>7s} {'F1mac':>7s} {'F1wt':>7s} {'Notes':<12s}",
            "-" * 88,
        ]
        for r in history:
            ts = r["trained_at"][:19].replace("T", " ")
            lines.append(
                f"{r['run_id']:<14s} {ts:<22s} {r['n_train']:>6d} {r['n_val']:>5d} "
                f"{r['accuracy']:>6.1%} {r['f1_macro']:>6.1%} {r['f1_weighted']:>6.1%} "
                f"{r.get('notes', ''):<12s}"
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
        if self._classifier.model_exists():
            self._model_status_label.configure(
                text="Model loaded ✓", foreground="green"
            )
        else:
            self._model_status_label.configure(
                text="No model trained", foreground="gray"
            )

    # ── Annotation stats ─────────────────────────────────────────

    def _refresh_stats(self) -> None:
        """Query the DB for annotation statistics and display them."""
        try:
            cur = self._store._conn.cursor()
            n_docs = cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            n_dets = cur.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
            n_corr = cur.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
            n_train = cur.execute("SELECT COUNT(*) FROM training_examples").fetchone()[
                0
            ]

            # Per-type breakdown
            rows = cur.execute(
                "SELECT element_type, COUNT(*) FROM detections GROUP BY element_type"
            ).fetchall()
            breakdown = "\n".join(f"  {r[0]}: {r[1]}" for r in rows)

            text = (
                f"Docs: {n_docs}  Dets: {n_dets}\n"
                f"Corrections: {n_corr}  Training: {n_train}\n"
                f"{breakdown}"
            )
            self._stats_label.configure(text=text)
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
