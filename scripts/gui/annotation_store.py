"""Annotation store, label registry, filter, and model training mixin."""
from __future__ import annotations

import json
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, messagebox, simpledialog, ttk
from typing import Any

from plancheck.corrections.classifier import ElementClassifier


class AnnotationStoreMixin:
    """Mixin providing label registry, filter, and model/stats methods."""

    def _normalize_element_type_name(self, name: str) -> str:
        return name.strip().lower().replace(" ", "_")

    def _label_registry_path(self) -> Path:
        # scripts/gui/tab_annotation.py -> repo root is parent.parent.parent
        return (
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / "label_registry.json"
        )

    def _load_label_registry_json(self) -> dict:
        path = self._label_registry_path()
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"version": "1.0", "label_registry": []}
        except Exception:
            return {"version": "1.0", "label_registry": []}

    def _save_label_registry_json(self, data: dict) -> None:
        path = self._label_registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
        tmp.replace(path)

    def _persist_element_type_to_registry(
        self, *, label: str, display_name: str, color: str
    ) -> None:
        data = self._load_label_registry_json()
        reg = data.get("label_registry")
        if not isinstance(reg, list):
            reg = []
            data["label_registry"] = reg

        existing: dict | None = None
        for entry in reg:
            if isinstance(entry, dict) and entry.get("label") == label:
                existing = entry
                break

        if existing is None:
            reg.append(
                {
                    "label": label,
                    "display_name": display_name,
                    "color": color,
                    "description": "",
                    "aliases": [],
                    "expected_zones": [],
                    "text_patterns": [],
                }
            )
        else:
            existing["display_name"] = display_name
            existing["color"] = color

        if "version" not in data:
            data["version"] = "1.0"
        self._save_label_registry_json(data)

    def _load_element_types_from_registry(self) -> None:
        data = self._load_label_registry_json()
        reg = data.get("label_registry", [])
        if not isinstance(reg, list):
            return
        for entry in reg:
            if not isinstance(entry, dict):
                continue
            label = entry.get("label", "")
            color = entry.get("color", "")
            if not label:
                continue
            if isinstance(color, str) and color.startswith("#") and len(color) == 7:
                self._register_element_type(label, color=color)
            else:
                self._register_element_type(label)

    def _register_element_type(self, name: str, *, color: str | None = None) -> None:
        """Register a new element type (optionally with explicit color).

        Updates LABEL_COLORS, ELEMENT_TYPES, the type combo boxes,
        and the filter checkboxes.
        """
        name = self._normalize_element_type_name(name)
        if not name or name in self.LABEL_COLORS:
            return

        # Auto-assign a distinct color from a palette
        _palette = [
            "#e6194b",
            "#3cb44b",
            "#4363d8",
            "#f58231",
            "#911eb4",
            "#42d4f4",
            "#f032e6",
            "#bfef45",
            "#fabebe",
            "#469990",
            "#9a6324",
            "#800000",
            "#aaffc3",
            "#808000",
            "#000075",
        ]
        if not (isinstance(color, str) and color.startswith("#") and len(color) == 7):
            idx = len(self.LABEL_COLORS) % len(_palette)
            color = _palette[idx]

        self.LABEL_COLORS[name] = color
        if name not in self.ELEMENT_TYPES:
            self.ELEMENT_TYPES.append(name)

        # Update combo boxes
        self._type_combo.configure(values=self.ELEMENT_TYPES)

        if name not in self._filter_label_vars:
            self._filter_label_vars[name] = tk.BooleanVar(value=True)
        self._rebuild_filter_controls()

    def _rebuild_filter_controls(self) -> None:
        """Rebuild per-type filter rows with checkboxes and clickable labels."""
        if self._filter_frame is None:
            return

        for child in self._filter_frame.winfo_children():
            child.destroy()

        if self._active_filter_color_type not in self.ELEMENT_TYPES:
            self._active_filter_color_type = (
                self.ELEMENT_TYPES[0] if self.ELEMENT_TYPES else None
            )

        for i, etype in enumerate(self.ELEMENT_TYPES):
            if etype not in self._filter_label_vars:
                self._filter_label_vars[etype] = tk.BooleanVar(value=True)

            is_active = etype == self._active_filter_color_type
            row_bg = (
                "SystemHighlight"
                if is_active
                else self._filter_frame.winfo_toplevel().cget("bg")
            )
            row_fg = "SystemHighlightText" if is_active else "SystemWindowText"

            row_frame = tk.Frame(self._filter_frame, bg=row_bg)
            row_frame.grid(row=i // 2, column=i % 2, sticky="w", pady=1, padx=(0, 8))

            cb = tk.Checkbutton(
                row_frame,
                variable=self._filter_label_vars[etype],
                command=self._apply_filters,
                bg=row_bg,
                activebackground=row_bg,
                highlightthickness=0,
                borderwidth=0,
            )
            cb.pack(side="left")

            lbl = tk.Label(
                row_frame,
                text=etype,
                bg=row_bg,
                fg=row_fg,
                cursor="hand2",
                padx=4,
            )
            lbl.pack(side="left")
            lbl.bind(
                "<Button-1>",
                lambda _e, label=etype: self._set_active_filter_color_type(label),
            )

        self._update_filter_color_button_label()

    def _set_active_filter_color_type(self, element_type: str) -> None:
        """Set the active element type target for the shared color picker."""
        if element_type not in self.ELEMENT_TYPES:
            return
        self._active_filter_color_type = element_type
        self._rebuild_filter_controls()

    def _update_filter_color_button_label(self) -> None:
        """Refresh shared color button text for the active element type."""
        if self._filter_color_btn is None:
            return
        if self._active_filter_color_type:
            self._filter_color_btn.configure(
                text=f"Pick Color: {self._active_filter_color_type}"
            )
            self._filter_color_btn.state(["!disabled"])
        else:
            self._filter_color_btn.configure(text="Pick Color")
            self._filter_color_btn.state(["disabled"])

    def _choose_active_filter_color(self) -> None:
        """Prompt for color of the currently selected filter label type."""
        element_type = self._active_filter_color_type
        if not element_type:
            return
        current = self.LABEL_COLORS.get(element_type, "#888888")
        _rgb, chosen = colorchooser.askcolor(
            color=current,
            title=f"Choose color for {element_type}",
            parent=self.root,
        )
        if not chosen:
            return

        self.LABEL_COLORS[element_type] = chosen
        self._rebuild_filter_controls()
        self._draw_all_boxes()
        self._apply_filters()

    def _select_all_filter_types(self) -> None:
        """Enable all element-type filter checkboxes."""
        for var in self._filter_label_vars.values():
            var.set(True)
        self._apply_filters()

    def _deselect_all_filter_types(self) -> None:
        """Disable all element-type filter checkboxes."""
        for var in self._filter_label_vars.values():
            var.set(False)
        self._apply_filters()

    def _on_add_element_type(self) -> None:
        """Prompt user to add a new element type."""
        name = simpledialog.askstring(
            "New Element Type",
            "Enter new element type name:",
            parent=self.root,
        )
        if name:
            self._register_element_type(name)
            self._type_var.set(name.strip().lower().replace(" ", "_"))
            self._status.configure(
                text=f"Added element type: {name.strip().lower().replace(' ', '_')}"
            )

    def _on_type_entered(self, _event: Any = None) -> None:
        """Handle Enter key in the type combo — register if new."""
        name = self._type_var.get().strip()
        if name and name not in self.ELEMENT_TYPES:
            self._register_element_type(name)
            self._status.configure(text=f"Added element type: {name}")

    def _deduplicate_boxes(self) -> None:
        """Remove near-duplicate canvas boxes (same type, IoU > 0.8).

        When the pipeline produces overlapping detections of the same
        element_type (e.g. two "header" boxes covering the same area),
        keep the one with the higher detection_id (most recently saved)
        and drop the other.
        """
        if len(self._canvas_boxes) < 2:
            return

        def _iou(a: tuple, b: tuple) -> float:
            x0 = max(a[0], b[0])
            y0 = max(a[1], b[1])
            x1 = min(a[2], b[2])
            y1 = min(a[3], b[3])
            inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
            area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0.0

        to_remove: set[int] = set()  # indices to remove
        n = len(self._canvas_boxes)
        for i in range(n):
            if i in to_remove:
                continue
            a = self._canvas_boxes[i]
            for j in range(i + 1, n):
                if j in to_remove:
                    continue
                b = self._canvas_boxes[j]
                if a.element_type != b.element_type:
                    continue
                if _iou(a.pdf_bbox, b.pdf_bbox) > 0.8:
                    # Keep the one with the higher detection_id
                    if (a.detection_id or 0) >= (b.detection_id or 0):
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break  # 'a' is removed, stop comparing

        if to_remove:
            self._canvas_boxes = [
                cb for idx, cb in enumerate(self._canvas_boxes) if idx not in to_remove
            ]

    # ── Rendering ──────────────────────────────────────────────────

    def _load_groups_for_page(self) -> None:
        """Load saved box groups from the DB for the current page."""
        self._groups.clear()
        if not self._doc_id:
            return
        for g in self._store.get_groups_for_page(self._doc_id, self._page):
            gid = g["group_id"]
            self._groups[gid] = {
                "label": g["group_label"],
                "root_detection_id": g["root_detection_id"],
                "members": [m["detection_id"] for m in g["members"]],
            }
            # Update canvas boxes
            for cb in self._canvas_boxes:
                if cb.detection_id in self._groups[gid]["members"]:
                    cb.group_id = gid
                    cb.is_group_root = cb.detection_id == g["root_detection_id"]

    def _apply_filters(self) -> None:
        """Show/hide boxes based on the current filter settings."""
        min_conf = self._filter_conf_min.get()
        uncorrected_only = self._filter_uncorrected_only.get()

        visible = 0
        for cb in self._canvas_boxes:
            show = True
            # Label type filter
            var = self._filter_label_vars.get(cb.element_type)
            if var and not var.get():
                show = False
            # Confidence filter
            if show and cb.confidence is not None and cb.confidence < min_conf:
                show = False
            # Uncorrected only
            if show and uncorrected_only and cb.corrected:
                show = False

            state = "normal" if show else "hidden"
            if cb.rect_id:
                self._canvas.itemconfigure(cb.rect_id, state=state)
            if cb.label_id:
                self._canvas.itemconfigure(cb.label_id, state=state)
            for hid in cb.handle_ids:
                self._canvas.itemconfigure(hid, state=state)

            if show:
                visible += 1

        self._status.configure(
            text=f"Showing {visible}/{len(self._canvas_boxes)} detections"
        )

    # ── Model suggestion ──────────────────────────────────────────

    def _apply_suggestion(self) -> None:
        """Apply the model's suggested label to the selected box."""
        if not self._selected_box or not self._model_suggestion or not self._doc_id:
            return

        cbox = self._selected_box
        new_label = self._model_suggestion

        self._push_undo("relabel", cbox, extra={"old_label": cbox.element_type})
        self._store.save_correction(
            doc_id=self._doc_id,
            page=self._page,
            correction_type="relabel",
            corrected_label=new_label,
            corrected_bbox=cbox.pdf_bbox,
            detection_id=cbox.detection_id,
            original_label=cbox.element_type,
            original_bbox=cbox.pdf_bbox,
            session_id=self._session_id,
        )
        old_label = cbox.element_type
        cbox.element_type = new_label
        cbox.corrected = True
        self._session_count += 1
        self._update_session_label()
        self._draw_box(cbox)

        self._model_suggestion = None
        self._suggest_label.configure(text="")
        self._suggest_btn.pack_forget()
        self._type_var.set(new_label)
        self._status.configure(text=f"Applied suggestion: {old_label} → {new_label}")

    # ── Page navigation ───────────────────────────────────────────

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

    # ── Annotation stats ──────────────────────────────────────────

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

    # ── Active learning ────────────────────────────────────────────

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

            result = suggest_next_page(self._store, self._classifier._model_path)
            if result is None:
                messagebox.showinfo(
                    "Active Learning",
                    "No unannotated pages with high uncertainty found.",
                )
                return

            doc_id, page = result
            # Look up PDF path from doc_id
            row = self._store._conn.execute(
                "SELECT pdf_path, filename FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if row:
                pdf_path_str = row["pdf_path"] or row["filename"]
                pdf_path = Path(pdf_path_str)
                if not pdf_path.exists():
                    # Try filename in current directory as fallback
                    pdf_path = Path(row["filename"])
                if pdf_path.exists():
                    self.state.set_pdf(pdf_path)
                    self._page_var.set(page)
                    self._navigate_to_page()
                    self._status.configure(
                        text=f"Active learning: page {page} (highest uncertainty)"
                    )
                else:
                    messagebox.showwarning(
                        "File not found",
                        f"PDF not found: {pdf_path_str}",
                    )
            else:
                messagebox.showwarning("Not Found", f"Document {doc_id} not in DB.")
        except Exception as exc:
            messagebox.showerror("Active Learning Error", str(exc))

    # ── Snapshots ──────────────────────────────────────────────────

    def _on_snapshot(self) -> None:
        """Create a snapshot of the corrections database."""
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
        listbox = tk.Listbox(win, width=55, height=10)
        listbox.pack(padx=10, fill="both", expand=True)

        for s in snaps:
            ts = s.get("timestamp", "?")
            tag = s.get("tag", "")
            size = s.get("size_kb", 0)
            listbox.insert("end", f"{ts}  [{tag}]  ({size:.0f} KB)")

        def on_restore():
            sel = listbox.curselection()
            if not sel:
                return
            idx = sel[0]
            snap_path = snaps[idx]["path"]
            if messagebox.askyesno(
                "Restore",
                "This will overwrite the current database. Continue?",
                parent=win,
            ):
                try:
                    self._store.restore_snapshot(snap_path)
                    self._status.configure(text="Snapshot restored")
                    win.destroy()
                    # Reload current view
                    if self._pdf_path:
                        self._navigate_to_page()
                except Exception as exc:
                    messagebox.showerror("Restore Error", str(exc), parent=win)

        ttk.Button(win, text="Restore", command=on_restore).pack(padx=10, pady=8)
