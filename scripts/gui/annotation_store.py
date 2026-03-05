"""Annotation store mixin composing label registry, filter, and model mixins.

This module provides a combined mixin that inherits from focused sub-mixins:
- LabelRegistryMixin: Label type management and persistence
- FilterControlsMixin: Filter UI for element types and confidence
- ModelTrainingMixin: ML model training, metrics, and active learning

Plus annotation-specific methods that don't fit elsewhere:
- _deduplicate_boxes: IoU-based duplicate removal
- _load_groups_for_page: Group persistence
- _apply_suggestion: Apply ML suggestion to selected box
"""

from __future__ import annotations

from .mixins import FilterControlsMixin, LabelRegistryMixin, ModelTrainingMixin


class AnnotationStoreMixin(LabelRegistryMixin, FilterControlsMixin, ModelTrainingMixin):
    """Composite mixin providing label registry, filter, and model methods.

    This class composes three focused mixins and adds annotation-specific
    methods for box deduplication, group loading, and suggestion application.
    """

    # ── Box deduplication ────────────────────────────────────────

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

    # ── Group loading ────────────────────────────────────────────

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

    # ── Model suggestion ─────────────────────────────────────────

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
