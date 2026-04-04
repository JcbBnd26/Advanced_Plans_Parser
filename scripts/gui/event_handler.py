"""Event handling mixin for the annotation tab.

Residual mixin retained for backward compatibility during the
Tool-pattern migration.  Methods are progressively replaced with
forwarding stubs to services and tools.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from .annotation_state import CanvasBox


class EventHandlerMixin:
    """Mixin providing all event handling, undo/redo, and interaction methods."""

    # ── ML prediction — delegated to MLPredictor service ──────────

    @staticmethod
    def _format_stage2_candidates(candidates: list[tuple[str, float]]) -> str:
        """Format Stage-2 alternatives — forwards to MLPredictor."""
        from .services.ml_predictor import MLPredictor

        return MLPredictor.format_stage2_candidates(candidates)

    def _get_configured_classifier(self):
        """Return the Stage-1 classifier — forwards to MLPredictor."""
        return self._ml_predictor.get_configured_classifier()

    def _predict_model_suggestion(
        self,
        features: dict[str, Any],
        *,
        text: str = "",
    ) -> tuple[str, float, str] | None:
        """Predict a GUI-facing label suggestion — forwards to MLPredictor."""
        return self._ml_predictor.predict(features, text=text)

    def _predict_model_suggestion_details(
        self,
        features: dict[str, Any],
        *,
        text: str = "",
    ) -> dict[str, Any] | None:
        """Return the suggestion label plus review metadata — forwards to MLPredictor."""
        return self._ml_predictor.predict_details(features, text=text)

    # ── Canvas event handlers — delegated to ToolManager ──────────

    def _on_word_click(self, event: tk.Event) -> str | None:
        """Ctrl+Click — delegated to ToolManager → SelectTool."""

    def _on_canvas_click(self, event: tk.Event) -> None:
        """Button-1 — delegated to ToolManager → active tool."""

    # ── Right-click copy / paste — delegated to Clipboard service ──

    def _copy_box(self, cbox: CanvasBox) -> None:
        """Copy box — forwards to Clipboard service."""
        self._clipboard.copy_box(cbox)

    def _paste_box(self, pdf_x: float, pdf_y: float) -> None:
        """Paste box — forwards to Clipboard service."""
        self._clipboard.paste_box(pdf_x, pdf_y)

    def _key_copy_box(self, event: tk.Event) -> None:
        """Ctrl+C — copy selected words' text, or the selected box."""
        if isinstance(event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)):
            return  # let native copy handle it
        if not self._is_active_tab():
            return
        # If words are selected, copy their text to clipboard
        if self._selected_word_rids and self._word_overlay_items:
            texts = []
            for rid in self._selected_word_rids:
                winfo = self._word_overlay_items.get(rid)
                if winfo and winfo.get("text"):
                    texts.append(winfo["text"])
            if texts:
                self.root.clipboard_clear()
                self.root.clipboard_append(" ".join(texts))
                self._status.configure(
                    text=f"Copied text from {len(texts)} words to clipboard"
                )
                return
        if self._selected_box:
            self._copy_box(self._selected_box)

    def _key_paste_box(self, event: tk.Event) -> None:
        """Ctrl+V — paste a copied box at the centre of the current view."""
        if isinstance(event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)):
            return  # let native paste handle it
        if not self._is_active_tab() or not self._copied_box_template:
            return
        # Place the box at the centre of the visible canvas area
        eff = self._effective_scale()
        # Visible region in canvas coords
        try:
            vx0 = self._canvas.canvasx(0)
            vy0 = self._canvas.canvasy(0)
            vx1 = self._canvas.canvasx(self._canvas.winfo_width())
            vy1 = self._canvas.canvasy(self._canvas.winfo_height())
            pdf_cx = ((vx0 + vx1) / 2) / eff
            pdf_cy = ((vy0 + vy1) / 2) / eff
        except Exception:  # noqa: BLE001 — use fallback center on any error
            pdf_cx, pdf_cy = 300.0, 300.0
        self._paste_box(pdf_cx, pdf_cy)

    # ── Group actions — delegated to actions.group_actions ────────

    def _create_group(self, cbox: CanvasBox) -> None:
        from .actions.group_actions import create_group

        create_group(self, cbox)

    def _on_create_group(self) -> None:
        if self._selected_box and not self._selected_box.group_id:
            self._create_group(self._selected_box)

    def _on_add_to_group(self) -> None:
        if not self._selected_box or not self._selected_box.is_group_root:
            self._status.configure(text="Select the group parent first")
            return
        targets = [
            cb
            for cb in self._multi_selected
            if cb is not self._selected_box and not cb.group_id
        ]
        if not targets:
            self._status.configure(
                text="Shift+click boxes to add, then click 'Add to Group'"
            )
            return
        self._add_children_to_group(targets)

    def _add_children_to_group(self, targets: list["CanvasBox"]) -> None:
        from .actions.group_actions import add_children_to_group

        add_children_to_group(self, targets)

    def _on_remove_from_group(self) -> None:
        if self._selected_box and self._selected_box.group_id:
            self._remove_from_group(self._selected_box)

    def _remove_from_group(self, cbox: CanvasBox) -> None:
        from .actions.group_actions import remove_from_group

        remove_from_group(self, cbox)

    def _update_group_inspector(self, cbox: CanvasBox | None) -> None:
        from .actions.group_actions import update_group_inspector

        update_group_inspector(self, cbox)

    def _select_box(self, cbox: CanvasBox) -> None:
        """Select a box and populate the inspector."""
        # Deselect previous
        if self._selected_box and self._selected_box is not cbox:
            self._selected_box.selected = False
            self._draw_box(self._selected_box)

        cbox.selected = True
        self._selected_box = cbox
        self._draw_box(cbox)

        # Populate inspector
        self._insp_id.configure(text=cbox.detection_id)
        self._set_active_element_type(cbox.element_type)
        conf_text = f"{cbox.confidence:.2%}" if cbox.confidence is not None else "—"
        self._insp_conf.configure(text=conf_text)

        self._insp_text.config(state="normal")
        self._insp_text.delete("1.0", "end")
        self._insp_text.insert("1.0", cbox.text_content)
        self._insp_text.config(state="disabled")
        self._model_suggestion = None
        self._suggest_label.configure(text="")
        self._suggest_detail_label.configure(text="")
        self._suggest_btn.pack_forget()
        if cbox.features:
            # Defer ML prediction to after the UI updates so the box
            # selection feels instant; the suggestion will appear a
            # moment later without blocking the main thread.
            self._canvas.after_idle(self._deferred_predict, cbox)

        # Update multi-select count
        self._update_multi_label()

        # Update group section
        self._update_group_inspector(cbox)

    def _deferred_predict(self, cbox: CanvasBox) -> None:
        """Run ML suggestion after the UI has painted the selection."""
        # Guard: box may have been deselected before this fires
        if self._selected_box is not cbox:
            return
        try:
            prediction = self._predict_model_suggestion_details(
                cbox.features,
                text=cbox.text_content,
            )
            if prediction is None:
                return
            pred_label = prediction["label"]
            pred_text = prediction["text"]
            show_prediction = pred_label != cbox.element_type or bool(
                prediction.get("detail_text")
            )
            if show_prediction:
                self._model_suggestion = pred_label
                self._suggest_label.configure(text=pred_text)
                self._suggest_detail_label.configure(
                    text=prediction.get("detail_text", "")
                )
                if pred_label != cbox.element_type:
                    self._suggest_btn.pack(side="left", padx=4)
                else:
                    self._model_suggestion = None
                    self._suggest_btn.pack_forget()
        except Exception:  # noqa: BLE001 — classifier suggestion is optional
            pass

    def _deselect(self) -> None:
        """Deselect the current box."""
        if self._selected_box:
            self._selected_box.selected = False
            self._draw_box(self._selected_box)
            self._selected_box = None

        self._insp_id.configure(text="—")
        self._set_active_element_type("")
        self._insp_conf.configure(text="—")
        self._insp_text.config(state="normal")
        self._insp_text.delete("1.0", "end")
        self._insp_text.config(state="disabled")

        # Clear suggestion
        self._model_suggestion = None
        self._suggest_label.configure(text="")
        self._suggest_detail_label.configure(text="")
        self._suggest_btn.pack_forget()

        # Clear group section
        self._update_group_inspector(None)

    # ── Drag / Release / Reshape / Move — delegated to ToolManager tools ──

    _DRAG_THROTTLE_MS = 33

    def _on_canvas_drag(self, event: tk.Event) -> None:
        """B1-Motion — delegated to ToolManager → active tool."""

    def _schedule_throttled_drag(self, cx: float, cy: float, handler: Any) -> None:
        """Legacy stub — throttling now lives inside each tool."""

    def _flush_throttled_drag(self, handler: Any) -> None:
        """Legacy stub."""

    def _do_move_drag(self, cx: float, cy: float) -> None:
        """Legacy stub — moved to MoveTool."""

    def _finalize_move(self) -> None:
        """Legacy stub — moved to MoveTool."""

    def _do_handle_drag(self, cx: float, cy: float) -> None:
        """Legacy stub — moved to ResizeTool."""

    def _on_canvas_release(self, event: tk.Event) -> None:
        """ButtonRelease-1 — delegated to ToolManager → active tool."""

    def _finalize_reshape(self) -> None:
        """Legacy stub — moved to ResizeTool."""

    def _finalize_add(self, cx: float, cy: float) -> None:
        """Legacy stub — moved to DrawTool."""

    # ── Inspector actions — delegated to BoxOperations service ─────

    def _auto_refresh_text(self, cbox: CanvasBox) -> None:
        """Re-extract text — forwards to BoxOperations."""
        self._box_ops.auto_refresh_text(cbox)

    def _on_rescan_text(self) -> None:
        """Rescan text — forwards to BoxOperations."""
        self._box_ops.rescan_text()

    def _on_accept(self) -> None:
        self._box_ops.accept()

    def _on_relabel(self) -> None:
        self._box_ops.relabel()

    def _on_delete(self) -> None:
        self._box_ops.delete()

    def _on_dismiss(self) -> None:
        self._box_ops.dismiss()

    def _on_mousewheel(self, event: tk.Event) -> None:
        """Scroll vertically with the mouse wheel — delegated to NavigationMixin."""

    def _on_shift_mousewheel(self, event: tk.Event) -> None:
        """Scroll horizontally — delegated to NavigationMixin."""

    def _on_pan_start(self, event: tk.Event) -> None:
        pass  # delegated to NavigationMixin

    def _on_pan_motion(self, event: tk.Event) -> None:
        pass  # delegated to NavigationMixin

    def _on_pan_end(self, event: tk.Event) -> None:
        pass  # delegated to NavigationMixin

    # ── Polygon-aware text extraction ────────────────────────────────

    def _is_active_tab(self) -> bool:
        """Check if the annotation tab is currently visible."""
        try:
            return self.notebook.index("current") == self.notebook.index(self.frame)
        except Exception:  # noqa: BLE001 — fallback to False if notebook unavailable
            return False

    def _key_accept(self, event: tk.Event) -> None:
        if self._is_active_tab() and not isinstance(
            event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)
        ):
            self._on_accept()

    def _key_delete(self, event: tk.Event) -> None:
        if self._is_active_tab() and not isinstance(
            event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)
        ):
            self._on_delete()

    def _key_dismiss(self, event: tk.Event) -> None:
        if self._is_active_tab() and not isinstance(
            event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)
        ):
            self._on_dismiss()

    def _key_relabel(self, event: tk.Event) -> None:
        if self._is_active_tab() and not isinstance(
            event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)
        ):
            self._type_combo.focus_set()

    def _key_deselect(self, event: tk.Event) -> None:
        if self._is_active_tab():
            self._deselect()

    def _key_prev_box(self, event: tk.Event) -> None:
        if not self._is_active_tab() or not self._canvas_boxes:
            return
        if isinstance(event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)):
            return
        if self._selected_box is None:
            self._select_box(self._canvas_boxes[-1])
        elif self._selected_box not in self._canvas_boxes:
            # Box was deleted; reset selection
            self._selected_box = None
            self._select_box(self._canvas_boxes[-1])
        else:
            idx = self._canvas_boxes.index(self._selected_box)
            prev_idx = (idx - 1) % len(self._canvas_boxes)
            self._select_box(self._canvas_boxes[prev_idx])

    def _key_next_box(self, event: tk.Event) -> None:
        if not self._is_active_tab() or not self._canvas_boxes:
            return
        if isinstance(event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)):
            return
        if self._selected_box is None:
            self._select_box(self._canvas_boxes[0])
        elif self._selected_box not in self._canvas_boxes:
            # Box was deleted; reset selection
            self._selected_box = None
            self._select_box(self._canvas_boxes[0])
        else:
            idx = self._canvas_boxes.index(self._selected_box)
            next_idx = (idx + 1) % len(self._canvas_boxes)
            self._select_box(self._canvas_boxes[next_idx])

    def _key_zoom_in(self, event: tk.Event) -> None:
        pass  # delegated to NavigationMixin

    def _key_zoom_out(self, event: tk.Event) -> None:
        pass  # delegated to NavigationMixin

    def _key_fit(self, event: tk.Event) -> None:
        pass  # delegated to NavigationMixin

    # ── Undo / Redo — delegated to UndoManager service ──────────

    def _push_undo(
        self,
        action: str,
        cbox: CanvasBox,
        *,
        extra: dict | None = None,
    ) -> None:
        """Push an undo record — forwards to UndoManager."""
        self._undo_mgr.push(action, cbox, extra=extra)

    def _undo(self) -> None:
        """Undo the last correction — forwards to UndoManager."""
        self._undo_mgr.undo()

    def _redo(self) -> None:
        """Redo the last undone action — forwards to UndoManager."""
        self._undo_mgr.redo()

    def _key_undo(self, event: tk.Event) -> None:
        if self._is_active_tab():
            self._undo()

    def _key_redo(self, event: tk.Event) -> None:
        if self._is_active_tab():
            self._redo()

    # ── Multi-select ───────────────────────────────────────────────

    def _toggle_multi_select(self, cbox: CanvasBox) -> None:
        """Toggle a box in the multi-selection set."""
        if cbox in self._multi_selected:
            self._multi_selected.remove(cbox)
        else:
            self._multi_selected.append(cbox)
        self._draw_box(cbox)
        self._update_multi_label()

    def _clear_multi_select(self) -> None:
        """Clear all multi-selected boxes."""
        prev = list(self._multi_selected)
        self._multi_selected.clear()
        for cb in prev:
            self._draw_box(cb)
        self._update_multi_label()

    def _update_multi_label(self) -> None:
        """Update the '3 selected' label in the inspector."""
        n = len(self._multi_selected)
        if self._selected_box and self._selected_box not in self._multi_selected:
            n += 1
        if n > 1:
            self._multi_label.configure(text=f"{n} selected")
        else:
            self._multi_label.configure(text="")

    # ── Word-box selection helpers ──────────────────────────────────

    def _set_word_selected(self, rid: int, selected: bool) -> None:
        """Highlight or un-highlight a single word overlay rectangle."""
        if selected:
            self._selected_word_rids.add(rid)
            self._canvas.itemconfigure(rid, outline="#00bfff", width=2)
        else:
            self._selected_word_rids.discard(rid)
            self._canvas.itemconfigure(rid, outline="#b0b0b0", width=1)

    def _toggle_word_selected(self, rid: int) -> None:
        """Toggle a word rectangle's selection state."""
        if rid in self._selected_word_rids:
            self._set_word_selected(rid, False)
        else:
            self._set_word_selected(rid, True)
        self._update_word_selection_label()

    def _clear_word_selection(self) -> None:
        """Deselect all word overlay rectangles."""
        for rid in list(self._selected_word_rids):
            self._set_word_selected(rid, False)
        self._selected_word_rids.clear()
        self._update_word_selection_label()

    def _update_word_selection_label(self) -> None:
        """Update the status bar with word selection count."""
        n = len(self._selected_word_rids)
        if n > 0:
            self._status.configure(text=f"{n} word(s) selected")

    def _key_select_all(self, event: tk.Event) -> None:
        """Ctrl+A — select all visible boxes (and all words if overlay is on)."""
        if not self._is_active_tab():
            return
        if isinstance(event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)):
            return
        self._multi_selected = [
            cb
            for cb in self._canvas_boxes
            if self._canvas.itemcget(cb.rect_id, "state") != "hidden"
        ]
        self._draw_all_boxes()
        self._update_multi_label()
        self._status.configure(text=f"Selected {len(self._multi_selected)} boxes")

    # ── Merge — delegated to actions.merge_actions ─────────────────

    def _on_merge(self) -> None:
        """Merge multi-selected boxes or reshape to word selection — forwards to merge_actions."""
        from .actions.merge_actions import merge_boxes_action

        merge_boxes_action(self)

    def _merge_words_into_detection(
        self, *, forced_type: str | None = None, force_create: bool = False
    ) -> None:
        """Reshape/create detection from word selection — forwards to merge_actions."""
        from .actions.merge_actions import merge_words_into_detection

        merge_words_into_detection(
            self, forced_type=forced_type, force_create=force_create
        )

    def _key_merge(self, event: tk.Event) -> None:
        """Keyboard shortcut M for merge."""
        self._on_merge()

    def _key_group(self, event: tk.Event) -> None:
        """Keyboard shortcut G for group creation.  Works with a selected
        detection box *or* with selected word boxes (creates a detection first)."""
        if isinstance(event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)):
            return
        if not self._is_active_tab():
            return

        # If words are selected, create a detection from them first
        if len(self._selected_word_rids) >= 2 and not self._selected_box:
            self._merge_words_into_detection()
            # _merge_words_into_detection selects the new box if created

        if not self._selected_box:
            self._status.configure(text="Select a box or Alt-click words first")
            return
        if self._selected_box.group_id:
            grp = self._groups.get(self._selected_box.group_id, {})
            label = grp.get("label", "?")
            self._status.configure(text=f"Already in group \u2039{label}\u203a")
        else:
            self._create_group(self._selected_box)

    def _key_link_column(self, event: tk.Event) -> None:
        """Keyboard shortcut L — create a notes_column from selected boxes."""
        if isinstance(event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)):
            return
        if not self._is_active_tab():
            return
        self._on_link_column()

    def _on_link_column(self) -> None:
        """Create a notes_column from selected boxes — forwards to merge_actions."""
        from .actions.merge_actions import link_column_action

        link_column_action(self)

    def _key_toggle_words(self, event: tk.Event) -> None:
        """Keyboard shortcut W for toggling word overlay."""
        if isinstance(event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)):
            return
        if not self._is_active_tab():
            return
        self._word_overlay_var.set(not self._word_overlay_var.get())
        self._toggle_word_overlay()

    # ── Lasso selection — delegated to LassoTool ─────────────────

    def _finalize_lasso(self, cx: float, cy: float) -> None:
        """Legacy stub — moved to LassoTool."""

    # ── Filters ────────────────────────────────────────────────────

    # ── Hover tooltip ──────────────────────────────────────────────

    def _on_canvas_motion(self, event: tk.Event) -> None:
        """Delegated to HoverTooltipMixin."""

    def _show_hover_tooltip(self, x_root: int, y_root: int, cbox: Any) -> None:
        """Delegated to HoverTooltipMixin."""

    def _hide_hover_tooltip(self) -> None:
        """Delegated to HoverTooltipMixin."""
