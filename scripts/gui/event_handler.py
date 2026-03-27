"""Event handling mixin for the annotation tab.

Contains all user interaction handlers: mouse clicks, drags, releases,
keyboard shortcuts, undo/redo, multi-select, word selection, merge,
group management, and copy/paste.
"""

from __future__ import annotations

import copy
import tkinter as tk
from tkinter import colorchooser, messagebox, simpledialog, ttk
from typing import Any

from plancheck.analysis.box_merge import merge_boxes, polygon_bbox
from plancheck.corrections.features import featurize_region
from plancheck.ingest.ingest import (
    extract_text_in_bbox,
    extract_text_in_polygon,
    point_in_polygon,
)

from .annotation_state import (
    HANDLE_POSITIONS,
    CanvasBox,
    _reshape_bbox_from_handle,
    _scale_polygon_to_bbox,
)


class EventHandlerMixin:
    """Mixin providing all event handling, undo/redo, and interaction methods."""

    @staticmethod
    def _format_stage2_candidates(candidates: list[tuple[str, float]]) -> str:
        """Format Stage-2 alternatives for compact inspector display."""
        if not candidates:
            return ""
        return ", ".join(
            f"{label} {confidence:.0%}" for label, confidence in candidates
        )

    def _get_configured_classifier(self):
        """Return the Stage-1 classifier selected by the current GUI config."""
        from pathlib import Path

        from plancheck.corrections.classifier import ElementClassifier

        cfg = getattr(getattr(self, "state", None), "config", None)
        model_path = Path(getattr(cfg, "ml_model_path", self._classifier.model_path))
        current_model_path = Path(getattr(self._classifier, "model_path", model_path))
        if current_model_path == model_path:
            return self._classifier
        return ElementClassifier(model_path=model_path)

    def _predict_model_suggestion(
        self,
        features: dict[str, Any],
        *,
        text: str = "",
    ) -> tuple[str, float, str] | None:
        """Predict a GUI-facing label suggestion using current ML settings."""
        suggestion = self._predict_model_suggestion_details(features, text=text)
        if suggestion is None:
            return None
        return suggestion["label"], suggestion["confidence"], suggestion["text"]

    def _predict_model_suggestion_details(
        self,
        features: dict[str, Any],
        *,
        text: str = "",
    ) -> dict[str, Any] | None:
        """Return the suggestion label plus review metadata for the inspector."""
        classifier = self._get_configured_classifier()
        if not classifier.model_exists():
            return None

        raw_label, raw_conf = classifier.predict(features)
        cfg = getattr(getattr(self, "state", None), "config", None)
        if not getattr(cfg, "ml_hierarchical_enabled", False):
            return {
                "label": raw_label,
                "confidence": raw_conf,
                "text": f"Model suggests: {raw_label} ({raw_conf:.0%})",
                "detail_text": "",
            }

        from pathlib import Path

        from plancheck.corrections.hierarchical_classifier import classify_element

        result = classify_element(
            features,
            text=text,
            stage1_model_path=Path(cfg.ml_model_path),
            stage2_model_path=Path(cfg.ml_stage2_model_path),
            enable_llm=False,
        )

        display_label = result.label
        display_conf = result.confidence
        stage_suffix = "Stage 1"
        detail_parts: list[str] = []

        if result.stage2_skipped and result.label == "title":
            display_label = raw_label
            display_conf = raw_conf
            detail_parts.append(
                f"Routing: Stage 1 {raw_label} ({raw_conf:.0%}) -> Stage 2 skipped."
            )
            detail_parts.append(
                "Review: Stage 2 title refinement is unavailable, so this remains a "
                f"Stage 1 {raw_label} suggestion."
            )
        elif result.subtype:
            stage_suffix = "Stage 2"
            detail_parts.append(
                f"Routing: Stage 1 {raw_label} ({raw_conf:.0%}) -> "
                f"Stage 2 {display_label} ({display_conf:.0%})."
            )

        if result.low_confidence and stage_suffix == "Stage 2":
            stage_suffix = "Stage 2, low confidence"
            alternatives = self._format_stage2_candidates(result.stage2_candidates)
            detail_parts.append("Review: low-confidence title subtype.")
            if alternatives:
                detail_parts.append(f"Alternatives: {alternatives}.")
        elif result.llm_used:
            detail_parts.append(
                f"Routing: Stage 1 {raw_label} ({raw_conf:.0%}) -> "
                f"LLM tiebreak {display_label} ({display_conf:.0%})."
            )
            detail_parts.append(
                "Review: resolved by LLM tiebreaker after close Stage 2 candidates."
            )

        detail_text = " ".join(detail_parts).strip()

        return {
            "label": display_label,
            "confidence": display_conf,
            "text": f"Model suggests: {display_label} ({display_conf:.0%}) [{stage_suffix}]",
            "detail_text": detail_text,
        }

    def _on_word_click(self, event: tk.Event) -> str | None:
        """Handle Ctrl+Click for word overlay selection or new-box drawing."""
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)

        if not (self._word_overlay_on and self._word_overlay_items):
            # Word overlay not active — Ctrl+click starts drawing a new box
            self._draw_start = (cx, cy)
            self._canvas.config(cursor="crosshair")
            return "break"

        # Hit-test using PDF coordinates
        eff = self._effective_scale()
        pdf_x = cx / eff
        pdf_y = cy / eff
        for rid, winfo in self._word_overlay_items.items():
            if (
                winfo["x0"] <= pdf_x <= winfo["x1"]
                and winfo["top"] <= pdf_y <= winfo["bottom"]
            ):
                self._toggle_word_selected(rid)
                return "break"

        # No word hit — start a word-lasso on empty space
        self._lasso_start = (cx, cy)
        self._lasso_word = True
        self._deselect()
        return "break"

    def _on_canvas_click(self, event: tk.Event) -> None:
        """Handle click on canvas — select a box or start drawing."""
        # Get canvas coordinates (accounting for scroll)
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)

        # Shift+click for multi-select
        shift_held = bool(event.state & 0x0001)

        # Check if clicking on a handle of the selected box
        if self._selected_box and self._selected_box.handle_ids:
            for i, hid in enumerate(self._selected_box.handle_ids):
                coords = self._canvas.coords(hid)
                if coords and len(coords) == 4:
                    hx0, hy0, hx1, hy1 = coords
                    if hx0 <= cx <= hx1 and hy0 <= cy <= hy1:
                        self._drag_handle = HANDLE_POSITIONS[i]
                        self._drag_orig_bbox = self._selected_box.pdf_bbox
                        self._drag_orig_polygon = (
                            list(self._selected_box.polygon)
                            if self._selected_box.polygon
                            else None
                        )
                        return

        # Find which box was clicked
        eff = self._effective_scale()
        pdf_x = cx / eff
        pdf_y = cy / eff

        clicked = None
        # Iterate in reverse so top-drawn boxes are selected first
        for cbox in reversed(self._canvas_boxes):
            if (
                cbox.rect_id
                and self._canvas.itemcget(cbox.rect_id, "state") == "hidden"
            ):
                continue
            if cbox.polygon:
                # Point-in-polygon test for merged boxes
                if point_in_polygon(pdf_x, pdf_y, cbox.polygon):
                    clicked = cbox
                    break
            else:
                bx0, by0, bx1, by1 = cbox.pdf_bbox
                if bx0 <= pdf_x <= bx1 and by0 <= pdf_y <= by1:
                    clicked = cbox
                    break

        if clicked:
            if shift_held:
                self._toggle_multi_select(clicked)
            else:
                # Select (or keep selected) and prepare for move-drag.
                # If the user drags, we move the box; if they just
                # click and release, it's a plain select.
                if clicked is not self._selected_box:
                    self._clear_multi_select()
                    self._select_box(clicked)
                self._move_dragging = True
                self._move_start_pdf = (pdf_x, pdf_y)
                self._move_orig_bbox = clicked.pdf_bbox
                self._move_orig_polygon = (
                    list(clicked.polygon) if clicked.polygon else None
                )
        else:
            if not shift_held:
                self._clear_multi_select()
                self._clear_word_selection()
            # Start lasso drag if on empty space in select mode
            self._lasso_start = (cx, cy)
            self._lasso_word = False
            self._deselect()

    # ── Right-click copy / paste ───────────────────────────────────

    def _copy_box(self, cbox: CanvasBox) -> None:
        """Copy box dimensions and type to the internal clipboard."""
        x0, y0, x1, y1 = cbox.pdf_bbox
        self._copied_box_template = {
            "element_type": cbox.element_type,
            "width": x1 - x0,
            "height": y1 - y0,
        }
        self._status.configure(
            text=f"Copied {cbox.element_type} box ({x1 - x0:.0f}×{y1 - y0:.0f} pt)"
        )

    def _paste_box(self, pdf_x: float, pdf_y: float) -> None:
        """Paste a copied box centred at a PDF-space location."""
        if not self._copied_box_template or not self._doc_id:
            return

        w = self._copied_box_template["width"]
        h = self._copied_box_template["height"]
        chosen_type = self._copied_box_template["element_type"]

        x0 = pdf_x - w / 2
        y0 = pdf_y - h / 2
        x1 = pdf_x + w / 2
        y1 = pdf_y + h / 2

        # Clamp to non-negative coordinates
        if x0 < 0:
            x1 -= x0
            x0 = 0
        if y0 < 0:
            y1 -= y0
            y0 = 0

        pdf_bbox = (x0, y0, x1, y1)

        # Extract text from PDF under the pasted box
        text_content = ""
        if self._pdf_path:
            text_content = extract_text_in_bbox(self._pdf_path, self._page, pdf_bbox)

        # Compute features and save
        features = featurize_region(chosen_type, pdf_bbox, None, 2448.0, 1584.0)
        det_id = self._store.save_detection(
            doc_id=self._doc_id,
            page=self._page,
            run_id=self._run_id or "manual",
            element_type=chosen_type,
            bbox=pdf_bbox,
            text_content=text_content,
            features=features,
        )
        self._store.save_correction(
            doc_id=self._doc_id,
            page=self._page,
            correction_type="add",
            corrected_label=chosen_type,
            corrected_bbox=pdf_bbox,
            detection_id=det_id,
            session_id=self._session_id,
        )

        cbox = CanvasBox(
            detection_id=det_id,
            element_type=chosen_type,
            confidence=None,
            text_content=text_content,
            features=features,
            pdf_bbox=pdf_bbox,
            corrected=True,
        )
        self._canvas_boxes.append(cbox)
        self._draw_box(cbox)
        self._select_box(cbox)
        self._session_count += 1
        self._update_session_label()
        self._update_page_summary()
        n_chars = len(text_content)
        self._status.configure(
            text=f"Pasted {chosen_type} detection ({n_chars} chars extracted)"
        )

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

    # ── Group actions ──────────────────────────────────────────────

    def _create_group(self, cbox: CanvasBox) -> None:
        """Create a new group with *cbox* as the root (parent)."""
        if not self._doc_id:
            return

        # Prompt for group name
        name_win = tk.Toplevel(self.root)
        name_win.title("New Group")
        name_win.transient(self.root)
        name_win.grab_set()
        name_win.resizable(True, True)
        name_win.minsize(280, 110)

        ttk.Label(name_win, text="Group name:").pack(padx=10, pady=(10, 4))
        name_var = tk.StringVar(value=cbox.element_type)
        entry = ttk.Entry(name_win, textvariable=name_var, width=28)
        entry.pack(padx=10)
        entry.selection_range(0, "end")
        entry.focus_set()

        result: list[str | None] = [None]

        def on_ok(_event: tk.Event | None = None) -> None:
            result[0] = name_var.get().strip()
            name_win.destroy()

        def on_cancel() -> None:
            name_win.destroy()

        entry.bind("<Return>", on_ok)
        name_win.bind("<Escape>", lambda e: on_cancel())
        btn_f = ttk.Frame(name_win)
        btn_f.pack(pady=8)
        ttk.Button(btn_f, text="OK", command=on_ok).pack(side="left", padx=4)
        ttk.Button(btn_f, text="Cancel", command=on_cancel).pack(side="left", padx=4)

        name_win.update_idletasks()
        name_win.geometry("")  # auto-size to content

        name_win.wait_window()

        label = result[0]
        if not label:
            return

        group_id = self._store.create_group(
            doc_id=self._doc_id,
            page=self._page,
            group_label=label,
            root_detection_id=cbox.detection_id,
        )
        cbox.group_id = group_id
        cbox.is_group_root = True
        self._groups[group_id] = {
            "label": label,
            "root_detection_id": cbox.detection_id,
            "members": [cbox.detection_id],
        }
        self._draw_box(cbox)
        self._select_box(cbox)
        self._status.configure(text=f"Created group \u2039{label}\u203a")

    def _on_create_group(self) -> None:
        """Inspector button: create group from selected box."""
        if self._selected_box and not self._selected_box.group_id:
            self._create_group(self._selected_box)

    def _on_add_to_group(self) -> None:
        """Inspector button: add multi-selected boxes to the active group."""
        if not self._selected_box or not self._selected_box.is_group_root:
            self._status.configure(text="Select the group parent first")
            return
        # Gather targets — multi-selected minus the root itself
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
        """Add a list of CanvasBoxes as children to the selected group."""
        if not self._selected_box or not self._selected_box.group_id:
            return
        gid = self._selected_box.group_id
        grp = self._groups.get(gid)
        if not grp:
            return

        count = 0
        next_order = len(grp["members"])
        for cb in targets:
            if cb.group_id:
                continue  # already in a group
            self._store.add_to_group(gid, cb.detection_id, sort_order=next_order)
            cb.group_id = gid
            cb.is_group_root = False
            grp["members"].append(cb.detection_id)
            next_order += 1
            count += 1
            self._draw_box(cb)

        if count:
            self._draw_group_links(gid)
            self._status.configure(
                text=f"Added {count} box(es) to group \u2039{grp['label']}\u203a"
            )

    def _on_remove_from_group(self) -> None:
        """Inspector button: remove selected box from its group."""
        if self._selected_box and self._selected_box.group_id:
            self._remove_from_group(self._selected_box)

    def _remove_from_group(self, cbox: CanvasBox) -> None:
        """Remove *cbox* from its group. Deletes group if root."""
        gid = cbox.group_id
        if not gid:
            return
        grp = self._groups.get(gid, {})
        label = grp.get("label", "?")
        is_root = cbox.is_group_root

        self._store.remove_from_group(gid, cbox.detection_id)

        if is_root:
            # Remove group from all canvas boxes
            for cb in self._canvas_boxes:
                if cb.group_id == gid:
                    cb.group_id = None
                    cb.is_group_root = False
                    self._draw_box(cb)
            self._groups.pop(gid, None)
            self._clear_group_links()
            self._status.configure(
                text=f"Deleted group \u2039{label}\u203a (parent removed)"
            )
        else:
            cbox.group_id = None
            cbox.is_group_root = False
            if gid in self._groups:
                members = self._groups[gid]["members"]
                if cbox.detection_id in members:
                    members.remove(cbox.detection_id)
            self._draw_box(cbox)
            self._draw_group_links(gid)
            self._status.configure(text=f"Removed from group \u2039{label}\u203a")

        # Refresh inspector
        if self._selected_box:
            self._select_box(self._selected_box)

    def _update_group_inspector(self, cbox: CanvasBox | None) -> None:
        """Populate the group section of the inspector."""
        self._clear_group_links()
        # Hide both buttons by default
        self._btn_create_group.pack_forget()
        self._btn_add_to_group.pack_forget()
        self._btn_remove_group.pack_forget()

        if cbox is None or not cbox.group_id:
            self._insp_group_label.configure(text="—")
            if cbox is not None:
                self._btn_create_group.pack(side="left", padx=(0, 3))
            return

        grp = self._groups.get(cbox.group_id, {})
        label = grp.get("label", "?")
        if cbox.is_group_root:
            n_members = len(grp.get("members", [])) - 1  # exclude root
            self._insp_group_label.configure(
                text=f"\u25cf {label} (parent \u2014 {n_members} children)"
            )
            self._btn_add_to_group.pack(side="left", padx=(0, 3))
        else:
            self._insp_group_label.configure(text=f"\u2192 {label}")
        self._btn_remove_group.pack(side="left", padx=(0, 3))
        self._draw_group_links(cbox.group_id)

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

    # ── Drag (reshape + move + add mode) ────────────────────────────

    # Throttle interval in milliseconds (~30 fps for drag redraws)
    _DRAG_THROTTLE_MS = 33

    def _on_canvas_drag(self, event: tk.Event) -> None:
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)

        # Reshape and move drags are throttled to avoid redrawing on
        # every single mouse-motion event (>100/s on most systems).
        if self._drag_handle and self._selected_box:
            self._schedule_throttled_drag(cx, cy, self._do_handle_drag)
            return

        if self._move_dragging and self._selected_box and self._move_start_pdf:
            self._schedule_throttled_drag(cx, cy, self._do_move_drag)
            return

        # Handle add-mode drag (Ctrl+drag to draw new box)
        if self._draw_start:
            if self._draw_rect_id:
                self._canvas.delete(self._draw_rect_id)
            sx, sy = self._draw_start
            self._draw_rect_id = self._canvas.create_rectangle(
                sx,
                sy,
                cx,
                cy,
                outline="#cccccc",
                width=2,
                dash=(4, 4),
            )
            return

        # Lasso selection drag
        if self._lasso_start:
            if self._lasso_rect_id:
                self._canvas.delete(self._lasso_rect_id)
            sx, sy = self._lasso_start
            self._lasso_rect_id = self._canvas.create_rectangle(
                sx,
                sy,
                cx,
                cy,
                outline="#00bfff",
                width=1,
                dash=(3, 3),
            )

    def _schedule_throttled_drag(
        self,
        cx: float,
        cy: float,
        handler: Any,
    ) -> None:
        """Coalesce rapid drag events into at most one redraw per throttle tick."""
        self._drag_pending_coords = (cx, cy)
        if self._drag_after_id is not None:
            # A redraw is already scheduled — just update coords
            return
        self._drag_after_id = self._canvas.after(
            self._DRAG_THROTTLE_MS, self._flush_throttled_drag, handler
        )

    def _flush_throttled_drag(self, handler: Any) -> None:
        """Execute the latest drag coordinates."""
        self._drag_after_id = None
        coords = self._drag_pending_coords
        if coords is not None:
            self._drag_pending_coords = None
            handler(coords[0], coords[1])

    def _do_move_drag(self, cx: float, cy: float) -> None:
        """Update the box position during a move drag."""
        if (
            not self._selected_box
            or not self._move_start_pdf
            or not self._move_orig_bbox
        ):
            return

        eff = self._effective_scale()
        pdf_x = cx / eff
        pdf_y = cy / eff

        dx = pdf_x - self._move_start_pdf[0]
        dy = pdf_y - self._move_start_pdf[1]

        ox0, oy0, ox1, oy1 = self._move_orig_bbox
        nx0 = ox0 + dx
        ny0 = oy0 + dy
        nx1 = ox1 + dx
        ny1 = oy1 + dy

        # Clamp to non-negative
        if nx0 < 0:
            nx1 -= nx0
            nx0 = 0
        if ny0 < 0:
            ny1 -= ny0
            ny0 = 0

        self._selected_box.pdf_bbox = (nx0, ny0, nx1, ny1)

        # Move polygon too if merged box
        if self._move_orig_polygon:
            self._selected_box.polygon = [
                (px + dx, py + dy) for px, py in self._move_orig_polygon
            ]

        self._draw_box(self._selected_box)

    def _finalize_move(self) -> None:
        """Save a reshape correction after move drag completes."""
        cbox = self._selected_box
        if not cbox or not self._move_orig_bbox or not self._doc_id:
            return

        orig_bbox = self._move_orig_bbox
        new_bbox = cbox.pdf_bbox

        if orig_bbox == new_bbox:
            return  # no change

        self._push_undo("reshape", cbox, extra={"orig_bbox": orig_bbox})
        self._store.save_correction(
            doc_id=self._doc_id,
            page=self._page,
            correction_type="reshape",
            corrected_label=cbox.element_type,
            corrected_bbox=new_bbox,
            detection_id=cbox.detection_id,
            original_label=cbox.element_type,
            original_bbox=orig_bbox,
            session_id=self._session_id,
        )
        cbox.corrected = True
        self._session_count += 1
        self._update_session_label()
        # Persist updated polygon to DB if present
        if cbox.polygon:
            self._store.update_detection_polygon(
                cbox.detection_id, cbox.polygon, new_bbox
            )
        self._draw_box(cbox)
        self._auto_refresh_text(cbox)
        self._status.configure(text="Moved box to new position")

    def _do_handle_drag(self, cx: float, cy: float) -> None:
        """Update the box rectangle during a handle drag."""
        if not self._selected_box or not self._drag_orig_bbox:
            return

        eff = self._effective_scale()
        ox0, oy0, ox1, oy1 = self._drag_orig_bbox
        # Convert current mouse to PDF coordinates
        px = cx / eff
        py = cy / eff

        # Compute new bbox based on which handle is being dragged
        if not self._drag_handle:
            return
        nx0, ny0, nx1, ny1 = _reshape_bbox_from_handle(
            self._drag_orig_bbox, self._drag_handle, px, py, min_size=1.0
        )

        # If this is a merged box rendered as a polygon, reshape the polygon too.
        if self._selected_box.polygon and self._drag_orig_polygon:
            self._selected_box.polygon = _scale_polygon_to_bbox(
                self._drag_orig_bbox,
                self._drag_orig_polygon,
                (nx0, ny0, nx1, ny1),
            )

        # Live update on canvas
        self._selected_box.pdf_bbox = (nx0, ny0, nx1, ny1)
        self._draw_box(self._selected_box)

    def _on_canvas_release(self, event: tk.Event) -> None:
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)

        # Flush any pending throttled drag so the final position is accurate
        if self._drag_after_id is not None:
            self._canvas.after_cancel(self._drag_after_id)
            self._drag_after_id = None
        if self._drag_pending_coords is not None:
            pcx, pcy = self._drag_pending_coords
            self._drag_pending_coords = None
            if self._drag_handle and self._selected_box:
                self._do_handle_drag(pcx, pcy)
            elif self._move_dragging and self._selected_box:
                self._do_move_drag(pcx, pcy)

        # Finalize move drag
        if self._move_dragging and self._selected_box:
            self._finalize_move()
            self._move_dragging = False
            self._move_start_pdf = None
            self._move_orig_bbox = None
            self._move_orig_polygon = None
            return

        # Finalize reshape
        if self._drag_handle and self._selected_box:
            self._finalize_reshape()
            self._drag_handle = None
            self._drag_orig_bbox = None
            self._drag_orig_polygon = None
            return

        # Finalize add-mode draw (Ctrl+drag)
        if self._draw_start:
            self._finalize_add(cx, cy)
            return

        # Finalize lasso selection
        if self._lasso_start and self._lasso_rect_id:
            self._finalize_lasso(cx, cy)
            return
        self._lasso_start = None

    def _finalize_reshape(self) -> None:
        """Save a reshape correction after handle drag completes."""
        cbox = self._selected_box
        if not cbox or not self._drag_orig_bbox or not self._doc_id:
            return

        orig_bbox = self._drag_orig_bbox
        new_bbox = cbox.pdf_bbox

        if orig_bbox == new_bbox:
            return  # no change

        extra: dict = {"orig_bbox": orig_bbox}
        if self._drag_orig_polygon is not None:
            extra["orig_polygon"] = copy.deepcopy(self._drag_orig_polygon)
            extra["polygon"] = copy.deepcopy(cbox.polygon)

        self._push_undo("reshape", cbox, extra=extra)
        self._store.save_correction(
            doc_id=self._doc_id,
            page=self._page,
            correction_type="reshape",
            corrected_label=cbox.element_type,
            corrected_bbox=new_bbox,
            detection_id=cbox.detection_id,
            original_label=cbox.element_type,
            original_bbox=orig_bbox,
            session_id=self._session_id,
        )
        cbox.corrected = True
        self._session_count += 1
        self._update_session_label()
        if cbox.polygon:
            self._store.update_detection_polygon(
                cbox.detection_id, cbox.polygon, new_bbox
            )
        self._draw_box(cbox)
        self._auto_refresh_text(cbox)

    def _finalize_add(self, cx: float, cy: float) -> None:
        """Complete adding a new detection after drawing a rectangle."""
        if self._draw_rect_id:
            self._canvas.delete(self._draw_rect_id)
            self._draw_rect_id = None

        self._canvas.config(cursor="")

        if not self._draw_start or not self._doc_id:
            self._draw_start = None
            return

        sx, sy = self._draw_start
        self._draw_start = None

        eff = self._effective_scale()
        x0 = min(sx, cx) / eff
        y0 = min(sy, cy) / eff
        x1 = max(sx, cx) / eff
        y1 = max(sy, cy) / eff

        # Minimum size check
        if (x1 - x0) < 10 or (y1 - y0) < 10:
            self._status.configure(text="Box too small (min 10pt in each dimension)")
            return

        pdf_bbox = (x0, y0, x1, y1)

        # Ask for element type
        type_win = tk.Toplevel(self.root)
        type_win.title("New Element Type")
        type_win.transient(self.root)
        type_win.grab_set()
        type_win.resizable(True, True)
        type_win.minsize(280, 220)

        ttk.Label(type_win, text="Element type:").pack(padx=10, pady=(10, 4))
        type_var = tk.StringVar(value=self.ELEMENT_TYPES[0])
        combo = ttk.Combobox(
            type_win,
            textvariable=type_var,
            values=self.ELEMENT_TYPES,
            width=20,
        )
        combo.pack(padx=10)

        ttk.Label(type_win, text="Title subtype:").pack(padx=10, pady=(8, 4))
        subtype_var = tk.StringVar(value="")
        subtype_combo = ttk.Combobox(
            type_win,
            textvariable=subtype_var,
            values=self._title_subtypes(),
            width=20,
            state="disabled",
        )
        subtype_combo.pack(padx=10)

        def _sync_add_dialog(*_args) -> None:
            label = self._normalize_element_type_name(type_var.get())
            if label in self._title_subtypes():
                subtype_var.set(label)
                subtype_combo.configure(state="readonly")
            elif label == "title_block":
                subtype_var.set("")
                subtype_combo.configure(state="readonly")
            else:
                subtype_var.set("")
                subtype_combo.configure(state="disabled")

        def _apply_add_subtype(_event=None) -> None:
            subtype = self._normalize_element_type_name(subtype_var.get())
            if subtype in self._title_subtypes():
                type_var.set(subtype)

        type_var.trace_add("write", _sync_add_dialog)
        subtype_combo.bind("<<ComboboxSelected>>", _apply_add_subtype)
        _sync_add_dialog()

        result: list[str | None] = [None]

        def on_ok():
            subtype = self._normalize_element_type_name(subtype_var.get())
            if subtype in self._title_subtypes():
                result[0] = subtype
            else:
                result[0] = type_var.get()
            type_win.destroy()

        def on_cancel():
            type_win.destroy()

        btn_f = ttk.Frame(type_win)
        btn_f.pack(pady=10)
        ok_btn = ttk.Button(btn_f, text="OK", command=on_ok)
        ok_btn.pack(side="left", padx=4)
        ttk.Button(btn_f, text="Cancel", command=on_cancel).pack(side="left", padx=4)

        # Keyboard shortcuts
        type_win.bind("<Return>", lambda e: on_ok())
        type_win.bind("<Escape>", lambda e: on_cancel())

        # Let tkinter compute the needed size, then enforce it
        type_win.update_idletasks()
        type_win.geometry("")  # auto-size to content
        combo.focus_set()

        type_win.wait_window()

        chosen_type = result[0]
        if not chosen_type:
            return

        # Extract text from PDF under the drawn box
        text_content = ""
        if self._pdf_path:
            text_content = extract_text_in_bbox(self._pdf_path, self._page, pdf_bbox)

        # Compute features and save
        features = featurize_region(
            chosen_type, pdf_bbox, None, 2448.0, 1584.0  # fallback dims
        )
        det_id = self._store.save_detection(
            doc_id=self._doc_id,
            page=self._page,
            run_id=self._run_id or "manual",
            element_type=chosen_type,
            bbox=pdf_bbox,
            text_content=text_content,
            features=features,
        )
        # Also save as an "add" correction
        self._store.save_correction(
            doc_id=self._doc_id,
            page=self._page,
            correction_type="add",
            corrected_label=chosen_type,
            corrected_bbox=pdf_bbox,
            detection_id=det_id,
            session_id=self._session_id,
        )

        cbox = CanvasBox(
            detection_id=det_id,
            element_type=chosen_type,
            confidence=None,
            text_content=text_content,
            features=features,
            pdf_bbox=pdf_bbox,
            corrected=True,
        )
        self._canvas_boxes.append(cbox)
        self._draw_box(cbox)
        self._select_box(cbox)
        self._session_count += 1
        self._update_session_label()
        self._update_page_summary()
        n_chars = len(text_content)
        self._status.configure(
            text=f"Added {chosen_type} detection ({n_chars} chars extracted)"
        )

    # ── Inspector actions ──────────────────────────────────────────

    def _auto_refresh_text(self, cbox: CanvasBox) -> None:
        """Re-extract text from the PDF for a box and persist to the DB.

        Called automatically after move/reshape so the box's text content
        stays in sync with the region it encompasses.
        """
        if not self._pdf_path:
            return
        try:
            new_text = self._extract_text_for_box(cbox)
        except Exception:
            return

        cbox.text_content = new_text

        # Recompute features with the updated bbox
        features = featurize_region(
            cbox.element_type, cbox.pdf_bbox, None, 2448.0, 1584.0
        )
        cbox.features = features

        # Persist text and features to the database
        self._store.update_detection_text_and_features(
            cbox.detection_id, new_text, features
        )

        # If this box is currently selected, refresh the inspector
        if self._selected_box is cbox:
            self._insp_text.config(state="normal")
            self._insp_text.delete("1.0", "end")
            self._insp_text.insert("1.0", new_text)
            self._insp_text.config(state="disabled")

    def _on_rescan_text(self) -> None:
        """Re-extract text from the PDF under the selected box's current bbox."""
        if not self._selected_box:
            self._status.configure(text="No box selected")
            return
        if not self._pdf_path:
            self._status.configure(text="No PDF loaded")
            return

        cbox = self._selected_box
        try:
            new_text = self._extract_text_for_box(cbox)
        except Exception as exc:
            self._status.configure(text=f"Rescan failed: {exc}")
            return

        cbox.text_content = new_text

        # Recompute features and persist to DB
        features = featurize_region(
            cbox.element_type, cbox.pdf_bbox, None, 2448.0, 1584.0
        )
        cbox.features = features
        self._store.update_detection_text_and_features(
            cbox.detection_id, new_text, features
        )

        # Update the inspector text widget
        self._insp_text.config(state="normal")
        self._insp_text.delete("1.0", "end")
        self._insp_text.insert("1.0", new_text)
        self._insp_text.config(state="disabled")

        mode = "polygon" if cbox.polygon else "rect"
        n_chars = len(new_text)
        self._status.configure(
            text=f"Rescanned text for {cbox.element_type} ({mode}) — {n_chars} chars"
        )

    def _on_accept(self) -> None:
        # Batch-aware: apply to all multi-selected + selected box
        targets = list(self._multi_selected)
        if self._selected_box and self._selected_box not in targets:
            targets.append(self._selected_box)
        if not targets or not self._doc_id:
            self._status.configure(text="No box selected")
            return

        for cbox in targets:
            self._push_undo("accept", cbox)
            self._store.accept_detection(cbox.detection_id, self._doc_id, self._page)
            cbox.corrected = True
            self._session_count += 1
            self._draw_box(cbox)

        self._update_session_label()
        self._clear_multi_select()
        self._status.configure(text=f"Accepted {len(targets)} box(es)")

    def _on_relabel(self) -> None:
        # Batch-aware: apply to all multi-selected + selected box
        targets = list(self._multi_selected)
        if self._selected_box and self._selected_box not in targets:
            targets.append(self._selected_box)
        if not targets or not self._doc_id:
            self._status.configure(text="No box selected")
            return

        new_label = self._normalize_element_type_name(self._type_var.get())
        if not new_label:
            return

        # Validate label is a known type
        if new_label not in self.ELEMENT_TYPES:
            self._status.configure(text=f"Unknown element type: {new_label}")
            return

        count = 0
        for cbox in targets:
            if new_label == cbox.element_type:
                # No change — treat as accept for this box
                self._push_undo("accept", cbox)
                self._store.accept_detection(
                    cbox.detection_id, self._doc_id, self._page
                )
                cbox.corrected = True
                self._session_count += 1
                self._draw_box(cbox)
                continue

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
            cbox.element_type = new_label
            cbox.corrected = True
            self._session_count += 1
            self._draw_box(cbox)
            count += 1

        self._update_session_label()
        self._update_page_summary()
        self._clear_multi_select()
        self._status.configure(text=f"Relabelled {count} box(es) → {new_label}")

    def _on_delete(self) -> None:
        # Batch-aware: apply to all multi-selected + selected box
        targets = list(self._multi_selected)
        if self._selected_box and self._selected_box not in targets:
            targets.append(self._selected_box)
        if not targets or not self._doc_id:
            self._status.configure(text="No box selected")
            return

        n = len(targets)
        msg = (
            "Mark this detection as a false positive?"
            if n == 1
            else f"Reject {n} selected detections?"
        )
        if not messagebox.askyesno("Reject Detection", msg):
            return

        for cbox in targets:
            self._push_undo("delete", cbox)
            self._store.save_correction(
                doc_id=self._doc_id,
                page=self._page,
                correction_type="delete",
                corrected_label=cbox.element_type,
                corrected_bbox=cbox.pdf_bbox,
                detection_id=cbox.detection_id,
                original_label=cbox.element_type,
                original_bbox=cbox.pdf_bbox,
                session_id=self._session_id,
            )

            # Remove from canvas
            if cbox.rect_id:
                self._canvas.delete(cbox.rect_id)
            if cbox.label_id:
                self._canvas.delete(cbox.label_id)
            if cbox.conf_dot_id:
                self._canvas.delete(cbox.conf_dot_id)
            for hid in cbox.handle_ids:
                self._canvas.delete(hid)
            if cbox in self._canvas_boxes:
                self._canvas_boxes.remove(cbox)
            self._session_count += 1

        self._selected_box = None
        self._multi_selected.clear()
        self._deselect()
        self._update_multi_label()
        self._update_session_label()
        self._update_page_summary()
        self._status.configure(text=f"Rejected {n} detection(s)")

    def _on_dismiss(self) -> None:
        """Remove detection(s) from the canvas without creating training data."""
        targets = list(self._multi_selected)
        if self._selected_box and self._selected_box not in targets:
            targets.append(self._selected_box)
        if not targets or not self._doc_id:
            self._status.configure(text="No box selected")
            return

        for cbox in targets:
            self._push_undo("dismiss", cbox)
            self._store.dismiss_detection(
                detection_id=cbox.detection_id,
                doc_id=self._doc_id,
                page=self._page,
                session_id=self._session_id,
            )

            # Remove from canvas (same cleanup as _on_delete)
            if cbox.rect_id:
                self._canvas.delete(cbox.rect_id)
            if cbox.label_id:
                self._canvas.delete(cbox.label_id)
            if cbox.conf_dot_id:
                self._canvas.delete(cbox.conf_dot_id)
            for hid in cbox.handle_ids:
                self._canvas.delete(hid)
            if cbox in self._canvas_boxes:
                self._canvas_boxes.remove(cbox)
            # NOTE: Do NOT increment _session_count.
            # Dismiss is not a correction — it shouldn't count toward
            # the "corrections this session" metric.

        n = len(targets)
        self._selected_box = None
        self._multi_selected.clear()
        self._deselect()
        self._update_multi_label()
        self._update_page_summary()
        self._status.configure(text=f"Dismissed {n} detection(s)")

    def _on_mousewheel(self, event: tk.Event) -> None:
        """Scroll vertically with the mouse wheel."""
        self._canvas.yview_scroll(-1 * (event.delta // 120), "units")

    def _on_shift_mousewheel(self, event: tk.Event) -> None:
        """Scroll horizontally with Shift + mouse wheel."""
        self._canvas.xview_scroll(-1 * (event.delta // 120), "units")

    def _on_pan_start(self, event: tk.Event) -> None:
        self._pan_start = (event.x, event.y)
        self._canvas.config(cursor="fleur")

    def _on_pan_motion(self, event: tk.Event) -> None:
        if self._pan_start:
            dx = event.x - self._pan_start[0]
            dy = event.y - self._pan_start[1]
            self._canvas.xview_scroll(-dx, "units")
            self._canvas.yview_scroll(-dy, "units")
            self._pan_start = (event.x, event.y)

    def _on_pan_end(self, event: tk.Event) -> None:
        self._pan_start = None
        self._canvas.config(cursor="")

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
        if self._is_active_tab():
            self._apply_zoom(1.2)

    def _key_zoom_out(self, event: tk.Event) -> None:
        if self._is_active_tab():
            self._apply_zoom(1 / 1.2)

    def _key_fit(self, event: tk.Event) -> None:
        if isinstance(event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)):
            return
        if self._is_active_tab():
            self._fit_to_window()

    # ── Undo / Redo ────────────────────────────────────────────────

    def _push_undo(
        self,
        action: str,
        cbox: CanvasBox,
        *,
        extra: dict | None = None,
    ) -> None:
        """Push an undo record and clear the redo stack."""
        record: dict = {
            "action": action,
            "detection_id": cbox.detection_id,
            "element_type": cbox.element_type,
            "pdf_bbox": cbox.pdf_bbox,
            "confidence": cbox.confidence,
            "corrected": cbox.corrected,
        }
        if extra:
            record.update(extra)
        self._undo_stack.append(record)
        self._redo_stack.clear()

    def _undo(self) -> None:
        """Undo the last correction (visual only — DB is append-only)."""
        if not self._undo_stack:
            self._status.configure(text="Nothing to undo")
            return

        rec = self._undo_stack.pop()
        self._redo_stack.append(rec)
        action = rec["action"]

        # Find the affected canvas box
        target = None
        for cb in self._canvas_boxes:
            if cb.detection_id == rec["detection_id"]:
                target = cb
                break

        if action == "relabel" and target:
            old_label = rec.get("old_label", target.element_type)
            target.element_type = old_label
            target.corrected = rec.get("corrected", False)
            self._draw_box(target)
            self._status.configure(text=f"Undo relabel → {old_label}")
        elif action == "reshape" and target:
            orig = rec.get("orig_bbox")
            if orig:
                target.pdf_bbox = orig
                target.corrected = rec.get("corrected", False)
                # Restore polygon if the undo record saved one
                if "orig_polygon" in rec:
                    target.polygon = copy.deepcopy(rec["orig_polygon"])
                self._draw_box(target)
            self._status.configure(text="Undo reshape")
        elif action == "delete":
            # Check if box with this detection_id already exists (avoid duplicate)
            already_exists = any(
                cb.detection_id == rec["detection_id"] for cb in self._canvas_boxes
            )
            if already_exists:
                self._status.configure(text="Undo reject (box already restored)")
            else:
                # Re-add the box visually
                cbox = CanvasBox(
                    detection_id=rec["detection_id"],
                    element_type=rec["element_type"],
                    confidence=rec.get("confidence"),
                    text_content="",
                    features={},
                    pdf_bbox=rec["pdf_bbox"],
                    corrected=rec.get("corrected", False),
                )
                self._canvas_boxes.append(cbox)
                self._draw_box(cbox)
                self._status.configure(text="Undo reject")
        elif action == "dismiss":
            self._store.undismiss_detection(rec["detection_id"])
            already_exists = any(
                cb.detection_id == rec["detection_id"] for cb in self._canvas_boxes
            )
            if already_exists:
                self._status.configure(text="Undo dismiss (box already restored)")
            else:
                cbox = CanvasBox(
                    detection_id=rec["detection_id"],
                    element_type=rec["element_type"],
                    confidence=rec.get("confidence"),
                    text_content="",
                    features={},
                    pdf_bbox=rec["pdf_bbox"],
                    corrected=rec.get("corrected", False),
                )
                self._canvas_boxes.append(cbox)
                self._draw_box(cbox)
                self._status.configure(text="Undo dismiss")
        elif action == "accept" and target:
            target.corrected = rec.get("corrected", False)
            self._draw_box(target)
            self._status.configure(text="Undo accept")
        elif action == "merge" and target:
            # Restore all original boxes from the merge record
            merged_boxes = rec.get("merged_boxes", [])
            # Remove the merged survivor from canvas
            if target.rect_id:
                self._canvas.delete(target.rect_id)
            if target.label_id:
                self._canvas.delete(target.label_id)
            for hid in target.handle_ids:
                self._canvas.delete(hid)
            self._canvas_boxes.remove(target)
            # Re-add all original boxes
            for mb in merged_boxes:
                cbox = CanvasBox(
                    detection_id=mb["detection_id"],
                    element_type=mb["element_type"],
                    confidence=mb.get("confidence"),
                    text_content=mb.get("text_content", ""),
                    features={},
                    pdf_bbox=mb["pdf_bbox"],
                    polygon=mb.get("polygon"),
                    corrected=mb.get("corrected", False),
                )
                self._canvas_boxes.append(cbox)
                self._draw_box(cbox)
            self._status.configure(
                text=f"Undo merge → restored {len(merged_boxes)} boxes"
            )
        else:
            self._status.configure(text="Undo (no visual change)")
        self._update_page_summary()

    def _redo(self) -> None:
        """Redo the last undone action."""
        if not self._redo_stack:
            self._status.configure(text="Nothing to redo")
            return

        rec = self._redo_stack.pop()
        self._undo_stack.append(rec)
        action = rec["action"]

        target = None
        for cb in self._canvas_boxes:
            if cb.detection_id == rec["detection_id"]:
                target = cb
                break

        if action == "relabel" and target:
            # rec stores the old_label; the "new" label was the type at the
            # time _push_undo was called, which is stored in element_type.
            target.element_type = rec["element_type"]
            target.corrected = True
            self._draw_box(target)
            self._status.configure(text=f"Redo relabel → {rec['element_type']}")
        elif action == "reshape" and target:
            target.pdf_bbox = rec["pdf_bbox"]
            target.corrected = True
            # If undo record includes a polygon, restore it. Otherwise keep legacy
            # behavior (some reshape flows intentionally clear the polygon).
            if "polygon" in rec:
                target.polygon = copy.deepcopy(rec["polygon"])
            elif "orig_polygon" in rec:
                target.polygon = None
            self._draw_box(target)
            self._status.configure(text="Redo reshape")
        elif action == "delete" and target:
            if target.rect_id:
                self._canvas.delete(target.rect_id)
            if target.label_id:
                self._canvas.delete(target.label_id)
            for hid in target.handle_ids:
                self._canvas.delete(hid)
            self._canvas_boxes.remove(target)
            if self._selected_box is target:
                self._deselect()
            self._status.configure(text="Redo reject")
        elif action == "dismiss" and target:
            self._store.dismiss_detection(
                detection_id=target.detection_id,
                doc_id=self._doc_id,
                page=self._page,
                session_id=self._session_id,
            )
            if target.rect_id:
                self._canvas.delete(target.rect_id)
            if target.label_id:
                self._canvas.delete(target.label_id)
            if target.conf_dot_id:
                self._canvas.delete(target.conf_dot_id)
            for hid in target.handle_ids:
                self._canvas.delete(hid)
            self._canvas_boxes.remove(target)
            if self._selected_box is target:
                self._deselect()
            self._status.configure(text="Redo dismiss")
        elif action == "accept" and target:
            target.corrected = True
            self._draw_box(target)
            self._status.configure(text="Redo accept")
        else:
            self._status.configure(text="Redo (no visual change)")
        self._update_page_summary()

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

    # ── Merge ──────────────────────────────────────────────────────

    def _on_merge(self) -> None:
        """Merge multi-selected overlapping boxes, or reshape a detection to
        the enclosing bbox of selected word-overlay rectangles."""

        # ── Word-merge path: reshape selected detection to word bbox ──
        if len(self._selected_word_rids) >= 2:
            self._merge_words_into_detection()
            return

        # ── Original detection-merge path ─────────────────────────────
        targets = list(self._multi_selected)
        if self._selected_box and self._selected_box not in targets:
            targets.append(self._selected_box)

        if len(targets) < 2:
            self._status.configure(
                text="Select at least 2 boxes to merge (Shift+Click)"
            )
            return
        if not self._doc_id:
            self._status.configure(text="No document loaded")
            return

        # Determine element type: use the largest box's type, or prompt if mixed
        types_seen = {cb.element_type for cb in targets}
        if len(types_seen) == 1:
            merged_type = types_seen.pop()
        else:
            # Find largest by area
            largest = max(
                targets,
                key=lambda cb: (
                    (cb.pdf_bbox[2] - cb.pdf_bbox[0])
                    * (cb.pdf_bbox[3] - cb.pdf_bbox[1])
                ),
            )
            merged_type = largest.element_type
            # Let user override
            answer = simpledialog.askstring(
                "Merge Type",
                f"Boxes have mixed types: {', '.join(sorted(types_seen))}.\n"
                f"Enter the merged element type (default: {merged_type}):",
                initialvalue=merged_type,
                parent=self.root,
            )
            if answer is None:
                return  # cancelled
            merged_type = answer.strip() or merged_type

        # Compute merged polygon
        bboxes = [cb.pdf_bbox for cb in targets]
        merged_poly = merge_boxes(bboxes)
        merged_bbox = polygon_bbox(merged_poly)

        # Re-extract text using the polygon boundary for precision
        if self._pdf_path:
            merged_text = extract_text_in_polygon(
                self._pdf_path, self._page, merged_poly
            )
        else:
            merged_text = "\n".join(
                cb.text_content for cb in targets if cb.text_content
            )

        # Aggregate features from the largest box
        largest = max(
            targets,
            key=lambda cb: (
                (cb.pdf_bbox[2] - cb.pdf_bbox[0]) * (cb.pdf_bbox[3] - cb.pdf_bbox[1])
            ),
        )
        merged_features = largest.features

        # Pick survivor (the first box) — will be reshaped
        survivor = targets[0]
        consumed = targets[1:]

        # Push undo record for the whole merge
        self._push_undo(
            "merge",
            survivor,
            extra={
                "merged_boxes": [
                    {
                        "detection_id": cb.detection_id,
                        "element_type": cb.element_type,
                        "pdf_bbox": cb.pdf_bbox,
                        "polygon": cb.polygon,
                        "confidence": cb.confidence,
                        "text_content": cb.text_content,
                        "corrected": cb.corrected,
                    }
                    for cb in targets
                ],
            },
        )

        # Persist: delete corrections for consumed boxes
        for cb in consumed:
            self._store.save_correction(
                doc_id=self._doc_id,
                page=self._page,
                correction_type="delete",
                corrected_label=cb.element_type,
                corrected_bbox=cb.pdf_bbox,
                detection_id=cb.detection_id,
                original_label=cb.element_type,
                original_bbox=cb.pdf_bbox,
                session_id=self._session_id,
            )
            # Remove from canvas
            if cb.rect_id:
                self._canvas.delete(cb.rect_id)
            if cb.label_id:
                self._canvas.delete(cb.label_id)
            for hid in cb.handle_ids:
                self._canvas.delete(hid)
            self._canvas_boxes.remove(cb)
            self._session_count += 1

        # Persist: reshape survivor to the merged bbox
        self._store.save_correction(
            doc_id=self._doc_id,
            page=self._page,
            correction_type="reshape",
            corrected_label=merged_type,
            corrected_bbox=merged_bbox,
            detection_id=survivor.detection_id,
            original_label=survivor.element_type,
            original_bbox=survivor.pdf_bbox,
            session_id=self._session_id,
        )

        # Update survivor in-place
        survivor.element_type = merged_type
        survivor.pdf_bbox = merged_bbox
        survivor.polygon = merged_poly
        survivor.text_content = merged_text
        survivor.features = merged_features
        survivor.corrected = True
        survivor.merged_from = [cb.detection_id for cb in targets]
        self._session_count += 1

        # Persist polygon to DB so it roundtrips on reload
        self._store.update_detection_polygon(
            survivor.detection_id, merged_poly, merged_bbox
        )

        # Redraw
        self._selected_box = None
        self._clear_multi_select()
        self._draw_box(survivor)

        self._update_session_label()
        self._update_page_summary()
        self._status.configure(
            text=f"Merged {len(targets)} boxes → {merged_type} (polygon with {len(merged_poly)} vertices)"
        )

    def _merge_words_into_detection(
        self, *, forced_type: str | None = None, force_create: bool = False
    ) -> None:
        """Reshape the currently selected detection to the enclosing bbox
        of the selected word-overlay rectangles, or create a new detection
        if no detection box is selected.  When multiple words are selected
        the result is a *union polygon* (via Shapely) so the outline hugs
        the words tightly — important when nearby elements are close."""
        if not self._doc_id:
            self._status.configure(text="No document loaded")
            return

        # Collect per-word bboxes (PDF coords)
        word_bboxes: list[tuple[float, float, float, float]] = []
        texts: list[str] = []
        for rid in self._selected_word_rids:
            winfo = self._word_overlay_items.get(rid)
            if not winfo:
                continue
            word_bboxes.append(
                (winfo["x0"], winfo["top"], winfo["x1"], winfo["bottom"])
            )
            if winfo.get("text"):
                texts.append(winfo["text"])

        if not word_bboxes:
            self._status.configure(text="No valid words selected")
            return

        # Compute union polygon from the word bboxes
        merged_poly: list[tuple[float, float]] | None = None
        if len(word_bboxes) >= 2:
            try:
                merged_poly = merge_boxes(word_bboxes)
            except Exception:  # noqa: BLE001 — fall back to bounding box merge
                merged_poly = None
        new_bbox = (
            polygon_bbox(merged_poly)
            if merged_poly
            else (
                min(b[0] for b in word_bboxes),
                min(b[1] for b in word_bboxes),
                max(b[2] for b in word_bboxes),
                max(b[3] for b in word_bboxes),
            )
        )

        # Re-extract text using the polygon or bbox for precision
        if self._pdf_path and merged_poly:
            merged_text = extract_text_in_polygon(
                self._pdf_path, self._page, merged_poly
            )
        elif self._pdf_path:
            merged_text = extract_text_in_bbox(self._pdf_path, self._page, new_bbox)
        else:
            merged_text = " ".join(texts)

        n_words = len(word_bboxes)

        if self._selected_box and not force_create and forced_type is None:
            # ── Reshape existing detection ───────────────────────
            cbox = self._selected_box
            orig_bbox = cbox.pdf_bbox
            orig_polygon = list(cbox.polygon) if cbox.polygon else None

            self._push_undo(
                "reshape",
                cbox,
                extra={"orig_bbox": orig_bbox, "orig_polygon": orig_polygon},
            )

            self._store.save_correction(
                doc_id=self._doc_id,
                page=self._page,
                correction_type="reshape",
                corrected_label=cbox.element_type,
                corrected_bbox=new_bbox,
                detection_id=cbox.detection_id,
                original_label=cbox.element_type,
                original_bbox=orig_bbox,
                corrected_text=merged_text,
                session_id=self._session_id,
            )

            cbox.pdf_bbox = new_bbox
            cbox.polygon = merged_poly
            cbox.text_content = merged_text
            cbox.corrected = True
            self._session_count += 1

            self._store.update_detection_polygon(
                cbox.detection_id, merged_poly, new_bbox
            )

            self._draw_box(cbox)
            self._clear_word_selection()
            self._update_session_label()
            self._update_page_summary()
            poly_info = f" (polygon {len(merged_poly)} pts)" if merged_poly else ""
            self._status.configure(
                text=f"Reshaped {cbox.element_type} to enclose {n_words} words{poly_info}"
            )
        else:
            # ── Create new detection from words ──────────────────
            # Auto-label via classifier if possible, unless user forced a type
            features = featurize_region("misc_title", new_bbox, None, 2448.0, 1584.0)
            chosen_type = forced_type or (self._type_var.get() or "misc_title")
            if forced_type is None and features:
                try:
                    prediction = self._predict_model_suggestion(
                        features,
                        text=merged_text,
                    )
                    if prediction is None:
                        raise ValueError("No configured model is available")
                    pred_label, pred_conf, _pred_text = prediction
                    if pred_conf and pred_conf > 0.5:
                        chosen_type = pred_label
                except Exception:  # noqa: BLE001 — classifier is optional
                    pass

            det_id = self._store.save_detection(
                doc_id=self._doc_id,
                page=self._page,
                run_id=self._run_id or "manual",
                element_type=chosen_type,
                bbox=new_bbox,
                text_content=merged_text,
                features=features,
            )
            self._store.save_correction(
                doc_id=self._doc_id,
                page=self._page,
                correction_type="add",
                corrected_label=chosen_type,
                corrected_bbox=new_bbox,
                detection_id=det_id,
                session_id=self._session_id,
            )

            cbox = CanvasBox(
                detection_id=det_id,
                element_type=chosen_type,
                confidence=None,
                text_content=merged_text,
                features=features,
                pdf_bbox=new_bbox,
                polygon=merged_poly,
                corrected=True,
            )
            # Persist polygon so it roundtrips on reload
            if merged_poly:
                self._store.update_detection_polygon(det_id, merged_poly, new_bbox)
            self._canvas_boxes.append(cbox)
            self._draw_box(cbox)
            self._select_box(cbox)
            self._session_count += 1
            self._update_session_label()
            self._update_page_summary()
            self._clear_word_selection()
            poly_info = f" (polygon {len(merged_poly)} pts)" if merged_poly else ""
            self._status.configure(
                text=f"Created {chosen_type} from {n_words} words{poly_info}"
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
        """Create a notes_column detection that encloses the selected
        header / notes_block boxes, then group them under it."""
        if not self._doc_id:
            return

        # ── Collect targets ────────────────────────────────────────
        targets = list(self._multi_selected)
        if self._selected_box and self._selected_box not in targets:
            targets.append(self._selected_box)

        if len(targets) < 2:
            self._status.configure(
                text="Shift+click ≥2 headers / notes blocks, then press L"
            )
            return

        # ── Validate types ─────────────────────────────────────────
        linkable_types = {"header", "notes_block"}
        non_linkable = [cb for cb in targets if cb.element_type not in linkable_types]
        if non_linkable:
            bad = ", ".join(sorted({cb.element_type for cb in non_linkable}))
            messagebox.showwarning(
                "Invalid Selection",
                f"Only header and notes_block boxes can be linked "
                f"into a notes column.\n\nFound: {bad}",
            )
            return

        # ── Reject boxes already in a group ────────────────────────
        already_grouped = [cb for cb in targets if cb.group_id]
        if already_grouped:
            self._status.configure(
                text=f"{len(already_grouped)} box(es) already in a group"
            )
            return

        # ── Compute tight enclosing bbox ───────────────────────────
        x0 = min(cb.pdf_bbox[0] for cb in targets)
        y0 = min(cb.pdf_bbox[1] for cb in targets)
        x1 = max(cb.pdf_bbox[2] for cb in targets)
        y1 = max(cb.pdf_bbox[3] for cb in targets)
        col_bbox = (x0, y0, x1, y1)

        # ── Extract text from the region ───────────────────────────
        text_content = ""
        if self._pdf_path:
            text_content = extract_text_in_bbox(self._pdf_path, self._page, col_bbox)

        # ── Create the notes_column detection ──────────────────────
        features = featurize_region("notes_column", col_bbox, None, 2448.0, 1584.0)
        det_id = self._store.save_detection(
            doc_id=self._doc_id,
            page=self._page,
            run_id=self._run_id or "manual",
            element_type="notes_column",
            bbox=col_bbox,
            text_content=text_content,
            features=features,
        )
        self._store.save_correction(
            doc_id=self._doc_id,
            page=self._page,
            correction_type="add",
            corrected_label="notes_column",
            corrected_bbox=col_bbox,
            detection_id=det_id,
            session_id=self._session_id,
        )

        col_box = CanvasBox(
            detection_id=det_id,
            element_type="notes_column",
            confidence=None,
            text_content=text_content,
            features=features,
            pdf_bbox=col_bbox,
            corrected=True,
        )
        self._canvas_boxes.append(col_box)
        self._draw_box(col_box)

        # Push notes_column rect behind children so they stay clickable
        if col_box.rect_id:
            self._canvas.tag_lower(col_box.rect_id, "det_box")

        # ── Group: notes_column = root, children = members ─────────
        group_id = self._store.create_group(
            doc_id=self._doc_id,
            page=self._page,
            group_label="notes_column",
            root_detection_id=det_id,
        )
        col_box.group_id = group_id
        col_box.is_group_root = True
        self._groups[group_id] = {
            "label": "notes_column",
            "root_detection_id": det_id,
            "members": [det_id],
        }

        # Sort children top-to-bottom
        targets.sort(key=lambda cb: cb.pdf_bbox[1])
        grp = self._groups[group_id]
        for i, cb in enumerate(targets, start=1):
            self._store.add_to_group(group_id, cb.detection_id, sort_order=i)
            cb.group_id = group_id
            cb.is_group_root = False
            grp["members"].append(cb.detection_id)
            self._draw_box(cb)

        self._draw_group_links(group_id)

        # ── Bookkeeping ────────────────────────────────────────────
        self._session_count += 1
        self._update_session_label()
        self._update_page_summary()
        self._clear_multi_select()
        self._select_box(col_box)
        n_children = len(targets)
        self._status.configure(text=f"Created notes_column from {n_children} boxes")

    def _key_toggle_words(self, event: tk.Event) -> None:
        """Keyboard shortcut W for toggling word overlay."""
        if isinstance(event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)):
            return
        if not self._is_active_tab():
            return
        self._word_overlay_var.set(not self._word_overlay_var.get())
        self._toggle_word_overlay()

    # ── Lasso selection ────────────────────────────────────────────

    def _finalize_lasso(self, cx: float, cy: float) -> None:
        """Select all boxes intersecting the lasso rectangle."""
        if self._lasso_rect_id:
            self._canvas.delete(self._lasso_rect_id)
            self._lasso_rect_id = None

        if not self._lasso_start:
            return

        sx, sy = self._lasso_start
        self._lasso_start = None

        eff = self._effective_scale()
        lx0 = min(sx, cx) / eff
        ly0 = min(sy, cy) / eff
        lx1 = max(sx, cx) / eff
        ly1 = max(sy, cy) / eff

        # Minimum drag distance to count as lasso
        if (lx1 - lx0) < 5 and (ly1 - ly0) < 5:
            return

        for cb in self._canvas_boxes:
            if self._canvas.itemcget(cb.rect_id, "state") == "hidden":
                continue
            bx0, by0, bx1, by1 = cb.pdf_bbox
            # Check intersection
            if bx0 < lx1 and bx1 > lx0 and by0 < ly1 and by1 > ly0:
                if cb not in self._multi_selected:
                    self._multi_selected.append(cb)
                    self._draw_box(cb)

        self._update_multi_label()

        # ── Alt-lasso also selects word overlay boxes ───────────
        word_count = 0
        if self._lasso_word and self._word_overlay_on and self._word_overlay_items:
            for rid, winfo in self._word_overlay_items.items():
                wx0, wy0 = winfo["x0"], winfo["top"]
                wx1, wy1 = winfo["x1"], winfo["bottom"]
                if wx0 < lx1 and wx1 > lx0 and wy0 < ly1 and wy1 > ly0:
                    if rid not in self._selected_word_rids:
                        self._set_word_selected(rid, True)
                    word_count += 1

        self._lasso_word = False

        parts = []
        if self._multi_selected:
            parts.append(f"{len(self._multi_selected)} boxes")
        if word_count or self._selected_word_rids:
            parts.append(f"{len(self._selected_word_rids)} words")
        self._status.configure(
            text=f"Lasso selected {', '.join(parts) if parts else '0 items'}"
        )

    # ── Filters ────────────────────────────────────────────────────

    # ── Hover tooltip ──────────────────────────────────────────────

    def _on_canvas_motion(self, event: tk.Event) -> None:
        """Show a floating tooltip when hovering over a detection box."""
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)

        # Cancel any pending tooltip timer
        pending = getattr(self, "_hover_after_id", None)
        if pending is not None:
            self._canvas.after_cancel(pending)
            self._hover_after_id = None

        # Hit-test in PDF coordinates
        eff = self._effective_scale()
        pdf_x = cx / eff
        pdf_y = cy / eff

        hit_box: Any = None
        for cb in self._canvas_boxes:
            if self._canvas.itemcget(cb.rect_id, "state") == "hidden":
                continue
            bx0, by0, bx1, by1 = cb.pdf_bbox
            if bx0 <= pdf_x <= bx1 and by0 <= pdf_y <= by1:
                hit_box = cb
                break

        if hit_box is None:
            self._hide_hover_tooltip()
            self._canvas.config(cursor="")
            return

        # Change cursor on box hover
        self._canvas.config(cursor="hand2")

        # Debounce tooltip display (50ms)
        self._hover_after_id = self._canvas.after(
            50,
            self._show_hover_tooltip,
            event.x_root,
            event.y_root,
            hit_box,
        )

    def _show_hover_tooltip(self, x_root: int, y_root: int, cbox: Any) -> None:
        """Display or update the hover tooltip near the cursor."""
        self._hide_hover_tooltip()
        if self._closing or not self._canvas.winfo_exists():
            return
        text_preview = (cbox.text_content or "")[:50]
        corrected = "Yes" if getattr(cbox, "corrected", False) else "No"
        lines = [
            f"Type: {cbox.element_type}",
            f"Confidence: {cbox.confidence:.0%}" if cbox.confidence else "",
            f"Text: {text_preview}" if text_preview else "",
            (
                f"ID: {cbox.detection_id[:12]}…"
                if len(cbox.detection_id) > 12
                else f"ID: {cbox.detection_id}"
            ),
            f"Corrected: {corrected}",
        ]
        tip_text = "\n".join(ln for ln in lines if ln)

        tip = tk.Toplevel(self._canvas)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x_root + 14}+{y_root + 10}")
        lbl = tk.Label(
            tip,
            text=tip_text,
            background="#ffffe0",
            foreground="#000",
            relief="solid",
            borderwidth=1,
            font=("TkDefaultFont", 9),
            justify="left",
        )
        lbl.pack()
        self._hover_tip_win = tip

    def _hide_hover_tooltip(self) -> None:
        """Destroy the hover tooltip if it exists."""
        tip = getattr(self, "_hover_tip_win", None)
        if tip is not None:
            try:
                tip.destroy()
            except Exception:  # noqa: BLE001 — widget may already be gone
                pass
            self._hover_tip_win = None
