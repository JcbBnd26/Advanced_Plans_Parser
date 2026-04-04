"""Draw tool — Ctrl+drag to create a new detection box.

Activated by SelectTool on Ctrl+click (when word overlay is off).
Reverts to SelectTool on release.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from plancheck.corrections.features import featurize_region
from plancheck.ingest.ingest import extract_text_in_bbox

from ..annotation_state import CanvasBox
from .base import BaseTool, ToolContext


class DrawTool(BaseTool):
    """Draw a rectangle to add a new detection."""

    name = "draw"

    def __init__(self, ctx: ToolContext) -> None:
        super().__init__(ctx)
        self._start_cx: float = 0.0
        self._start_cy: float = 0.0
        self._rect_id: int | None = None

    def enter(self, **kwargs: Any) -> None:
        self._start_cx = kwargs["start_cx"]
        self._start_cy = kwargs["start_cy"]
        self._rect_id = None
        self.ctx.canvas.config(cursor="crosshair")

    def exit(self) -> None:
        if self._rect_id is not None:
            self.ctx.canvas.delete(self._rect_id)
            self._rect_id = None
        self.ctx.canvas.config(cursor="")

    # ── Drag ───────────────────────────────────────────────────────

    def on_drag(
        self,
        pdf_x: float,
        pdf_y: float,
        cx: float,
        cy: float,
        event: tk.Event,
    ) -> None:
        canvas = self.ctx.canvas
        if self._rect_id is not None:
            canvas.delete(self._rect_id)
        self._rect_id = canvas.create_rectangle(
            self._start_cx,
            self._start_cy,
            cx,
            cy,
            outline="#cccccc",
            width=2,
            dash=(4, 4),
        )

    # ── Release ────────────────────────────────────────────────────

    def on_release(
        self,
        pdf_x: float,
        pdf_y: float,
        cx: float,
        cy: float,
        event: tk.Event,
    ) -> None:
        if self._rect_id is not None:
            self.ctx.canvas.delete(self._rect_id)
            self._rect_id = None
        self.ctx.canvas.config(cursor="")

        self._finalize_add(cx, cy)
        self.ctx.tool_manager.revert_to_default()

    def _finalize_add(self, cx: float, cy: float) -> None:
        """Complete adding a new detection via dialog."""
        tab = self.ctx.tab
        if not self.ctx.doc_id:
            return

        eff = self.ctx.effective_scale()
        sx, sy = self._start_cx, self._start_cy
        x0 = min(sx, cx) / eff
        y0 = min(sy, cy) / eff
        x1 = max(sx, cx) / eff
        y1 = max(sy, cy) / eff

        if (x1 - x0) < 10 or (y1 - y0) < 10:
            self.ctx.set_status("Box too small (min 10pt in each dimension)")
            return

        pdf_bbox = (x0, y0, x1, y1)

        # Ask user for element type
        type_win = tk.Toplevel(tab.root)
        type_win.title("New Element Type")
        type_win.transient(tab.root)
        type_win.grab_set()
        type_win.resizable(True, True)
        type_win.minsize(280, 220)

        ttk.Label(type_win, text="Element type:").pack(padx=10, pady=(10, 4))
        type_var = tk.StringVar(value=tab.ELEMENT_TYPES[0])
        combo = ttk.Combobox(
            type_win,
            textvariable=type_var,
            values=tab.ELEMENT_TYPES,
            width=20,
        )
        combo.pack(padx=10)

        ttk.Label(type_win, text="Title subtype:").pack(padx=10, pady=(8, 4))
        subtype_var = tk.StringVar(value="")
        subtype_combo = ttk.Combobox(
            type_win,
            textvariable=subtype_var,
            values=tab._title_subtypes(),
            width=20,
            state="disabled",
        )
        subtype_combo.pack(padx=10)

        def _sync_add_dialog(*_args) -> None:
            label = tab._normalize_element_type_name(type_var.get())
            if label in tab._title_subtypes():
                subtype_var.set(label)
                subtype_combo.configure(state="readonly")
            elif label == "title_block":
                subtype_var.set("")
                subtype_combo.configure(state="readonly")
            else:
                subtype_var.set("")
                subtype_combo.configure(state="disabled")

        def _apply_add_subtype(_event=None) -> None:
            subtype = tab._normalize_element_type_name(subtype_var.get())
            if subtype in tab._title_subtypes():
                type_var.set(subtype)

        type_var.trace_add("write", _sync_add_dialog)
        subtype_combo.bind("<<ComboboxSelected>>", _apply_add_subtype)
        _sync_add_dialog()

        result: list[str | None] = [None]

        def on_ok():
            subtype = tab._normalize_element_type_name(subtype_var.get())
            if subtype in tab._title_subtypes():
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
        ttk.Button(btn_f, text="Cancel", command=on_cancel).pack(
            side="left", padx=4
        )

        type_win.bind("<Return>", lambda e: on_ok())
        type_win.bind("<Escape>", lambda e: on_cancel())

        type_win.update_idletasks()
        type_win.geometry("")
        combo.focus_set()
        type_win.wait_window()

        chosen_type = result[0]
        if not chosen_type:
            return

        text_content = ""
        if self.ctx.pdf_path:
            text_content = extract_text_in_bbox(
                self.ctx.pdf_path, self.ctx.page, pdf_bbox
            )

        features = featurize_region(
            chosen_type, pdf_bbox, None, 2448.0, 1584.0
        )
        det_id = self.ctx.store.save_detection(
            doc_id=self.ctx.doc_id,
            page=self.ctx.page,
            run_id=self.ctx.run_id or "manual",
            element_type=chosen_type,
            bbox=pdf_bbox,
            text_content=text_content,
            features=features,
        )
        self.ctx.store.save_correction(
            doc_id=self.ctx.doc_id,
            page=self.ctx.page,
            correction_type="add",
            corrected_label=chosen_type,
            corrected_bbox=pdf_bbox,
            detection_id=det_id,
            session_id=self.ctx.session_id,
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
        self.ctx.canvas_boxes.append(cbox)
        self.ctx.draw_box(cbox)
        self.ctx.select_box(cbox)
        self.ctx.increment_session()
        self.ctx.update_session_label()
        self.ctx.update_page_summary()
        n_chars = len(text_content)
        self.ctx.set_status(
            f"Added {chosen_type} detection ({n_chars} chars extracted)"
        )
