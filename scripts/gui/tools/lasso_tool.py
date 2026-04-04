"""Lasso tool — rectangular lasso selection of boxes (and optionally words).

Activated by SelectTool when clicking on empty canvas space.
Reverts to SelectTool on release.
"""

from __future__ import annotations

import tkinter as tk
from typing import Any

from .base import BaseTool, ToolContext


class LassoTool(BaseTool):
    """Drag a rectangle to multi-select intersecting boxes / words."""

    name = "lasso"

    def __init__(self, ctx: ToolContext) -> None:
        super().__init__(ctx)
        self._start_cx: float = 0.0
        self._start_cy: float = 0.0
        self._word_mode: bool = False
        self._rect_id: int | None = None

    def enter(self, **kwargs: Any) -> None:
        self._start_cx = kwargs["start_cx"]
        self._start_cy = kwargs["start_cy"]
        self._word_mode = kwargs.get("word_mode", False)
        self._rect_id = None

    def exit(self) -> None:
        if self._rect_id is not None:
            self.ctx.canvas.delete(self._rect_id)
            self._rect_id = None

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
            outline="#00bfff",
            width=1,
            dash=(3, 3),
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

        self._finalize_lasso(cx, cy)
        self.ctx.tool_manager.revert_to_default()

    def _finalize_lasso(self, cx: float, cy: float) -> None:
        tab = self.ctx.tab
        canvas = self.ctx.canvas
        eff = self.ctx.effective_scale()

        sx, sy = self._start_cx, self._start_cy
        lx0 = min(sx, cx) / eff
        ly0 = min(sy, cy) / eff
        lx1 = max(sx, cx) / eff
        ly1 = max(sy, cy) / eff

        # Minimum drag distance
        if (lx1 - lx0) < 5 and (ly1 - ly0) < 5:
            return

        for cb in self.ctx.canvas_boxes:
            if canvas.itemcget(cb.rect_id, "state") == "hidden":
                continue
            bx0, by0, bx1, by1 = cb.pdf_bbox
            if bx0 < lx1 and bx1 > lx0 and by0 < ly1 and by1 > ly0:
                if cb not in tab._multi_selected:
                    tab._multi_selected.append(cb)
                    self.ctx.draw_box(cb)

        self.ctx.update_multi_label()

        # Word-lasso: also select word overlay boxes
        word_count = 0
        if self._word_mode and tab._word_overlay_on and tab._word_overlay_items:
            for rid, winfo in tab._word_overlay_items.items():
                wx0, wy0 = winfo["x0"], winfo["top"]
                wx1, wy1 = winfo["x1"], winfo["bottom"]
                if wx0 < lx1 and wx1 > lx0 and wy0 < ly1 and wy1 > ly0:
                    if rid not in tab._selected_word_rids:
                        tab._set_word_selected(rid, True)
                    word_count += 1

        parts = []
        if tab._multi_selected:
            parts.append(f"{len(tab._multi_selected)} boxes")
        if word_count or tab._selected_word_rids:
            parts.append(f"{len(tab._selected_word_rids)} words")
        self.ctx.set_status(
            f"Lasso selected {', '.join(parts) if parts else '0 items'}"
        )
