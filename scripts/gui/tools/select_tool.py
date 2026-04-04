"""Select tool — the default canvas interaction tool.

Handles click-to-select, shift+multi-select, and dispatches to
MoveTool, ResizeTool, LassoTool, or DrawTool based on what was
clicked.
"""

from __future__ import annotations

import tkinter as tk
from typing import TYPE_CHECKING, Any

from plancheck.ingest.ingest import point_in_polygon

from ..annotation_state import HANDLE_POSITIONS
from .base import BaseTool, ToolContext

if TYPE_CHECKING:
    from ..annotation_state import CanvasBox


class SelectTool(BaseTool):
    """Click-to-select and mode dispatcher."""

    name = "select"

    def __init__(self, ctx: ToolContext) -> None:
        super().__init__(ctx)

    # ── Event handlers ─────────────────────────────────────────────

    def on_click(
        self,
        pdf_x: float,
        pdf_y: float,
        cx: float,
        cy: float,
        event: tk.Event,
    ) -> None:
        tab = self.ctx.tab
        canvas = self.ctx.canvas

        # Ctrl+click → word selection or draw-new-box
        ctrl_held = bool(event.state & 0x0004)
        if ctrl_held:
            self._handle_ctrl_click(pdf_x, pdf_y, cx, cy, event)
            return

        shift_held = bool(event.state & 0x0001)

        # Check if clicking on a resize handle of the selected box
        sel = self.ctx.selected_box
        if sel and sel.handle_ids:
            for i, hid in enumerate(sel.handle_ids):
                coords = canvas.coords(hid)
                if coords and len(coords) == 4:
                    hx0, hy0, hx1, hy1 = coords
                    if hx0 <= cx <= hx1 and hy0 <= cy <= hy1:
                        # Switch to resize tool
                        self.ctx.tool_manager.switch_to(
                            "resize",
                            cbox=sel,
                            handle=HANDLE_POSITIONS[i],
                            orig_bbox=sel.pdf_bbox,
                            orig_polygon=(
                                list(sel.polygon) if sel.polygon else None
                            ),
                        )
                        return

        # Hit-test canvas boxes
        clicked = self._hit_test(pdf_x, pdf_y)

        if clicked:
            if shift_held:
                tab._toggle_multi_select(clicked)
            else:
                if clicked is not sel:
                    tab._clear_multi_select()
                    tab._select_box(clicked)
                # Prepare for possible move-drag
                self.ctx.tool_manager.switch_to(
                    "move",
                    cbox=clicked,
                    start_pdf=(pdf_x, pdf_y),
                    orig_bbox=clicked.pdf_bbox,
                    orig_polygon=(
                        list(clicked.polygon) if clicked.polygon else None
                    ),
                )
        else:
            if not shift_held:
                tab._clear_multi_select()
                tab._clear_word_selection()
            # Start lasso
            self.ctx.tool_manager.switch_to(
                "lasso",
                start_cx=cx,
                start_cy=cy,
                word_mode=False,
            )
            tab._deselect()

    def _handle_ctrl_click(
        self,
        pdf_x: float,
        pdf_y: float,
        cx: float,
        cy: float,
        event: tk.Event,
    ) -> None:
        """Handle Ctrl+Click: word selection overlay or new-box drawing."""
        tab = self.ctx.tab
        if tab._word_overlay_on and tab._word_overlay_items:
            # Hit-test word overlay
            for rid, winfo in tab._word_overlay_items.items():
                if (
                    winfo["x0"] <= pdf_x <= winfo["x1"]
                    and winfo["top"] <= pdf_y <= winfo["bottom"]
                ):
                    tab._toggle_word_selected(rid)
                    return
            # No word hit → start word-lasso
            self.ctx.tool_manager.switch_to(
                "lasso",
                start_cx=cx,
                start_cy=cy,
                word_mode=True,
            )
            tab._deselect()
        else:
            # Start drawing a new box
            self.ctx.tool_manager.switch_to(
                "draw",
                start_cx=cx,
                start_cy=cy,
            )

    def _hit_test(self, pdf_x: float, pdf_y: float) -> CanvasBox | None:
        """Find the topmost visible box under *(pdf_x, pdf_y)*."""
        canvas = self.ctx.canvas
        for cbox in reversed(self.ctx.canvas_boxes):
            if (
                cbox.rect_id
                and canvas.itemcget(cbox.rect_id, "state") == "hidden"
            ):
                continue
            if cbox.polygon:
                if point_in_polygon(pdf_x, pdf_y, cbox.polygon):
                    return cbox
            else:
                bx0, by0, bx1, by1 = cbox.pdf_bbox
                if bx0 <= pdf_x <= bx1 and by0 <= pdf_y <= by1:
                    return cbox
        return None
