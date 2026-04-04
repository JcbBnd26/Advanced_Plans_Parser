"""Move tool — drag a selected box to a new position.

Activated by SelectTool when clicking inside an already-selected box.
Reverts to SelectTool on release.
"""

from __future__ import annotations

import copy
import tkinter as tk
from typing import Any

from .base import BaseTool, ToolContext


class MoveTool(BaseTool):
    """Handles box move-drag with throttled redraws."""

    name = "move"

    # Throttle interval (~30 fps)
    _THROTTLE_MS = 33

    def __init__(self, ctx: ToolContext) -> None:
        super().__init__(ctx)
        self._cbox: Any = None
        self._start_pdf: tuple[float, float] | None = None
        self._orig_bbox: tuple[float, float, float, float] | None = None
        self._orig_polygon: list[tuple[float, float]] | None = None
        self._actually_moved = False
        self._after_id: int | None = None
        self._pending: tuple[float, float] | None = None

    def enter(self, **kwargs: Any) -> None:
        self._cbox = kwargs["cbox"]
        self._start_pdf = kwargs["start_pdf"]
        self._orig_bbox = kwargs["orig_bbox"]
        self._orig_polygon = kwargs.get("orig_polygon")
        self._actually_moved = False
        self._after_id = None
        self._pending = None

    def exit(self) -> None:
        if self._after_id is not None:
            self.ctx.canvas.after_cancel(self._after_id)
            self._after_id = None
        self._pending = None

    # ── Drag ───────────────────────────────────────────────────────

    def on_drag(
        self,
        pdf_x: float,
        pdf_y: float,
        cx: float,
        cy: float,
        event: tk.Event,
    ) -> None:
        self._pending = (cx, cy)
        if self._after_id is not None:
            return
        self._after_id = self.ctx.canvas.after(
            self._THROTTLE_MS, self._flush
        )

    def _flush(self) -> None:
        self._after_id = None
        coords = self._pending
        if coords is None:
            return
        self._pending = None
        self._do_move(coords[0], coords[1])

    def _do_move(self, cx: float, cy: float) -> None:
        cbox = self._cbox
        if not cbox or not self._start_pdf or not self._orig_bbox:
            return

        eff = self.ctx.effective_scale()
        pdf_x = cx / eff
        pdf_y = cy / eff

        dx = pdf_x - self._start_pdf[0]
        dy = pdf_y - self._start_pdf[1]

        ox0, oy0, ox1, oy1 = self._orig_bbox
        nx0 = ox0 + dx
        ny0 = oy0 + dy
        nx1 = ox1 + dx
        ny1 = oy1 + dy

        if nx0 < 0:
            nx1 -= nx0
            nx0 = 0
        if ny0 < 0:
            ny1 -= ny0
            ny0 = 0

        cbox.pdf_bbox = (nx0, ny0, nx1, ny1)

        if self._orig_polygon:
            cbox.polygon = [
                (px + dx, py + dy) for px, py in self._orig_polygon
            ]

        self._actually_moved = True
        self.ctx.draw_box(cbox)

    # ── Release ────────────────────────────────────────────────────

    def on_release(
        self,
        pdf_x: float,
        pdf_y: float,
        cx: float,
        cy: float,
        event: tk.Event,
    ) -> None:
        # Flush pending drag
        if self._after_id is not None:
            self.ctx.canvas.after_cancel(self._after_id)
            self._after_id = None
        if self._pending is not None:
            pcx, pcy = self._pending
            self._pending = None
            self._do_move(pcx, pcy)

        if self._actually_moved:
            self._finalize()

        self.ctx.tool_manager.revert_to_default()

    def _finalize(self) -> None:
        """Persist the move as a reshape correction."""
        tab = self.ctx.tab
        cbox = self._cbox
        orig_bbox = self._orig_bbox
        if not cbox or not orig_bbox or not self.ctx.doc_id:
            return

        new_bbox = cbox.pdf_bbox
        if orig_bbox == new_bbox:
            return

        tab._push_undo("reshape", cbox, extra={"orig_bbox": orig_bbox})
        self.ctx.store.save_correction(
            doc_id=self.ctx.doc_id,
            page=self.ctx.page,
            correction_type="reshape",
            corrected_label=cbox.element_type,
            corrected_bbox=new_bbox,
            detection_id=cbox.detection_id,
            original_label=cbox.element_type,
            original_bbox=orig_bbox,
            session_id=self.ctx.session_id,
        )
        cbox.corrected = True
        self.ctx.increment_session()
        self.ctx.update_session_label()
        if cbox.polygon:
            self.ctx.store.update_detection_polygon(
                cbox.detection_id, cbox.polygon, new_bbox
            )
        self.ctx.draw_box(cbox)
        tab._auto_refresh_text(cbox)
        self.ctx.set_status("Moved box to new position")
