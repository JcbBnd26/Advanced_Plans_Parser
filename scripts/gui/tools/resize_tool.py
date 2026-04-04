"""Resize tool — drag a handle to reshape the selected box.

Activated by SelectTool when clicking on a resize handle.
Reverts to SelectTool on release.
"""

from __future__ import annotations

import copy
import tkinter as tk
from typing import Any

from ..annotation_state import _reshape_bbox_from_handle, _scale_polygon_to_bbox
from .base import BaseTool, ToolContext


class ResizeTool(BaseTool):
    """Handles handle-drag reshape with throttled redraws."""

    name = "resize"

    _THROTTLE_MS = 33

    def __init__(self, ctx: ToolContext) -> None:
        super().__init__(ctx)
        self._cbox: Any = None
        self._handle: str | None = None
        self._orig_bbox: tuple[float, float, float, float] | None = None
        self._orig_polygon: list[tuple[float, float]] | None = None
        self._after_id: int | None = None
        self._pending: tuple[float, float] | None = None

    def enter(self, **kwargs: Any) -> None:
        self._cbox = kwargs["cbox"]
        self._handle = kwargs["handle"]
        self._orig_bbox = kwargs["orig_bbox"]
        self._orig_polygon = kwargs.get("orig_polygon")
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
        self._do_handle_drag(coords[0], coords[1])

    def _do_handle_drag(self, cx: float, cy: float) -> None:
        cbox = self._cbox
        if not cbox or not self._orig_bbox or not self._handle:
            return

        eff = self.ctx.effective_scale()
        px = cx / eff
        py = cy / eff

        nx0, ny0, nx1, ny1 = _reshape_bbox_from_handle(
            self._orig_bbox, self._handle, px, py, min_size=1.0
        )

        if cbox.polygon and self._orig_polygon:
            cbox.polygon = _scale_polygon_to_bbox(
                self._orig_bbox,
                self._orig_polygon,
                (nx0, ny0, nx1, ny1),
            )

        cbox.pdf_bbox = (nx0, ny0, nx1, ny1)
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
        # Flush pending
        if self._after_id is not None:
            self.ctx.canvas.after_cancel(self._after_id)
            self._after_id = None
        if self._pending is not None:
            pcx, pcy = self._pending
            self._pending = None
            self._do_handle_drag(pcx, pcy)

        self._finalize()
        self.ctx.tool_manager.revert_to_default()

    def _finalize(self) -> None:
        """Save a reshape correction after handle drag completes."""
        tab = self.ctx.tab
        cbox = self._cbox
        orig_bbox = self._orig_bbox
        if not cbox or not orig_bbox or not self.ctx.doc_id:
            return

        new_bbox = cbox.pdf_bbox
        if orig_bbox == new_bbox:
            return

        extra: dict = {"orig_bbox": orig_bbox}
        if self._orig_polygon is not None:
            extra["orig_polygon"] = copy.deepcopy(self._orig_polygon)
            extra["polygon"] = copy.deepcopy(cbox.polygon)

        tab._push_undo("reshape", cbox, extra=extra)
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
