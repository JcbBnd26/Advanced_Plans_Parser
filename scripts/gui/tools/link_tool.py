"""Link tool — interactive parent/child group wiring.

Activated from the context menu ("Create Parent") or by switching to
the ``"link"`` tool name.  Implements a two-phase state machine:

    PARENT_READY → (click parent) → LINKING → (right-click children) → …
        ↑                              │
        └────── Escape ◄───────────────┘
"""

from __future__ import annotations

import enum
import logging
import tkinter as tk
from typing import TYPE_CHECKING, Any

from plancheck.ingest.ingest import point_in_polygon

from .base import BaseTool, ToolContext

if TYPE_CHECKING:
    from ..annotation_state import CanvasBox

log = logging.getLogger(__name__)

# Visual constants
_PARENT_OUTLINE = "#ffaa00"  # bright amber for parent highlight
_PARENT_WIDTH = 4
_VALID_HOVER_OUTLINE = "#22cc44"
_VALID_HOVER_WIDTH = 3
_RUBBER_BAND_COLOR = "#ff6600"


class _LinkState(enum.Enum):
    PARENT_READY = "parent_ready"
    LINKING = "linking"


class LinkTool(BaseTool):
    """Interactive group-building tool.

    **PARENT_READY** — parent box is highlighted; left-click on it to
    begin linking.  **LINKING** — rubber band line follows cursor;
    right-click a valid child to attach it.  Escape exits at any
    point.
    """

    name = "link"

    def __init__(self, ctx: ToolContext) -> None:
        super().__init__(ctx)
        self._state: _LinkState = _LinkState.PARENT_READY
        self._parent: CanvasBox | None = None
        self._group_id: str | None = None
        self._group_label: str = ""
        self._rubber_band_id: int | None = None
        self._hover_box: CanvasBox | None = None
        self._connected_count: int = 0

    # ── Lifecycle ──────────────────────────────────────────────────

    def enter(
        self,
        *,
        parent: CanvasBox | None = None,
        group_id: str | None = None,
        group_label: str = "",
        **_kwargs: Any,
    ) -> None:
        """Activate the link tool with *parent* already designated."""
        self._parent = parent
        self._group_id = group_id
        self._group_label = group_label
        self._connected_count = 0
        self._rubber_band_id = None
        self._hover_box = None

        if parent is None:
            # Nothing to link — bail back to select
            self.ctx.tool_manager.revert_to_default()
            return

        self._state = _LinkState.PARENT_READY
        self._highlight_parent(True)
        self.ctx.set_status(
            f"Link mode: click the parent box \u2039{group_label}\u203a to start linking"
        )

    def exit(self) -> None:
        """Clean up visual artifacts."""
        self._remove_rubber_band()
        self._unhighlight_hover()
        if self._parent is not None:
            self._highlight_parent(False)
        summary = (
            f"Linked {self._connected_count} child(ren) to "
            f"\u2039{self._group_label}\u203a"
            if self._connected_count
            else "Link mode cancelled"
        )
        self.ctx.set_status(summary)
        self._parent = None
        self._group_id = None

    # ── Event handlers ─────────────────────────────────────────────

    def on_click(
        self,
        pdf_x: float,
        pdf_y: float,
        cx: float,
        cy: float,
        event: tk.Event,
    ) -> None:
        if self._state is _LinkState.PARENT_READY:
            # Must click on the parent box to begin linking
            hit = self._hit_test(pdf_x, pdf_y)
            if hit is self._parent:
                self._state = _LinkState.LINKING
                self.ctx.canvas.config(cursor="crosshair")
                self.ctx.set_status(
                    f"Linking: right-click children to attach to "
                    f"\u2039{self._group_label}\u203a  (Escape to finish)"
                )
            else:
                self.ctx.set_status("Click on the highlighted parent box first")

    def on_right_click(
        self,
        pdf_x: float,
        pdf_y: float,
        cx: float,
        cy: float,
        event: tk.Event,
    ) -> None:
        if self._state is not _LinkState.LINKING:
            return

        hit = self._hit_test(pdf_x, pdf_y)
        if hit is None or hit is self._parent:
            return
        if not self._is_valid_child(hit):
            self.ctx.set_status(
                f"Cannot link: {hit.element_type} "
                f"{'(already in a group)' if hit.group_id else ''}"
            )
            return

        self._connect_child(hit)

    def on_motion(
        self,
        pdf_x: float,
        pdf_y: float,
        cx: float,
        cy: float,
        event: tk.Event,
    ) -> None:
        if self._state is _LinkState.LINKING:
            self._update_rubber_band(cx, cy)
            # Highlight valid target under cursor
            hit = self._hit_test(pdf_x, pdf_y)
            if hit is not None and hit is not self._parent and self._is_valid_child(hit):
                self._highlight_hover(hit)
            else:
                self._unhighlight_hover()

    def on_key(self, key: str, event: tk.Event) -> bool:
        if key == "Escape":
            self.ctx.tool_manager.revert_to_default()
            return True
        return False

    # ── Internal helpers ───────────────────────────────────────────

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

    def _is_valid_child(self, cbox: CanvasBox) -> bool:
        """A box is a valid child if it isn't already in a group."""
        return cbox.group_id is None

    def _connect_child(self, child: CanvasBox) -> None:
        """Attach *child* to the current group."""
        if self._group_id is None or self._parent is None:
            return

        tab = self.ctx.tab
        grp = self.ctx.groups.get(self._group_id)
        if grp is None:
            return

        next_order = len(grp["members"])
        self.ctx.store.add_to_group(
            self._group_id, child.detection_id, sort_order=next_order
        )
        child.group_id = self._group_id
        child.is_group_root = False
        grp["members"].append(child.detection_id)

        self.ctx.draw_box(child)
        self.ctx.draw_group_links(self._group_id)
        self._connected_count += 1

        self._unhighlight_hover()
        self.ctx.set_status(
            f"Linked {child.element_type} to \u2039{self._group_label}\u203a "
            f"({self._connected_count} total)  — right-click more or Escape to finish"
        )

    # ── Visual helpers ─────────────────────────────────────────────

    def _highlight_parent(self, on: bool) -> None:
        """Draw or remove the thick amber border on the parent box."""
        if self._parent is None or self._parent.rect_id is None:
            return
        canvas = self.ctx.canvas
        if on:
            canvas.itemconfigure(
                self._parent.rect_id,
                outline=_PARENT_OUTLINE,
                width=_PARENT_WIDTH,
            )
        else:
            # Redraw normally
            self.ctx.draw_box(self._parent)

    def _update_rubber_band(self, cx: float, cy: float) -> None:
        """Draw a line from the parent's centre to *(cx, cy)*."""
        if self._parent is None or self._parent.rect_id is None:
            return

        canvas = self.ctx.canvas
        eff = self.ctx.effective_scale()
        px0, py0, px1, py1 = self._parent.pdf_bbox
        parent_cx = ((px0 + px1) / 2) * eff
        parent_cy = ((py0 + py1) / 2) * eff

        if self._rubber_band_id is None:
            self._rubber_band_id = canvas.create_line(
                parent_cx,
                parent_cy,
                cx,
                cy,
                fill=_RUBBER_BAND_COLOR,
                width=2,
                dash=(6, 4),
                tags=("link_rubber",),
            )
        else:
            canvas.coords(
                self._rubber_band_id, parent_cx, parent_cy, cx, cy
            )

    def _remove_rubber_band(self) -> None:
        if self._rubber_band_id is not None:
            self.ctx.canvas.delete(self._rubber_band_id)
            self._rubber_band_id = None

    def _highlight_hover(self, cbox: CanvasBox) -> None:
        """Show a green highlight on a valid hover target."""
        if cbox is self._hover_box:
            return
        self._unhighlight_hover()
        self._hover_box = cbox
        if cbox.rect_id is not None:
            self.ctx.canvas.itemconfigure(
                cbox.rect_id,
                outline=_VALID_HOVER_OUTLINE,
                width=_VALID_HOVER_WIDTH,
            )

    def _unhighlight_hover(self) -> None:
        """Remove hover highlight from the previous box."""
        if self._hover_box is not None:
            self.ctx.draw_box(self._hover_box)
            self._hover_box = None
