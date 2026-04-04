"""Tool manager — routes canvas events to the active tool."""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, Any

from .base import BaseTool, ToolContext

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# Widget types that should receive keypresses instead of tools
_TEXT_WIDGETS = (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)


class ToolManager:
    """Owns the active tool and forwards canvas events to it.

    Coordinate conversion (canvas → PDF) happens once here so every
    tool receives clean ``(pdf_x, pdf_y)`` without boilerplate.
    """

    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx
        ctx.tool_manager = self
        self._tools: dict[str, BaseTool] = {}
        self._active: BaseTool | None = None
        self._default_name: str = "select"

    # ── Tool registration ──────────────────────────────────────────

    def register(self, tool: BaseTool) -> None:
        """Register a tool by its ``name`` attribute."""
        self._tools[tool.name] = tool

    def set_default(self, name: str) -> None:
        """Set which tool name is used by :meth:`revert_to_default`."""
        self._default_name = name

    # ── Switching ──────────────────────────────────────────────────

    def switch_to(self, name: str, **kwargs: Any) -> None:
        """Deactivate the current tool, activate *name*."""
        tool = self._tools.get(name)
        if tool is None:
            log.warning("Unknown tool: %s", name)
            return
        if self._active is not None:
            self._active.exit()
        self._active = tool
        tool.enter(**kwargs)

    def revert_to_default(self) -> None:
        """Switch back to the default tool (usually SelectTool)."""
        self.switch_to(self._default_name)

    @property
    def active(self) -> BaseTool | None:
        return self._active

    @property
    def active_name(self) -> str:
        return self._active.name if self._active else ""

    # ── Coordinate helpers ─────────────────────────────────────────

    def _to_pdf(self, cx: float, cy: float) -> tuple[float, float]:
        eff = self.ctx.effective_scale()
        return cx / eff, cy / eff

    def _canvas_coords(self, event: tk.Event) -> tuple[float, float]:
        return (
            self.ctx.canvas.canvasx(event.x),
            self.ctx.canvas.canvasy(event.y),
        )

    # ── Event routing ──────────────────────────────────────────────

    def on_click(self, event: tk.Event) -> None:
        """Route <Button-1> to the active tool."""
        if self._active is None:
            return
        cx, cy = self._canvas_coords(event)
        pdf_x, pdf_y = self._to_pdf(cx, cy)
        self._active.on_click(pdf_x, pdf_y, cx, cy, event)

    def on_ctrl_click(self, event: tk.Event) -> str | None:
        """Route <Control-Button-1> to the active tool."""
        if self._active is None:
            return None
        cx, cy = self._canvas_coords(event)
        pdf_x, pdf_y = self._to_pdf(cx, cy)
        self._active.on_click(pdf_x, pdf_y, cx, cy, event)
        return "break"

    def on_drag(self, event: tk.Event) -> None:
        """Route <B1-Motion> to the active tool."""
        if self._active is None:
            return
        cx, cy = self._canvas_coords(event)
        pdf_x, pdf_y = self._to_pdf(cx, cy)
        self._active.on_drag(pdf_x, pdf_y, cx, cy, event)

    def on_release(self, event: tk.Event) -> None:
        """Route <ButtonRelease-1> to the active tool."""
        if self._active is None:
            return
        cx, cy = self._canvas_coords(event)
        pdf_x, pdf_y = self._to_pdf(cx, cy)
        self._active.on_release(pdf_x, pdf_y, cx, cy, event)

    def on_right_click(self, event: tk.Event) -> None:
        """Route <Button-3> to the active tool."""
        if self._active is None:
            return
        cx, cy = self._canvas_coords(event)
        pdf_x, pdf_y = self._to_pdf(cx, cy)
        self._active.on_right_click(pdf_x, pdf_y, cx, cy, event)

    def on_motion(self, event: tk.Event) -> None:
        """Route <Motion> to the active tool."""
        if self._active is None:
            return
        cx, cy = self._canvas_coords(event)
        pdf_x, pdf_y = self._to_pdf(cx, cy)
        self._active.on_motion(pdf_x, pdf_y, cx, cy, event)

    def on_key(self, key: str, event: tk.Event) -> bool:
        """Route a keypress to the active tool.

        Returns True if the tool consumed the key.
        """
        if self._active is None:
            return False
        return self._active.on_key(key, event)

    def handle_escape(self, event: tk.Event) -> None:
        """Escape: if not on default tool, revert. Otherwise deselect."""
        if not self.ctx.is_active_tab():
            return
        if self._active and self._active.name != self._default_name:
            self.revert_to_default()
        else:
            self.ctx.deselect()
