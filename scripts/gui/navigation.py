"""Navigation mixin — pan, zoom, and scroll handlers.

These bypass the ToolManager entirely and stay bound directly to the
canvas because they must work regardless of which tool is active.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class NavigationMixin:
    """Pan / zoom / scroll handlers extracted from EventHandlerMixin."""

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
