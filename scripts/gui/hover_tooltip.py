"""Hover tooltip for detection boxes on the canvas.

Always active regardless of the current tool.  The ToolManager gives
the active tool first crack at motion events; if the tool does not
consume the event, the tooltip logic runs.
"""

from __future__ import annotations

import tkinter as tk
from typing import Any


class HoverTooltipMixin:
    """Hover tooltip handlers extracted from EventHandlerMixin."""

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
                f"ID: {cbox.detection_id[:12]}\u2026"
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
