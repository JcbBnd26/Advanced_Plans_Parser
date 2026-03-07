"""Filter controls mixin for GUI annotation tab.

Handles filter checkbox UI, color picker, and visibility filtering
for detection boxes.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import colorchooser
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tkinter import ttk


class FilterControlsMixin:
    """Mixin providing filter controls and box visibility management.

    Expected attributes on the host class:
        LABEL_COLORS: dict[str, str]
        ELEMENT_TYPES: list[str]
        _filter_frame: tk.Frame | None
        _filter_label_vars: dict[str, tk.BooleanVar]
        _filter_color_btn: ttk.Button | None
        _active_filter_color_type: str | None
        _filter_conf_min: tk.DoubleVar
        _filter_uncorrected_only: tk.BooleanVar
        _canvas_boxes: list[CanvasBox]
        _canvas: tk.Canvas
        _status: ttk.Label
        root: tk.Tk
        _filter_row_widgets: dict[str, tuple[tk.Frame, tk.Checkbutton, tk.Label]]
    """

    # ── Filter UI rebuilding ─────────────────────────────────────

    def _rebuild_filter_controls(self) -> None:
        """Rebuild per-type filter rows with checkboxes and clickable labels."""
        if self._filter_frame is None:
            return

        for child in self._filter_frame.winfo_children():
            child.destroy()

        # Initialize widget storage if missing
        if not hasattr(self, "_filter_row_widgets"):
            self._filter_row_widgets = {}
        self._filter_row_widgets.clear()

        if self._active_filter_color_type not in self.ELEMENT_TYPES:
            self._active_filter_color_type = (
                self.ELEMENT_TYPES[0] if self.ELEMENT_TYPES else None
            )

        for i, etype in enumerate(self.ELEMENT_TYPES):
            if etype not in self._filter_label_vars:
                self._filter_label_vars[etype] = tk.BooleanVar(value=True)

            is_active = etype == self._active_filter_color_type
            row_bg = (
                "SystemHighlight"
                if is_active
                else self._filter_frame.winfo_toplevel().cget("bg")
            )
            row_fg = "SystemHighlightText" if is_active else "SystemWindowText"

            row_frame = tk.Frame(self._filter_frame, bg=row_bg)
            row_frame.grid(row=i // 2, column=i % 2, sticky="w", pady=1, padx=(0, 8))

            cb = tk.Checkbutton(
                row_frame,
                variable=self._filter_label_vars[etype],
                command=self._apply_filters,
                bg=row_bg,
                activebackground=row_bg,
                highlightthickness=0,
                borderwidth=0,
            )
            cb.pack(side="left")

            lbl = tk.Label(
                row_frame,
                text=etype,
                bg=row_bg,
                fg=row_fg,
                cursor="hand2",
                padx=4,
            )
            lbl.pack(side="left")
            lbl.bind(
                "<Button-1>",
                lambda _e, label=etype: self._set_active_filter_color_type(label),
            )

            # Store references for in-place updates
            self._filter_row_widgets[etype] = (row_frame, cb, lbl)

        self._update_filter_color_button_label()

    def _set_active_filter_color_type(self, element_type: str) -> None:
        """Set the active element type target for the shared color picker."""
        if element_type not in self.ELEMENT_TYPES:
            return
        old_active = self._active_filter_color_type
        self._active_filter_color_type = element_type

        # Update colors in-place instead of full rebuild to avoid flicker
        if hasattr(self, "_filter_row_widgets") and self._filter_row_widgets:
            default_bg = self._filter_frame.winfo_toplevel().cget("bg")
            for etype, (row_frame, cb, lbl) in self._filter_row_widgets.items():
                is_active = etype == element_type
                row_bg = "SystemHighlight" if is_active else default_bg
                row_fg = "SystemHighlightText" if is_active else "SystemWindowText"
                row_frame.configure(bg=row_bg)
                cb.configure(bg=row_bg, activebackground=row_bg)
                lbl.configure(bg=row_bg, fg=row_fg)
            self._update_filter_color_button_label()
        else:
            # Fallback to full rebuild if widgets not tracked yet
            self._rebuild_filter_controls()

    def _update_filter_color_button_label(self) -> None:
        """Refresh shared color button text for the active element type."""
        if self._filter_color_btn is None:
            return
        if self._active_filter_color_type:
            self._filter_color_btn.configure(
                text=f"Pick Color: {self._active_filter_color_type}"
            )
            self._filter_color_btn.state(["!disabled"])
        else:
            self._filter_color_btn.configure(text="Pick Color")
            self._filter_color_btn.state(["disabled"])

    # ── Color picking ────────────────────────────────────────────

    def _choose_active_filter_color(self) -> None:
        """Prompt for color of the currently selected filter label type."""
        element_type = self._active_filter_color_type
        if not element_type:
            return
        current = self.LABEL_COLORS.get(element_type, "#888888")
        _rgb, chosen = colorchooser.askcolor(
            color=current,
            title=f"Choose color for {element_type}",
            parent=self.root,
        )
        if not chosen:
            return

        self.LABEL_COLORS[element_type] = chosen
        self._rebuild_filter_controls()
        self._draw_all_boxes()
        self._apply_filters()

    # ── Bulk selection ───────────────────────────────────────────

    def _select_all_filter_types(self) -> None:
        """Enable all element-type filter checkboxes."""
        for var in self._filter_label_vars.values():
            var.set(True)
        self._apply_filters()

    def _deselect_all_filter_types(self) -> None:
        """Disable all element-type filter checkboxes."""
        for var in self._filter_label_vars.values():
            var.set(False)
        self._apply_filters()

    # ── Visibility filtering ─────────────────────────────────────

    def _apply_filters(self) -> None:
        """Show/hide boxes based on the current filter settings."""
        min_conf = self._filter_conf_min.get()
        uncorrected_only = self._filter_uncorrected_only.get()

        visible = 0
        for cb in self._canvas_boxes:
            show = True
            # Label type filter
            var = self._filter_label_vars.get(cb.element_type)
            if var and not var.get():
                show = False
            # Confidence filter
            if show and cb.confidence is not None and cb.confidence < min_conf:
                show = False
            # Uncorrected only
            if show and uncorrected_only and cb.corrected:
                show = False

            state = "normal" if show else "hidden"
            if cb.rect_id:
                self._canvas.itemconfigure(cb.rect_id, state=state)
            if cb.label_id:
                self._canvas.itemconfigure(cb.label_id, state=state)
            for hid in cb.handle_ids:
                self._canvas.itemconfigure(hid, state=state)

            if show:
                visible += 1

        self._status.configure(
            text=f"Showing {visible}/{len(self._canvas_boxes)} detections"
        )
