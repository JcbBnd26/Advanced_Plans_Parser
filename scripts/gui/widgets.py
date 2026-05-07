"""Reusable GUI widgets for the Advanced Plan Parser.

Provides:
- ``CollapsibleFrame`` – expandable/collapsible section with ▶/▼ indicator
- ``LogPanel`` – scrollable, color-coded log text area
- ``KnobGrid`` – scrollable grid of labelled config parameter entries
- ``StatusBar`` – bottom status strip with context info
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Optional

# ---------------------------------------------------------------------------
# CollapsibleFrame
# ---------------------------------------------------------------------------


class CollapsibleFrame(ttk.Frame):
    """A LabelFrame that can expand/collapse its content area.

    Parameters
    ----------
    parent : tk widget
    title : str – section header text
    initially_open : bool – whether to start expanded (default False)
    """

    def __init__(
        self,
        parent: tk.Widget,
        title: str = "",
        initially_open: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, **kwargs)
        self.columnconfigure(0, weight=1)
        self._is_open = initially_open

        # Header row
        self._header = ttk.Frame(self)
        self._header.grid(row=0, column=0, sticky="ew")
        self._header.columnconfigure(1, weight=1)

        self._indicator_var = tk.StringVar(
            value="\u25bc" if initially_open else "\u25b6"
        )
        self._indicator = ttk.Label(
            self._header,
            textvariable=self._indicator_var,
            width=2,
            cursor="hand2",
        )
        self._indicator.grid(row=0, column=0, padx=(4, 0))

        self._title_label = ttk.Label(
            self._header,
            text=title,
            font=("TkDefaultFont", 9, "bold"),
            cursor="hand2",
        )
        self._title_label.grid(row=0, column=1, sticky="w", padx=(2, 0))

        # Clickable header
        for w in (self._header, self._indicator, self._title_label):
            w.bind("<Button-1>", self._toggle)

        # Content area
        self.content = ttk.Frame(self)
        if initially_open:
            self.content.grid(row=1, column=0, sticky="nsew", padx=4, pady=(2, 4))

    @property
    def is_open(self) -> bool:
        return self._is_open

    def _toggle(self, _event: tk.Event | None = None) -> None:
        if self._is_open:
            self.collapse()
        else:
            self.expand()

    def expand(self) -> None:
        self._is_open = True
        self._indicator_var.set("\u25bc")
        self.content.grid(row=1, column=0, sticky="nsew", padx=4, pady=(2, 4))

    def collapse(self) -> None:
        self._is_open = False
        self._indicator_var.set("\u25b6")
        self.content.grid_remove()

    def set_title(self, title: str) -> None:
        self._title_label.config(text=title)

    def add_header_widget(self, widget_cls, **kwargs) -> tk.Widget:
        """Add an extra widget (e.g. checkbox) into the header row."""
        w = widget_cls(self._header, **kwargs)
        w.grid(row=0, column=2, padx=(8, 0))
        return w


# ---------------------------------------------------------------------------
# LogPanel
# ---------------------------------------------------------------------------


class LogPanel(ttk.Frame):
    """Scrollable, color-coded log output panel.

    Call ``write(text, level)`` to append coloured text.
    Levels: ``"INFO"``, ``"WARNING"``, ``"ERROR"``, ``"DEBUG"``, ``"STAGE"``.
    """

    MAX_LINES = 5000

    def __init__(self, parent: tk.Widget, height: int = 12, **kwargs: Any) -> None:
        super().__init__(parent, **kwargs)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._text = tk.Text(
            self,
            height=height,
            wrap="word",
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="#d4d4d4",
            font=("Consolas", 9),
            state="disabled",
            relief="sunken",
            borderwidth=1,
        )
        scroll = ttk.Scrollbar(self, orient="vertical", command=self._text.yview)
        self._text.configure(yscrollcommand=scroll.set)
        self._text.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

        # Color tags
        self._text.tag_configure("INFO", foreground="#d4d4d4")
        self._text.tag_configure("WARNING", foreground="#e5c07b")
        self._text.tag_configure("ERROR", foreground="#e06c75")
        self._text.tag_configure("DEBUG", foreground="#7f848e")
        self._text.tag_configure("STAGE", foreground="#61afef")
        self._text.tag_configure("SUCCESS", foreground="#98c379")

        # Toolbar
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        ttk.Button(btn_frame, text="Clear", command=self.clear).pack(
            side="left", padx=2
        )
        ttk.Button(btn_frame, text="Copy All", command=self._copy_all).pack(
            side="left", padx=2
        )
        self._auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            btn_frame, text="Auto-scroll", variable=self._auto_scroll_var
        ).pack(side="left", padx=(8, 0))

    def write(self, text: str, level: str = "INFO") -> None:
        """Append *text* with the given colour level."""
        self._text.config(state="normal")
        self._text.insert("end", text, level)
        if not text.endswith("\n"):
            self._text.insert("end", "\n", level)
        # Trim excess lines
        line_count = int(self._text.index("end-1c").split(".")[0])
        if line_count > self.MAX_LINES:
            self._text.delete("1.0", f"{line_count - self.MAX_LINES}.0")
        self._text.config(state="disabled")
        if self._auto_scroll_var.get():
            self._text.see("end")

    def clear(self) -> None:
        self._text.config(state="normal")
        self._text.delete("1.0", "end")
        self._text.config(state="disabled")

    def _copy_all(self) -> None:
        content = self._text.get("1.0", "end-1c")
        self._text.clipboard_clear()
        self._text.clipboard_append(content)


# ---------------------------------------------------------------------------
# KnobGrid  –  scrollable grid of labelled config parameter entries
# ---------------------------------------------------------------------------


class KnobGrid(ttk.Frame):
    """Scrollable grid of labelled config knob entries.

    Parameters
    ----------
    parent : tk widget
    field_names : list of GroupingConfig field names to expose
    defaults : GroupingConfig instance used for default values
    height : canvas scroll height in pixels
    """

    def __init__(
        self,
        parent: tk.Widget,
        field_names: list[str],
        defaults: Any,
        height: int = 160,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, **kwargs)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self._defaults = defaults

        self._canvas = tk.Canvas(self, height=height, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._inner = ttk.Frame(self._canvas)
        self._inner.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        self._canvas.create_window((0, 0), window=self._inner, anchor="nw")
        self._canvas.configure(yscrollcommand=sb.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self.knob_vars: dict[str, tk.StringVar] = {}
        for row_i, name in enumerate(field_names):
            default_val = getattr(defaults, name, "")
            ttk.Label(self._inner, text=name, width=30, anchor="w").grid(
                row=row_i, column=0, sticky="w", padx=(2, 6)
            )
            sv = tk.StringVar(value=str(default_val))
            self.knob_vars[name] = sv
            ttk.Entry(self._inner, textvariable=sv, width=12).grid(
                row=row_i, column=1, sticky="w"
            )
            ttk.Label(
                self._inner, text=f"(default: {default_val})", foreground="gray"
            ).grid(row=row_i, column=2, sticky="w", padx=(6, 0))

    def reset(self) -> None:
        """Reset all knobs to defaults."""
        for name, sv in self.knob_vars.items():
            sv.set(str(getattr(self._defaults, name, "")))

    def collect(self, target: Any) -> Any:
        """Read knob values into *target* object attributes.

        Returns the modified target.
        """
        for name, sv in self.knob_vars.items():
            raw = sv.get().strip()
            if not raw:
                continue
            default_val = getattr(target, name, None)
            if default_val is None:
                continue
            try:
                if isinstance(default_val, bool):
                    setattr(target, name, raw.lower() in ("1", "true", "yes"))
                elif isinstance(default_val, int):
                    setattr(target, name, int(raw))
                elif isinstance(default_val, float):
                    setattr(target, name, float(raw))
                else:
                    setattr(target, name, raw)
            except (ValueError, TypeError):
                pass
        return target


# ---------------------------------------------------------------------------
# StageProgressBar  –  pipeline stage progress indicator
# ---------------------------------------------------------------------------


class StageProgressBar(ttk.Frame):
    """Horizontal stage indicator showing pipeline progress.

    Each stage lights up as it completes: grey → blue (running) → green (done) / red (error).
    A timer label beneath each stage shows elapsed time (counting up while running).
    """

    STAGES = [
        "ingest",
        "tocr",
        "grouping",
        "analysis",
        "checks",
        "export",
    ]

    def __init__(self, parent: tk.Widget, **kwargs: Any) -> None:
        super().__init__(parent, **kwargs)
        self._labels: dict[str, ttk.Label] = {}
        self._timer_labels: dict[str, tk.Label] = {}
        self._stage_start_times: dict[str, float] = {}
        self._timer_after_id: Optional[str] = None
        self._stage_styles = {
            "pending": {"background": "#3c3c3c", "foreground": "#888888"},
            "running": {"background": "#264f78", "foreground": "#ffffff"},
            "done": {"background": "#2d5a2d", "foreground": "#98c379"},
            "skipped": {"background": "#3c3c3c", "foreground": "#555555"},
            "error": {"background": "#5a2d2d", "foreground": "#e06c75"},
        }

        for i, stage in enumerate(self.STAGES):
            # Stage name label (row 0)
            lbl = tk.Label(
                self,
                text=stage,
                font=("Consolas", 8),
                relief="groove",
                padx=4,
                pady=1,
                bg="#3c3c3c",
                fg="#888888",
            )
            lbl.grid(row=0, column=i, padx=1, sticky="ew")
            self.columnconfigure(i, weight=1)
            self._labels[stage] = lbl

            # Timer label beneath (row 1) - centered under stage box
            timer_lbl = tk.Label(
                self,
                text="--:--.-",
                font=("Consolas", 7),
                bg="#1e1e1e",
                fg="#888888",
                anchor="center",
            )
            timer_lbl.grid(row=1, column=i, padx=1, sticky="ew")
            self._timer_labels[stage] = timer_lbl

    def _format_time(self, seconds: float) -> str:
        """Format elapsed seconds as M:SS.m (with tenths of a second)."""
        mins, remainder = divmod(seconds, 60)
        secs = int(remainder)
        tenths = int((remainder - secs) * 10)
        if mins < 10:
            return f"{int(mins)}:{secs:02d}.{tenths}"
        return f"{int(mins):02d}:{secs:02d}.{tenths}"

    def _update_timers(self) -> None:
        """Update all running stage timers (called every 100ms for smooth millisecond display)."""
        import time

        now = time.perf_counter()
        has_running = False
        for stage, start_time in self._stage_start_times.items():
            if start_time is not None and stage in self._timer_labels:
                elapsed = now - start_time
                self._timer_labels[stage].config(
                    text=self._format_time(elapsed),
                    fg="#ffffff",  # Bright white for active timer
                )
                has_running = True

        # Continue polling if any stage is still running (100ms for smooth updates)
        if has_running:
            self._timer_after_id = self.after(100, self._update_timers)
        else:
            self._timer_after_id = None

    def set_stage(self, stage: str, status: str) -> None:
        """Set *stage* to *status* ('pending'|'running'|'done'|'skipped'|'error')."""
        import time

        if stage not in self._labels:
            return
        style = self._stage_styles.get(status, self._stage_styles["pending"])
        self._labels[stage].config(**style)

        # Timer management
        if status == "running":
            # Start timing this stage
            self._stage_start_times[stage] = time.perf_counter()
            self._timer_labels[stage].config(text="0:00.0", fg="#ffffff")
            # Start the polling loop if not already running (100ms for smooth updates)
            if self._timer_after_id is None:
                self._timer_after_id = self.after(100, self._update_timers)
        elif status in ("done", "error", "skipped"):
            # Freeze the timer if we were tracking this stage
            if stage in self._stage_start_times:
                start_time = self._stage_start_times.pop(stage, None)
                if start_time is not None:
                    elapsed = time.perf_counter() - start_time
                    final_time = self._format_time(elapsed)
                    # Color based on status
                    if status == "done":
                        fg_color = "#98c379"  # Green
                    elif status == "error":
                        fg_color = "#e06c75"  # Red
                    else:
                        fg_color = "#666666"  # Dim for skipped
                    self._timer_labels[stage].config(text=final_time, fg=fg_color)
        elif status == "pending":
            # Reset timer for this stage
            self._stage_start_times.pop(stage, None)
            self._timer_labels[stage].config(text="--:--.-", fg="#888888")

    def reset(self) -> None:
        """Reset all stages to pending and clear timers."""
        # Cancel any pending timer update
        if self._timer_after_id is not None:
            self.after_cancel(self._timer_after_id)
            self._timer_after_id = None
        # Clear start times
        self._stage_start_times.clear()
        # Reset all stage labels and timers
        for stage in self.STAGES:
            self.set_stage(stage, "pending")

    def set_running(self, stage: str) -> None:
        """Set *stage* to running and all previous stages to done."""
        idx = self.STAGES.index(stage) if stage in self.STAGES else -1
        for i, s in enumerate(self.STAGES):
            if i < idx:
                self.set_stage(s, "done")
            elif i == idx:
                self.set_stage(s, "running")
            # Leave later stages as-is

    def get_times_text(self) -> str:
        """Return all stage names and their displayed times as formatted text."""
        lines = []
        for stage in self.STAGES:
            if stage in self._timer_labels:
                time_text = self._timer_labels[stage].cget("text")
                if time_text and time_text != "--:--.-":
                    lines.append(f"{stage}: {time_text}")
        return "\n".join(lines) if lines else "No times recorded"

    def copy_times(self) -> None:
        """Copy all stage times to clipboard."""
        text = self.get_times_text()
        self.clipboard_clear()
        self.clipboard_append(text)


# ---------------------------------------------------------------------------
# StatusBar
# ---------------------------------------------------------------------------


class StatusBar(ttk.Frame):
    """Persistent three-zone status bar for the bottom of the main window.

    Layout:  [left: project / doc]  [center: last action]  [right: ML status]
    """

    def __init__(self, parent: tk.Widget, **kwargs: Any) -> None:
        super().__init__(parent, **kwargs)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=0)

        self._left_var = tk.StringVar(value="Ready")
        self._center_var = tk.StringVar(value="")
        self._right_var = tk.StringVar(value="")

        ttk.Label(
            self,
            textvariable=self._left_var,
            relief="sunken",
            anchor="w",
            padding=(6, 2),
        ).grid(row=0, column=0, sticky="ew")

        ttk.Label(
            self,
            textvariable=self._center_var,
            relief="sunken",
            anchor="center",
            padding=(6, 2),
        ).grid(row=0, column=1, sticky="ew")

        ttk.Label(
            self,
            textvariable=self._right_var,
            relief="sunken",
            anchor="e",
            padding=(6, 2),
            width=30,
        ).grid(row=0, column=2, sticky="e")

    def set_status(self, text: str) -> None:
        self._left_var.set(text)

    def set_center(self, text: str) -> None:
        self._center_var.set(text)

    def set_right(self, text: str) -> None:
        self._right_var.set(text)

    def set_pdf(self, name: str) -> None:
        self._right_var.set(f"PDF: {name}")


# ---------------------------------------------------------------------------
# ErrorPanel  –  collapsible error display with navigation
# ---------------------------------------------------------------------------


class ErrorPanel(ttk.Frame):
    """Inline error panel with navigation for displaying pipeline errors.

    Errors are sorted by severity (CRITICAL > ERROR > WARNING) and displayed
    one at a time with prev/next navigation buttons.

    Parameters
    ----------
    parent : tk widget
    """

    # Severity ordering (lower = more severe)
    SEVERITY = {"CRITICAL": 1, "ERROR": 2, "WARNING": 3, "INFO": 4}

    def __init__(self, parent: tk.Widget, **kwargs: Any) -> None:
        super().__init__(parent, **kwargs)
        self.columnconfigure(0, weight=1)

        self._errors: list[tuple[str, str]] = []  # [(message, level), ...]
        self._current_index: int = 0

        # Main container (hidden by default)
        self._container = ttk.LabelFrame(self, text="⚠ Errors", padding=8)
        self._container.columnconfigure(0, weight=1)

        # Navigation row
        nav_frame = ttk.Frame(self._container)
        nav_frame.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        nav_frame.columnconfigure(1, weight=1)

        self._prev_btn = ttk.Button(
            nav_frame, text="◄ Prev", width=8, command=self._prev_error
        )
        self._prev_btn.grid(row=0, column=0, padx=(0, 4))

        self._counter_var = tk.StringVar(value="")
        ttk.Label(
            nav_frame,
            textvariable=self._counter_var,
            font=("TkDefaultFont", 9, "bold"),
            anchor="center",
        ).grid(row=0, column=1)

        self._next_btn = ttk.Button(
            nav_frame, text="Next ►", width=8, command=self._next_error
        )
        self._next_btn.grid(row=0, column=2, padx=(4, 0))

        self._copy_btn = ttk.Button(
            nav_frame, text="📋 Copy", width=8, command=self._copy_error
        )
        self._copy_btn.grid(row=0, column=3, padx=(8, 0))

        self._copy_all_btn = ttk.Button(
            nav_frame, text="📋 Copy All", width=10, command=self._copy_all_errors
        )
        self._copy_all_btn.grid(row=0, column=4, padx=(4, 0))

        self._dismiss_btn = ttk.Button(
            nav_frame, text="✕ Dismiss", width=10, command=self.hide
        )
        self._dismiss_btn.grid(row=0, column=5, padx=(4, 0))

        # Error text display (read-only, selectable)
        self._text = tk.Text(
            self._container,
            height=4,
            wrap="word",
            font=("Consolas", 9),
            state="disabled",
            bg="#2d2d2d",
            fg="#e06c75",
            relief="sunken",
            borderwidth=1,
            padx=6,
            pady=4,
        )
        self._text.grid(row=1, column=0, sticky="nsew")

        # Scrollbar for text
        scrollbar = ttk.Scrollbar(
            self._container, orient="vertical", command=self._text.yview
        )
        scrollbar.grid(row=1, column=1, sticky="ns")
        self._text.configure(yscrollcommand=scrollbar.set)

        # Severity-colored tags
        self._text.tag_configure("CRITICAL", foreground="#ff6b6b")
        self._text.tag_configure("ERROR", foreground="#e06c75")
        self._text.tag_configure("WARNING", foreground="#e5c07b")

        # Copy binding
        self._text.bind("<Control-c>", self._copy_selection)

    def add_error(self, message: str, level: str = "ERROR") -> None:
        """Add an error message and show the panel."""
        level = level.upper()
        if level not in self.SEVERITY:
            level = "ERROR"

        self._errors.append((message, level))
        # Sort by severity (most severe first)
        self._errors.sort(key=lambda e: self.SEVERITY.get(e[1], 99))
        self._current_index = 0  # Reset to show most severe
        self._update_display()
        self.show()

    def clear(self) -> None:
        """Clear all errors and hide the panel."""
        self._errors.clear()
        self._current_index = 0
        self._update_display()
        self.hide()

    def show(self) -> None:
        """Show the error panel."""
        if self._errors:
            self._container.grid(row=0, column=0, sticky="ew", pady=(4, 0))

    def hide(self) -> None:
        """Hide the error panel."""
        self._container.grid_remove()

    def _prev_error(self) -> None:
        """Navigate to previous error."""
        if self._current_index > 0:
            self._current_index -= 1
            self._update_display()

    def _next_error(self) -> None:
        """Navigate to next error."""
        if self._current_index < len(self._errors) - 1:
            self._current_index += 1
            self._update_display()

    def _update_display(self) -> None:
        """Update the text display and navigation state."""
        total = len(self._errors)

        if total == 0:
            self._counter_var.set("")
            self._text.config(state="normal")
            self._text.delete("1.0", "end")
            self._text.config(state="disabled")
            self._prev_btn.config(state="disabled")
            self._next_btn.config(state="disabled")
            return

        # Update counter
        self._counter_var.set(f"Error {self._current_index + 1} of {total}")

        # Update navigation buttons
        self._prev_btn.config(state="normal" if self._current_index > 0 else "disabled")
        self._next_btn.config(
            state="normal" if self._current_index < total - 1 else "disabled"
        )

        # Update text
        message, level = self._errors[self._current_index]
        self._text.config(state="normal")
        self._text.delete("1.0", "end")
        self._text.insert("1.0", f"[{level}] {message}", level)
        self._text.config(state="disabled")

    def _copy_selection(self, event: tk.Event | None = None) -> str:
        """Copy selected text to clipboard."""
        try:
            selection = self._text.get("sel.first", "sel.last")
            self._text.clipboard_clear()
            self._text.clipboard_append(selection)
        except tk.TclError:
            pass  # No selection
        return "break"

    def _copy_error(self) -> None:
        """Copy the current error message to clipboard."""
        if not self._errors:
            return
        message, level = self._errors[self._current_index]
        full_text = f"[{level}] {message}"
        self._text.clipboard_clear()
        self._text.clipboard_append(full_text)

    def _copy_all_errors(self) -> None:
        """Copy all error messages to clipboard."""
        if not self._errors:
            return
        lines = [f"[{level}] {message}" for message, level in self._errors]
        full_text = "\n".join(lines)
        self._text.clipboard_clear()
        self._text.clipboard_append(full_text)

    @property
    def error_count(self) -> int:
        """Return the number of errors."""
        return len(self._errors)


# ---------------------------------------------------------------------------
# TOCRProgressBar  –  per-page visual progress indicator
# ---------------------------------------------------------------------------


class TOCRProgressBar(ttk.Frame):
    """Canvas-based per-page progress display for TOCR processing.

    Shows one small coloured box per page. Boxes fill with colour as TOCR
    completes: gray (pending) → green (success) → yellow (warning) → red (error).
    Clicking a box fires an optional callback with the page number.
    """

    # Minimum and default box sizes
    _MIN_BOX = 6
    _DEFAULT_BOX = 16
    _BOX_PAD = 2

    # Status → colour mapping
    _STATUS_COLORS = {
        "pending": "#3c3c3c",
        "running": "#264f78",
        "success": "#2d5a2d",
        "warning": "#8b7300",
        "error": "#5a2d2d",
    }

    def __init__(
        self,
        parent: tk.Widget,
        on_page_click: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, **kwargs)
        self._on_page_click = on_page_click
        self._page_count = 0
        self._page_statuses: list[str] = []
        self._page_info: list[dict] = []
        self._rects: list[int] = []
        self._box_size = self._DEFAULT_BOX

        self._canvas = tk.Canvas(
            self,
            height=self._DEFAULT_BOX + 8,
            bg="#1e1e1e",
            highlightthickness=0,
        )
        self._canvas.pack(fill="x", expand=True)
        self._canvas.bind("<Configure>", self._redraw)
        self._canvas.bind("<Button-1>", self._on_click)
        self._canvas.bind("<Motion>", self._on_motion)

        # Tooltip
        self._tooltip: tk.Toplevel | None = None

    def set_page_count(self, count: int) -> None:
        """Initialize the progress bar for *count* pages."""
        self._page_count = count
        self._page_statuses = ["pending"] * count
        self._page_info = [{}] * count
        self._rects.clear()
        self._redraw()

    def set_page_status(
        self,
        page: int,
        status: str,
        *,
        info: dict | None = None,
    ) -> None:
        """Update the status of a single page (0-indexed)."""
        if 0 <= page < self._page_count:
            self._page_statuses[page] = status
            if info:
                self._page_info[page] = info
            self._update_rect(page)

    def reset(self) -> None:
        """Clear all pages."""
        self._page_count = 0
        self._page_statuses.clear()
        self._page_info.clear()
        self._rects.clear()
        self._canvas.delete("all")

    def _redraw(self, _event: tk.Event | None = None) -> None:
        """Recalculate layout and redraw all boxes."""
        self._canvas.delete("all")
        self._rects.clear()
        if self._page_count == 0:
            return

        canvas_w = self._canvas.winfo_width()
        if canvas_w < 10:
            canvas_w = 400

        # Calculate box size to fit in available width
        total_pad = self._BOX_PAD * (self._page_count + 1)
        available = canvas_w - total_pad
        box = max(self._MIN_BOX, min(self._DEFAULT_BOX, available // self._page_count))
        self._box_size = box

        # Calculate rows needed
        boxes_per_row = max(1, (canvas_w - self._BOX_PAD) // (box + self._BOX_PAD))
        rows = (self._page_count + boxes_per_row - 1) // boxes_per_row
        canvas_h = rows * (box + self._BOX_PAD) + self._BOX_PAD
        self._canvas.configure(height=max(canvas_h, box + 8))

        for i in range(self._page_count):
            row_idx = i // boxes_per_row
            col_idx = i % boxes_per_row
            x = self._BOX_PAD + col_idx * (box + self._BOX_PAD)
            y = self._BOX_PAD + row_idx * (box + self._BOX_PAD)
            color = self._STATUS_COLORS.get(
                self._page_statuses[i], self._STATUS_COLORS["pending"]
            )
            rect = self._canvas.create_rectangle(
                x, y, x + box, y + box, fill=color, outline="#555555", width=1
            )
            self._rects.append(rect)

    def _update_rect(self, page: int) -> None:
        """Update a single rectangle's colour without full redraw."""
        if page < len(self._rects):
            color = self._STATUS_COLORS.get(
                self._page_statuses[page], self._STATUS_COLORS["pending"]
            )
            self._canvas.itemconfigure(self._rects[page], fill=color)

    def _page_at(self, x: int, y: int) -> int | None:
        """Return the page index at canvas coordinates, or None."""
        if self._page_count == 0:
            return None
        canvas_w = self._canvas.winfo_width()
        box = self._box_size
        boxes_per_row = max(1, (canvas_w - self._BOX_PAD) // (box + self._BOX_PAD))
        col = (x - self._BOX_PAD) // (box + self._BOX_PAD)
        row = (y - self._BOX_PAD) // (box + self._BOX_PAD)
        if col < 0 or row < 0 or col >= boxes_per_row:
            return None
        idx = row * boxes_per_row + col
        if 0 <= idx < self._page_count:
            return idx
        return None

    def _on_click(self, event: tk.Event) -> None:
        page = self._page_at(event.x, event.y)
        if page is not None and self._on_page_click:
            self._on_page_click(page)

    def _on_motion(self, event: tk.Event) -> None:
        page = self._page_at(event.x, event.y)
        if page is None:
            self._hide_tooltip()
            return
        info = self._page_info[page] if page < len(self._page_info) else {}
        status = self._page_statuses[page] if page < len(self._page_statuses) else "?"
        tip_lines = [f"Page {page + 1} — {status}"]
        if info.get("duration"):
            tip_lines.append(f"Duration: {info['duration']:.1f}s")
        if info.get("word_count"):
            tip_lines.append(f"Words: {info['word_count']}")
        self._show_tooltip(event, "\n".join(tip_lines))

    def _show_tooltip(self, event: tk.Event, text: str) -> None:
        self._hide_tooltip()
        self._tooltip = tw = tk.Toplevel(self)
        tw.wm_overrideredirect(True)
        tw.attributes("-topmost", True)
        x = event.x_root + 12
        y = event.y_root + 8
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=text,
            background="#333333",
            foreground="#d4d4d4",
            font=("Consolas", 8),
            relief="solid",
            borderwidth=1,
            padx=4,
            pady=2,
        )
        label.pack()

    def _hide_tooltip(self) -> None:
        if self._tooltip:
            self._tooltip.destroy()
            self._tooltip = None
