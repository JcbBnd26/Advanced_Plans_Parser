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
from typing import Any

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
    """

    STAGES = [
        "ingest",
        "tocr",
        "vocrpp",
        "vocr",
        "reconcile",
        "grouping",
        "analysis",
        "checks",
        "export",
    ]

    def __init__(self, parent: tk.Widget, **kwargs: Any) -> None:
        super().__init__(parent, **kwargs)
        self._labels: dict[str, ttk.Label] = {}
        self._stage_styles = {
            "pending": {"background": "#3c3c3c", "foreground": "#888888"},
            "running": {"background": "#264f78", "foreground": "#ffffff"},
            "done": {"background": "#2d5a2d", "foreground": "#98c379"},
            "skipped": {"background": "#3c3c3c", "foreground": "#555555"},
            "error": {"background": "#5a2d2d", "foreground": "#e06c75"},
        }

        for i, stage in enumerate(self.STAGES):
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

        # Arrow separators
        for i in range(len(self.STAGES) - 1):
            sep = tk.Label(
                self,
                text="\u25b8",
                font=("Consolas", 7),
                fg="#555555",
                bg=(
                    self.cget("background")
                    if self.cget("background") != "SystemButtonFace"
                    else "#f0f0f0"
                ),
            )
            # Place between stage labels – use a sub-grid approach via column offset
            # Actually easier to just put arrows as part of stage labels
            # We'll skip arrow labels for simplicity – the groove relief provides separation

    def set_stage(self, stage: str, status: str) -> None:
        """Set *stage* to *status* ('pending'|'running'|'done'|'skipped'|'error')."""
        if stage not in self._labels:
            return
        style = self._stage_styles.get(status, self._stage_styles["pending"])
        self._labels[stage].config(**style)

    def reset(self) -> None:
        """Reset all stages to pending."""
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


# ---------------------------------------------------------------------------
# StatusBar
# ---------------------------------------------------------------------------


class StatusBar(ttk.Frame):
    """Persistent status bar for the bottom of the main window."""

    def __init__(self, parent: tk.Widget, **kwargs: Any) -> None:
        super().__init__(parent, **kwargs)
        self.columnconfigure(0, weight=1)

        self._left_var = tk.StringVar(value="Ready")
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
            textvariable=self._right_var,
            relief="sunken",
            anchor="e",
            padding=(6, 2),
            width=30,
        ).grid(row=0, column=1, sticky="e")

    def set_status(self, text: str) -> None:
        self._left_var.set(text)

    def set_right(self, text: str) -> None:
        self._right_var.set(text)

    def set_pdf(self, name: str) -> None:
        self._right_var.set(f"PDF: {name}")
