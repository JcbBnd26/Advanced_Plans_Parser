"""Tab 4 – Diagnostics: thin composer that assembles diagnostic sections.

Each section lives in its own module under ``diagnostics/``:

* ``font_benchmark`` – Font metrics and A/B/C/D benchmark runner
* ``ml_tools``       – ML calibration, runtime summary, training charts,
                       model comparison
* ``external_tools`` – LayoutLMv3, sentence-transformer, LLM checks,
                       cross-page GNN

Note: VOCRPP preview, config tuning harness, and grouping playground have
been removed from the GUI – those functions will be managed by the LLM
layer in a future release.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from .diagnostics import (
    BenchmarkSection,
    CrossPageGNNSection,
    FontDiagnosticsSection,
    LayoutModelSection,
    LLMSemanticChecksSection,
    MLCalibrationSection,
    MLRuntimeSummarySection,
    ModelComparisonSection,
    TextEmbeddingsSection,
    TrainingProgressSection,
)
from .widgets import LogPanel


class DiagnosticsTab:
    """Tab 4: Diagnostics tools."""

    def __init__(self, notebook: ttk.Notebook, gui_state: Any) -> None:
        self.notebook = notebook
        self.state = gui_state
        self.root = notebook.winfo_toplevel()

        self.frame = ttk.Frame(notebook)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=3)
        self.frame.rowconfigure(1, weight=1)
        notebook.add(self.frame, text="Diagnostics")

        # Mousewheel scroll state (avoid unbind_all, which breaks other tabs)
        self._wheel_active: bool = False
        self._build_ui()

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 4}

        # ── Scrollable top ───────────────────────────────────────────
        self._canvas = tk.Canvas(self.frame, highlightthickness=0)
        sb = ttk.Scrollbar(self.frame, orient="vertical", command=self._canvas.yview)
        self._inner = ttk.Frame(self._canvas)
        self._inner.columnconfigure(0, weight=1)
        self._inner.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        cw = self._canvas.create_window((0, 0), window=self._inner, anchor="nw")
        self._canvas.configure(yscrollcommand=sb.set)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        def _resize(event):
            self._canvas.itemconfig(cw, width=event.width)

        self._canvas.bind("<Configure>", _resize)
        self._canvas.bind("<Enter>", lambda e: setattr(self, "_wheel_active", True))
        self._canvas.bind("<Leave>", lambda e: setattr(self, "_wheel_active", False))

        # Bind once globally; handler is gated by _wheel_active
        self.root.bind_all("<MouseWheel>", self._on_mousewheel, add="+")

        # ── Log panel ────────────────────────────────────────────────
        self.log_panel = LogPanel(self.frame, height=8)
        self.log_panel.grid(
            row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 6)
        )

        # ── Instantiate section widgets ──────────────────────────────
        section_kwargs = dict(
            log_panel=self.log_panel, state=self.state, root=self.root
        )

        sections = [
            FontDiagnosticsSection(self._inner, **section_kwargs),
            BenchmarkSection(self._inner, **section_kwargs),
            MLCalibrationSection(self._inner, **section_kwargs),
            MLRuntimeSummarySection(self._inner, **section_kwargs),
            TrainingProgressSection(self._inner, **section_kwargs),
            ModelComparisonSection(self._inner, **section_kwargs),
            LayoutModelSection(self._inner, **section_kwargs),
            TextEmbeddingsSection(self._inner, **section_kwargs),
            LLMSemanticChecksSection(self._inner, **section_kwargs),
            CrossPageGNNSection(self._inner, **section_kwargs),
        ]

        for row, section in enumerate(sections):
            section.grid(row=row, column=0, sticky="ew", **pad)

    def _on_mousewheel(self, event) -> None:
        if not self._wheel_active or not self._canvas.winfo_ismapped():
            return
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
