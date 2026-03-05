"""Tab 1 – Pipeline: PDF input, OCR stages, config, run button, embedded console.

Preserves all original GUI functionality (PDF selection, page range, OCR
toggles, tag management) and adds:
- Collapsible advanced-config section for Grouping & Geometry
- Config file Load / Save buttons (YAML / TOML)
- Embedded log console with stage-progress bar (replaces PowerShell windows)
- Cancel button for in-progress runs

Note: TOCR / VOCRPP / VOCR / Reconcile advanced knobs are intentionally
omitted from the GUI – they will be managed by the LLM layer in a future
release.
"""

from __future__ import annotations

import json
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from plancheck.config import GroupingConfig

from .widgets import ErrorPanel, LogPanel, StageProgressBar
from .worker import PipelineWorker

# ---------------------------------------------------------------------------
# Note: All advanced field lists (TOCR / VOCRPP / VOCR / Reconcile / Geometry)
# have been removed from the GUI – those knobs will be managed by the LLM
# layer in a future release.
# ---------------------------------------------------------------------------


class PipelineTab:
    """Tab 1: Pipeline configuration and execution."""

    def __init__(self, notebook: ttk.Notebook, gui_state: Any) -> None:
        self.notebook = notebook
        self.state = gui_state
        self.root = notebook.winfo_toplevel()

        self.frame = ttk.Frame(notebook)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=3)  # top scrollable area
        self.frame.rowconfigure(1, weight=0)  # stage bar
        self.frame.rowconfigure(2, weight=1)  # log panel
        notebook.add(self.frame, text="Pipeline")

        # PDF state
        self.pdf_files: list[Path] = []

        # Worker
        self._worker: PipelineWorker | None = None

        # Mousewheel scroll state (avoid unbind_all, which breaks other tabs)
        self._wheel_active: bool = False

        self._build_ui()

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 4}

        # ── Scrollable top area ──────────────────────────────────────
        self._canvas = tk.Canvas(self.frame, highlightthickness=0)
        self._scrollbar = ttk.Scrollbar(
            self.frame, orient="vertical", command=self._canvas.yview
        )
        self._inner = ttk.Frame(self._canvas)
        self._inner.columnconfigure(0, weight=1)
        self._inner.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        self._canvas_window = self._canvas.create_window(
            (0, 0), window=self._inner, anchor="nw"
        )
        self._canvas.configure(yscrollcommand=self._scrollbar.set)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._scrollbar.grid(row=0, column=1, sticky="ns")

        def _on_canvas_configure(event):
            self._canvas.itemconfig(self._canvas_window, width=event.width)

        self._canvas.bind("<Configure>", _on_canvas_configure)
        self._canvas.bind("<Enter>", lambda e: setattr(self, "_wheel_active", True))
        self._canvas.bind("<Leave>", lambda e: setattr(self, "_wheel_active", False))

        # Bind once globally; handler is gated by _wheel_active
        self.root.bind_all("<MouseWheel>", self._on_mousewheel, add="+")

        row = 0

        # ── PDF File Selection ───────────────────────────────────────
        file_frame = ttk.LabelFrame(self._inner, text="PDF File", padding=10)
        file_frame.grid(row=row, column=0, sticky="ew", **pad)
        file_frame.columnconfigure(1, weight=1)

        ttk.Button(file_frame, text="Select File...", command=self._select_file).grid(
            row=0, column=0, padx=(0, 5)
        )
        self.file_label_var = tk.StringVar(value="No file selected")
        ttk.Label(file_frame, textvariable=self.file_label_var, foreground="gray").grid(
            row=0, column=1, sticky="w"
        )
        ttk.Button(file_frame, text="Clear", command=self._clear_file).grid(
            row=0, column=2, padx=(5, 0)
        )
        row += 1

        # ── Page Selection ───────────────────────────────────────────
        page_frame = ttk.LabelFrame(self._inner, text="Page Selection", padding=10)
        page_frame.grid(row=row, column=0, sticky="ew", **pad)
        page_frame.columnconfigure(2, weight=1)

        self.page_mode_var = tk.StringVar(value="all")

        ttk.Radiobutton(
            page_frame,
            text="All Pages",
            variable=self.page_mode_var,
            value="all",
            command=self._update_page_mode,
        ).grid(row=0, column=0, sticky="w", pady=2, columnspan=3)

        ttk.Radiobutton(
            page_frame,
            text="Single Page:",
            variable=self.page_mode_var,
            value="single",
            command=self._update_page_mode,
        ).grid(row=1, column=0, sticky="w", pady=2)
        self.single_page_var = tk.StringVar(value="1")
        self.single_page_entry = ttk.Entry(
            page_frame, textvariable=self.single_page_var, width=10
        )
        self.single_page_entry.grid(row=1, column=1, sticky="w", pady=2)
        self.single_page_hint = ttk.Label(page_frame, text="(1 = first page)")
        self.single_page_hint.grid(row=1, column=2, sticky="w", padx=(5, 0))

        ttk.Radiobutton(
            page_frame,
            text="Page Range:",
            variable=self.page_mode_var,
            value="range",
            command=self._update_page_mode,
        ).grid(row=2, column=0, sticky="w", pady=2)

        range_inner = ttk.Frame(page_frame)
        range_inner.grid(row=2, column=1, columnspan=2, sticky="w", pady=2)
        self.start_page_var = tk.StringVar(value="1")
        self.start_entry = ttk.Entry(
            range_inner, textvariable=self.start_page_var, width=6
        )
        self.start_entry.grid(row=0, column=0)
        ttk.Label(range_inner, text=" to ").grid(row=0, column=1)
        self.end_page_var = tk.StringVar(value="")
        self.end_entry = ttk.Entry(range_inner, textvariable=self.end_page_var, width=6)
        self.end_entry.grid(row=0, column=2)
        ttk.Label(range_inner, text="  (blank end = last page)").grid(row=0, column=3)

        self._update_page_mode()
        row += 1

        # ── OCR Stage Toggles ───────────────────────────────────────
        stages_frame = ttk.LabelFrame(self._inner, text="Pipeline Stages", padding=10)
        stages_frame.grid(row=row, column=0, sticky="ew", **pad)
        stages_frame.columnconfigure(1, weight=1)

        # TOCR
        self.tocr_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            stages_frame,
            text="TOCR (pdfplumber text extraction)",
            variable=self.tocr_var,
        ).grid(row=0, column=0, sticky="w", pady=2)
        ttk.Label(
            stages_frame,
            text="Extract word boxes from the PDF text layer",
            foreground="gray",
        ).grid(row=0, column=1, sticky="w", padx=(10, 0))

        # VOCRPP
        self.ocr_preprocess_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            stages_frame,
            text="VOCRPP (Image Preprocessing)",
            variable=self.ocr_preprocess_var,
        ).grid(row=1, column=0, sticky="w", pady=2)
        ttk.Label(
            stages_frame,
            text="Grayscale, contrast, denoising for better OCR",
            foreground="gray",
        ).grid(row=1, column=1, sticky="w", padx=(10, 0))

        # VOCR
        self.vocr_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            stages_frame,
            text="VOCR (PaddleOCR extraction)",
            variable=self.vocr_var,
        ).grid(row=2, column=0, sticky="w", pady=2)
        ttk.Label(
            stages_frame,
            text="Full-page PaddleOCR visual token extraction",
            foreground="gray",
        ).grid(row=2, column=1, sticky="w", padx=(10, 0))

        # Reconcile
        self.ocr_reconcile_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            stages_frame,
            text="Reconcile (Symbol injection)",
            variable=self.ocr_reconcile_var,
        ).grid(row=3, column=0, sticky="w", pady=2)
        ttk.Label(
            stages_frame,
            text="Inject missing %, /, °, ± from VOCR into text layer",
            foreground="gray",
        ).grid(row=3, column=1, sticky="w", padx=(10, 0))

        # OCR DPI
        dpi_row = ttk.Frame(stages_frame)
        dpi_row.grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 2))
        ttk.Label(dpi_row, text="OCR/Preprocess DPI:").pack(side="left")
        self.ocr_dpi_var = tk.StringVar(value="300")
        ttk.Spinbox(
            dpi_row,
            textvariable=self.ocr_dpi_var,
            values=(120, 150, 180, 200, 220, 300, 400),
            width=8,
            state="readonly",
        ).pack(side="left", padx=(8, 0))
        ttk.Label(dpi_row, text="Render DPI:", foreground="gray").pack(
            side="left", padx=(20, 0)
        )
        self.resolution_var = tk.StringVar(value="200")
        ttk.Spinbox(
            dpi_row,
            textvariable=self.resolution_var,
            values=(72, 150, 200, 300),
            width=8,
            state="readonly",
        ).pack(side="left", padx=(8, 0))

        row += 1

        # ── Run Button ───────────────────────────────────────────────
        btn_frame = ttk.Frame(self._inner)
        btn_frame.grid(row=row, column=0, sticky="ew", **pad)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=0)

        run_style = ttk.Style()
        run_style.configure(
            "Run.TButton", font=("TkDefaultFont", 12, "bold"), padding=(20, 10)
        )

        self.run_button = ttk.Button(
            btn_frame,
            text="Run Processing",
            command=self._run_processing,
            style="Run.TButton",
        )
        self.run_button.grid(row=0, column=0, sticky="ew", pady=(0, 4), padx=(0, 8))

        self.cancel_button = ttk.Button(
            btn_frame,
            text="ABORT",
            command=self._cancel_processing,
            state="disabled",
        )
        self.cancel_button.grid(row=0, column=1, sticky="e", pady=(0, 4))

        row += 1

        # ── Error Panel (hidden by default) ──────────────────────────
        self.error_panel = ErrorPanel(self._inner)
        self.error_panel.grid(row=row, column=0, sticky="ew", **pad)

        row += 1

        # ── Stage Progress Bar ───────────────────────────────────────
        self.stage_bar = StageProgressBar(self.frame)
        self.stage_bar.grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(4, 2)
        )

        # ── Embedded Log Console ─────────────────────────────────────
        self.log_panel = LogPanel(self.frame, height=10)
        self.log_panel.grid(
            row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 6)
        )

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _collect_config(self) -> GroupingConfig:
        """Build a GroupingConfig from all current UI knobs + toggles."""
        cfg = GroupingConfig()
        # Master toggles
        cfg.enable_tocr = self.tocr_var.get()
        cfg.enable_vocr = self.vocr_var.get()
        cfg.enable_ocr_reconcile = self.ocr_reconcile_var.get()
        cfg.enable_ocr_preprocess = self.ocr_preprocess_var.get()

        try:
            cfg.ocr_reconcile_resolution = int(self.ocr_dpi_var.get())
        except ValueError:
            pass

        return cfg

    def _apply_config(self, cfg: GroupingConfig) -> None:
        """Push a GroupingConfig into all UI controls."""
        self.tocr_var.set(cfg.enable_tocr)
        self.vocr_var.set(cfg.enable_vocr)
        self.ocr_reconcile_var.set(cfg.enable_ocr_reconcile)
        self.ocr_preprocess_var.set(cfg.enable_ocr_preprocess)
        self.ocr_dpi_var.set(str(cfg.ocr_reconcile_resolution))

    # ------------------------------------------------------------------
    # PDF / Page selection (preserved from original)
    # ------------------------------------------------------------------

    def _select_file(self) -> None:
        f = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")],
            initialdir=str(Path("input")),
        )
        if f:
            path = Path(f)
            self.pdf_files = [path]
            self.file_label_var.set(path.name)
            # Update shared state
            self.state.pdf_path = path
            self.state.notify("pdf_changed")

    def _clear_file(self) -> None:
        self.pdf_files.clear()
        self.file_label_var.set("No file selected")
        self.state.pdf_path = None
        self.state.notify("pdf_changed")

    def _update_page_mode(self) -> None:
        mode = self.page_mode_var.get()
        self.single_page_entry.config(
            state="normal" if mode == "single" else "disabled"
        )
        self.start_entry.config(state="normal" if mode == "range" else "disabled")
        self.end_entry.config(state="normal" if mode == "range" else "disabled")

    def _parse_page_range(self) -> tuple[int, int | None]:
        mode = self.page_mode_var.get()
        if mode == "all":
            return 0, None
        if mode == "single":
            page_str = self.single_page_var.get().strip()
            if not page_str:
                raise ValueError("Please enter a page number")
            page = int(page_str)
            if page < 1:
                raise ValueError("Page number must be 1 or greater")
            return page - 1, page
        start_str = self.start_page_var.get().strip()
        end_str = self.end_page_var.get().strip()
        start = int(start_str) - 1 if start_str else 0
        if start < 0:
            start = 0
        end = int(end_str) if end_str else None
        return start, end

    def _on_mousewheel(self, event) -> None:
        if not self._wheel_active or not self._canvas.winfo_ismapped():
            return
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ------------------------------------------------------------------
    # Pipeline Execution (embedded – replaces subprocess)
    # ------------------------------------------------------------------

    def _run_processing(self) -> None:
        if not self.pdf_files:
            messagebox.showwarning("No Files", "Please add at least one PDF file.")
            return
        try:
            start, end = self._parse_page_range()
        except ValueError as exc:
            messagebox.showerror("Invalid Input", str(exc) or "Invalid page range")
            return
        try:
            resolution = int(self.resolution_var.get())
        except ValueError:
            resolution = 200

        cfg = self._collect_config()

        runs_root = Path("runs")

        self.log_panel.clear()
        self.error_panel.clear()
        self.stage_bar.reset()
        self.run_button.config(state="disabled")
        self.cancel_button.config(state="normal")

        self._worker = PipelineWorker(
            self.root, self.log_panel, self.stage_bar, self.error_panel
        )
        worker = self._worker

        def target():
            from ..runners.run_pdf_batch import cleanup_old_runs, run_pdf

            results = []
            for pdf_path in self.pdf_files:
                if worker and worker.cancel_event.is_set():
                    break
                run_prefix = pdf_path.stem.replace(" ", "_")[:20]
                run_dir = run_pdf(
                    pdf=pdf_path,
                    start=start,
                    end=end,
                    resolution=resolution,
                    run_root=runs_root,
                    run_prefix=run_prefix,
                    cfg=cfg,
                    cancel_event=worker.cancel_event if worker else None,
                    stage_callback=worker.post_stage if worker else None,
                )
                results.append(run_dir)
            cleanup_old_runs(runs_root, keep=50)
            return results

        def on_done(result, error, elapsed):
            self.run_button.config(state="normal")
            self.cancel_button.config(state="disabled")
            if error:
                self.error_panel.add_error(str(error), "ERROR")
            if result and not error:
                self.state.last_run_dir = (
                    result[-1] if isinstance(result, list) else result
                )
                self.state.notify("run_completed")

        self._worker.run(target, on_done=on_done)

    def _cancel_processing(self) -> None:
        if self._worker:
            self._worker.cancel()
