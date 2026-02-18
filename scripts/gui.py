"""
Lightweight GUI for Advanced Plan Parser.

Select one or more PDF plan sets, set page range and resolution,
then run batch processing. Results go to the runs/ folder.
"""

import sys
from pathlib import Path

# Add src to path for plancheck imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from overlay_viewer import OverlayViewerTab
from run_pdf_batch import cleanup_old_runs, run_pdf
from tag_list import TAG_DESCRIPTIONS, TAG_LIST


class PlanParserGUI:
    """Main GUI window for the Plan Parser."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Advanced Plans Parser")
        self.root.geometry("650x760")
        self.root.minsize(650, 760)
        self.root.resizable(True, True)

        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)  # File list expands

        self.pdf_files: list[Path] = []
        self.runs_root = Path(__file__).parent.parent / "runs"

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the main UI components."""
        # Padding for all widgets
        pad = {"padx": 10, "pady": 5}

        # --- Tab Container ---
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.tab1_frame = ttk.Frame(self.notebook)
        self.tab1_frame.columnconfigure(0, weight=1)
        self.tab1_frame.rowconfigure(0, weight=1)
        self.notebook.add(self.tab1_frame, text="Tab 1")

        # Scrollable canvas inside Tab 1
        self._tab1_canvas = tk.Canvas(self.tab1_frame, highlightthickness=0)
        self._tab1_scrollbar = ttk.Scrollbar(
            self.tab1_frame, orient="vertical", command=self._tab1_canvas.yview
        )
        self._tab1_inner = ttk.Frame(self._tab1_canvas)
        self._tab1_inner.columnconfigure(0, weight=1)

        self._tab1_inner.bind(
            "<Configure>",
            lambda e: self._tab1_canvas.configure(
                scrollregion=self._tab1_canvas.bbox("all")
            ),
        )
        self._tab1_canvas_window = self._tab1_canvas.create_window(
            (0, 0), window=self._tab1_inner, anchor="nw"
        )
        self._tab1_canvas.configure(yscrollcommand=self._tab1_scrollbar.set)

        self._tab1_canvas.grid(row=0, column=0, sticky="nsew")
        self._tab1_scrollbar.grid(row=0, column=1, sticky="ns")

        # Resize inner frame width to match canvas
        def _on_tab1_canvas_configure(event):
            self._tab1_canvas.itemconfig(self._tab1_canvas_window, width=event.width)

        self._tab1_canvas.bind("<Configure>", _on_tab1_canvas_configure)

        # Mousewheel scrolling
        def _on_mousewheel(event):
            self._tab1_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self._tab1_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self.tab2_frame = ttk.Frame(self.notebook)
        self.tab2_frame.columnconfigure(0, weight=1)
        self.notebook.add(self.tab2_frame, text="Tab 2")

        # Tab 3: Visual Debug overlay viewer
        self._overlay_tab = OverlayViewerTab(self.notebook)

        # --- Tab 2: Standalone Diagnostics ---
        diag_frame = ttk.LabelFrame(
            self.tab2_frame, text="Font Diagnostics", padding=10
        )
        diag_frame.grid(row=0, column=0, sticky="ew", **pad)
        diag_frame.columnconfigure(0, weight=1)

        self.font_metrics_var = tk.BooleanVar(value=False)
        self.visual_metrics_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(
            diag_frame,
            text="FontMetricsAnalyzer (Heuristic font width analysis)",
            variable=self.font_metrics_var,
            command=self._toggle_diagnostics_button,
        ).grid(row=0, column=0, sticky="w", pady=2)

        ttk.Checkbutton(
            diag_frame,
            text="VisualMetricsAnalyzer (Pixel-accurate font width analysis)",
            variable=self.visual_metrics_var,
            command=self._toggle_diagnostics_button,
        ).grid(row=1, column=0, sticky="w", pady=2)

        self.run_diagnostics_button = ttk.Button(
            diag_frame,
            text="Run Selected Diagnostics...",
            command=self._run_font_diagnostics,
            state="disabled",
        )
        self.run_diagnostics_button.grid(row=2, column=0, sticky="w", pady=(8, 2))

        ttk.Label(
            diag_frame,
            text="Runs as standalone tests and writes JSON reports.",
            foreground="gray",
        ).grid(row=3, column=0, sticky="w")

        # --- File Selection Section ---
        file_frame = ttk.LabelFrame(self._tab1_inner, text="PDF File", padding=10)
        file_frame.grid(row=0, column=0, sticky="ew", **pad)
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

        # --- Page Selection Section ---
        page_frame = ttk.LabelFrame(self._tab1_inner, text="Page Selection", padding=10)
        page_frame.grid(row=1, column=0, sticky="nsew", **pad)
        page_frame.columnconfigure(2, weight=1)

        # Radio button variable: "all", "single", or "range"
        self.page_mode_var = tk.StringVar(value="all")

        # All Pages option
        ttk.Radiobutton(
            page_frame,
            text="All Pages",
            variable=self.page_mode_var,
            value="all",
            command=self._update_page_mode,
        ).grid(row=0, column=0, sticky="w", pady=2, columnspan=3)

        # Single Page option
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

        # Page Range option
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
        self.start_entry.grid(row=0, column=0, sticky="w")
        self.range_to_label = ttk.Label(range_inner, text=" to ")
        self.range_to_label.grid(row=0, column=1)
        self.end_page_var = tk.StringVar(value="")
        self.end_entry = ttk.Entry(range_inner, textvariable=self.end_page_var, width=6)
        self.end_entry.grid(row=0, column=2, sticky="w")
        self.range_hint = ttk.Label(range_inner, text="  (blank end = last page)")
        self.range_hint.grid(row=0, column=3, sticky="w")

        # Initialize entry states
        self._update_page_mode()

        # --- Settings Sections ---
        settings_container = ttk.Frame(self._tab1_inner)
        settings_container.grid(row=2, column=0, sticky="nsew", **pad)
        settings_container.columnconfigure(0, weight=1)

        optical_ocr_frame = ttk.LabelFrame(
            settings_container, text="Optical OCR", padding=10
        )
        optical_ocr_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        optical_ocr_frame.columnconfigure(1, weight=1)
        optical_ocr_frame.columnconfigure(2, weight=1)

        text_ocr_frame = ttk.LabelFrame(settings_container, text="Text OCR", padding=10)
        text_ocr_frame.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        text_ocr_frame.columnconfigure(0, weight=1)

        reconcile_frame = ttk.LabelFrame(
            settings_container, text="Post Processing", padding=10
        )
        reconcile_frame.grid(row=2, column=0, sticky="ew")
        reconcile_frame.columnconfigure(0, weight=1)

        # Optical OCR settings
        self.ocr_preprocess_var = tk.BooleanVar(value=True)
        self.ocr_preprocess_check = ttk.Checkbutton(
            optical_ocr_frame,
            text="VOCRPP (Preprocessing)",
            variable=self.ocr_preprocess_var,
            command=self._toggle_preprocess_button,
        )
        self.ocr_preprocess_check.grid(row=0, column=0, sticky="w", pady=2)

        ttk.Label(
            optical_ocr_frame,
            text="(Grayscale, contrast, denoising for better OCR)",
            foreground="gray",
        ).grid(row=0, column=1, sticky="w", padx=(10, 0))

        # Run Preprocessing Only button (enabled when VOCRPP checkbox is ticked)
        self.preprocess_only_button = ttk.Button(
            optical_ocr_frame,
            text="Run Preprocessing Only...",
            command=self._run_preprocess_only,
            state="normal",
        )
        self.preprocess_only_button.grid(row=0, column=2, sticky="e", pady=2)

        # VOCR checkbox (PaddleOCR full-page extraction)
        self.vocr_var = tk.BooleanVar(value=True)
        self.vocr_check = ttk.Checkbutton(
            optical_ocr_frame,
            text="VOCR (PaddleOCR extraction)",
            variable=self.vocr_var,
        )
        self.vocr_check.grid(row=1, column=0, sticky="w", pady=2)

        ttk.Label(
            optical_ocr_frame,
            text="(Full-page PaddleOCR visual token extraction)",
            foreground="gray",
        ).grid(row=1, column=1, sticky="w", padx=(10, 0))

        # OCR/Preprocess DPI selection (used by VOCR + preprocessing stages)
        ttk.Label(optical_ocr_frame, text="OCR/Preprocess DPI:").grid(
            row=2, column=0, sticky="w", pady=(6, 2)
        )
        self.ocr_dpi_var = tk.StringVar(value="300")
        self.ocr_dpi_spinbox = ttk.Spinbox(
            optical_ocr_frame,
            textvariable=self.ocr_dpi_var,
            values=(120, 150, 180, 200, 220, 300, 400),
            width=8,
            state="readonly",
        )
        self.ocr_dpi_spinbox.grid(
            row=2, column=1, sticky="w", padx=(10, 0), pady=(6, 2)
        )

        ttk.Label(
            optical_ocr_frame,
            text="(Applies to Run Processing OCR and Run Preprocessing Only)",
            foreground="gray",
        ).grid(row=2, column=2, sticky="e", pady=(6, 2))

        # Text OCR (TOCR) toggle — on by default
        self.tocr_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            text_ocr_frame,
            text="TOCR (pdfplumber text extraction)",
            variable=self.tocr_var,
        ).grid(row=0, column=0, sticky="w", pady=2)

        ttk.Label(
            text_ocr_frame,
            text="(Extract word boxes from the PDF text layer)",
            foreground="gray",
        ).grid(row=0, column=1, sticky="w", padx=(10, 0))

        # Reconcile checkbox (merge VOCR tokens into TOCR)
        self.ocr_reconcile_var = tk.BooleanVar(value=True)
        self.ocr_reconcile_check = ttk.Checkbutton(
            reconcile_frame,
            text="Reconcile",
            variable=self.ocr_reconcile_var,
        )
        self.ocr_reconcile_check.grid(row=0, column=0, sticky="w", pady=2)

        ttk.Label(
            reconcile_frame,
            text="(Inject missing %, /, °, ± from VOCR into text layer)",
            foreground="gray",
        ).grid(row=0, column=1, sticky="w", padx=(10, 0))

        # --- Run Button ---
        btn_run_frame = ttk.Frame(self._tab1_inner)
        btn_run_frame.grid(row=5, column=0, sticky="ew", **pad)
        btn_run_frame.columnconfigure(0, weight=1)

        run_style = ttk.Style()
        run_style.configure(
            "Run.TButton", font=("TkDefaultFont", 12, "bold"), padding=(20, 10)
        )

        self.run_button = ttk.Button(
            btn_run_frame,
            text="Run Processing",
            command=self._run_processing,
            style="Run.TButton",
        )
        self.run_button.grid(row=0, column=0, sticky="ew", pady=(0, 10), padx=20)

        # --- Tag Selection & Color Debugger Section ---
        tag_frame = ttk.LabelFrame(
            self._tab1_inner, text="Visual Debug Overlays", padding=10
        )
        tag_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=5)
        tag_frame.columnconfigure(1, weight=1)

        # Tag dropdown
        self.tag_var = tk.StringVar(value=TAG_LIST[0])
        tag_dropdown = ttk.Combobox(
            tag_frame,
            textvariable=self.tag_var,
            values=TAG_LIST,
            state="readonly",
            width=24,
        )
        tag_dropdown.grid(row=0, column=0, sticky="w")
        ttk.Button(tag_frame, text="Add Tag", command=self._add_tag_to_list).grid(
            row=0, column=1, sticky="w", padx=(5, 0)
        )

        # Overlay resolution (DPI) — same row as tag dropdown
        ttk.Label(tag_frame, text="Overlay DPI:").grid(
            row=0, column=2, sticky="e", padx=(20, 0)
        )
        self.resolution_var = tk.StringVar(value="200")
        self.resolution_spinbox = ttk.Spinbox(
            tag_frame,
            textvariable=self.resolution_var,
            values=(72, 150, 200, 300),
            width=8,
            state="readonly",
        )
        self.resolution_spinbox.grid(row=0, column=3, sticky="w", padx=(5, 0))

        # Scrollable frame for tag checkboxes
        tag_canvas = tk.Canvas(tag_frame, height=60, highlightthickness=0)
        tag_scrollbar = ttk.Scrollbar(
            tag_frame, orient="vertical", command=tag_canvas.yview
        )
        self.tag_inner_frame = ttk.Frame(tag_canvas)
        self.tag_inner_frame.bind(
            "<Configure>",
            lambda e: tag_canvas.configure(scrollregion=tag_canvas.bbox("all")),
        )
        tag_canvas.create_window((0, 0), window=self.tag_inner_frame, anchor="nw")
        tag_canvas.configure(yscrollcommand=tag_scrollbar.set)
        tag_canvas.grid(row=1, column=0, columnspan=4, sticky="nsew", pady=(5, 0))
        tag_scrollbar.grid(row=1, column=4, sticky="ns", pady=(5, 0))

        # Tag data: {tag: {"color": hex, "ignored": bool, "selected": BooleanVar, "widgets": {...}}}
        self.tag_data = {}

        # Mass action buttons
        action_frame = ttk.Frame(tag_frame)
        action_frame.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(5, 0))
        ttk.Button(
            action_frame, text="Set Color", command=self._set_color_selected
        ).grid(row=0, column=0, padx=2)
        ttk.Button(action_frame, text="Remove", command=self._clear_selected_tags).grid(
            row=0, column=1, padx=2
        )
        ttk.Button(
            action_frame, text="Ignore", command=self._ignore_selected_tags
        ).grid(row=0, column=2, padx=2)
        ttk.Button(
            action_frame, text="Unignore", command=self._unignore_selected_tags
        ).grid(row=0, column=3, padx=2)
        ttk.Button(action_frame, text="Select All", command=self._select_all_tags).grid(
            row=0, column=4, padx=2
        )
        ttk.Button(
            action_frame, text="Deselect All", command=self._deselect_all_tags
        ).grid(row=0, column=5, padx=2)

        # Tooltip state
        self._tag_tooltip = None
        self._tooltip_after_id = None
        self._tooltip_tag = None

    def _select_file(self) -> None:
        """Open file dialog to select a single PDF file."""
        f = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")],
            initialdir=Path(__file__).parent.parent / "input",
        )
        if f:
            path = Path(f)
            self.pdf_files = [path]
            self.file_label_var.set(path.name)

    def _clear_file(self) -> None:
        """Clear the selected file."""
        self.pdf_files.clear()
        self.file_label_var.set("No file selected")

    def _update_page_mode(self) -> None:
        """Enable/disable page entry fields based on selected mode."""
        mode = self.page_mode_var.get()

        # Single page entries
        single_state = "normal" if mode == "single" else "disabled"
        self.single_page_entry.config(state=single_state)

        # Range entries
        range_state = "normal" if mode == "range" else "disabled"
        self.start_entry.config(state=range_state)
        self.end_entry.config(state=range_state)

    def _parse_page_range(self) -> tuple[int, int | None]:
        """Parse page values based on selected mode. Returns 0-indexed values."""
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
            # Convert to 0-indexed: page 1 -> start=0, end=1
            return page - 1, page

        # mode == "range"
        start_str = self.start_page_var.get().strip()
        end_str = self.end_page_var.get().strip()

        start = int(start_str) - 1 if start_str else 0
        if start < 0:
            start = 0

        # End is inclusive for user, exclusive internally
        end = int(end_str) if end_str else None

        return start, end

    def _set_ui_enabled(self, enabled: bool) -> None:
        """Enable or disable UI elements during processing."""
        state = "normal" if enabled else "disabled"
        self.run_button.config(state=state)
        self.resolution_spinbox.config(state="readonly" if enabled else "disabled")
        self.ocr_dpi_spinbox.config(state="readonly" if enabled else "disabled")

        if enabled:
            # Restore proper state based on page mode
            self._update_page_mode()
        else:
            # Disable all page entry fields
            self.single_page_entry.config(state="disabled")
            self.start_entry.config(state="disabled")
            self.end_entry.config(state="disabled")

    def _run_processing(self) -> None:
        """Launch a new PowerShell window to run batch processing with selected options."""
        if not self.pdf_files:
            messagebox.showwarning("No Files", "Please add at least one PDF file.")
            return
        try:
            start, end = self._parse_page_range()
        except ValueError:
            messagebox.showerror("Invalid Input", "Page numbers must be integers.")
            return
        try:
            resolution = int(self.resolution_var.get())
        except ValueError:
            resolution = 200
        try:
            ocr_dpi = int(self.ocr_dpi_var.get())
        except ValueError:
            ocr_dpi = 300
        mode = self.page_mode_var.get()
        pdf_args = " ".join(f'"{str(p)}"' for p in self.pdf_files)

        # Build color overrides from non-ignored tags and write to temp file
        import json
        import tempfile

        color_dict = {}
        for tag, data in self.tag_data.items():
            if not data["ignored"]:
                color_dict[tag] = data["color"]  # hex string like "#D3D3D3"

        args = [
            f"--pdfs {pdf_args}",
            f"--mode {mode}",
            f"--resolution {resolution}",
            f"--ocr-resolution {ocr_dpi}",
            f'--run-root "{str(self.runs_root)}"',
        ]
        # Add OCR flags based on checkboxes
        if not self.tocr_var.get():
            args.append("--no-tocr")
        if self.vocr_var.get():
            args.append("--vocr")
        if self.ocr_preprocess_var.get():
            args.append("--ocr-preprocess")
        if self.ocr_reconcile_var.get():
            args.append("--ocr-full-reconcile")
        if color_dict:
            # Write colors to temp file to avoid PowerShell escaping issues
            color_file = Path(tempfile.gettempdir()) / "planparser_colors.json"
            color_file.write_text(json.dumps(color_dict))
            args.append(f'--colors-file "{str(color_file)}"')
        if mode == "single":
            args.append(f"--single {start+1}")
        elif mode == "range":
            args.append(f"--start {start+1}")
            if end is not None:
                args.append(f"--end {end}")
        cmd = f'python scripts/run_from_args.py {" ".join(args)}'
        subprocess.Popen(
            ["powershell.exe", "-NoExit", "-Command", cmd],
            cwd=str(Path(__file__).parent.parent),
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
        # GUI stays open for more jobs

    def _toggle_preprocess_button(self) -> None:
        """Enable/disable the 'Run Preprocessing Only' button based on checkbox."""
        state = "normal" if self.ocr_preprocess_var.get() else "disabled"
        self.preprocess_only_button.config(state=state)

    def _toggle_diagnostics_button(self) -> None:
        """Enable diagnostics run button when at least one analyzer is selected."""
        enabled = self.font_metrics_var.get() or self.visual_metrics_var.get()
        self.run_diagnostics_button.config(state=("normal" if enabled else "disabled"))

    def _run_preprocess_only(self) -> None:
        """Run standalone OCR preprocessing and let the user pick an output folder."""
        if not self.pdf_files:
            messagebox.showwarning("No Files", "Please add at least one PDF file.")
            return
        try:
            start, end = self._parse_page_range()
        except ValueError:
            messagebox.showerror("Invalid Input", "Page numbers must be integers.")
            return
        try:
            resolution = int(self.resolution_var.get())
        except ValueError:
            resolution = 200
        try:
            ocr_dpi = int(self.ocr_dpi_var.get())
        except ValueError:
            ocr_dpi = 300

        # Ask user to pick output folder
        out_dir = filedialog.askdirectory(
            title="Select Output Folder for Processed PDF Files",
            initialdir=str(self.runs_root),
        )
        if not out_dir:
            return  # User cancelled

        mode = self.page_mode_var.get()
        for pdf_path in self.pdf_files:
            output_pdf = Path(out_dir) / f"{pdf_path.stem}_ocr_preprocessed.pdf"
            args = [
                f'"{str(pdf_path)}"',
                f"--render-dpi {ocr_dpi}",
                f"--start {start}",
                f'--run-root "{out_dir}"',
                "--pdf-only",
                f'--output-pdf "{str(output_pdf)}"',
            ]
            if mode == "single":
                args.append(f"--end {start + 1}")
            elif mode == "range" and end is not None:
                args.append(f"--end {end}")

            cmd = f'python scripts/run_ocr_preprocess.py {" ".join(args)}'
            subprocess.Popen(
                ["powershell.exe", "-NoExit", "-Command", cmd],
                cwd=str(Path(__file__).parent.parent),
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )

    def _run_font_diagnostics(self) -> None:
        """Run selected font diagnostics as standalone tests.

        Output goes into a timestamped run subfolder under runs/ by default.
        """
        if not self.pdf_files:
            messagebox.showwarning("No Files", "Please add at least one PDF file.")
            return

        if not (self.font_metrics_var.get() or self.visual_metrics_var.get()):
            messagebox.showinfo(
                "No Diagnostics Selected",
                "Enable at least one diagnostics tool in Tab 2.",
            )
            return

        try:
            start, end = self._parse_page_range()
        except ValueError:
            messagebox.showerror("Invalid Input", "Page numbers must be integers.")
            return

        for pdf_path in self.pdf_files:
            args = [
                f'"{str(pdf_path)}"',
                f'--run-root "{str(self.runs_root)}"',
                f"--start {start}",
            ]

            if end is not None:
                args.append(f"--end {end}")

            if self.font_metrics_var.get():
                args.append("--heuristic")
            if self.visual_metrics_var.get():
                args.append("--visual")

            cmd = f'python scripts/run_font_metrics_diagnostics.py {" ".join(args)}'
            subprocess.Popen(
                ["powershell.exe", "-NoExit", "-Command", cmd],
                cwd=str(Path(__file__).parent.parent),
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )

    def _add_tag_to_list(self):
        tag = self.tag_var.get()
        if tag in self.tag_data:
            return  # Already added
        # Create checkbox row
        row_frame = ttk.Frame(self.tag_inner_frame)
        row_frame.pack(fill="x", pady=1)

        selected_var = tk.BooleanVar(value=False)
        chk = ttk.Checkbutton(row_frame, variable=selected_var)
        chk.pack(side="left")

        color_btn = tk.Button(
            row_frame,
            text="  ",
            bg="#D3D3D3",
            width=2,
            command=lambda t=tag: self._pick_tag_color(t),
        )
        color_btn.pack(side="left", padx=(0, 5))

        label = ttk.Label(row_frame, text=tag)
        label.pack(side="left")

        status_label = ttk.Label(row_frame, text="", foreground="gray")
        status_label.pack(side="left", padx=(5, 0))

        self.tag_data[tag] = {
            "color": "#D3D3D3",
            "ignored": False,
            "selected": selected_var,
            "widgets": {
                "row": row_frame,
                "color_btn": color_btn,
                "label": label,
                "status": status_label,
            },
        }

        # Bind tooltip events
        label.bind("<Enter>", lambda e, t=tag: self._schedule_tooltip(t, e))
        label.bind("<Leave>", lambda e: self._cancel_tooltip())

    def _pick_tag_color(self, tag):
        from tkinter import colorchooser

        color = colorchooser.askcolor(
            title=f"Choose color for {tag}", initialcolor=self.tag_data[tag]["color"]
        )[1]
        if color:
            self.tag_data[tag]["color"] = color
            self.tag_data[tag]["widgets"]["color_btn"].configure(bg=color)

    def _set_color_selected(self):
        from tkinter import colorchooser

        selected_tags = [t for t, d in self.tag_data.items() if d["selected"].get()]
        if not selected_tags:
            messagebox.showinfo("No Selection", "Please check one or more tags first.")
            return
        color = colorchooser.askcolor(title="Choose color for selected tags")[1]
        if color:
            for tag in selected_tags:
                self.tag_data[tag]["color"] = color
                self.tag_data[tag]["widgets"]["color_btn"].configure(bg=color)

    def _clear_selected_tags(self):
        selected_tags = [t for t, d in self.tag_data.items() if d["selected"].get()]
        for tag in selected_tags:
            self.tag_data[tag]["widgets"]["row"].destroy()
            del self.tag_data[tag]

    def _ignore_selected_tags(self):
        for tag, data in self.tag_data.items():
            if data["selected"].get():
                data["ignored"] = True
                data["widgets"]["status"].configure(text="[IGNORED]")

    def _unignore_selected_tags(self):
        for tag, data in self.tag_data.items():
            if data["selected"].get():
                data["ignored"] = False
                data["widgets"]["status"].configure(text="")

    def _select_all_tags(self):
        for data in self.tag_data.values():
            data["selected"].set(True)

    def _deselect_all_tags(self):
        for data in self.tag_data.values():
            data["selected"].set(False)

    def _schedule_tooltip(self, tag, event):
        self._cancel_tooltip()
        self._tooltip_tag = tag
        self._tooltip_after_id = self.root.after(
            500, lambda: self._show_tag_tooltip(tag, event)
        )

    def _cancel_tooltip(self):
        if self._tooltip_after_id:
            self.root.after_cancel(self._tooltip_after_id)
            self._tooltip_after_id = None
        self._hide_tag_tooltip()
        self._tooltip_tag = None

    def _show_tag_tooltip(self, tag, event):
        desc = TAG_DESCRIPTIONS.get(tag, "No description available.")
        x = event.widget.winfo_rootx() + 20
        y = event.widget.winfo_rooty() + event.widget.winfo_height() + 5
        self._hide_tag_tooltip()
        self._tag_tooltip = tk.Toplevel(self.root)
        self._tag_tooltip.wm_overrideredirect(True)
        self._tag_tooltip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self._tag_tooltip,
            text=desc,
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            font=(None, 10),
        )
        label.pack(ipadx=4, ipady=2)

    def _hide_tag_tooltip(self):
        if self._tag_tooltip:
            self._tag_tooltip.destroy()
            self._tag_tooltip = None


def main() -> None:
    """Launch the GUI application."""
    root = tk.Tk()
    app = PlanParserGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
