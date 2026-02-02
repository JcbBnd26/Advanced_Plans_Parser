"""
Lightweight GUI for Advanced Plan Parser.

Select one or more PDF plan sets, set page range and resolution,
then run batch processing. Results go to the runs/ folder.
"""

import os
import sys
from pathlib import Path

# Add src to path for plancheck imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from run_pdf_batch import cleanup_old_runs, run_pdf


class PlanParserGUI:
    """Main GUI window for the Plan Parser."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Advanced Plans Parser")
        self.root.geometry("550x480")
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

        # --- File Selection Section ---
        file_frame = ttk.LabelFrame(self.root, text="PDF Files", padding=10)
        file_frame.grid(row=0, column=0, sticky="ew", **pad)
        file_frame.columnconfigure(0, weight=1)

        btn_frame = ttk.Frame(file_frame)
        btn_frame.grid(row=0, column=0, sticky="ew")
        btn_frame.columnconfigure(1, weight=1)

        ttk.Button(btn_frame, text="Add Files...", command=self._add_files).grid(
            row=0, column=0, padx=(0, 5)
        )
        ttk.Button(
            btn_frame, text="Remove Selected", command=self._remove_selected
        ).grid(row=0, column=1, sticky="w")
        ttk.Button(btn_frame, text="Clear All", command=self._clear_all).grid(
            row=0, column=2, padx=(5, 0)
        )

        # Listbox with scrollbar for selected files
        list_frame = ttk.Frame(file_frame)
        list_frame.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        file_frame.rowconfigure(1, weight=1)

        self.file_listbox = tk.Listbox(list_frame, height=6, selectmode=tk.EXTENDED)
        self.file_listbox.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            list_frame, orient="vertical", command=self.file_listbox.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.file_listbox.config(yscrollcommand=scrollbar.set)

        # --- Page Selection Section ---
        page_frame = ttk.LabelFrame(self.root, text="Page Selection", padding=10)
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

        # --- Settings Section ---
        settings_frame = ttk.LabelFrame(self.root, text="Settings", padding=10)
        settings_frame.grid(row=2, column=0, sticky="nsew", **pad)
        settings_frame.columnconfigure(1, weight=1)

        # Resolution
        ttk.Label(settings_frame, text="Resolution (DPI):").grid(
            row=0, column=0, sticky="w", pady=2
        )
        self.resolution_var = tk.StringVar(value="200")
        self.resolution_spinbox = ttk.Spinbox(
            settings_frame,
            textvariable=self.resolution_var,
            values=(72, 150, 200, 300),
            width=8,
            state="readonly",
        )
        self.resolution_spinbox.grid(row=0, column=1, sticky="w", pady=2)

        # --- Status Section ---
        status_frame = ttk.Frame(self.root, padding=5)
        status_frame.grid(row=3, column=0, sticky="ew", **pad)
        status_frame.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(
            status_frame, textvariable=self.status_var, anchor="w"
        )
        self.status_label.grid(row=0, column=0, sticky="ew")

        # --- Run Button ---
        btn_run_frame = ttk.Frame(self.root)
        btn_run_frame.grid(row=4, column=0, sticky="ew", **pad)
        btn_run_frame.columnconfigure(0, weight=1)

        self.run_button = ttk.Button(
            btn_run_frame, text="Run Processing", command=self._run_processing
        )
        self.run_button.grid(row=0, column=0, pady=(0, 10))

    def _add_files(self) -> None:
        """Open file dialog to add PDF files."""
        files = filedialog.askopenfilenames(
            title="Select PDF Files",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")],
            initialdir=Path(__file__).parent.parent / "input",
        )
        for f in files:
            path = Path(f)
            if path not in self.pdf_files:
                self.pdf_files.append(path)
                self.file_listbox.insert(tk.END, path.name)

    def _remove_selected(self) -> None:
        """Remove selected files from the list."""
        selected = list(self.file_listbox.curselection())
        # Remove in reverse order to preserve indices
        for idx in reversed(selected):
            self.file_listbox.delete(idx)
            del self.pdf_files[idx]

    def _clear_all(self) -> None:
        """Clear all files from the list."""
        self.file_listbox.delete(0, tk.END)
        self.pdf_files.clear()

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
        self.file_listbox.config(state=state)
        self.resolution_spinbox.config(state="readonly" if enabled else "disabled")

        if enabled:
            # Restore proper state based on page mode
            self._update_page_mode()
        else:
            # Disable all page entry fields
            self.single_page_entry.config(state="disabled")
            self.start_entry.config(state="disabled")
            self.end_entry.config(state="disabled")

    def _run_processing(self) -> None:
        """Run batch processing on all selected PDFs."""
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

        # Ensure runs directory exists
        self.runs_root.mkdir(parents=True, exist_ok=True)

        # Disable UI during processing
        self._set_ui_enabled(False)
        self.root.update()

        processed_runs = []
        total = len(self.pdf_files)

        try:
            for i, pdf_path in enumerate(self.pdf_files, start=1):
                self.status_var.set(f"Processing {i}/{total}: {pdf_path.name}...")
                self.root.update()

                run_prefix = pdf_path.stem.replace(" ", "_")[:20]
                print(
                    f"DEBUG: pdf={pdf_path}, start={start}, end={end}, resolution={resolution}",
                    flush=True,
                )
                run_dir = run_pdf(
                    pdf=pdf_path,
                    start=start,
                    end=end,
                    resolution=resolution,
                    run_root=self.runs_root,
                    run_prefix=run_prefix,
                )
                processed_runs.append(run_dir)

            # Cleanup old runs
            cleanup_old_runs(self.runs_root, keep=50)

            self.status_var.set(f"Complete! Processed {total} file(s).")
            messagebox.showinfo(
                "Processing Complete",
                f"Processed {total} PDF file(s).\n\nOutput saved to:\n{self.runs_root}",
            )

            # Auto-open the runs folder
            os.startfile(self.runs_root)

        except Exception as e:
            self.status_var.set(f"Error: {e}")
            messagebox.showerror("Processing Error", str(e))

        finally:
            self._set_ui_enabled(True)


def main() -> None:
    """Launch the GUI application."""
    root = tk.Tk()
    app = PlanParserGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
