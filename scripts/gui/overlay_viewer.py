"""Visual Debug tab – interactive overlay viewer (GUI only).

``OverlayViewerTab`` plugs into the tkinter Notebook and exposes PDF /
page / DPI pickers, layer toggles, zoom controls, and a "Load from Run…"
action.  All pure rendering logic lives in ``overlay_renderer.py``.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from PIL import Image

from plancheck import GroupingConfig

from .overlay_renderer import render_overlay
from ..utils.run_utils import latest_overlays_dir


def _import_tk():
    """Lazy-import tkinter so headless ``render_overlay`` works without it."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    return tk, ttk, filedialog, messagebox


class OverlayViewerTab:
    """Visual Debug tab that plugs into the existing tkinter Notebook."""

    def __init__(self, notebook, gui_state=None) -> None:
        tk, ttk, filedialog, messagebox = _import_tk()
        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.state = gui_state

        self.frame = ttk.Frame(notebook)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(2, weight=1)  # image canvas row expands
        notebook.add(self.frame, text="Visual Debug")

        self._pdf_path: Path | None = None
        self._last_img: Image.Image | None = None
        self._photo = None  # prevent GC
        self._render_thread: threading.Thread | None = None
        self._zoom: float = 1.0  # current zoom factor
        self._zoom_levels = [
            0.1,
            0.15,
            0.2,
            0.25,
            0.33,
            0.5,
            0.67,
            0.75,
            1.0,
            1.25,
            1.5,
            2.0,
            3.0,
            4.0,
        ]

        self._build_controls(notebook)
        self._build_image_canvas()
        self._build_status_bar()

    # ── Controls ─────────────────────────────────────────────────────

    def _build_controls(self, notebook) -> None:
        tk, ttk = self.tk, self.ttk
        pad = {"padx": 8, "pady": 4}

        # --- Top bar: PDF + page + DPI ---
        top = ttk.LabelFrame(self.frame, text="Source", padding=6)
        top.grid(row=0, column=0, sticky="ew", **pad)
        top.columnconfigure(1, weight=1)

        ttk.Button(top, text="Select PDF...", command=self._pick_pdf).grid(
            row=0, column=0, padx=(0, 4)
        )
        self._pdf_label_var = tk.StringVar(value="No file selected")
        ttk.Label(top, textvariable=self._pdf_label_var, foreground="gray").grid(
            row=0, column=1, sticky="w"
        )

        ttk.Label(top, text="Page (0-based):").grid(row=0, column=2, padx=(12, 2))
        self._page_var = tk.StringVar(value="0")
        ttk.Spinbox(top, textvariable=self._page_var, from_=0, to=999, width=5).grid(
            row=0, column=3
        )

        ttk.Label(top, text="DPI:").grid(row=0, column=4, padx=(12, 2))
        self._dpi_var = tk.StringVar(value="150")
        ttk.Spinbox(
            top,
            textvariable=self._dpi_var,
            values=(72, 100, 150, 200, 300),
            width=5,
            state="readonly",
        ).grid(row=0, column=5)

        # --- Layer toggles + render button ---
        mid = ttk.LabelFrame(self.frame, text="Layers & Knobs", padding=6)
        mid.grid(row=1, column=0, sticky="ew", **pad)
        mid.columnconfigure(3, weight=1)

        # Core layer toggles (row 0)
        self._green_var = tk.BooleanVar(value=True)
        self._purple_var = tk.BooleanVar(value=True)
        self._red_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(mid, text="Green: notes", variable=self._green_var).grid(
            row=0, column=0, sticky="w", padx=(0, 6)
        )
        ttk.Checkbutton(mid, text="Purple: columns", variable=self._purple_var).grid(
            row=0, column=1, sticky="w", padx=(0, 6)
        )
        ttk.Checkbutton(mid, text="Red: headers", variable=self._red_var).grid(
            row=0, column=2, sticky="w", padx=(0, 6)
        )

        # Extended layer toggles (row 0 continued + row 0b)
        self._glyph_var = tk.BooleanVar(value=False)
        self._blocks_var = tk.BooleanVar(value=False)
        self._structural_var = tk.BooleanVar(value=False)
        self._zones_var = tk.BooleanVar(value=False)
        self._legends_var = tk.BooleanVar(value=False)
        self._abbrev_var = tk.BooleanVar(value=False)
        self._revisions_var = tk.BooleanVar(value=False)
        self._stddet_var = tk.BooleanVar(value=False)

        ext_layer_frame = ttk.Frame(mid)
        ext_layer_frame.grid(row=0, column=3, sticky="w", padx=(6, 0))

        ext_layers_btn = ttk.Menubutton(ext_layer_frame, text="More Layers ▾")
        ext_layers_btn.pack(side="left")
        ext_menu = tk.Menu(ext_layers_btn, tearoff=False)
        ext_layers_btn["menu"] = ext_menu
        ext_menu.add_checkbutton(label="Glyph Boxes", variable=self._glyph_var)
        ext_menu.add_checkbutton(label="Block Outlines", variable=self._blocks_var)
        ext_menu.add_separator()
        ext_menu.add_checkbutton(
            label="Structural Boxes", variable=self._structural_var
        )
        ext_menu.add_checkbutton(label="Zones", variable=self._zones_var)
        ext_menu.add_separator()
        ext_menu.add_checkbutton(label="Legends", variable=self._legends_var)
        ext_menu.add_checkbutton(label="Abbreviations", variable=self._abbrev_var)
        ext_menu.add_checkbutton(label="Revisions", variable=self._revisions_var)
        ext_menu.add_checkbutton(label="Standard Details", variable=self._stddet_var)

        self._render_btn = ttk.Button(mid, text="Render", command=self._on_render)
        self._render_btn.grid(row=0, column=4, padx=(8, 0))

        ttk.Button(mid, text="Save PNG...", command=self._on_save).grid(
            row=0, column=5, padx=(4, 0)
        )

        ttk.Button(mid, text="Load from Run...", command=self._load_from_run).grid(
            row=0, column=6, padx=(4, 0)
        )

        # --- Zoom controls ---
        zoom_frame = ttk.Frame(mid)
        zoom_frame.grid(row=0, column=7, padx=(12, 0))

        ttk.Button(zoom_frame, text="−", width=2, command=self._zoom_out).grid(
            row=0, column=0
        )
        self._zoom_label_var = tk.StringVar(value="100%")
        ttk.Label(
            zoom_frame, textvariable=self._zoom_label_var, width=6, anchor="center"
        ).grid(row=0, column=1, padx=2)
        ttk.Button(zoom_frame, text="+", width=2, command=self._zoom_in).grid(
            row=0, column=2
        )
        ttk.Button(zoom_frame, text="Fit", width=3, command=self._zoom_fit).grid(
            row=0, column=3, padx=(4, 0)
        )
        ttk.Button(zoom_frame, text="1:1", width=3, command=self._zoom_reset).grid(
            row=0, column=4, padx=(2, 0)
        )

        # Note: GroupingConfig knobs removed – will be managed by LLM layer.
        self._knob_vars: dict[str, tk.StringVar] = {}

    # ── Image canvas ─────────────────────────────────────────────────

    def _build_image_canvas(self) -> None:
        tk, ttk = self.tk, self.ttk

        canvas_frame = ttk.Frame(self.frame)
        canvas_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=4)
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self._canvas = tk.Canvas(canvas_frame, bg="#2b2b2b", highlightthickness=0)
        self._h_scroll = ttk.Scrollbar(
            canvas_frame, orient="horizontal", command=self._canvas.xview
        )
        self._v_scroll = ttk.Scrollbar(
            canvas_frame, orient="vertical", command=self._canvas.yview
        )
        self._canvas.configure(
            xscrollcommand=self._h_scroll.set,
            yscrollcommand=self._v_scroll.set,
        )
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._v_scroll.grid(row=0, column=1, sticky="ns")
        self._h_scroll.grid(row=1, column=0, sticky="ew")

        self._canvas_img_id = None

        # Mouse-wheel zoom (bind on canvas)
        self._canvas.bind("<MouseWheel>", self._on_mousewheel_zoom)
        # Also allow standard scroll when not zooming (Shift+wheel = horizontal)
        self._canvas.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)

    # ── Status bar ───────────────────────────────────────────────────

    def _build_status_bar(self) -> None:
        tk, ttk = self.tk, self.ttk
        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(
            self.frame,
            textvariable=self._status_var,
            relief="sunken",
            anchor="w",
            padding=(6, 2),
        ).grid(row=3, column=0, sticky="ew")

    # ── Actions ──────────────────────────────────────────────────────

    def _pick_pdf(self) -> None:
        f = self.filedialog.askopenfilename(
            title="Select PDF",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")],
            initialdir=Path(__file__).parent.parent / "input",
        )
        if f:
            self._pdf_path = Path(f)
            self._pdf_label_var.set(self._pdf_path.name)

    def _collect_cfg(self) -> GroupingConfig:
        cfg = GroupingConfig()
        for name, sv in self._knob_vars.items():
            raw = sv.get().strip()
            if not raw:
                continue
            default_val = getattr(cfg, name)
            try:
                if isinstance(default_val, bool):
                    setattr(cfg, name, raw.lower() in ("1", "true", "yes"))
                elif isinstance(default_val, int):
                    setattr(cfg, name, int(raw))
                elif isinstance(default_val, float):
                    setattr(cfg, name, float(raw))
                else:
                    setattr(cfg, name, raw)
            except (ValueError, TypeError):
                pass  # keep default
        return cfg

    def _collect_layers(self) -> dict[str, bool]:
        return {
            "green": self._green_var.get(),
            "purple": self._purple_var.get(),
            "red": self._red_var.get(),
            "glyph_boxes": self._glyph_var.get(),
            "block_outlines": self._blocks_var.get(),
            "structural": self._structural_var.get(),
            "zones": self._zones_var.get(),
            "legends": self._legends_var.get(),
            "abbreviations": self._abbrev_var.get(),
            "revisions": self._revisions_var.get(),
            "std_details": self._stddet_var.get(),
        }

    def _on_render(self) -> None:
        if self._pdf_path is None:
            self.messagebox.showwarning("No PDF", "Select a PDF file first.")
            return
        if self._render_thread and self._render_thread.is_alive():
            return  # already rendering

        self._status_var.set("Rendering...")
        self._render_btn.config(state="disabled")

        page_idx = int(self._page_var.get())
        resolution = int(self._dpi_var.get())
        cfg = self._collect_cfg()
        layers = self._collect_layers()
        pdf_path = self._pdf_path

        def worker():
            t0 = time.perf_counter()
            try:
                img = render_overlay(
                    pdf_path,
                    page_idx,
                    cfg=cfg,
                    layers=layers,
                    resolution=resolution,
                )
                elapsed = time.perf_counter() - t0
                try:
                    if (
                        hasattr(self.frame, "winfo_exists")
                        and not self.frame.winfo_exists()
                    ):
                        return
                    self.frame.after(0, lambda: self._show_image(img, elapsed))
                except Exception:  # noqa: BLE001 — GUI callback is best-effort
                    return
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                try:
                    if (
                        hasattr(self.frame, "winfo_exists")
                        and not self.frame.winfo_exists()
                    ):
                        return
                    self.frame.after(0, lambda: self._render_error(str(exc), elapsed))
                except Exception:  # noqa: BLE001 — GUI callback is best-effort
                    return

        self._render_thread = threading.Thread(target=worker, daemon=True)
        self._render_thread.start()

    def _show_image(self, img: Image.Image, elapsed: float) -> None:
        self._last_img = img
        self._zoom = 1.0
        self._apply_zoom()
        self._status_var.set(f"Rendered {img.width}×{img.height} in {elapsed:.1f}s")
        self._render_btn.config(state="normal")

    def _apply_zoom(self) -> None:
        """Redraw the canvas image at the current zoom level."""
        from PIL import ImageTk

        if self._last_img is None:
            return
        img = self._last_img
        z = self._zoom
        disp_w = max(1, int(img.width * z))
        disp_h = max(1, int(img.height * z))
        resized = img.resize((disp_w, disp_h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        if self._canvas_img_id is not None:
            self._canvas.delete(self._canvas_img_id)
        self._canvas_img_id = self._canvas.create_image(
            0, 0, anchor="nw", image=self._photo
        )
        self._canvas.configure(scrollregion=(0, 0, disp_w, disp_h))
        self._zoom_label_var.set(f"{int(z * 100)}%")

    # ── Zoom helpers ─────────────────────────────────────────────────

    def _zoom_in(self) -> None:
        if self._last_img is None:
            return
        for lvl in self._zoom_levels:
            if lvl > self._zoom + 0.001:
                self._zoom = lvl
                self._apply_zoom()
                return

    def _zoom_out(self) -> None:
        if self._last_img is None:
            return
        for lvl in reversed(self._zoom_levels):
            if lvl < self._zoom - 0.001:
                self._zoom = lvl
                self._apply_zoom()
                return

    def _zoom_reset(self) -> None:
        if self._last_img is None:
            return
        self._zoom = 1.0
        self._apply_zoom()

    def _zoom_fit(self) -> None:
        """Fit the image within the visible canvas area."""
        if self._last_img is None:
            return
        self._canvas.update_idletasks()
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        iw, ih = self._last_img.width, self._last_img.height
        self._zoom = min(cw / iw, ch / ih)
        self._apply_zoom()

    def _on_mousewheel_zoom(self, event) -> None:
        """Ctrl-free mouse wheel zoom (standard zoom behaviour)."""
        if self._last_img is None:
            return
        if event.delta > 0:
            self._zoom_in()
        else:
            self._zoom_out()

    def _on_shift_mousewheel(self, event) -> None:
        """Shift+wheel for horizontal scroll."""
        self._canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

    def _render_error(self, msg: str, elapsed: float) -> None:
        self._status_var.set(f"Error ({elapsed:.1f}s): {msg}")
        self._render_btn.config(state="normal")

    def _on_save(self) -> None:
        if self._last_img is None:
            self.messagebox.showinfo("Nothing to save", "Render an overlay first.")
            return
        path = self.filedialog.asksaveasfilename(
            title="Save overlay PNG",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")],
            initialdir=latest_overlays_dir(),
            initialfile=f"page_{self._page_var.get()}_debug_overlay.png",
        )
        if path:
            self._last_img.save(path)
            self._status_var.set(f"Saved: {path}")

    def _load_from_run(self) -> None:
        """Load pre-computed extraction JSON from a run directory and render."""
        run_dir = self.filedialog.askdirectory(
            title="Select Run Directory",
            initialdir=str(Path(__file__).resolve().parent.parent.parent / "runs"),
        )
        if not run_dir:
            return
        run_dir = Path(run_dir)
        artifacts = run_dir / "artifacts"
        if not artifacts.is_dir():
            self.messagebox.showwarning(
                "No Artifacts", f"No artifacts/ folder in {run_dir.name}."
            )
            return

        # Find extraction JSON files
        jsons = sorted(artifacts.glob("page_*_extraction.json"))
        if not jsons:
            jsons = sorted(artifacts.glob("*extraction*.json"))
        if not jsons:
            self.messagebox.showwarning(
                "No Data", "No extraction JSON found in artifacts/."
            )
            return

        # If multiple pages, let user pick
        if len(jsons) == 1:
            json_path = jsons[0]
        else:
            page_names = [j.stem for j in jsons]
            pick_win = self.tk.Toplevel(self.frame)
            pick_win.title("Select extraction file")
            pick_win.geometry("400x300")
            lb = self.tk.Listbox(pick_win, selectmode="single")
            for n in page_names:
                lb.insert("end", n)
            lb.pack(fill="both", expand=True, padx=10, pady=10)
            chosen = [None]

            def on_ok():
                sel = lb.curselection()
                if sel:
                    chosen[0] = jsons[sel[0]]
                pick_win.destroy()

            self.ttk.Button(pick_win, text="OK", command=on_ok).pack(pady=5)
            pick_win.grab_set()
            pick_win.wait_window()
            json_path = chosen[0]
            if json_path is None:
                return

        # Find PDF path from manifest
        manifest_path = run_dir / "manifest.json"
        pdf_path = None
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                pdf_path_str = manifest.get("pdf_path") or manifest.get("pdf")
                if pdf_path_str:
                    pdf_path = Path(pdf_path_str)
                    if not pdf_path.is_file():
                        pdf_path = None
            except Exception:  # noqa: BLE001 — manifest parsing is best-effort
                pass

        if pdf_path is None:
            f = self.filedialog.askopenfilename(
                title="Select matching PDF (for background raster)",
                filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")],
            )
            if not f:
                return
            pdf_path = Path(f)

        self._pdf_path = pdf_path
        self._pdf_label_var.set(pdf_path.name)
        self._status_var.set(f"Rendering from {json_path.name}...")
        self._render_btn.config(state="disabled")

        page_idx = int(self._page_var.get())
        resolution = int(self._dpi_var.get())
        cfg = self._collect_cfg()
        layers = self._collect_layers()

        def worker():
            t0 = time.perf_counter()
            try:
                img = render_overlay(
                    pdf_path,
                    page_idx,
                    cfg=cfg,
                    layers=layers,
                    resolution=resolution,
                    json_path=str(json_path),
                )
                elapsed = time.perf_counter() - t0
                self.frame.after(0, lambda: self._show_image(img, elapsed))
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                self.frame.after(0, lambda: self._render_error(str(exc), elapsed))

        self._render_thread = threading.Thread(target=worker, daemon=True)
        self._render_thread.start()
