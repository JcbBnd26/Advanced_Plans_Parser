"""New Project / Open Project dialogs for the Advanced Plan Parser GUI.

Provides two dialog classes:
- ``NewProjectDialog``  – create a new project folder with class selection
- ``open_project_dialog`` – simple folder chooser that validates and activates a project

Both use the core functions from :mod:`plancheck.config.project`.
"""

from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .gui import GuiState

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# New Project Dialog
# ═══════════════════════════════════════════════════════════════════════


class NewProjectDialog(tk.Toplevel):
    """Dialog for creating a new project.

    Sections
    --------
    1. Name & Location  – project name, description, parent directory
    2. Class Selection   – checklist of label types from master registry
    3. Config Overrides  – optional ML tuning parameters
    """

    def __init__(
        self,
        parent: tk.Widget,
        state: "GuiState",
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, **kwargs)
        self.title("New Project")
        self.geometry("600x650")
        self.minsize(500, 500)
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        self._state = state
        self._result: Path | None = None

        # Default parent directory for projects
        self._default_parent = Path.cwd() / "projects"

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

    # ── UI construction ───────────────────────────────────────────

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)  # class selection gets the stretch

        pad = {"padx": 8, "pady": 4}

        # ── Section 1: Name & Location ────────────────────────────
        sec1 = ttk.LabelFrame(self, text="Project Details")
        sec1.grid(row=0, column=0, sticky="ew", **pad)
        sec1.columnconfigure(1, weight=1)

        ttk.Label(sec1, text="Name:").grid(row=0, column=0, sticky="w", **pad)
        self._name_var = tk.StringVar()
        ttk.Entry(sec1, textvariable=self._name_var, width=40).grid(
            row=0, column=1, sticky="ew", **pad
        )

        ttk.Label(sec1, text="Description:").grid(row=1, column=0, sticky="w", **pad)
        self._desc_var = tk.StringVar()
        ttk.Entry(sec1, textvariable=self._desc_var, width=40).grid(
            row=1, column=1, sticky="ew", **pad
        )

        ttk.Label(sec1, text="Location:").grid(row=2, column=0, sticky="w", **pad)
        loc_frame = ttk.Frame(sec1)
        loc_frame.grid(row=2, column=1, sticky="ew", **pad)
        loc_frame.columnconfigure(0, weight=1)

        self._location_var = tk.StringVar(value=str(self._default_parent))
        ttk.Entry(loc_frame, textvariable=self._location_var).grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Button(loc_frame, text="Browse…", command=self._browse_location).grid(
            row=0, column=1, padx=(4, 0)
        )

        # ── Section 2: Class Selection ────────────────────────────
        sec2 = ttk.LabelFrame(self, text="Label Classes")
        sec2.grid(row=1, column=0, sticky="nsew", **pad)
        sec2.columnconfigure(0, weight=1)
        sec2.rowconfigure(0, weight=1)

        # Scrollable checklist
        canvas = tk.Canvas(sec2, highlightthickness=0)
        scrollbar = ttk.Scrollbar(sec2, orient="vertical", command=canvas.yview)
        self._class_frame = ttk.Frame(canvas)

        self._class_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self._class_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Populate checklist from master registry
        from plancheck.config.project import get_master_label_defs

        self._label_defs = get_master_label_defs()
        self._class_vars: list[tuple[tk.BooleanVar, dict]] = []

        for label_def in self._label_defs:
            var = tk.BooleanVar(value=True)
            row_frame = ttk.Frame(self._class_frame)
            row_frame.pack(fill="x", padx=4, pady=2)

            cb = ttk.Checkbutton(row_frame, variable=var)
            cb.pack(side="left")

            # Color swatch
            color = label_def.get("color", "#888888")
            swatch = tk.Canvas(row_frame, width=16, height=16, highlightthickness=0)
            swatch.create_rectangle(1, 1, 15, 15, fill=color, outline="#333")
            swatch.pack(side="left", padx=(2, 6))

            display = label_def.get("display_name", label_def.get("label", "?"))
            desc = label_def.get("description", "")
            text = f"{display}"
            if desc:
                text += f"  —  {desc[:60]}"
            ttk.Label(row_frame, text=text, wraplength=400).pack(
                side="left", fill="x", expand=True
            )

            self._class_vars.append((var, label_def))

        # Select All / Deselect All buttons
        btn_row = ttk.Frame(sec2)
        btn_row.grid(row=1, column=0, columnspan=2, sticky="ew", padx=4, pady=2)
        ttk.Button(btn_row, text="Select All", command=self._select_all).pack(
            side="left", padx=2
        )
        ttk.Button(btn_row, text="Deselect All", command=self._deselect_all).pack(
            side="left", padx=2
        )

        # ── Section 3: Config Overrides (collapsible) ─────────────
        sec3 = ttk.LabelFrame(self, text="Config Overrides (optional)")
        sec3.grid(row=2, column=0, sticky="ew", **pad)
        sec3.columnconfigure(1, weight=1)

        ttk.Label(sec3, text="Retrain threshold:").grid(
            row=0, column=0, sticky="w", **pad
        )
        self._retrain_var = tk.IntVar(value=50)
        ttk.Spinbox(
            sec3, textvariable=self._retrain_var, from_=5, to=500, width=8
        ).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(sec3, text="ML confidence:").grid(row=1, column=0, sticky="w", **pad)
        self._confidence_var = tk.DoubleVar(value=0.8)
        ttk.Spinbox(
            sec3,
            textvariable=self._confidence_var,
            from_=0.1,
            to=1.0,
            increment=0.05,
            width=8,
        ).grid(row=1, column=1, sticky="w", **pad)

        # ── Buttons ───────────────────────────────────────────────
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=3, column=0, sticky="ew", **pad)

        ttk.Button(btn_frame, text="Create Project", command=self._on_create).pack(
            side="right", padx=4
        )
        ttk.Button(btn_frame, text="Cancel", command=self._on_cancel).pack(
            side="right", padx=4
        )

    # ── Callbacks ─────────────────────────────────────────────────

    def _browse_location(self) -> None:
        d = filedialog.askdirectory(
            title="Select Parent Directory for Projects",
            initialdir=self._location_var.get(),
        )
        if d:
            self._location_var.set(d)

    def _select_all(self) -> None:
        for var, _ in self._class_vars:
            var.set(True)

    def _deselect_all(self) -> None:
        for var, _ in self._class_vars:
            var.set(False)

    def _on_create(self) -> None:
        name = self._name_var.get().strip()
        if not name:
            messagebox.showwarning(
                "Missing Name", "Please enter a project name.", parent=self
            )
            return

        # Gather selected labels
        selected_labels = [ldef for var, ldef in self._class_vars if var.get()]
        if not selected_labels:
            messagebox.showwarning(
                "No Classes",
                "Select at least one label class for the project.",
                parent=self,
            )
            return

        # Build config overrides
        overrides: dict = {}
        retrain = self._retrain_var.get()
        if retrain != 50:
            overrides["ml_retrain_threshold"] = retrain
        confidence = self._confidence_var.get()
        if confidence != 0.8:
            overrides["ml_relabel_confidence"] = confidence

        # Build project directory path
        from plancheck.config.project import create_project, slugify

        parent_dir = Path(self._location_var.get())
        project_dir = parent_dir / slugify(name)

        try:
            create_project(
                project_dir,
                name,
                selected_labels,
                description=self._desc_var.get().strip(),
                config_overrides=overrides or None,
            )
        except FileExistsError:
            messagebox.showerror(
                "Already Exists",
                f"A project already exists at:\n{project_dir}",
                parent=self,
            )
            return
        except Exception as exc:  # noqa: BLE001 — surface creation errors
            log.error("Failed to create project: %s", exc, exc_info=True)
            messagebox.showerror(
                "Error",
                f"Failed to create project:\n{exc}",
                parent=self,
            )
            return

        # Activate the new project
        try:
            self._state.set_project(project_dir)
        except Exception as exc:  # noqa: BLE001 — surface activation errors
            log.error("Failed to activate project: %s", exc, exc_info=True)
            messagebox.showerror(
                "Error",
                f"Project created but failed to activate:\n{exc}",
                parent=self,
            )
            return

        self._result = project_dir
        self.destroy()

    def _on_cancel(self) -> None:
        self._result = None
        self.destroy()

    @property
    def result(self) -> Path | None:
        """The project directory that was created, or None if cancelled."""
        return self._result


# ═══════════════════════════════════════════════════════════════════════
# Open Project (simple function)
# ═══════════════════════════════════════════════════════════════════════


def open_project_dialog(parent: tk.Widget, state: "GuiState") -> Path | None:
    """Show a folder chooser and activate the selected project.

    Returns the project directory on success, or ``None`` if cancelled
    or the folder is not a valid project.
    """
    folder = filedialog.askdirectory(
        title="Open Project Folder",
        initialdir=str(Path.cwd() / "projects"),
    )
    if not folder:
        return None

    project_dir = Path(folder)
    project_json = project_dir / "project.json"

    if not project_json.exists():
        messagebox.showerror(
            "Invalid Project",
            f"Not a valid project folder:\n{project_dir}\n\n" "(No project.json found)",
            parent=parent,
        )
        return None

    try:
        state.set_project(project_dir)
    except Exception as exc:  # noqa: BLE001 — surface errors to user
        log.error("Failed to open project: %s", exc, exc_info=True)
        messagebox.showerror(
            "Error",
            f"Failed to open project:\n{exc}",
            parent=parent,
        )
        return None

    return project_dir


# ═══════════════════════════════════════════════════════════════════════
# Export / Import dialogs
# ═══════════════════════════════════════════════════════════════════════


def export_project_dialog(parent: tk.Widget, state: "GuiState") -> Path | None:
    """Show a save dialog to export the active project as a zip archive.

    Returns the output path on success, or ``None`` if cancelled.
    """
    if not state.project_dir:
        messagebox.showwarning(
            "No Project",
            "No project is currently open.\nCreate or open a project first.",
            parent=parent,
        )
        return None

    project_name = (state.project_meta or {}).get("name", "project")
    default_name = f"{project_name.replace(' ', '_')}.plancheck"

    output = filedialog.asksaveasfilename(
        title="Export Project",
        initialfile=default_name,
        defaultextension=".plancheck",
        filetypes=[
            ("PlanCheck Project", "*.plancheck"),
            ("Zip Archive", "*.zip"),
            ("All Files", "*.*"),
        ],
        parent=parent,
    )
    if not output:
        return None

    try:
        from plancheck.config.project import export_project

        result = export_project(state.project_dir, Path(output))
        messagebox.showinfo(
            "Export Complete",
            f"Project exported to:\n{result}",
            parent=parent,
        )
        return result
    except Exception as exc:  # noqa: BLE001 — surface errors
        log.error("Project export failed: %s", exc, exc_info=True)
        messagebox.showerror(
            "Export Failed",
            f"Failed to export project:\n{exc}",
            parent=parent,
        )
        return None


def import_project_dialog(parent: tk.Widget, state: "GuiState") -> Path | None:
    """Show a file chooser to import a project from a zip archive.

    Returns the imported project directory on success, or ``None``.
    """
    zip_path = filedialog.askopenfilename(
        title="Import Project",
        filetypes=[
            ("PlanCheck Project", "*.plancheck"),
            ("Zip Archive", "*.zip"),
            ("All Files", "*.*"),
        ],
        parent=parent,
    )
    if not zip_path:
        return None

    # Choose where to extract
    target_dir = filedialog.askdirectory(
        title="Select Destination for Imported Project",
        initialdir=str(Path.cwd() / "projects"),
    )
    if not target_dir:
        return None

    try:
        from plancheck.config.project import import_project

        project_dir = import_project(Path(zip_path), Path(target_dir))
        state.set_project(project_dir)
        messagebox.showinfo(
            "Import Complete",
            f"Project imported and activated:\n{project_dir}",
            parent=parent,
        )
        return project_dir
    except Exception as exc:  # noqa: BLE001 — surface errors
        log.error("Project import failed: %s", exc, exc_info=True)
        messagebox.showerror(
            "Import Failed",
            f"Failed to import project:\n{exc}",
            parent=parent,
        )
        return None
