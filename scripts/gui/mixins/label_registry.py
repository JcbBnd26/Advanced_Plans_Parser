"""Label registry mixin for GUI annotation tab.

Handles element type management: registration, persistence to JSON,
color assignment, and combo box updates.
"""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import simpledialog
from typing import TYPE_CHECKING, Any

from plancheck.corrections.subtype_classifier import TITLE_SUBTYPES

if TYPE_CHECKING:
    from tkinter import ttk


class LabelRegistryMixin:
    """Mixin providing label registry persistence and element type management.

    Expected attributes on the host class:
        LABEL_COLORS: dict[str, str]
        ELEMENT_TYPES: list[str]
        _type_combo: ttk.Combobox
        _type_var: tk.StringVar
        _filter_label_vars: dict[str, tk.BooleanVar]
        _status: ttk.Label
        root: tk.Tk
    """

    TITLE_SUBTYPE_COLORS: dict[str, str] = {
        "page_title": "#6f42c1",
        "plan_title": "#8f5bd6",
        "detail_title": "#b26ee5",
        "section_title": "#5a8dee",
        "graph_title": "#2b8a78",
        "map_title": "#1f9d55",
        "box_title": "#d97706",
    }

    # ── Normalization ─────────────────────────────────────────────

    def _normalize_element_type_name(self, name: str) -> str:
        """Normalize element type name to lowercase_underscore format."""
        return name.strip().lower().replace(" ", "_")

    def _title_subtypes(self) -> list[str]:
        """Return the canonical Stage-2 title subtype labels."""
        return list(TITLE_SUBTYPES)

    def _is_title_family_label(self, label: str) -> bool:
        """Return True when the label is the title family or a subtype."""
        normalized = self._normalize_element_type_name(label) if label else ""
        return normalized == "title_block" or normalized in self._title_subtypes()

    def _ensure_title_subtype_labels(self) -> None:
        """Register Stage-2 subtype labels with stable default colors."""
        for label in self._title_subtypes():
            self._register_element_type(
                label,
                color=self.TITLE_SUBTYPE_COLORS.get(label),
            )

    def _sync_title_subtype_controls(self, label: str) -> None:
        """Keep the subtype selector aligned with the active element label."""
        subtype_combo = getattr(self, "_subtype_combo", None)
        subtype_var = getattr(self, "_subtype_var", None)
        if subtype_combo is None or subtype_var is None:
            return

        normalized = self._normalize_element_type_name(label) if label else ""
        subtype_value = normalized if normalized in self._title_subtypes() else ""
        state = "readonly" if self._is_title_family_label(normalized) else "disabled"

        self._syncing_title_subtype_controls = True
        try:
            subtype_var.set(subtype_value)
            subtype_combo.configure(state=state)
        finally:
            self._syncing_title_subtype_controls = False

    def _set_active_element_type(self, label: str) -> None:
        """Set the active type field and synchronize subtype controls."""
        self._type_var.set(label)
        self._sync_title_subtype_controls(label)

    def _on_type_selection_changed(self, *_args: Any) -> None:
        """React to manual edits in the main type control."""
        if getattr(self, "_syncing_title_subtype_controls", False):
            return
        self._sync_title_subtype_controls(self._type_var.get())

    def _on_subtype_selected(self, _event: Any = None) -> None:
        """Promote the active type to the chosen title subtype."""
        if getattr(self, "_syncing_title_subtype_controls", False):
            return
        subtype_var = getattr(self, "_subtype_var", None)
        if subtype_var is None:
            return
        subtype = self._normalize_element_type_name(subtype_var.get())
        if subtype in self._title_subtypes():
            self._set_active_element_type(subtype)

    # ── Registry path ────────────────────────────────────────────

    def _label_registry_path(self) -> Path:
        """Path to the shared label_registry.json in the data/ folder."""
        # scripts/gui/mixins/label_registry.py -> repo root is parent^4
        return (
            Path(__file__).resolve().parent.parent.parent.parent
            / "data"
            / "label_registry.json"
        )

    # ── JSON I/O ─────────────────────────────────────────────────

    def _load_label_registry_json(self) -> dict:
        """Load the label registry JSON, returning empty structure on error."""
        path = self._label_registry_path()
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"version": "1.0", "label_registry": []}
        except Exception:  # noqa: BLE001 — return default on any error
            return {"version": "1.0", "label_registry": []}

    def _save_label_registry_json(self, data: dict) -> None:
        """Save label registry JSON atomically."""
        path = self._label_registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
        tmp.replace(path)

    # ── Persistence ──────────────────────────────────────────────

    def _persist_element_type_to_registry(
        self, *, label: str, display_name: str, color: str
    ) -> None:
        """Persist an element type to the label_registry.json."""
        data = self._load_label_registry_json()
        reg = data.get("label_registry")
        if not isinstance(reg, list):
            reg = []
            data["label_registry"] = reg

        existing: dict | None = None
        for entry in reg:
            if isinstance(entry, dict) and entry.get("label") == label:
                existing = entry
                break

        if existing is None:
            reg.append(
                {
                    "label": label,
                    "display_name": display_name,
                    "color": color,
                    "description": "",
                    "aliases": [],
                    "expected_zones": [],
                    "text_patterns": [],
                }
            )
        else:
            existing["display_name"] = display_name
            existing["color"] = color

        if "version" not in data:
            data["version"] = "1.0"
        self._save_label_registry_json(data)

    def _load_element_types_from_registry(self) -> None:
        """Load element types from the registry JSON into LABEL_COLORS."""
        data = self._load_label_registry_json()
        reg = data.get("label_registry", [])
        if not isinstance(reg, list):
            return
        for entry in reg:
            if not isinstance(entry, dict):
                continue
            label = entry.get("label", "")
            color = entry.get("color", "")
            if not label:
                continue
            if isinstance(color, str) and color.startswith("#") and len(color) == 7:
                self._register_element_type(label, color=color)
            else:
                self._register_element_type(label)
        self._ensure_title_subtype_labels()
        self._sync_title_subtype_controls(self._type_var.get())

    # ── Registration ─────────────────────────────────────────────

    def _register_element_type(self, name: str, *, color: str | None = None) -> None:
        """Register a new element type with optional color.

        Updates LABEL_COLORS, ELEMENT_TYPES, the type combo boxes,
        and the filter checkboxes.
        """
        name = self._normalize_element_type_name(name)
        if not name or name in self.LABEL_COLORS:
            return

        # Auto-assign a distinct color from a palette
        _palette = [
            "#e6194b",
            "#3cb44b",
            "#4363d8",
            "#f58231",
            "#911eb4",
            "#42d4f4",
            "#f032e6",
            "#bfef45",
            "#fabebe",
            "#469990",
            "#9a6324",
            "#800000",
            "#aaffc3",
            "#808000",
            "#000075",
        ]
        if not (isinstance(color, str) and color.startswith("#") and len(color) == 7):
            idx = len(self.LABEL_COLORS) % len(_palette)
            color = _palette[idx]

        self.LABEL_COLORS[name] = color
        if name not in self.ELEMENT_TYPES:
            self.ELEMENT_TYPES.append(name)

        # Update combo boxes
        self._type_combo.configure(values=self.ELEMENT_TYPES)
        subtype_combo = getattr(self, "_subtype_combo", None)
        if subtype_combo is not None:
            subtype_combo.configure(values=self._title_subtypes())

        if name not in self._filter_label_vars:
            self._filter_label_vars[name] = tk.BooleanVar(value=True)
        self._rebuild_filter_controls()

    # ── UI callbacks ─────────────────────────────────────────────

    def _on_add_element_type(self) -> None:
        """Prompt user to add a new element type."""
        name = simpledialog.askstring(
            "New Element Type",
            "Enter new element type name:",
            parent=self.root,
        )
        if name:
            self._register_element_type(name)
            self._set_active_element_type(name.strip().lower().replace(" ", "_"))
            self._status.configure(
                text=f"Added element type: {name.strip().lower().replace(' ', '_')}"
            )

    def _on_type_entered(self, _event: Any = None) -> None:
        """Handle Enter key in the type combo — register if new."""
        name = self._type_var.get().strip()
        if name and name not in self.ELEMENT_TYPES:
            self._register_element_type(name)
            self._status.configure(text=f"Added element type: {name}")
