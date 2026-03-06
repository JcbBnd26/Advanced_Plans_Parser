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

    # ── Normalization ─────────────────────────────────────────────

    def _normalize_element_type_name(self, name: str) -> str:
        """Normalize element type name to lowercase_underscore format."""
        return name.strip().lower().replace(" ", "_")

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
            self._type_var.set(name.strip().lower().replace(" ", "_"))
            self._status.configure(
                text=f"Added element type: {name.strip().lower().replace(' ', '_')}"
            )

    def _on_type_entered(self, _event: Any = None) -> None:
        """Handle Enter key in the type combo — register if new."""
        name = self._type_var.get().strip()
        if name and name not in self.ELEMENT_TYPES:
            self._register_element_type(name)
            self._status.configure(text=f"Added element type: {name}")
