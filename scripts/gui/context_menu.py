"""Right-click context menu mixin for the annotation tab."""

from __future__ import annotations

import tkinter as tk
from tkinter import colorchooser, messagebox, simpledialog

from plancheck.ingest.ingest import point_in_polygon

from .annotation_state import CanvasBox


class ContextMenuMixin:
    """Mixin providing right-click context menu functionality."""

    def _on_canvas_right_click(self, event: tk.Event) -> None:
        """Show a context menu — word actions if words selected, else box actions."""
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)

        eff = self._effective_scale()
        pdf_x = cx / eff
        pdf_y = cy / eff

        # ── If word overlay is active, check for right-click on a word ──
        if self._word_overlay_on and self._word_overlay_items:
            for rid, winfo in self._word_overlay_items.items():
                if (
                    winfo["x0"] <= pdf_x <= winfo["x1"]
                    and winfo["top"] <= pdf_y <= winfo["bottom"]
                ):
                    # Auto-select the word under cursor if not already
                    if rid not in self._selected_word_rids:
                        self._set_word_selected(rid, True)
                    break

        # ── Word context menu ──────────────────────────────────────
        if self._selected_word_rids:
            self._show_word_context_menu(event, pdf_x, pdf_y)
            return

        # ── Detection box context menu (original) ─────────────────
        clicked: CanvasBox | None = None
        for cbox in reversed(self._canvas_boxes):
            if (
                cbox.rect_id
                and self._canvas.itemcget(cbox.rect_id, "state") == "hidden"
            ):
                continue
            if cbox.polygon:
                if point_in_polygon(pdf_x, pdf_y, cbox.polygon):
                    clicked = cbox
                    break
            else:
                bx0, by0, bx1, by1 = cbox.pdf_bbox
                if bx0 <= pdf_x <= bx1 and by0 <= pdf_y <= by1:
                    clicked = cbox
                    break

        menu = tk.Menu(self._canvas, tearoff=0)
        if clicked:
            menu.add_command(
                label=f"Copy Box ({clicked.element_type})",
                command=lambda: self._copy_box(clicked),
            )
        if self._copied_box_template:
            t = self._copied_box_template["element_type"]
            menu.add_command(
                label=f"Paste Box ({t})",
                command=lambda: self._paste_box(pdf_x, pdf_y),
            )

        # ── Link to notes column ──────────────────────────────────
        n_multi = len(self._multi_selected)
        if clicked and clicked not in self._multi_selected:
            n_multi += 1
        if n_multi >= 2:
            # Check if selection contains linkable types
            link_targets = list(self._multi_selected)
            if clicked and clicked not in link_targets:
                link_targets.append(clicked)
            linkable_types = {"header", "notes_block"}
            if any(cb.element_type in linkable_types for cb in link_targets):
                menu.add_separator()
                menu.add_command(
                    label="Create Notes Column from Selection (L)",
                    command=self._on_link_column,
                )

        # ── Group actions ──────────────────────────────────────────
        has_group_items = False
        if clicked:
            if not clicked.group_id:
                menu.add_separator()
                menu.add_command(
                    label="Create Group (Set as Parent)",
                    command=lambda: self._create_group(clicked),
                )
                has_group_items = True
                if (
                    self._selected_box
                    and self._selected_box.is_group_root
                    and self._selected_box is not clicked
                ):
                    g_label = self._groups.get(
                        self._selected_box.group_id or "", {}
                    ).get("label", "?")
                    menu.add_command(
                        label=f"Add to Group \u2039{g_label}\u203a",
                        command=lambda: self._add_children_to_group([clicked]),
                    )
            else:
                if not has_group_items:
                    menu.add_separator()
                menu.add_command(
                    label="Remove from Group",
                    command=lambda: self._remove_from_group(clicked),
                )

        if menu.index("end") is not None:
            menu.tk_popup(event.x_root, event.y_root)
        else:
            menu.destroy()

    def _show_word_context_menu(
        self, event: tk.Event, pdf_x: float, pdf_y: float
    ) -> None:
        """Show context menu with all actions for selected words."""
        n = len(self._selected_word_rids)
        # Gather selected word texts
        texts: list[str] = []
        for rid in self._selected_word_rids:
            winfo = self._word_overlay_items.get(rid)
            if winfo and winfo.get("text"):
                texts.append(winfo["text"])
        preview = " ".join(texts)
        if len(preview) > 50:
            preview = preview[:47] + "..."

        menu = tk.Menu(self._canvas, tearoff=0)

        # ── Header ─────────────────────────────────────────────────
        menu.add_command(
            label=f"{n} word{'s' if n != 1 else ''} selected",
            state="disabled",
        )
        if preview:
            menu.add_command(
                label=f'"{preview}"',
                state="disabled",
            )
        menu.add_separator()

        # ── Copy text ──────────────────────────────────────────────
        def _copy_word_text():
            if texts:
                self.root.clipboard_clear()
                self.root.clipboard_append(" ".join(texts))
                self._status.configure(text=f"Copied text from {len(texts)} words")

        menu.add_command(label="Copy Text  (Ctrl+C)", command=_copy_word_text)

        # ── Create New Type ──────────────────────────────────────
        menu.add_separator()
        menu.add_command(
            label="Create New Type…",
            command=self._on_create_new_type_from_words,
        )

        # ── Merge / Create Detection ──────────────────────────────
        if n >= 2:
            menu.add_separator()
            if self._selected_box:
                menu.add_command(
                    label=f"Reshape \u2039{self._selected_box.element_type}\u203a to Words  (M)",
                    command=self._merge_words_into_detection,
                )
            else:
                menu.add_command(
                    label="Create Detection from Words  (M)",
                    command=self._merge_words_into_detection,
                )

            menu.add_separator()

            # ── Create as specific type ─────────────────────────
            type_menu = tk.Menu(menu, tearoff=0)
            for etype in self.ELEMENT_TYPES:
                color = self.LABEL_COLORS.get(etype, "#888")
                type_menu.add_command(
                    label=etype,
                    command=lambda t=etype: self._create_words_as_type(t),
                )
            menu.add_cascade(label="Create as Type \u25b6", menu=type_menu)

        # ── Group from words ──────────────────────────────────────
        if n >= 2:
            menu.add_separator()
            menu.add_command(
                label="Group Words  (G)",
                command=lambda: self._key_group(
                    type("E", (), {"widget": self._canvas, "state": 0})()
                ),
            )

        # ── Select controls ──────────────────────────────────────
        menu.add_separator()
        menu.add_command(
            label="Select All Words",
            command=self._select_all_words,
        )
        menu.add_command(
            label="Clear Selection",
            command=self._clear_word_selection,
        )

        menu.tk_popup(event.x_root, event.y_root)

    def _on_create_new_type_from_words(self) -> None:
        """Prompt for a new type name + color, persist it, then create from selected words."""
        if not self._selected_word_rids:
            return

        raw = simpledialog.askstring(
            "Create New Type",
            "Enter new type name:",
            parent=self.root,
        )
        if not raw:
            return

        label = self._normalize_element_type_name(raw)
        if not label:
            return

        if label in self.LABEL_COLORS:
            messagebox.showwarning(
                "Type Exists",
                f'Type "{label}" already exists.',
                parent=self.root,
            )
            return

        _rgb, hex_color = colorchooser.askcolor(
            title="Choose Type Color",
            parent=self.root,
        )
        if not hex_color:
            return

        self._register_element_type(label, color=hex_color)
        try:
            self._persist_element_type_to_registry(
                label=label,
                display_name=raw.strip(),
                color=hex_color,
            )
        except Exception:  # noqa: BLE001 — registry write is best-effort
            self._status.configure(text="Warning: failed to write label_registry.json")

        # Create detection from words using this type (force create, no classifier override)
        self._set_active_element_type(label)
        self._merge_words_into_detection(forced_type=label, force_create=True)

    def _create_words_as_type(self, element_type: str) -> None:
        """Create a new detection from selected words with a specific type."""
        old_type = self._type_var.get()
        self._set_active_element_type(element_type)
        self._merge_words_into_detection(forced_type=element_type, force_create=True)
        self._set_active_element_type(old_type)

    def _select_all_words(self) -> None:
        """Select every word in the overlay."""
        if not self._word_overlay_items:
            return
        for rid in self._word_overlay_items:
            if rid not in self._selected_word_rids:
                self._set_word_selected(rid, True)
        self._status.configure(
            text=f"Selected all {len(self._selected_word_rids)} words"
        )
