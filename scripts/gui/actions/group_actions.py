"""Group management actions for the annotation tab.

Functions accept the ``tab`` (AnnotationTab) instance and operate on
its state.  Extracted from EventHandlerMixin to keep that module
thin.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..annotation_state import CanvasBox


def create_group(tab: Any, cbox: "CanvasBox") -> None:
    """Create a new group with *cbox* as the root (parent)."""
    if not tab._doc_id:
        return

    name_win = tk.Toplevel(tab.root)
    name_win.title("New Group")
    name_win.transient(tab.root)
    name_win.grab_set()
    name_win.resizable(True, True)
    name_win.minsize(280, 110)

    ttk.Label(name_win, text="Group name:").pack(padx=10, pady=(10, 4))
    name_var = tk.StringVar(value=cbox.element_type)
    entry = ttk.Entry(name_win, textvariable=name_var, width=28)
    entry.pack(padx=10)
    entry.selection_range(0, "end")
    entry.focus_set()

    result: list[str | None] = [None]

    def on_ok(_event: tk.Event | None = None) -> None:
        result[0] = name_var.get().strip()
        name_win.destroy()

    def on_cancel() -> None:
        name_win.destroy()

    entry.bind("<Return>", on_ok)
    name_win.bind("<Escape>", lambda e: on_cancel())
    btn_f = ttk.Frame(name_win)
    btn_f.pack(pady=8)
    ttk.Button(btn_f, text="OK", command=on_ok).pack(side="left", padx=4)
    ttk.Button(btn_f, text="Cancel", command=on_cancel).pack(side="left", padx=4)

    name_win.update_idletasks()
    name_win.geometry("")

    name_win.wait_window()

    label = result[0]
    if not label:
        return

    group_id = tab._store.create_group(
        doc_id=tab._doc_id,
        page=tab._page,
        group_label=label,
        root_detection_id=cbox.detection_id,
    )
    cbox.group_id = group_id
    cbox.is_group_root = True
    tab._groups[group_id] = {
        "label": label,
        "root_detection_id": cbox.detection_id,
        "members": [cbox.detection_id],
    }
    tab._draw_box(cbox)
    tab._select_box(cbox)
    tab._status.configure(text=f"Created group \u2039{label}\u203a")


def add_children_to_group(tab: Any, targets: list["CanvasBox"]) -> None:
    """Add a list of CanvasBoxes as children to the selected group."""
    if not tab._selected_box or not tab._selected_box.group_id:
        return
    gid = tab._selected_box.group_id
    grp = tab._groups.get(gid)
    if not grp:
        return

    count = 0
    next_order = len(grp["members"])
    for cb in targets:
        if cb.group_id:
            continue
        tab._store.add_to_group(gid, cb.detection_id, sort_order=next_order)
        cb.group_id = gid
        cb.is_group_root = False
        grp["members"].append(cb.detection_id)
        next_order += 1
        count += 1
        tab._draw_box(cb)

    if count:
        tab._draw_group_links(gid)
        tab._status.configure(
            text=f"Added {count} box(es) to group \u2039{grp['label']}\u203a"
        )


def remove_from_group(tab: Any, cbox: "CanvasBox") -> None:
    """Remove *cbox* from its group.  Deletes group if root."""
    gid = cbox.group_id
    if not gid:
        return
    grp = tab._groups.get(gid, {})
    label = grp.get("label", "?")
    is_root = cbox.is_group_root

    tab._store.remove_from_group(gid, cbox.detection_id)

    if is_root:
        for cb in tab._canvas_boxes:
            if cb.group_id == gid:
                cb.group_id = None
                cb.is_group_root = False
                tab._draw_box(cb)
        tab._groups.pop(gid, None)
        tab._clear_group_links()
        tab._status.configure(
            text=f"Deleted group \u2039{label}\u203a (parent removed)"
        )
    else:
        cbox.group_id = None
        cbox.is_group_root = False
        if gid in tab._groups:
            members = tab._groups[gid]["members"]
            if cbox.detection_id in members:
                members.remove(cbox.detection_id)
        tab._draw_box(cbox)
        tab._draw_group_links(gid)
        tab._status.configure(text=f"Removed from group \u2039{label}\u203a")

    if tab._selected_box:
        tab._select_box(tab._selected_box)


def update_group_inspector(tab: Any, cbox: "CanvasBox | None") -> None:
    """Populate the group section of the inspector."""
    tab._clear_group_links()
    tab._btn_create_group.pack_forget()
    tab._btn_add_to_group.pack_forget()
    tab._btn_remove_group.pack_forget()

    if cbox is None or not cbox.group_id:
        tab._insp_group_label.configure(text="\u2014")
        if cbox is not None:
            tab._btn_create_group.pack(side="left", padx=(0, 3))
        return

    grp = tab._groups.get(cbox.group_id, {})
    label = grp.get("label", "?")
    if cbox.is_group_root:
        n_members = len(grp.get("members", [])) - 1
        tab._insp_group_label.configure(
            text=f"\u25cf {label} (parent \u2014 {n_members} children)"
        )
        tab._btn_add_to_group.pack(side="left", padx=(0, 3))
    else:
        tab._insp_group_label.configure(text=f"\u2192 {label}")
    tab._btn_remove_group.pack(side="left", padx=(0, 3))
    tab._draw_group_links(cbox.group_id)
