"""Grouper event handler mixin.

Provides all mouse event handlers for the Grouper tab canvas.

Learn mode gestures
-------------------
- ``Shift+Left-click`` — toggle a GlyphBox in/out of the active selection.
- ``Left-click`` (no Shift) on a *selected* glyph — open the
  **Group popup** with a "Confirm Group ✓" button.
- ``Left-click`` (no Shift) on an *unselected* glyph — inspect: highlight
  the group (confirmed or machine) that the glyph belongs to.
- ``Right-click`` — clear selection and inspect state.

Edit mode gestures
------------------
All of the above plus:
- ``Shift+Left-click`` an *ungrouped* glyph → popup offers "Add to Group"
  (then click any existing group to target it) or "New Group".
- ``Shift+Left-click`` a *grouped* glyph that is **already selected** →
  popup offers "Remove from Group" and "Split into New Group".
- ``Shift+Left-click`` a glyph from a *second* group when another group
  is already inspected → popup offers "Merge Groups".

All edit operations persist a ``group_corrections`` row with
``signal="edit"`` and a ``delta`` dict.

Phase F + G — implemented in Write 3.
"""

from __future__ import annotations

import logging
import tkinter as tk
from typing import Any, List, Optional, Tuple

log = logging.getLogger(__name__)

# Pixels of movement before a press becomes a drag rather than a click
_DRAG_THRESHOLD: int = 5


class GrouperEventsMixin:
    """Mixin providing all Grouper mouse-event handlers.

    Requires the host class to expose:

    - ``self._canvas``         — ``tk.Canvas`` widget
    - ``self._gsession``       — :class:`.GrouperSessionState` instance
    - ``self._store``          — :class:`.CorrectionStore` instance
    - ``self._cfg``            — :class:`.GroupingConfig` instance
    - ``self._render_canvas()``— full canvas redraw callable
    - ``self._hit_test_glyph(cx, cy)`` — from :class:`.GrouperCanvasMixin`
    - ``self._set_status(text)``       — status bar update
    """

    # ── Binding entry point ───────────────────────────────────────────────────

    def _bind_canvas_events(self) -> None:
        """Attach all canvas mouse bindings.  Called after canvas is built."""
        # Drag-selection state (initialised here so mixins share one init point)
        self._drag_origin: Optional[Tuple[float, float]] = None
        self._drag_event: Any = None
        self._is_dragging: bool = False

        # Use a single <Button-1> press/motion/release triplet so that clicks
        # and rubber-band drags share one consistent state machine.
        self._canvas.bind("<Button-1>", self._on_button_press)
        self._canvas.bind("<B1-Motion>", self._on_drag_motion)
        self._canvas.bind("<ButtonRelease-1>", self._on_button_release)
        self._canvas.bind("<Control-Button-1>", self._on_ctrl_click)
        self._canvas.bind("<Button-3>", self._on_right_click)
        self._canvas.bind("<MouseWheel>", self._on_mousewheel)

    # ── Press → start of click-or-drag ───────────────────────────────────────

    def _on_button_press(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Record press position; drag vs click is resolved on release."""
        if not self._gsession.active:
            self._set_status("Open a PDF to start a Grouper session.")
            return
        self._canvas.focus_set()
        self._drag_origin = (
            self._canvas.canvasx(event.x),
            self._canvas.canvasy(event.y),
        )
        self._drag_event = event
        self._is_dragging = False

    def _handle_shift_click(
        self, cx: float, cy: float, event: tk.Event  # type: ignore[type-arg]
    ) -> None:
        """Shift+click: toggle selection or trigger Edit-mode gesture."""
        glyph_idx = self._hit_test_glyph(cx, cy)
        if glyph_idx is None:
            return

        if self._gsession.mode == "edit":
            self._handle_edit_shift_click(glyph_idx, event, cx, cy)
            return

        # Learn mode — simple toggle
        sel = self._gsession.selected_indices
        if glyph_idx in sel:
            sel.discard(glyph_idx)
        else:
            sel.add(glyph_idx)
        self._render_canvas()
        self._set_status(f"{len(sel)} token(s) selected")

    def _handle_plain_click(
        self, cx: float, cy: float, event: tk.Event  # type: ignore[type-arg]
    ) -> None:
        """Plain click: confirm selection, inspect group, or clear."""
        glyph_idx = self._hit_test_glyph(cx, cy)

        if glyph_idx is None:
            # Clicked blank space — clear everything
            self._gsession.selected_indices.clear()
            self._gsession.inspected_group_idx = None
            self._render_canvas()
            return

        # Glyph already selected → open confirm popup
        if self._gsession.is_glyph_selected(glyph_idx):
            self._show_group_popup(event, cx, cy)
            return

        # Glyph belongs to a confirmed group → inspect it
        confirmed_gi = self._gsession.index_in_confirmed_group(glyph_idx)
        if confirmed_gi is not None:
            self._gsession.inspected_group_idx = confirmed_gi
            self._gsession.selected_indices.clear()
            self._render_canvas()
            n = len(self._gsession.confirmed_groups[confirmed_gi].indices)
            self._set_status(f"Group {confirmed_gi + 1} — {n} token(s)")
            return

        # Edit mode: check machine groups
        if self._gsession.mode == "edit" and self._gsession.show_machine_groups:
            pd = self._gsession.page_data
            if pd:
                for mi, blk in enumerate(pd.machine_groups):
                    blk_boxes = blk.get_all_boxes()
                    if any(b is pd.boxes[glyph_idx] for b in blk_boxes):
                        self._gsession.inspected_group_idx = -(mi + 1)
                        self._gsession.selected_indices.clear()
                        self._render_canvas()
                        self._set_status(
                            f"Machine group {mi + 1} — {len(blk_boxes)} token(s)"
                        )
                        return

        # Ungrouped glyph, nothing special — clear state and hint
        self._gsession.selected_indices.clear()
        self._gsession.inspected_group_idx = None
        self._render_canvas()
        self._set_status("Shift+click to add tokens to a selection.")

    # ── Drag motion: rubber-band rectangle ───────────────────────────────────

    def _on_drag_motion(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Draw a dashed rubber-band rect while the mouse button is held."""
        if self._drag_origin is None:
            return
        ox, oy = self._drag_origin
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)

        if not self._is_dragging:
            if abs(cx - ox) > _DRAG_THRESHOLD or abs(cy - oy) > _DRAG_THRESHOLD:
                self._is_dragging = True

        if self._is_dragging:
            self._canvas.delete("lasso")
            self._canvas.create_rectangle(
                ox,
                oy,
                cx,
                cy,
                outline="#FFD700",
                dash=(4, 4),
                width=2,
                tags="lasso",
            )

    # ── Button release: finalise click or lasso ───────────────────────────────

    def _on_button_release(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """On release: lasso-select if dragged, else treat as a plain click."""
        if self._drag_origin is None:
            return

        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)
        ox, oy = self._drag_origin

        if self._is_dragging:
            self._canvas.delete("lasso")
            shift_held = bool(event.state & 0x0001)
            self._lasso_select(
                min(ox, cx),
                min(oy, cy),
                max(ox, cx),
                max(oy, cy),
                add=shift_held,
            )
        else:
            # Short click — route to existing click logic
            shift_held = bool(
                self._drag_event.state & 0x0001
                if self._drag_event is not None
                else event.state & 0x0001
            )
            if shift_held:
                self._handle_shift_click(cx, cy, event)
            else:
                self._handle_plain_click(cx, cy, event)

        self._drag_origin = None
        self._drag_event = None
        self._is_dragging = False

    # ── Lasso selection helper ────────────────────────────────────────────────

    def _lasso_select(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        add: bool,
    ) -> None:
        """Select all glyphs overlapping canvas rect [x0,y0,x1,y1].

        Parameters
        ----------
        x0, y0, x1, y1:
            Bounding box in *canvas* pixel coordinates (already normalised so
            x0 <= x1 and y0 <= y1).
        add:
            If True (Shift held), add hits to the existing selection.
            If False, replace the selection.
        """
        pd = self._gsession.page_data
        if pd is None:
            return

        # Convert lasso corners to PDF point coordinates
        px0, py0 = self._canvas_to_pdf(x0, y0)
        px1, py1 = self._canvas_to_pdf(x1, y1)

        hits: set[int] = set()
        for i, box in enumerate(pd.boxes):
            # Include any box whose area overlaps the lasso rect
            if box.x0 < px1 and box.x1 > px0 and box.y0 < py1 and box.y1 > py0:
                hits.add(i)

        if not add:
            self._gsession.selected_indices.clear()
            self._gsession.inspected_group_idx = None

        self._gsession.selected_indices.update(hits)
        self._render_canvas()

        n = len(self._gsession.selected_indices)
        if n:
            self._set_status(
                f"{n} token(s) selected — right-click or click any to confirm group"
            )
        else:
            self._set_status("No tokens in selection area.")

    # ── Right-click: clear ────────────────────────────────────────────────────

    def _on_right_click(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Right-click: context-sensitive popup depending on current state.

        Priority:
        1. Token selection active → confirm-group popup.
        2. 2+ groups marked for meta-grouping → meta-group popup.
        3. Confirmed group inspected → confirm/mark popup.
        4. Machine group inspected → confirm machine group popup.
        5. Nothing → clear state.
        """
        if not self._gsession.active:
            return
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)

        if self._gsession.selected_indices:
            self._show_group_popup(event, cx, cy)
            return

        if len(self._gsession.selected_group_indices) >= 2:
            self._show_meta_group_popup(event)
            return

        idx = self._gsession.inspected_group_idx
        if idx is not None:
            if idx >= 0:
                self._show_confirmed_group_options_popup(event, idx)
            else:
                self._show_machine_group_popup(event, -(idx + 1))
            return

        # Nothing active — clear
        self._gsession.selected_group_indices.clear()
        self._gsession.inspected_group_idx = None
        self._render_canvas()
        self._set_status("Selection cleared.")

    # ── Ctrl+click: toggle a confirmed group into the meta-group selection ──────

    def _on_ctrl_click(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Ctrl+click: toggle a confirmed group in/out of the meta-group queue."""
        if not self._gsession.active:
            return
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)
        glyph_idx = self._hit_test_glyph(cx, cy)
        if glyph_idx is None:
            return
        gi = self._gsession.index_in_confirmed_group(glyph_idx)
        if gi is None:
            return
        sel_gi = self._gsession.selected_group_indices
        if gi in sel_gi:
            sel_gi.discard(gi)
        else:
            sel_gi.add(gi)
        self._render_canvas()
        n = len(sel_gi)
        if n >= 2:
            self._set_status(
                f"{n} groups marked (purple) — right-click to create meta-group"
            )
        elif n == 1:
            self._set_status("1 group marked — Ctrl+click another to add it")
        else:
            self._set_status("Group unmarked.")

    # ── Mouse wheel: zoom (Ctrl) or scroll (plain) ────────────────────────────

    def _on_mousewheel(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """MouseWheel: Ctrl held → zoom; plain → vertical scroll."""
        if event.state & 0x0004:  # Ctrl
            factor = 1.1 if event.delta > 0 else 0.9
            new_zoom = max(0.25, min(4.0, self._zoom * factor))
            if abs(new_zoom - self._zoom) < 0.001:
                return
            self._zoom = new_zoom
            self._update_zoom_label()
            self._render_canvas()
        else:
            self._canvas.yview_scroll(-1 * (event.delta // 120), "units")

    # ── Confirmed group options popup ──────────────────────────────────────

    def _show_confirmed_group_options_popup(
        self, event: tk.Event, gi: int  # type: ignore[type-arg]
    ) -> None:
        """Right-click on an inspected confirmed group: mark or unmark for meta-grouping."""
        popup = tk.Toplevel(self._canvas)
        popup.title("")
        popup.resizable(False, False)
        popup.overrideredirect(True)
        popup.geometry(f"+{event.x_root + 8}+{event.y_root + 8}")

        sel_gi = self._gsession.selected_group_indices
        is_marked = gi in sel_gi

        tk.Label(
            popup,
            text=f"Group G{gi + 1}",
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=4,
        ).pack()

        btn_frame = tk.Frame(popup, padx=6, pady=(0, 6))
        btn_frame.pack()

        def _close() -> None:
            popup.destroy()

        def _toggle_mark() -> None:
            _close()
            if is_marked:
                sel_gi.discard(gi)
                self._set_status(f"Group G{gi + 1} unmarked.")
            else:
                sel_gi.add(gi)
                n = len(sel_gi)
                if n >= 2:
                    self._set_status(
                        f"{n} groups marked — right-click to create meta-group"
                    )
                else:
                    self._set_status("1 group marked — Ctrl+click another to add it")
            self._render_canvas()

        label = "Unmark" if is_marked else "Mark for Meta-Group"
        tk.Button(
            btn_frame,
            text=label,
            command=_toggle_mark,
            bg="#AA44CC" if not is_marked else "#888888",
            fg="white",
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left", padx=(0, 4))

        if len(sel_gi) >= 1 and not is_marked:

            def _mark_and_create() -> None:
                _close()
                sel_gi.add(gi)
                self._confirm_meta_group()

            tk.Button(
                btn_frame,
                text=f"Mark + Create Meta-Group ({len(sel_gi) + 1} groups)",
                command=_mark_and_create,
                bg="#AA44CC",
                fg="white",
            ).pack(side="left", padx=(0, 4))

        tk.Button(btn_frame, text="Cancel", command=_close).pack(side="left")
        popup.bind("<Escape>", lambda _e: _close())
        popup.grab_set()
        popup.focus_set()

    # ── Machine group popup ─────────────────────────────────────────────────

    def _show_machine_group_popup(
        self, event: tk.Event, mi: int  # type: ignore[type-arg]
    ) -> None:
        """Right-click on an inspected machine group: offer one-click confirmation."""
        pd = self._gsession.page_data
        if pd is None or mi >= len(pd.machine_groups):
            return

        blk = pd.machine_groups[mi]
        blk_boxes = blk.get_all_boxes()
        n = len(blk_boxes)

        popup = tk.Toplevel(self._canvas)
        popup.title("")
        popup.resizable(False, False)
        popup.overrideredirect(True)
        popup.geometry(f"+{event.x_root + 8}+{event.y_root + 8}")

        tk.Label(
            popup,
            text=f"Machine group {mi + 1} — {n} token{'s' if n != 1 else ''}",
            font=("Segoe UI", 9),
            padx=8,
            pady=4,
        ).pack()

        def _close() -> None:
            popup.destroy()

        def _confirm() -> None:
            _close()
            self._confirm_machine_group(mi)

        btn_frame = tk.Frame(popup, padx=6, pady=(0, 6))
        btn_frame.pack()
        tk.Button(
            btn_frame,
            text="Confirm Machine Group ✓",
            command=_confirm,
            bg="#4488CC",
            fg="white",
            activebackground="#2266aa",
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left", padx=(0, 4))
        tk.Button(btn_frame, text="Cancel", command=_close).pack(side="left")
        popup.bind("<Escape>", lambda _e: _close())
        popup.grab_set()
        popup.focus_set()

    def _confirm_machine_group(self, mi: int) -> None:
        """Accept a machine group as-is and write it to the DB."""
        pd = self._gsession.page_data
        if pd is None or mi >= len(pd.machine_groups):
            return

        blk = pd.machine_groups[mi]
        blk_boxes = blk.get_all_boxes()

        # Map BlockCluster boxes back to pd.boxes indices by identity
        id_to_idx = {id(b): i for i, b in enumerate(pd.boxes)}
        indices = [id_to_idx[id(b)] for b in blk_boxes if id(b) in id_to_idx]
        if not indices:
            self._set_status(f"Machine group {mi + 1} has no matching tokens.")
            return

        grp_boxes = [pd.boxes[i] for i in indices]
        bbox = (
            min(b.x0 for b in grp_boxes),
            min(b.y0 for b in grp_boxes),
            max(b.x1 for b in grp_boxes),
            max(b.y1 for b in grp_boxes),
        )
        example_id = self._write_group_to_db(grp_boxes, bbox, pd)

        from .grouper_state import ConfirmedGroup

        self._gsession.confirmed_groups.append(
            ConfirmedGroup(indices=indices, bbox=bbox, example_id=example_id)
        )
        self._gsession.inspected_group_idx = None
        self._render_canvas()
        n_total = len(self._gsession.confirmed_groups)
        self._set_status(
            f"Machine group {mi + 1} confirmed as G{n_total} ({len(indices)} tokens)."
        )
        log.info(
            "grouper: machine group %d confirmed — %d tokens, example_id=%s",
            mi,
            len(indices),
            example_id or "(none)",
        )

    # ── Meta-group popup + confirm ────────────────────────────────────────────

    def _show_meta_group_popup(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Popup to create a meta-group from all currently marked groups."""
        sel_gis = sorted(self._gsession.selected_group_indices)
        n = len(sel_gis)
        if n < 2:
            return

        # Count total tokens
        total_tokens = sum(
            len(self._gsession.confirmed_groups[gi].indices)
            for gi in sel_gis
            if gi < len(self._gsession.confirmed_groups)
        )

        popup = tk.Toplevel(self._canvas)
        popup.title("")
        popup.resizable(False, False)
        popup.overrideredirect(True)
        popup.geometry(f"+{event.x_root + 8}+{event.y_root + 8}")

        names = ", ".join(f"G{gi + 1}" for gi in sel_gis)
        tk.Label(
            popup,
            text=f"Group {names} → Meta-Group ({total_tokens} tokens)",
            font=("Segoe UI", 9),
            padx=8,
            pady=4,
        ).pack()

        def _close() -> None:
            popup.destroy()

        def _confirm() -> None:
            _close()
            self._confirm_meta_group()

        btn_frame = tk.Frame(popup, padx=6, pady=(0, 6))
        btn_frame.pack()
        tk.Button(
            btn_frame,
            text="Create Meta-Group ✓",
            command=_confirm,
            bg="#AA44CC",
            fg="white",
            activebackground="#882299",
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left", padx=(0, 4))

        def _clear_marks() -> None:
            _close()
            self._gsession.selected_group_indices.clear()
            self._render_canvas()
            self._set_status("Group marks cleared.")

        tk.Button(btn_frame, text="Clear Marks", command=_clear_marks).pack(
            side="left", padx=(0, 4)
        )
        tk.Button(btn_frame, text="Cancel", command=_close).pack(side="left")
        popup.bind("<Escape>", lambda _e: _close())
        popup.grab_set()
        popup.focus_set()

    def _confirm_meta_group(self) -> None:
        """Collapse all marked groups into one new meta-group and persist it."""
        sel_gis = sorted(self._gsession.selected_group_indices, reverse=True)
        pd = self._gsession.page_data
        if pd is None or len(sel_gis) < 2:
            return

        # Gather all token indices (de-duplicate, preserve order)
        seen: set[int] = set()
        all_indices: List[int] = []
        for gi in sorted(sel_gis):  # ascending so order is spatial
            if gi < len(self._gsession.confirmed_groups):
                for idx in self._gsession.confirmed_groups[gi].indices:
                    if idx not in seen:
                        seen.add(idx)
                        all_indices.append(idx)

        if not all_indices:
            return

        # Remove constituent groups (descending to keep earlier indices valid)
        for gi in sel_gis:
            if gi < len(self._gsession.confirmed_groups):
                self._gsession.confirmed_groups.pop(gi)

        grp_boxes = [pd.boxes[i] for i in all_indices if i < len(pd.boxes)]
        bbox = (
            min(b.x0 for b in grp_boxes),
            min(b.y0 for b in grp_boxes),
            max(b.x1 for b in grp_boxes),
            max(b.y1 for b in grp_boxes),
        )
        example_id = self._write_group_to_db(grp_boxes, bbox, pd)

        from .grouper_state import ConfirmedGroup

        self._gsession.confirmed_groups.append(
            ConfirmedGroup(indices=all_indices, bbox=bbox, example_id=example_id)
        )
        self._gsession.selected_group_indices.clear()
        self._gsession.inspected_group_idx = None
        self._render_canvas()

        n_grps = len(self._gsession.confirmed_groups)
        self._set_status(
            f"Meta-group created as G{n_grps} — {len(all_indices)} tokens "
            f"from {len(sel_gis)} groups."
        )
        log.info(
            "grouper: meta-group confirmed as G%d — %d tokens from %d sub-groups",
            n_grps,
            len(all_indices),
            len(sel_gis),
        )

    # ── Group popup ───────────────────────────────────────────────────────────

    def _show_group_popup(
        self, event: tk.Event, cx: float, cy: float  # type: ignore[type-arg]
    ) -> None:
        """Show a small Toplevel near the cursor for confirming a group."""
        sel = self._gsession.selected_indices
        if not sel:
            return

        popup = tk.Toplevel(self._canvas)
        popup.title("")
        popup.resizable(False, False)
        popup.overrideredirect(True)

        # Position near cursor
        rx = event.x_root + 8
        ry = event.y_root + 8
        popup.geometry(f"+{rx}+{ry}")

        n = len(sel)
        tk.Label(
            popup,
            text=f"{n} token{'s' if n != 1 else ''} selected",
            font=("Segoe UI", 9),
            padx=8,
            pady=4,
        ).pack()

        def _confirm() -> None:
            popup.destroy()
            self._confirm_group()

        def _cancel() -> None:
            popup.destroy()

        btn_frame = tk.Frame(popup)
        btn_frame.pack(padx=6, pady=(0, 6))
        tk.Button(
            btn_frame,
            text="Confirm Group ✓",
            command=_confirm,
            bg="#22AA44",
            fg="white",
            activebackground="#1a8834",
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left", padx=(0, 4))
        tk.Button(btn_frame, text="Cancel", command=_cancel).pack(side="left")

        popup.bind("<Escape>", lambda _e: _cancel())
        popup.grab_set()
        popup.focus_set()

    # ── Confirm a group (Learn + Edit) ────────────────────────────────────────

    def _confirm_group(self) -> None:
        """Finalise the current selection as a confirmed group and persist it."""
        sel = list(self._gsession.selected_indices)
        if not sel:
            return

        pd = self._gsession.page_data
        if pd is None:
            return

        grp_boxes = [pd.boxes[i] for i in sel if i < len(pd.boxes)]
        if not grp_boxes:
            return

        # Compute bounding box
        bx0 = min(b.x0 for b in grp_boxes)
        by0 = min(b.y0 for b in grp_boxes)
        bx1 = max(b.x1 for b in grp_boxes)
        by1 = max(b.y1 for b in grp_boxes)

        # Write to DB
        example_id = self._write_group_to_db(grp_boxes, (bx0, by0, bx1, by1), pd)

        # Add to session state
        from .grouper_state import ConfirmedGroup

        cg = ConfirmedGroup(
            indices=sel,
            bbox=(bx0, by0, bx1, by1),
            example_id=example_id,
        )
        self._gsession.confirmed_groups.append(cg)
        self._gsession.selected_indices.clear()
        self._render_canvas()

        n = len(self._gsession.confirmed_groups)
        self._set_status(
            f"Group {n} confirmed ({len(sel)} tokens). " f"Total on page: {n}"
        )
        log.info(
            "grouper: confirmed group %d — %d tokens, example_id=%s",
            n,
            len(sel),
            example_id or "(none)",
        )

    def _write_group_to_db(
        self,
        grp_boxes: list,
        bbox: tuple,
        pd: Any,
    ) -> str:
        """Compute geometry and write snapshot + correction rows.  Returns example_id."""
        try:
            from plancheck.grouping.snapshot_geometry import (
                compute_group_geometry,
                compute_normalized_geometry,
                compute_page_context,
            )

            group_geom = compute_group_geometry(grp_boxes, pd.page_w, pd.page_h)
            norm_geom = compute_normalized_geometry(grp_boxes)
            page_ctx = compute_page_context(
                grp_boxes,
                self._gsession.current_page,
                pd.page_w,
                pd.page_h,
                pd.machine_groups,
            )

            boxes_dicts = [b.to_dict() for b in grp_boxes]

            example_id = self._store.save_group_snapshot(
                session_id=self._gsession.session_id,
                pdf_filename=(
                    self._gsession.pdf_path.name if self._gsession.pdf_path else ""
                ),
                page_number=self._gsession.current_page,
                boxes_dicts=boxes_dicts,
                group_geometry=group_geom,
                normalized_geom=norm_geom,
                page_context=page_ctx,
            )

            self._store.save_group_correction(
                session_id=self._gsession.session_id,
                doc_id=self._gsession.doc_id,
                page_number=self._gsession.current_page,
                signal="is_group",
                machine_grouping=None,
                corrected_grouping={"bbox": list(bbox), "n_boxes": len(grp_boxes)},
                example_id=example_id,
            )
            return example_id

        except Exception:  # noqa: BLE001
            log.error("grouper: failed to persist group to DB", exc_info=True)
            return ""

    # ── Edit mode — shift-click gesture state machine ─────────────────────────

    def _handle_edit_shift_click(
        self, glyph_idx: int, event: tk.Event, cx: float, cy: float  # type: ignore[type-arg]
    ) -> None:
        """Handle Shift+click in Edit mode with add/remove/split/merge logic."""
        confirmed_gi = self._gsession.index_in_confirmed_group(glyph_idx)
        inspected_gi = self._gsession.inspected_group_idx
        sel = self._gsession.selected_indices

        # Case 1: glyph is ungrouped
        if confirmed_gi is None:
            sel.add(glyph_idx)
            self._render_canvas()
            self._set_status(f"{len(sel)} token(s) selected")
            return

        # Case 2: glyph is in *same* group as inspected — select for removal/split
        if inspected_gi is not None and inspected_gi == confirmed_gi:
            if glyph_idx in sel:
                sel.discard(glyph_idx)
            else:
                sel.add(glyph_idx)
            self._render_canvas()
            if sel:
                self._show_edit_popup(event, confirmed_gi, mode="remove_split")
            return

        # Case 3: glyph is in a *different* group from the one being inspected → merge
        if inspected_gi is not None and inspected_gi != confirmed_gi:
            self._show_edit_popup(
                event, confirmed_gi, mode="merge", other_gi=inspected_gi
            )
            return

        # Case 4: nothing inspected yet — inspect the group this glyph belongs to
        self._gsession.inspected_group_idx = confirmed_gi
        self._gsession.selected_indices.clear()
        self._render_canvas()
        n = len(self._gsession.confirmed_groups[confirmed_gi].indices)
        self._set_status(f"Group {confirmed_gi + 1} inspected ({n} tokens)")

    # ── Edit popup ────────────────────────────────────────────────────────────

    def _show_edit_popup(
        self,
        event: tk.Event,  # type: ignore[type-arg]
        group_idx: int,
        mode: str,
        other_gi: Optional[int] = None,
    ) -> None:
        """Show context popup for add/remove/split/merge operations."""
        popup = tk.Toplevel(self._canvas)
        popup.title("")
        popup.resizable(False, False)
        popup.overrideredirect(True)
        popup.geometry(f"+{event.x_root + 8}+{event.y_root + 8}")

        def _close() -> None:
            popup.destroy()

        btn_frame = tk.Frame(popup, padx=6, pady=6)
        btn_frame.pack()

        if mode == "remove_split":
            sel = list(self._gsession.selected_indices)
            tk.Label(
                btn_frame, text=f"{len(sel)} token(s) selected in group {group_idx + 1}"
            ).pack(pady=(0, 4))

            def _remove() -> None:
                _close()
                self._edit_remove_tokens(
                    group_idx, list(self._gsession.selected_indices)
                )

            def _split() -> None:
                _close()
                self._edit_split_group(group_idx, list(self._gsession.selected_indices))

            tk.Button(btn_frame, text="Remove from Group", command=_remove).pack(
                fill="x", pady=2
            )
            tk.Button(btn_frame, text="Split into New Group", command=_split).pack(
                fill="x", pady=2
            )

        elif mode == "merge" and other_gi is not None:
            gi_a, gi_b = min(group_idx, other_gi), max(group_idx, other_gi)
            tk.Label(
                btn_frame, text=f"Merge group {gi_a + 1} and group {gi_b + 1}?"
            ).pack(pady=(0, 4))

            def _merge() -> None:
                _close()
                self._edit_merge_groups(gi_a, gi_b)

            tk.Button(
                btn_frame, text="Merge Groups", command=_merge, bg="#4488CC", fg="white"
            ).pack(fill="x", pady=2)

        tk.Button(btn_frame, text="Cancel", command=_close).pack(fill="x", pady=(4, 0))
        popup.bind("<Escape>", lambda _e: _close())
        popup.grab_set()
        popup.focus_set()

    # ── Edit operations ───────────────────────────────────────────────────────

    def _edit_remove_tokens(self, group_idx: int, indices_to_remove: List[int]) -> None:
        """Remove selected tokens from an existing confirmed group."""
        if group_idx >= len(self._gsession.confirmed_groups):
            return
        cg = self._gsession.confirmed_groups[group_idx]
        before = list(cg.indices)
        cg.indices = [i for i in cg.indices if i not in indices_to_remove]

        # Recompute bbox
        pd = self._gsession.page_data
        if pd and cg.indices:
            remaining = [pd.boxes[i] for i in cg.indices if i < len(pd.boxes)]
            cg.bbox = (
                min(b.x0 for b in remaining),
                min(b.y0 for b in remaining),
                max(b.x1 for b in remaining),
                max(b.y1 for b in remaining),
            )

        self._gsession.selected_indices.clear()
        self._gsession.inspected_group_idx = None

        delta = {
            "op": "remove",
            "removed": indices_to_remove,
            "before": before,
            "after": list(cg.indices),
        }
        self._persist_edit_correction(delta)
        self._render_canvas()
        self._set_status(
            f"Removed {len(indices_to_remove)} token(s) from group {group_idx + 1}."
        )

    def _edit_split_group(self, group_idx: int, split_indices: List[int]) -> None:
        """Split selected tokens out of a group into a new group."""
        if group_idx >= len(self._gsession.confirmed_groups):
            return
        cg = self._gsession.confirmed_groups[group_idx]
        before = list(cg.indices)

        remain = [i for i in cg.indices if i not in split_indices]
        if not remain:
            self._set_status("Cannot split: no tokens would remain in original group.")
            return

        pd = self._gsession.page_data
        if pd is None:
            return

        # Update original group
        cg.indices = remain
        rem_boxes = [pd.boxes[i] for i in remain if i < len(pd.boxes)]
        cg.bbox = (
            min(b.x0 for b in rem_boxes),
            min(b.y0 for b in rem_boxes),
            max(b.x1 for b in rem_boxes),
            max(b.y1 for b in rem_boxes),
        )

        # Create new group from split tokens
        spl_boxes = [pd.boxes[i] for i in split_indices if i < len(pd.boxes)]
        new_bbox = (
            min(b.x0 for b in spl_boxes),
            min(b.y0 for b in spl_boxes),
            max(b.x1 for b in spl_boxes),
            max(b.y1 for b in spl_boxes),
        )

        # Persist snapshot + correction for new group
        example_id = self._write_group_to_db(spl_boxes, new_bbox, pd)

        from .grouper_state import ConfirmedGroup

        new_cg = ConfirmedGroup(
            indices=split_indices, bbox=new_bbox, example_id=example_id
        )
        self._gsession.confirmed_groups.append(new_cg)

        self._gsession.selected_indices.clear()
        self._gsession.inspected_group_idx = None

        delta = {
            "op": "split",
            "from_group": group_idx,
            "split_indices": split_indices,
            "before": before,
        }
        self._persist_edit_correction(delta)
        self._render_canvas()
        self._set_status(
            f"Split {len(split_indices)} token(s) into new group "
            f"{len(self._gsession.confirmed_groups)}."
        )

    def _edit_merge_groups(self, gi_a: int, gi_b: int) -> None:
        """Merge two confirmed groups (gi_b absorbed into gi_a)."""
        cgs = self._gsession.confirmed_groups
        if gi_a >= len(cgs) or gi_b >= len(cgs):
            return

        before_a = list(cgs[gi_a].indices)
        before_b = list(cgs[gi_b].indices)

        merged_indices = sorted(set(cgs[gi_a].indices + cgs[gi_b].indices))
        pd = self._gsession.page_data
        if pd is None:
            return

        merged_boxes = [pd.boxes[i] for i in merged_indices if i < len(pd.boxes)]
        new_bbox = (
            min(b.x0 for b in merged_boxes),
            min(b.y0 for b in merged_boxes),
            max(b.x1 for b in merged_boxes),
            max(b.y1 for b in merged_boxes),
        )
        cgs[gi_a].indices = merged_indices
        cgs[gi_a].bbox = new_bbox

        # Remove gi_b (higher index is always gi_b after sorting above)
        cgs.pop(gi_b)

        self._gsession.selected_indices.clear()
        self._gsession.inspected_group_idx = None

        delta = {
            "op": "merge",
            "groups": [gi_a, gi_b],
            "before_a": before_a,
            "before_b": before_b,
        }
        self._persist_edit_correction(delta)
        self._render_canvas()
        self._set_status(f"Merged groups {gi_a + 1} and {gi_b + 1} → group {gi_a + 1}.")

    def _persist_edit_correction(self, delta: dict) -> None:
        """Write a ``signal='edit'`` group correction row."""
        try:
            self._store.save_group_correction(
                session_id=self._gsession.session_id,
                doc_id=self._gsession.doc_id,
                page_number=self._gsession.current_page,
                signal="edit",
                delta=delta,
            )
        except Exception:  # noqa: BLE001
            log.error("grouper: failed to persist edit correction", exc_info=True)
