"""Inspector button actions: accept, relabel, delete, dismiss, rescan.

These are not mouse-driven tools — they're triggered by buttons and
keyboard shortcuts.  They read from the current selection and call
through to the :class:`CorrectionStore`.
"""

from __future__ import annotations

import logging
from tkinter import messagebox
from typing import TYPE_CHECKING, Any

from plancheck.corrections.features import featurize_region

if TYPE_CHECKING:
    from ..annotation_state import CanvasBox

log = logging.getLogger(__name__)


class BoxOperations:
    """Inspector actions extracted from EventHandlerMixin."""

    def __init__(self, tab: Any) -> None:
        self._tab = tab

    # ── Accept ─────────────────────────────────────────────────────

    def accept(self) -> None:
        tab = self._tab
        targets = list(tab._multi_selected)
        if tab._selected_box and tab._selected_box not in targets:
            targets.append(tab._selected_box)
        if not targets or not tab._doc_id:
            tab._status.configure(text="No box selected")
            return

        for cbox in targets:
            tab._push_undo("accept", cbox)
            tab._store.accept_detection(cbox.detection_id, tab._doc_id, tab._page)
            cbox.corrected = True
            tab._session_count += 1
            tab._draw_box(cbox)

        tab._update_session_label()
        tab._clear_multi_select()
        tab._status.configure(text=f"Accepted {len(targets)} box(es)")

    # ── Relabel ────────────────────────────────────────────────────

    def relabel(self) -> None:
        tab = self._tab
        targets = list(tab._multi_selected)
        if tab._selected_box and tab._selected_box not in targets:
            targets.append(tab._selected_box)
        if not targets or not tab._doc_id:
            tab._status.configure(text="No box selected")
            return

        new_label = tab._normalize_element_type_name(tab._type_var.get())
        if not new_label:
            return

        if new_label not in tab.ELEMENT_TYPES:
            tab._status.configure(text=f"Unknown element type: {new_label}")
            return

        count = 0
        for cbox in targets:
            if new_label == cbox.element_type:
                tab._push_undo("accept", cbox)
                tab._store.accept_detection(
                    cbox.detection_id, tab._doc_id, tab._page
                )
                cbox.corrected = True
                tab._session_count += 1
                tab._draw_box(cbox)
                continue

            tab._push_undo("relabel", cbox, extra={"old_label": cbox.element_type})
            tab._store.save_correction(
                doc_id=tab._doc_id,
                page=tab._page,
                correction_type="relabel",
                corrected_label=new_label,
                corrected_bbox=cbox.pdf_bbox,
                detection_id=cbox.detection_id,
                original_label=cbox.element_type,
                original_bbox=cbox.pdf_bbox,
                session_id=tab._session_id,
            )
            cbox.element_type = new_label
            cbox.corrected = True
            tab._session_count += 1
            tab._draw_box(cbox)
            count += 1

        tab._update_session_label()
        tab._update_page_summary()
        tab._clear_multi_select()
        tab._status.configure(text=f"Relabelled {count} box(es) \u2192 {new_label}")

    # ── Delete (reject) ───────────────────────────────────────────

    def delete(self) -> None:
        tab = self._tab
        targets = list(tab._multi_selected)
        if tab._selected_box and tab._selected_box not in targets:
            targets.append(tab._selected_box)
        if not targets or not tab._doc_id:
            tab._status.configure(text="No box selected")
            return

        n = len(targets)
        msg = (
            "Mark this detection as a false positive?"
            if n == 1
            else f"Reject {n} selected detections?"
        )
        if not messagebox.askyesno("Reject Detection", msg):
            return

        for cbox in targets:
            tab._push_undo("delete", cbox)
            tab._store.save_correction(
                doc_id=tab._doc_id,
                page=tab._page,
                correction_type="delete",
                corrected_label=cbox.element_type,
                corrected_bbox=cbox.pdf_bbox,
                detection_id=cbox.detection_id,
                original_label=cbox.element_type,
                original_bbox=cbox.pdf_bbox,
                session_id=tab._session_id,
            )

            if cbox.rect_id:
                tab._canvas.delete(cbox.rect_id)
            if cbox.label_id:
                tab._canvas.delete(cbox.label_id)
            if cbox.conf_dot_id:
                tab._canvas.delete(cbox.conf_dot_id)
            for hid in cbox.handle_ids:
                tab._canvas.delete(hid)
            if cbox in tab._canvas_boxes:
                tab._canvas_boxes.remove(cbox)
            tab._session_count += 1

        tab._selected_box = None
        tab._multi_selected.clear()
        tab._deselect()
        tab._update_multi_label()
        tab._update_session_label()
        tab._update_page_summary()
        tab._status.configure(text=f"Rejected {n} detection(s)")

    # ── Dismiss ────────────────────────────────────────────────────

    def dismiss(self) -> None:
        tab = self._tab
        targets = list(tab._multi_selected)
        if tab._selected_box and tab._selected_box not in targets:
            targets.append(tab._selected_box)
        if not targets or not tab._doc_id:
            tab._status.configure(text="No box selected")
            return

        for cbox in targets:
            tab._push_undo("dismiss", cbox)
            tab._store.dismiss_detection(
                detection_id=cbox.detection_id,
                doc_id=tab._doc_id,
                page=tab._page,
                session_id=tab._session_id,
            )

            if cbox.rect_id:
                tab._canvas.delete(cbox.rect_id)
            if cbox.label_id:
                tab._canvas.delete(cbox.label_id)
            if cbox.conf_dot_id:
                tab._canvas.delete(cbox.conf_dot_id)
            for hid in cbox.handle_ids:
                tab._canvas.delete(hid)
            if cbox in tab._canvas_boxes:
                tab._canvas_boxes.remove(cbox)

        n = len(targets)
        tab._selected_box = None
        tab._multi_selected.clear()
        tab._deselect()
        tab._update_multi_label()
        tab._update_page_summary()
        tab._status.configure(text=f"Dismissed {n} detection(s)")

    # ── Text re-extraction ─────────────────────────────────────────

    def auto_refresh_text(self, cbox: CanvasBox) -> None:
        """Re-extract text from PDF after move/reshape."""
        tab = self._tab
        if not tab._pdf_path:
            return
        try:
            new_text = tab._extract_text_for_box(cbox)
        except Exception:
            return

        cbox.text_content = new_text

        features = featurize_region(
            cbox.element_type, cbox.pdf_bbox, None, 2448.0, 1584.0
        )
        cbox.features = features

        tab._store.update_detection_text_and_features(
            cbox.detection_id, new_text, features
        )

        if tab._selected_box is cbox:
            tab._insp_text.config(state="normal")
            tab._insp_text.delete("1.0", "end")
            tab._insp_text.insert("1.0", new_text)
            tab._insp_text.config(state="disabled")

    def rescan_text(self) -> None:
        """Re-extract text from PDF under the selected box."""
        tab = self._tab
        if not tab._selected_box:
            tab._status.configure(text="No box selected")
            return
        if not tab._pdf_path:
            tab._status.configure(text="No PDF loaded")
            return

        cbox = tab._selected_box
        try:
            new_text = tab._extract_text_for_box(cbox)
        except Exception as exc:
            tab._status.configure(text=f"Rescan failed: {exc}")
            return

        cbox.text_content = new_text

        features = featurize_region(
            cbox.element_type, cbox.pdf_bbox, None, 2448.0, 1584.0
        )
        cbox.features = features
        tab._store.update_detection_text_and_features(
            cbox.detection_id, new_text, features
        )

        tab._insp_text.config(state="normal")
        tab._insp_text.delete("1.0", "end")
        tab._insp_text.insert("1.0", new_text)
        tab._insp_text.config(state="disabled")

        mode = "polygon" if cbox.polygon else "rect"
        n_chars = len(new_text)
        tab._status.configure(
            text=f"Rescanned text for {cbox.element_type} ({mode}) \u2014 {n_chars} chars"
        )
