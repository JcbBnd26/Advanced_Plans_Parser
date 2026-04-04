"""Undo / redo manager extracted from EventHandlerMixin.

Owns the undo and redo stacks and the logic for reverting each action
type (relabel, reshape, delete, dismiss, accept, merge).
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..annotation_state import CanvasBox

log = logging.getLogger(__name__)


class UndoManager:
    """Manages undo/redo stacks and applies reversals.

    Parameters
    ----------
    tab : AnnotationTab
        The host tab — used for canvas-box access and drawing helpers.
    """

    def __init__(self, tab: Any) -> None:
        self._tab = tab

    # ── Stack access (stacks live on the tab for backward compat) ──

    @property
    def _undo_stack(self) -> list[dict]:
        return self._tab._undo_stack

    @property
    def _redo_stack(self) -> list[dict]:
        return self._tab._redo_stack

    # ── Push ───────────────────────────────────────────────────────

    def push(
        self,
        action: str,
        cbox: CanvasBox,
        *,
        extra: dict | None = None,
    ) -> None:
        """Push an undo record and clear the redo stack."""
        record: dict = {
            "action": action,
            "detection_id": cbox.detection_id,
            "element_type": cbox.element_type,
            "pdf_bbox": cbox.pdf_bbox,
            "confidence": cbox.confidence,
            "corrected": cbox.corrected,
        }
        if extra:
            record.update(extra)
        self._undo_stack.append(record)
        self._redo_stack.clear()

    # ── Undo ───────────────────────────────────────────────────────

    def undo(self) -> None:
        """Undo the last correction (visual only — DB is append-only)."""
        tab = self._tab
        if not self._undo_stack:
            tab._status.configure(text="Nothing to undo")
            return

        rec = self._undo_stack.pop()
        self._redo_stack.append(rec)
        action = rec["action"]

        # Find the affected canvas box
        target = None
        for cb in tab._canvas_boxes:
            if cb.detection_id == rec["detection_id"]:
                target = cb
                break

        if action == "relabel" and target:
            old_label = rec.get("old_label", target.element_type)
            target.element_type = old_label
            target.corrected = rec.get("corrected", False)
            tab._draw_box(target)
            tab._status.configure(text=f"Undo relabel \u2192 {old_label}")
        elif action == "reshape" and target:
            orig = rec.get("orig_bbox")
            if orig:
                target.pdf_bbox = orig
                target.corrected = rec.get("corrected", False)
                if "orig_polygon" in rec:
                    target.polygon = copy.deepcopy(rec["orig_polygon"])
                tab._draw_box(target)
            tab._status.configure(text="Undo reshape")
        elif action == "delete":
            already_exists = any(
                cb.detection_id == rec["detection_id"] for cb in tab._canvas_boxes
            )
            if already_exists:
                tab._status.configure(text="Undo reject (box already restored)")
            else:
                from ..annotation_state import CanvasBox as CBox

                cbox = CBox(
                    detection_id=rec["detection_id"],
                    element_type=rec["element_type"],
                    confidence=rec.get("confidence"),
                    text_content="",
                    features={},
                    pdf_bbox=rec["pdf_bbox"],
                    corrected=rec.get("corrected", False),
                )
                tab._canvas_boxes.append(cbox)
                tab._draw_box(cbox)
                tab._status.configure(text="Undo reject")
        elif action == "dismiss":
            tab._store.undismiss_detection(rec["detection_id"])
            already_exists = any(
                cb.detection_id == rec["detection_id"] for cb in tab._canvas_boxes
            )
            if already_exists:
                tab._status.configure(text="Undo dismiss (box already restored)")
            else:
                from ..annotation_state import CanvasBox as CBox

                cbox = CBox(
                    detection_id=rec["detection_id"],
                    element_type=rec["element_type"],
                    confidence=rec.get("confidence"),
                    text_content="",
                    features={},
                    pdf_bbox=rec["pdf_bbox"],
                    corrected=rec.get("corrected", False),
                )
                tab._canvas_boxes.append(cbox)
                tab._draw_box(cbox)
                tab._status.configure(text="Undo dismiss")
        elif action == "accept" and target:
            target.corrected = rec.get("corrected", False)
            tab._draw_box(target)
            tab._status.configure(text="Undo accept")
        elif action == "merge" and target:
            merged_boxes = rec.get("merged_boxes", [])
            if target.rect_id:
                tab._canvas.delete(target.rect_id)
            if target.label_id:
                tab._canvas.delete(target.label_id)
            for hid in target.handle_ids:
                tab._canvas.delete(hid)
            tab._canvas_boxes.remove(target)
            from ..annotation_state import CanvasBox as CBox

            for mb in merged_boxes:
                cbox = CBox(
                    detection_id=mb["detection_id"],
                    element_type=mb["element_type"],
                    confidence=mb.get("confidence"),
                    text_content=mb.get("text_content", ""),
                    features={},
                    pdf_bbox=mb["pdf_bbox"],
                    polygon=mb.get("polygon"),
                    corrected=mb.get("corrected", False),
                )
                tab._canvas_boxes.append(cbox)
                tab._draw_box(cbox)
            tab._status.configure(
                text=f"Undo merge \u2192 restored {len(merged_boxes)} boxes"
            )
        else:
            tab._status.configure(text="Undo (no visual change)")
        tab._update_page_summary()

    # ── Redo ───────────────────────────────────────────────────────

    def redo(self) -> None:
        """Redo the last undone action."""
        tab = self._tab
        if not self._redo_stack:
            tab._status.configure(text="Nothing to redo")
            return

        rec = self._redo_stack.pop()
        self._undo_stack.append(rec)
        action = rec["action"]

        target = None
        for cb in tab._canvas_boxes:
            if cb.detection_id == rec["detection_id"]:
                target = cb
                break

        if action == "relabel" and target:
            target.element_type = rec["element_type"]
            target.corrected = True
            tab._draw_box(target)
            tab._status.configure(text=f"Redo relabel \u2192 {rec['element_type']}")
        elif action == "reshape" and target:
            target.pdf_bbox = rec["pdf_bbox"]
            target.corrected = True
            if "polygon" in rec:
                target.polygon = copy.deepcopy(rec["polygon"])
            elif "orig_polygon" in rec:
                target.polygon = None
            tab._draw_box(target)
            tab._status.configure(text="Redo reshape")
        elif action == "delete" and target:
            if target.rect_id:
                tab._canvas.delete(target.rect_id)
            if target.label_id:
                tab._canvas.delete(target.label_id)
            for hid in target.handle_ids:
                tab._canvas.delete(hid)
            tab._canvas_boxes.remove(target)
            if tab._selected_box is target:
                tab._deselect()
            tab._status.configure(text="Redo reject")
        elif action == "dismiss" and target:
            tab._store.dismiss_detection(
                detection_id=target.detection_id,
                doc_id=tab._doc_id,
                page=tab._page,
                session_id=tab._session_id,
            )
            if target.rect_id:
                tab._canvas.delete(target.rect_id)
            if target.label_id:
                tab._canvas.delete(target.label_id)
            if target.conf_dot_id:
                tab._canvas.delete(target.conf_dot_id)
            for hid in target.handle_ids:
                tab._canvas.delete(hid)
            tab._canvas_boxes.remove(target)
            if tab._selected_box is target:
                tab._deselect()
            tab._status.configure(text="Redo dismiss")
        elif action == "accept" and target:
            target.corrected = True
            tab._draw_box(target)
            tab._status.configure(text="Redo accept")
        else:
            tab._status.configure(text="Redo (no visual change)")
        tab._update_page_summary()
