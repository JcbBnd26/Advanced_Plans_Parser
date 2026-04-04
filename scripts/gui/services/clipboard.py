"""Clipboard service for box copy/paste."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from plancheck.corrections.features import featurize_region
from plancheck.ingest.ingest import extract_text_in_bbox

if TYPE_CHECKING:
    from ..annotation_state import CanvasBox

log = logging.getLogger(__name__)


class Clipboard:
    """Manages the internal box clipboard (copy + paste)."""

    def __init__(self, tab: Any) -> None:
        self._tab = tab

    @property
    def _template(self) -> dict | None:
        return self._tab._copied_box_template

    @_template.setter
    def _template(self, value: dict | None) -> None:
        self._tab._copied_box_template = value

    @property
    def has_content(self) -> bool:
        return self._template is not None

    def copy_box(self, cbox: CanvasBox) -> None:
        """Copy box dimensions and type to the internal clipboard."""
        x0, y0, x1, y1 = cbox.pdf_bbox
        self._template = {
            "element_type": cbox.element_type,
            "width": x1 - x0,
            "height": y1 - y0,
        }
        self._tab._status.configure(
            text=f"Copied {cbox.element_type} box ({x1 - x0:.0f}\u00d7{y1 - y0:.0f} pt)"
        )

    def paste_box(self, pdf_x: float, pdf_y: float) -> None:
        """Paste a copied box centred at a PDF-space location."""
        tab = self._tab
        if not self._template or not tab._doc_id:
            return

        w = self._template["width"]
        h = self._template["height"]
        chosen_type = self._template["element_type"]

        x0 = pdf_x - w / 2
        y0 = pdf_y - h / 2
        x1 = pdf_x + w / 2
        y1 = pdf_y + h / 2

        # Clamp to non-negative coordinates
        if x0 < 0:
            x1 -= x0
            x0 = 0
        if y0 < 0:
            y1 -= y0
            y0 = 0

        pdf_bbox = (x0, y0, x1, y1)

        # Extract text from PDF under the pasted box
        text_content = ""
        if tab._pdf_path:
            text_content = extract_text_in_bbox(tab._pdf_path, tab._page, pdf_bbox)

        # Compute features and save
        features = featurize_region(chosen_type, pdf_bbox, None, 2448.0, 1584.0)
        det_id = tab._store.save_detection(
            doc_id=tab._doc_id,
            page=tab._page,
            run_id=tab._run_id or "manual",
            element_type=chosen_type,
            bbox=pdf_bbox,
            text_content=text_content,
            features=features,
        )
        tab._store.save_correction(
            doc_id=tab._doc_id,
            page=tab._page,
            correction_type="add",
            corrected_label=chosen_type,
            corrected_bbox=pdf_bbox,
            detection_id=det_id,
            session_id=tab._session_id,
        )

        from ..annotation_state import CanvasBox as CBox

        cbox = CBox(
            detection_id=det_id,
            element_type=chosen_type,
            confidence=None,
            text_content=text_content,
            features=features,
            pdf_bbox=pdf_bbox,
            corrected=True,
        )
        tab._canvas_boxes.append(cbox)
        tab._draw_box(cbox)
        tab._select_box(cbox)
        tab._session_count += 1
        tab._update_session_label()
        tab._update_page_summary()
        n_chars = len(text_content)
        tab._status.configure(
            text=f"Pasted {chosen_type} detection ({n_chars} chars extracted)"
        )
