"""Shared selection state for annotation tools.

Multiple tools read/write the selection state: SelectTool sets it,
MoveTool reads it to know what to move, ResizeTool reads it to know
what to resize, etc.  Centralising it here avoids ownership ambiguity.
"""

from __future__ import annotations

import logging
import tkinter as tk
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..annotation_state import CanvasBox

log = logging.getLogger(__name__)


class SelectionState:
    """Owns box/word selection state shared across tools.

    This is a *thin facade* over the existing ``self._selected_box`` /
    ``self._multi_selected`` / ``self._selected_word_rids`` attributes
    on ``AnnotationTab``.  During the migration the tab still holds the
    canonical data; this class provides the API that new tool code uses.
    """

    def __init__(self, tab: Any) -> None:
        self._tab = tab

    # ── Box selection ──────────────────────────────────────────────

    @property
    def selected_box(self) -> CanvasBox | None:
        return self._tab._selected_box

    @selected_box.setter
    def selected_box(self, value: CanvasBox | None) -> None:
        self._tab._selected_box = value

    @property
    def multi_selected(self) -> list[CanvasBox]:
        return self._tab._multi_selected

    def select_box(self, cbox: CanvasBox) -> None:
        self._tab._select_box(cbox)

    def deselect(self) -> None:
        self._tab._deselect()

    def toggle_multi_select(self, cbox: CanvasBox) -> None:
        self._tab._toggle_multi_select(cbox)

    def clear_multi_select(self) -> None:
        self._tab._clear_multi_select()

    def update_multi_label(self) -> None:
        self._tab._update_multi_label()

    # ── Word selection ─────────────────────────────────────────────

    @property
    def selected_word_rids(self) -> set[int]:
        return self._tab._selected_word_rids

    @property
    def word_overlay_on(self) -> bool:
        return self._tab._word_overlay_on

    @property
    def word_overlay_items(self) -> dict[int, dict]:
        return self._tab._word_overlay_items

    def set_word_selected(self, rid: int, selected: bool) -> None:
        self._tab._set_word_selected(rid, selected)

    def toggle_word_selected(self, rid: int) -> None:
        self._tab._toggle_word_selected(rid)

    def clear_word_selection(self) -> None:
        self._tab._clear_word_selection()

    def update_word_selection_label(self) -> None:
        self._tab._update_word_selection_label()
