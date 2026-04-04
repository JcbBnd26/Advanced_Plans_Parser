"""Base tool interface and shared context for the Tool pattern."""

from __future__ import annotations

import tkinter as tk
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..annotation_state import CanvasBox


@dataclass
class ToolContext:
    """Shared references passed to every tool.

    Instead of passing 10+ constructor arguments, tools receive a single
    ``ctx`` that gives them access to the canvas, state, services, and
    the host tab's helper methods.
    """

    tab: Any  # AnnotationTab — the coordinator that owns everything
    canvas: tk.Canvas = field(init=False)
    tool_manager: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.canvas = self.tab._canvas

    # ── Convenience accessors ──────────────────────────────────────

    @property
    def canvas_boxes(self) -> list[CanvasBox]:
        return self.tab._canvas_boxes

    @property
    def selected_box(self) -> CanvasBox | None:
        return self.tab._selected_box

    @selected_box.setter
    def selected_box(self, value: CanvasBox | None) -> None:
        self.tab._selected_box = value

    @property
    def multi_selected(self) -> list[CanvasBox]:
        return self.tab._multi_selected

    @property
    def store(self) -> Any:
        return self.tab._store

    @property
    def doc_id(self) -> str | None:
        return self.tab._doc_id

    @property
    def page(self) -> int:
        return self.tab._page

    @property
    def run_id(self) -> str:
        return self.tab._run_id

    @property
    def session_id(self) -> str:
        return self.tab._session_id

    @property
    def pdf_path(self) -> Any:
        return self.tab._pdf_path

    @property
    def groups(self) -> dict:
        return self.tab._groups

    @property
    def closing(self) -> bool:
        return self.tab._closing

    def effective_scale(self) -> float:
        return self.tab._effective_scale()

    def draw_box(self, cbox: CanvasBox) -> None:
        self.tab._draw_box(cbox)

    def draw_all_boxes(self) -> None:
        self.tab._draw_all_boxes()

    def draw_group_links(self, group_id: str) -> None:
        self.tab._draw_group_links(group_id)

    def clear_group_links(self) -> None:
        self.tab._clear_group_links()

    def select_box(self, cbox: CanvasBox) -> None:
        self.tab._select_box(cbox)

    def deselect(self) -> None:
        self.tab._deselect()

    def increment_session(self) -> None:
        self.tab._session_count += 1

    def update_session_label(self) -> None:
        self.tab._update_session_label()

    def update_page_summary(self) -> None:
        self.tab._update_page_summary()

    def update_multi_label(self) -> None:
        self.tab._update_multi_label()

    def set_status(self, text: str) -> None:
        self.tab._status.configure(text=text)

    def is_active_tab(self) -> bool:
        return self.tab._is_active_tab()


class BaseTool(ABC):
    """Interface that every tool implements.

    All event methods have no-op defaults so a tool only overrides the
    events it cares about.
    """

    name: str = "base"

    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx

    # ── Lifecycle ──────────────────────────────────────────────────

    def enter(self, **kwargs: Any) -> None:
        """Called when this tool becomes active. Override for setup."""

    def exit(self) -> None:
        """Called when this tool deactivates. Override for cleanup."""

    # ── Event handlers ─────────────────────────────────────────────

    def on_click(
        self,
        pdf_x: float,
        pdf_y: float,
        cx: float,
        cy: float,
        event: tk.Event,
    ) -> None:
        """Left mouse button pressed on the canvas."""

    def on_drag(
        self,
        pdf_x: float,
        pdf_y: float,
        cx: float,
        cy: float,
        event: tk.Event,
    ) -> None:
        """Left mouse button held and moving."""

    def on_release(
        self,
        pdf_x: float,
        pdf_y: float,
        cx: float,
        cy: float,
        event: tk.Event,
    ) -> None:
        """Left mouse button released."""

    def on_right_click(
        self,
        pdf_x: float,
        pdf_y: float,
        cx: float,
        cy: float,
        event: tk.Event,
    ) -> None:
        """Right mouse button pressed."""

    def on_motion(
        self,
        pdf_x: float,
        pdf_y: float,
        cx: float,
        cy: float,
        event: tk.Event,
    ) -> None:
        """Mouse movement with no button held."""

    def on_key(self, key: str, event: tk.Event) -> bool:
        """Key press while this tool is active.

        Return True if the key was consumed (prevents further handling).
        """
        return False
