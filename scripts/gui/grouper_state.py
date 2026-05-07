"""Session state dataclass for the Grouper tab.

Holds all mutable state for one Grouper session:  selected GlyphBoxes,
confirmed groups, page navigation, and cross-page persistence.  Passed
by reference to the canvas and event mixins so they all share the same
object.

Deliberately contains only data — no tkinter widgets, no DB calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set
from uuid import uuid4

if TYPE_CHECKING:
    from plancheck.grouping.grouper_pipeline import GrouperPageData

# Allowed Grouper tab modes
GROUPER_MODES = ("learn", "edit")


@dataclass
class ConfirmedGroup:
    """One human-confirmed group on a canvas page.

    Attributes
    ----------
    indices:
        Indices into ``GrouperSessionState.page_data.boxes`` for the
        GlyphBoxes that belong to this group.
    bbox:
        Cached bounding box ``(x0, y0, x1, y1)`` in PDF points.
    example_id:
        ID written to the ``group_snapshots`` table when this group was
        confirmed.  Empty string until the DB write succeeds.
    """

    indices: List[int] = field(default_factory=list)
    bbox: tuple = field(default_factory=lambda: (0.0, 0.0, 0.0, 0.0))
    example_id: str = ""


@dataclass
class GrouperSessionState:
    """All mutable state for one Grouper tab session.

    A new instance is created when the user starts a Learn or Edit
    session on a PDF.  ``saved_pages`` persists across page navigation
    and is kept even if the user clicks Abort (so previously saved pages
    are never lost).

    Attributes
    ----------
    session_id:
        UUID-based identifier generated at session start.  Written to
        every ``group_snapshots`` and ``group_corrections`` row produced
        during this session.
    pdf_path:
        Path to the source PDF.  ``None`` before a PDF is loaded.
    current_page:
        Current zero-based page index being displayed.
    page_count:
        Total number of pages in the PDF.
    mode:
        Active Grouper mode — ``"learn"`` or ``"edit"``.
    page_data:
        TOCR tokens and machine groups for the current page.  ``None``
        while a page is loading.
    doc_id:
        ``sha256:…`` document ID from the ``documents`` table.  Populated
        after the PDF is registered in ``CorrectionStore``.
    selected_indices:
        GlyphBox indices currently highlighted by Shift+click.  Cleared
        after a group is confirmed or the user right-clicks to cancel.
    inspected_group_idx:
        Index into ``confirmed_groups`` (or machine group list in Edit
        mode) that is currently highlighted by a single click.  ``None``
        when nothing is inspected.
    confirmed_groups:
        Groups confirmed on the *current page* since the last
        Save & Next.  Copied into ``saved_pages`` by Save & Next.
    saved_pages:
        Permanent record of confirmed groups indexed by page number.
        Populated by Save & Next; unaffected by Abort Session.
    show_machine_groups:
        When ``True``, dashed machine group overlays are drawn on canvas.
    active:
        ``False`` after Abort Session — signals the tab to disable
        edit controls while preserving ``saved_pages``.
    """

    session_id: str = field(default_factory=lambda: uuid4().hex)
    pdf_path: Optional[Path] = None
    current_page: int = 0
    page_count: int = 0
    mode: str = "learn"
    page_data: Optional["GrouperPageData"] = None
    doc_id: str = ""
    selected_indices: Set[int] = field(default_factory=set)
    inspected_group_idx: Optional[int] = None
    selected_group_indices: Set[int] = field(
        default_factory=set
    )  # groups marked for meta-grouping
    confirmed_groups: List[ConfirmedGroup] = field(default_factory=list)
    saved_pages: Dict[int, List[ConfirmedGroup]] = field(default_factory=dict)
    show_machine_groups: bool = False
    active: bool = False

    # ── Convenience helpers ───────────────────────────────────────────

    def reset_page_state(self) -> None:
        """Clear per-page selection and inspection state.

        Called when navigating to a new page so stale highlights from
        the previous page are not carried over.
        """
        self.selected_indices = set()
        self.inspected_group_idx = None
        self.selected_group_indices = set()
        self.confirmed_groups = []
        self.page_data = None

    def index_in_confirmed_group(self, glyph_idx: int) -> Optional[int]:
        """Return the confirmed-group index that contains *glyph_idx*.

        Returns ``None`` if the glyph belongs to no confirmed group.
        """
        for gi, cg in enumerate(self.confirmed_groups):
            if glyph_idx in cg.indices:
                return gi
        return None

    def is_glyph_selected(self, glyph_idx: int) -> bool:
        """Return ``True`` if *glyph_idx* is in the current shift-click selection."""
        return glyph_idx in self.selected_indices

    def commit_page(self) -> None:
        """Copy ``confirmed_groups`` for ``current_page`` into ``saved_pages``.

        Called by Save & Next before advancing the page counter.
        """
        self.saved_pages[self.current_page] = list(self.confirmed_groups)
