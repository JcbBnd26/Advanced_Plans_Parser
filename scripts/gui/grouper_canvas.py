"""Grouper canvas rendering mixin.

Provides all drawing methods for the Grouper tab's independent
tkinter Canvas.  Renders:

- PDF page background image (via ``render_page_image``)
- Grey GlyphBox outlines for all tokens
- Yellow Shift+click selection highlight
- Green bounding boxes for confirmed groups (with group index label)
- Dashed blue machine-group overlays (when Show Machine Groups is on)
- Cyan single-click inspect highlight for Edit mode

Coordinate system
-----------------
pdfplumber ``extract_words`` yields ``top``/``bottom`` for the y-axis
(origin at the *top* of the page, same as image pixel coordinates).
GlyphBox stores these as ``y0`` / ``y1``.  The canvas transform is
therefore simply ``canvas_px = pdf_pt * scale * zoom``.

Phase E — implemented in Write 3.
"""

from __future__ import annotations

import logging
import tkinter as tk
from typing import Any, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── Render resolution (DPI) ───────────────────────────────────────────────────
_RENDER_DPI: int = 144  # 2× at 72 pt/in → scale = 2.0 by default

# ── Colour palette (shared with tab_grouper constants) ───────────────────────
_COLOR_GLYPH_OUTLINE = "#888888"
_COLOR_SELECTION = "#FFD700"
_COLOR_CONFIRMED = "#22AA44"
_COLOR_MACHINE = "#4488CC"
_COLOR_INSPECT = "#00CCCC"
_COLOR_META_MARKED = "#AA44CC"  # purple — group selected for meta-grouping


class GrouperCanvasMixin:
    """Mixin providing all Grouper canvas drawing methods.

    Requires the host class to expose:

    - ``self._canvas``   — ``tk.Canvas`` widget
    - ``self._gsession`` — :class:`.GrouperSessionState` instance
    - ``self._scale``    — float, PDF-point → canvas-pixel factor (set here)
    - ``self._zoom``     — float, additional zoom multiplier
    - ``self._photo``    — ``ImageTk.PhotoImage`` reference (kept alive here)
    """

    # ── Public entry point ────────────────────────────────────────────────────

    def _render_canvas(self) -> None:
        """Full canvas redraw: background → glyphs → groups → overlays."""
        self._canvas.delete("all")

        pd = self._gsession.page_data
        if pd is None:
            return

        self._render_page_background(pd.page_w, pd.page_h)
        self._draw_glyph_boxes(pd.boxes)
        self._draw_confirmed_groups(pd.boxes)
        if self._gsession.show_machine_groups:
            self._draw_machine_groups(pd.machine_groups, pd.boxes)
        if self._gsession.inspected_group_idx is not None:
            self._draw_inspect_highlight(pd.boxes)

    # ── Background ────────────────────────────────────────────────────────────

    def _render_page_background(self, page_w: float, page_h: float) -> None:
        """Render the PDF page as a background image, update ``_scale``.

        The raw page render is cached in ``self._bg_image_base`` so that
        zoom changes only resize rather than re-open the PDF.
        ``tab_grouper`` sets ``_bg_image_base = None`` on every page change
        to force a fresh render.
        """
        pdf_path = self._gsession.pdf_path
        page_num = self._gsession.current_page

        if pdf_path is None or not pdf_path.exists():
            cw = int(page_w * self._zoom)
            ch = int(page_h * self._zoom)
            self._canvas.create_rectangle(
                0, 0, cw, ch, fill="#ffffff", outline="", tags="bg"
            )
            self._canvas.configure(scrollregion=(0, 0, cw, ch))
            self._scale = self._zoom
            return

        # Fetch base image (cached per page load — invalidated by tab_grouper)
        if self._bg_image_base is None:
            try:
                from plancheck.ingest.ingest import render_page_image

                self._bg_image_base = render_page_image(
                    pdf_path, page_num, resolution=_RENDER_DPI
                )
            except Exception:  # noqa: BLE001
                log.warning("grouper canvas: failed to render page background")
                self._scale = self._zoom
                return

        base_img = self._bg_image_base
        base_scale = _RENDER_DPI / 72.0
        eff_scale = base_scale * self._zoom

        if self._zoom != 1.0:
            from PIL import Image

            new_w = int(base_img.width * self._zoom)
            new_h = int(base_img.height * self._zoom)
            display = base_img.resize((new_w, new_h), Image.LANCZOS)
        else:
            display = base_img

        from PIL import ImageTk

        self._photo = ImageTk.PhotoImage(display)
        self._canvas.create_image(0, 0, anchor="nw", image=self._photo, tags="bg")
        self._canvas.configure(scrollregion=(0, 0, display.width, display.height))
        self._scale = eff_scale

    # ── Glyph boxes ───────────────────────────────────────────────────────────

    def _draw_glyph_boxes(self, boxes: list) -> None:
        """Draw grey outlines for every GlyphBox; yellow if selected."""
        for i, box in enumerate(boxes):
            in_confirmed = self._gsession.index_in_confirmed_group(i) is not None
            if in_confirmed:
                continue  # confirmed groups are drawn by _draw_confirmed_groups

            if self._gsession.is_glyph_selected(i):
                color = _COLOR_SELECTION
                width = 2
            else:
                color = _COLOR_GLYPH_OUTLINE
                width = 1

            x0, y0, x1, y1 = self._pdf_to_canvas_rect(box.x0, box.y0, box.x1, box.y1)
            self._canvas.create_rectangle(
                x0, y0, x1, y1, outline=color, width=width, fill="", tags="glyph"
            )

    # ── Confirmed groups ──────────────────────────────────────────────────────

    def _draw_confirmed_groups(self, boxes: list) -> None:
        """Draw green bounding boxes for every confirmed group.

        Groups in ``selected_group_indices`` are drawn in purple to show
        they are staged for meta-grouping.
        """
        for gi, grp in enumerate(self._gsession.confirmed_groups):
            grp_boxes = [boxes[i] for i in grp.indices if i < len(boxes)]
            if not grp_boxes:
                continue

            is_marked = gi in self._gsession.selected_group_indices
            color = _COLOR_META_MARKED if is_marked else _COLOR_CONFIRMED
            outline_w = 3 if is_marked else 2

            # Individual box fills (semi-transparent via stipple)
            for box in grp_boxes:
                x0, y0, x1, y1 = self._pdf_to_canvas_rect(
                    box.x0, box.y0, box.x1, box.y1
                )
                self._canvas.create_rectangle(
                    x0,
                    y0,
                    x1,
                    y1,
                    outline=color,
                    width=1,
                    fill=color,
                    stipple="gray12",
                    tags="confirmed_glyph",
                )

            # Group bounding box
            bx0, by0, bx1, by1 = self._pdf_to_canvas_rect(
                grp.bbox[0], grp.bbox[1], grp.bbox[2], grp.bbox[3]
            )
            self._canvas.create_rectangle(
                bx0,
                by0,
                bx1,
                by1,
                outline=color,
                width=outline_w,
                fill="",
                tags="confirmed_group",
            )
            # Small index label in top-left corner of group bbox
            label = f"G{gi + 1}" if not is_marked else f"G{gi + 1}✦"
            self._canvas.create_text(
                bx0 + 2,
                by0 + 1,
                text=label,
                anchor="nw",
                fill=color,
                font=("Segoe UI", 8, "bold"),
                tags="confirmed_label",
            )

    # ── Machine group overlays ────────────────────────────────────────────────

    def _draw_machine_groups(self, machine_groups: list, boxes: list) -> None:
        """Draw dashed blue overlays for machine-produced BlockClusters."""
        for blk in machine_groups:
            try:
                bx0, by0, bx1, by1 = blk.bbox()
            except Exception:  # noqa: BLE001
                continue
            cx0, cy0, cx1, cy1 = self._pdf_to_canvas_rect(bx0, by0, bx1, by1)
            self._canvas.create_rectangle(
                cx0,
                cy0,
                cx1,
                cy1,
                outline=_COLOR_MACHINE,
                width=1,
                fill="",
                dash=(4, 3),
                tags="machine_group",
            )

    # ── Inspect highlight ─────────────────────────────────────────────────────

    def _draw_inspect_highlight(self, boxes: list) -> None:
        """Highlight the currently inspected group in cyan."""
        idx = self._gsession.inspected_group_idx
        if idx is None:
            return

        # inspected_group_idx may point to a confirmed group (>= 0) or a
        # machine group (negative: -(machine_idx + 1)).
        if idx >= 0 and idx < len(self._gsession.confirmed_groups):
            indices = self._gsession.confirmed_groups[idx].indices
        else:
            return

        for i in indices:
            if i >= len(boxes):
                continue
            box = boxes[i]
            x0, y0, x1, y1 = self._pdf_to_canvas_rect(box.x0, box.y0, box.x1, box.y1)
            self._canvas.create_rectangle(
                x0, y0, x1, y1, outline=_COLOR_INSPECT, width=2, fill="", tags="inspect"
            )

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _pdf_to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        """Convert a PDF-point coordinate to canvas pixels."""
        s = self._scale
        return x * s, y * s

    def _pdf_to_canvas_rect(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> Tuple[float, float, float, float]:
        """Convert a PDF-point bounding box to canvas pixel coordinates."""
        s = self._scale
        return x0 * s, y0 * s, x1 * s, y1 * s

    def _canvas_to_pdf(self, cx: float, cy: float) -> Tuple[float, float]:
        """Convert canvas pixel coordinates back to PDF points."""
        s = self._scale
        if s == 0:
            return 0.0, 0.0
        return cx / s, cy / s

    def _hit_test_glyph(self, cx: float, cy: float) -> Optional[int]:
        """Return the index of the GlyphBox under canvas point ``(cx, cy)``.

        Returns ``None`` if no box is hit.  When multiple boxes overlap the
        point, returns the smallest one (most specific match).
        """
        pd = self._gsession.page_data
        if pd is None:
            return None

        px, py = self._canvas_to_pdf(cx, cy)
        best_idx: Optional[int] = None
        best_area: float = float("inf")
        for i, box in enumerate(pd.boxes):
            if box.x0 <= px <= box.x1 and box.y0 <= py <= box.y1:
                area = (box.x1 - box.x0) * (box.y1 - box.y0)
                if area < best_area:
                    best_area = area
                    best_idx = i
        return best_idx
