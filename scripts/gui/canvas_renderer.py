"""Canvas rendering mixin for the annotation tab.

Contains all methods that draw boxes, handles, overlays, and the
background page image onto the tkinter Canvas.
"""

from __future__ import annotations

import tkinter as tk

from PIL import Image, ImageTk

from .annotation_state import HANDLE_SIZE, CanvasBox


class CanvasRendererMixin:
    """Mixin providing all canvas drawing methods."""

    def _render_background(self) -> None:
        """Display the background image on the canvas."""
        if self._bg_image is None:
            return

        eff = self._effective_scale()
        # Scale relative to the base DPI rendering
        if self._zoom != 1.0:
            # Re-use a cached resize if available at this zoom level
            cached = self._zoom_image_cache.get(self._zoom)
            if cached is not None:
                display = cached
            else:
                new_w = int(self._bg_image.width * self._zoom)
                new_h = int(self._bg_image.height * self._zoom)
                display = self._bg_image.resize((new_w, new_h), Image.LANCZOS)
                # Evict oldest entries when the cache is full
                if len(self._zoom_image_cache) >= self._zoom_cache_max:
                    oldest = next(iter(self._zoom_image_cache))
                    del self._zoom_image_cache[oldest]
                self._zoom_image_cache[self._zoom] = display
        else:
            display = self._bg_image

        self._photo = ImageTk.PhotoImage(display)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self._canvas.configure(scrollregion=(0, 0, display.width, display.height))

    def _draw_all_boxes(self) -> None:
        """Redraw every detection box on the canvas."""
        # Remove existing box items (but not the background image)
        self._canvas.delete("det_box")
        self._canvas.delete("det_label")
        self._canvas.delete("det_handle")
        self._canvas.delete("pipeline_prompt")
        for cbox in self._canvas_boxes:
            self._draw_box(cbox)
        # Refresh word overlay if active
        if self._word_overlay_on:
            self._draw_word_overlay()

    def _draw_pipeline_prompt(self) -> None:
        """Draw a centered watermark prompting the user to run the pipeline."""
        self._canvas.delete("pipeline_prompt")
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        cx, cy = cw // 2, ch // 2
        self._canvas.create_text(
            cx,
            cy,
            text="Run pipeline to detect elements",
            font=("Segoe UI", 18, "bold"),
            fill="#aaaaaa",
            tags="pipeline_prompt",
        )

    def _confidence_color(self, conf: float | None) -> str:
        """Return a hex color for the confidence badge.

        Green (>0.9) → yellow (0.5–0.9) → red (<0.5) → grey (None).
        """
        if conf is None:
            return "#888888"
        if conf >= 0.9:
            return "#28a745"
        if conf >= 0.5:
            # Interpolate yellow-ish
            return "#d4a017"
        return "#dc3545"

    def _draw_box(self, cbox: CanvasBox) -> None:
        """Draw (or redraw) a single CanvasBox on the canvas."""
        # Remove old items if they exist
        if cbox.rect_id:
            self._canvas.delete(cbox.rect_id)
        if cbox.label_id:
            self._canvas.delete(cbox.label_id)
        if cbox.conf_dot_id:
            self._canvas.delete(cbox.conf_dot_id)
            cbox.conf_dot_id = 0
        for hid in cbox.handle_ids:
            self._canvas.delete(hid)
        cbox.handle_ids.clear()

        eff = self._effective_scale()
        x0, y0, x1, y1 = cbox.pdf_bbox
        cx0 = x0 * eff
        cy0 = y0 * eff
        cx1 = x1 * eff
        cy1 = y1 * eff

        color = self.LABEL_COLORS.get(cbox.element_type, "#888888")
        in_multi = cbox in self._multi_selected
        lw = 3 if (cbox.selected or in_multi) else 2

        # Fill for selected box (translucent approximation via stipple)
        fill_kw: dict = {}
        if cbox.selected or in_multi:
            fill_kw = {"fill": color, "stipple": "gray25"}

        # Render as polygon (merged) or rectangle (normal)
        if cbox.polygon:
            flat_coords: list[float] = []
            for px, py in cbox.polygon:
                flat_coords.append(px * eff)
                flat_coords.append(py * eff)
            if "fill" not in fill_kw:
                fill_kw["fill"] = ""  # transparent (create_polygon defaults to black)
            cbox.rect_id = self._canvas.create_polygon(
                *flat_coords,
                outline=color,
                width=lw,
                tags="det_box",
                **fill_kw,
            )
        else:
            cbox.rect_id = self._canvas.create_rectangle(
                cx0,
                cy0,
                cx1,
                cy1,
                outline=color,
                width=lw,
                tags="det_box",
                **fill_kw,
            )

        # Confidence badge background
        conf_str = f" ({cbox.confidence:.0%})" if cbox.confidence is not None else ""
        check = " ✓" if cbox.corrected else ""
        label_text = f"{cbox.element_type}{conf_str}{check}"

        # Draw confidence colour dot
        if cbox.confidence is not None:
            conf_color = self._confidence_color(cbox.confidence)
            dot_r = 4
            cbox.conf_dot_id = self._canvas.create_oval(
                cx0 - dot_r - 2,
                cy0 - 12 - dot_r,
                cx0 - 2 + dot_r,
                cy0 - 12 + dot_r,
                fill=conf_color,
                outline=conf_color,
                tags="det_label",
            )

        cbox.label_id = self._canvas.create_text(
            cx0 + 2,
            cy0 - 2,
            anchor="sw",
            text=label_text,
            fill=color,
            font=("TkDefaultFont", 8, "bold"),
            tags="det_label",
        )

        # Draw handles if selected
        if cbox.selected:
            self._draw_handles(cbox)

    def _draw_handles(self, cbox: CanvasBox) -> None:
        """Draw 8 resize handles around the selected box."""
        eff = self._effective_scale()
        x0, y0, x1, y1 = cbox.pdf_bbox
        cx0 = x0 * eff
        cy0 = y0 * eff
        cx1 = x1 * eff
        cy1 = y1 * eff
        mx = (cx0 + cx1) / 2
        my = (cy0 + cy1) / 2
        hs = HANDLE_SIZE

        positions = {
            "nw": (cx0, cy0),
            "n": (mx, cy0),
            "ne": (cx1, cy0),
            "e": (cx1, my),
            "se": (cx1, cy1),
            "s": (mx, cy1),
            "sw": (cx0, cy1),
            "w": (cx0, my),
        }

        color = self.LABEL_COLORS.get(cbox.element_type, "#888888")
        for pos_name, (hx, hy) in positions.items():
            hid = self._canvas.create_rectangle(
                hx - hs,
                hy - hs,
                hx + hs,
                hy + hs,
                fill=color,
                outline="white",
                tags="det_handle",
            )
            cbox.handle_ids.append(hid)

    # ── Selection ──────────────────────────────────────────────────

    def _draw_group_links(self, group_id: str) -> None:
        """Draw dashed lines from child boxes to the group root."""
        self._clear_group_links()
        grp = self._groups.get(group_id)
        if not grp:
            return

        root_id = grp["root_detection_id"]
        root_box: CanvasBox | None = None
        children: list[CanvasBox] = []
        for cb in self._canvas_boxes:
            if cb.detection_id == root_id:
                root_box = cb
            elif cb.group_id == group_id:
                children.append(cb)

        if not root_box:
            return

        eff = self._effective_scale()
        rx = (root_box.pdf_bbox[0] + root_box.pdf_bbox[2]) / 2 * eff
        ry = (root_box.pdf_bbox[1] + root_box.pdf_bbox[3]) / 2 * eff
        color = self.LABEL_COLORS.get(root_box.element_type, "#888888")

        for child in children:
            cx = (child.pdf_bbox[0] + child.pdf_bbox[2]) / 2 * eff
            cy = (child.pdf_bbox[1] + child.pdf_bbox[3]) / 2 * eff
            lid = self._canvas.create_line(
                rx,
                ry,
                cx,
                cy,
                dash=(6, 4),
                fill=color,
                width=1,
                tags="group_link",
            )
            self._group_link_ids.append(lid)

    def _clear_group_links(self) -> None:
        """Remove all group connector lines from the canvas."""
        for lid in self._group_link_ids:
            self._canvas.delete(lid)
        self._group_link_ids.clear()

    def _draw_word_overlay(self) -> None:
        """Draw light-gray rectangles around every word on the current page."""
        self._clear_word_overlay()
        if not self._pdf_path:
            return

        cache_key = (str(self._pdf_path), self._page)
        words = self._word_cache.get(cache_key)
        if words is None:
            try:
                from plancheck.ingest.ingest import extract_page_words

                words = extract_page_words(self._pdf_path, self._page)
            except Exception as exc:
                self._status.configure(text=f"Word overlay failed: {exc}")
                return
            self._word_cache[cache_key] = words

        eff = self._effective_scale()
        for w in words:
            cx0 = w["x0"] * eff
            cy0 = w["top"] * eff
            cx1 = w["x1"] * eff
            cy1 = w["bottom"] * eff
            rid = self._canvas.create_rectangle(
                cx0,
                cy0,
                cx1,
                cy1,
                outline="#b0b0b0",
                width=1,
                tags="word_overlay",
            )
            self._word_overlay_ids.append(rid)
            # Store word metadata keyed by canvas item id for hit-testing
            self._word_overlay_items[rid] = {
                "x0": w["x0"],
                "top": w["top"],
                "x1": w["x1"],
                "bottom": w["bottom"],
                "text": w.get("text", ""),
            }

        n = len(words)
        self._status.configure(text=f"Word overlay: {n} words on page {self._page}")

    def _clear_word_overlay(self) -> None:
        """Remove all word overlay rectangles from the canvas."""
        self._canvas.delete("word_overlay")
        self._word_overlay_ids.clear()
        self._word_overlay_items.clear()
        self._selected_word_rids.clear()

    # ── Keyboard shortcuts ─────────────────────────────────────────
