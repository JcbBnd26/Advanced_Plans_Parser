"""PDF loading, page navigation, and zoom mixin for the annotation tab."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from tkinter import filedialog
from typing import Any

from plancheck.ingest.ingest import extract_text_in_bbox, extract_text_in_polygon

from .annotation_state import CanvasBox

# Foreground color used to prompt the user to run the pipeline
_PIPELINE_PENDING_COLOR = "#cc6600"


class PdfLoaderMixin:
    """Mixin providing PDF loading, navigation, and zoom methods."""

    def _browse_pdf(self) -> None:
        f = filedialog.askopenfilename(
            title="Select PDF",
            filetypes=[("PDF", "*.pdf"), ("All", "*.*")],
            initialdir=str(Path("input")),
        )
        if f:
            self.state.set_pdf(Path(f))

    def _on_pdf_changed(self) -> None:
        self._pdf_path = self.state.pdf_path
        if self._pdf_path:
            self._pdf_label.configure(text=self._pdf_path.name)
            # Count pages
            try:
                import pdfplumber

                with pdfplumber.open(self._pdf_path) as pdf:
                    self._page_count = len(pdf.pages)
            except Exception:  # noqa: BLE001 — best-effort page count
                self._page_count = 0
            # Reset page selection to first page
            self._page_var.set(0)
            self._page_spin.configure(to=max(0, self._page_count - 1))
            self._page_count_label.configure(text=f"/ {self._page_count}")
            # Clear prior state
            self._canvas_boxes.clear()
            self._selected_box = None
            self._multi_selected.clear()
            # Reset word overlay — it should only be active after pipeline runs
            self._word_overlay_var.set(False)
            self._word_overlay_on = False
            self._pipeline_ran_for_doc = False
            self._canvas.delete("all")
            # Render first page preview
            self._navigate_to_page()
        else:
            self._pdf_label.configure(text="(none)")
            self._page_count = 0
            self._page_count_label.configure(text="/ ?")
            self._page_spin.configure(to=999)

    def _effective_scale(self) -> float:
        """Combined DPI scale × zoom factor."""
        return self._scale * self._zoom

    def _update_session_label(self) -> None:
        self._session_label.configure(text=f"Session: {self._session_count} saved")

    # ── Zoom ───────────────────────────────────────────────────────

    def _apply_zoom(self, factor: float) -> None:
        new_zoom = self._zoom * factor
        new_zoom = max(0.25, min(new_zoom, 5.0))
        if new_zoom == self._zoom:
            return
        self._zoom = new_zoom
        self._render_background()
        self._draw_all_boxes()
        self._status.configure(text=f"Zoom: {self._zoom:.0%}")

    def _fit_to_window(self) -> None:
        """Set zoom so the full page fits inside the visible canvas."""
        if self._bg_image is None:
            return
        # Force geometry update so winfo returns real sizes
        self._canvas.update_idletasks()
        canvas_w = self._canvas.winfo_width()
        canvas_h = self._canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            return
        img_w = self._bg_image.width  # rendered at current DPI (zoom 1.0)
        img_h = self._bg_image.height
        zoom_w = canvas_w / img_w
        zoom_h = canvas_h / img_h
        new_zoom = min(zoom_w, zoom_h)
        new_zoom = max(0.25, min(new_zoom, 5.0))
        self._zoom = new_zoom
        self._render_background()
        self._draw_all_boxes()
        self._status.configure(text=f"Fit: {self._zoom:.0%}")

    # ── Pan (middle mouse button) ──────────────────────────────────

    def _extract_text_for_box(self, cbox: CanvasBox) -> str:
        """Extract text using the polygon boundary when available,
        falling back to the rectangular bbox otherwise."""
        if not self._pdf_path:
            return ""
        if cbox.polygon:
            return extract_text_in_polygon(self._pdf_path, self._page, cbox.polygon)
        return extract_text_in_bbox(self._pdf_path, self._page, cbox.pdf_bbox)

    # ── Word overlay ───────────────────────────────────────────────

    def _toggle_word_overlay(self) -> None:
        """Toggle the pdfplumber word-boxes overlay on or off."""
        self._word_overlay_on = self._word_overlay_var.get()
        if self._word_overlay_on:
            if not self._pipeline_ran_for_doc:
                self._word_overlay_var.set(False)
                self._word_overlay_on = False
                self._status.configure(
                    text="Run the pipeline first to enable word overlay"
                )
                return
            # Only allow word overlay on pages that have detections
            if not self._canvas_boxes:
                self._word_overlay_var.set(False)
                self._word_overlay_on = False
                self._status.configure(
                    text="Word overlay only available on processed pages"
                )
                return
            self._draw_word_overlay()
        else:
            self._clear_word_overlay()

    def _on_prev_page(self) -> None:
        """Navigate to the previous page."""
        cur = self._page_var.get()
        if cur > 0:
            self._page_var.set(cur - 1)
            self._navigate_to_page()

    def _on_next_page(self) -> None:
        """Navigate to the next page."""
        upper = self._page_count - 1 if self._page_count > 0 else 0
        cur = self._page_var.get()
        if cur < upper:
            self._page_var.set(cur + 1)
            self._navigate_to_page()

    def _on_page_spin_enter(self, _event: Any = None) -> None:
        """Handle Enter key or manual spinbox changes."""
        self._navigate_to_page()

    def _navigate_to_page(self) -> None:
        """Render the current page preview and load any existing detections.

        Shows a clear status message indicating whether the pipeline
        has been run for this page.
        """
        if not self._pdf_path:
            return

        self._page = self._page_var.get()
        # Clamp to valid range
        if self._page_count > 0 and self._page >= self._page_count:
            self._page = self._page_count - 1
            self._page_var.set(self._page)
        if self._page < 0:
            self._page = 0
            self._page_var.set(0)

        self._resolution = self._dpi_var.get()
        self._scale = self._resolution / 72.0

        self._status.configure(text=f"Loading page {self._page}…")
        self.root.update_idletasks()

        # Render page preview
        try:
            from plancheck.ingest.ingest import render_page_image

            self._bg_image = render_page_image(
                self._pdf_path, self._page, resolution=self._resolution
            )
            self._render_background()
        except Exception as exc:
            self._status.configure(text=f"Error rendering page: {exc}")
            return

        # Load detections only after pipeline has run for this doc this session
        self._doc_id = self._store.register_document(self._pdf_path)
        self._canvas_boxes.clear()
        self._selected_box = None
        self._multi_selected.clear()

        if self._pipeline_ran_for_doc:
            # Refresh to see latest data from pipeline runs in other threads/connections
            self._store.refresh()
            dets = self._store.get_latest_detections_for_page(self._doc_id, self._page)
            for d in dets:
                self._canvas_boxes.append(
                    CanvasBox(
                        detection_id=d["detection_id"],
                        element_type=d["element_type"],
                        confidence=d["confidence"],
                        text_content=d["text_content"],
                        features=d["features"],
                        pdf_bbox=d["bbox"],
                        polygon=d.get("polygon"),
                    )
                )

        # Deduplicate overlapping same-type boxes
        self._deduplicate_boxes()

        # Load any saved groups for this page
        self._load_groups_for_page()

        # Check for drift on loaded detections
        self._check_drift_on_detections()

        self._draw_all_boxes()
        n = len(self._canvas_boxes)
        page_label = f"Page {self._page}"
        if self._page_count > 0:
            page_label += f" of {self._page_count}"
        if n > 0:
            self._status.configure(text=f"{page_label} — {n} detections")
        elif self._pipeline_ran_for_doc:
            # This page was NOT processed in the latest run — clear word overlay
            if self._word_overlay_on:
                self._word_overlay_var.set(False)
                self._word_overlay_on = False
                self._clear_word_overlay()
            self._status.configure(text=f"{page_label} — not processed in latest run")
            self._draw_pipeline_prompt()
        else:
            self._status.configure(
                text=f"{page_label} — run pipeline to load detections"
            )
            # Draw a watermark on the canvas
            self._draw_pipeline_prompt()
        self._update_page_summary()

    def _check_drift_on_detections(self) -> None:
        """Check for drift on current page detections and update indicator."""
        # Clear previous drift indicator
        if hasattr(self, "_drift_indicator"):
            self._drift_indicator.configure(text="")

        if not self._canvas_boxes:
            return

        try:
            from pathlib import Path as _Path

            import numpy as np

            drift_stats_path = _Path("data") / "drift_stats.json"
            if not drift_stats_path.exists():
                return

            from plancheck.corrections.drift_detection import DriftDetector

            detector = DriftDetector.load(drift_stats_path)

            # Check each detection's features for drift
            drifted_count = 0
            total = 0
            for cb in self._canvas_boxes:
                if cb.features:
                    try:
                        vec = np.array(cb.features, dtype=float)
                        result = detector.check(vec)
                        if result.is_drifted:
                            drifted_count += 1
                        total += 1
                    except Exception:  # noqa: BLE001 — skip malformed detection
                        pass

            if drifted_count > 0:
                self._drift_indicator.configure(
                    text=f"⚠ Drift detected on {drifted_count}/{total} detections"
                )
        except Exception:  # noqa: BLE001 — drift check is best-effort
            # Silently ignore drift check failures
            pass

    # ── Page element summary ───────────────────────────────────────

    def _update_page_summary(self) -> None:
        """Refresh the per-page element type summary in the sidebar."""
        if not self._canvas_boxes:
            if self._pipeline_ran_for_doc:
                self._page_elements_label.configure(
                    text="(no detections on this page)", foreground="gray"
                )
            else:
                self._page_elements_label.configure(
                    text="Run pipeline first", foreground=_PIPELINE_PENDING_COLOR
                )
            return
        counts = Counter(cb.element_type for cb in self._canvas_boxes)
        total = sum(counts.values())
        lines = [f"Total: {total}"]
        for etype, n in counts.most_common():
            color = self.LABEL_COLORS.get(etype, "#888888")
            lines.append(f"  {etype}: {n}")
        self._page_elements_label.configure(text="\n".join(lines), foreground="#222222")

    # ── Model training ─────────────────────────────────────────────
