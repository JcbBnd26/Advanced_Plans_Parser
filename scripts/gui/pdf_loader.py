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

    @property
    def _doc_has_detections(self) -> bool:
        """Check the database for any pipeline detections for the current doc."""
        return bool(
            getattr(self, "_store", None)
            and getattr(self, "_doc_id", None)
            and self._store.has_detections_for_doc(self._doc_id)
        )

    def _browse_pdf(self) -> None:
        f = filedialog.askopenfilename(
            title="Select PDF",
            filetypes=[("PDF", "*.pdf"), ("All", "*.*")],
            initialdir=str(Path("input")),
        )
        if f:
            self.state.set_pdf(Path(f))

    def _load_document(
        self,
        pdf_path: Path | None,
        *,
        doc_id: str | None = None,
        page_count: int | None = None,
    ) -> None:
        """Unified document-loading entry point used by both Browse and Project flows.

        Parameters
        ----------
        pdf_path:
            Path to the PDF on disk, or ``None`` for offline/PNG mode.
        doc_id:
            Pre-resolved document ID (from the Project dropdown).  When
            ``None`` the ID will be obtained by registering *pdf_path* in
            the correction store during the first ``_navigate_to_page``.
        page_count:
            Page count from the database.  When ``None`` the count is
            derived by opening the PDF.
        """
        self._pdf_path = pdf_path
        self._doc_id = doc_id

        # ── Page count ─────────────────────────────────────────────
        if page_count is not None:
            self._page_count = page_count
        elif self._pdf_path:
            try:
                import pdfplumber

                with pdfplumber.open(self._pdf_path) as pdf:
                    self._page_count = len(pdf.pages)
            except Exception:  # noqa: BLE001 — best-effort page count
                self._page_count = 0
        else:
            self._page_count = 0

        # ── UI reset ───────────────────────────────────────────────
        self._page_var.set(0)
        self._page_spin.configure(to=max(0, self._page_count - 1))
        self._page_count_label.configure(text=f"/ {self._page_count}")

        # Offline mode
        if self._pdf_path:
            self._offline_mode_var.set(False)
        elif doc_id:
            self._offline_mode_var.set(True)
        # (no else — offline_mode_var stays unchanged when clearing)

        # Clear prior canvas / selection state
        self._canvas_boxes.clear()
        self._selected_box = None
        self._multi_selected.clear()
        self._word_overlay_var.set(False)
        self._word_overlay_on = False
        self._canvas.delete("all")

        if not self._pdf_path and not self._doc_id:
            self._page_spin.configure(to=999)
            return

        # Show loading cue
        self._status.configure(text="Loading PDF\u2026")
        self._canvas.create_text(
            10,
            10,
            text="Loading PDF, please wait\u2026",
            anchor="nw",
            fill="#888888",
            font=("TkDefaultFont", 11),
        )
        self.root.update_idletasks()

        # Populate dropdowns via the tab (Project ➜ Runs)
        self._sync_dropdowns()

        # Render first page
        self._navigate_to_page()

    def _sync_dropdowns(self) -> None:
        """Refresh Project and Run dropdowns to match the current document.

        Overridden in AnnotationTab where the widgets exist.
        """

    def _on_pdf_changed(self) -> None:
        pdf_path = self.state.pdf_path
        if pdf_path:
            self._load_document(pdf_path)
        else:
            self._load_document(None)

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
            if not self._doc_has_detections:
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
        has been run for this page.  Falls back to a pre-rendered PNG
        when the original PDF is not available on disk.
        """
        # Allow navigation when we have either a PDF or a doc_id (offline mode)
        if not self._pdf_path and not getattr(self, "_doc_id", None):
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

        # Render page preview — try PDF first, then PNG fallback
        self._png_fallback_mode = False
        _offline = (
            getattr(self, "_offline_mode_var", None) and self._offline_mode_var.get()
        )  # noqa: SIM102
        if not _offline and self._pdf_path and self._pdf_path.exists():
            try:
                from plancheck.ingest.ingest import render_page_image

                self._bg_image = render_page_image(
                    self._pdf_path, self._page, resolution=self._resolution
                )
                self._zoom_image_cache.clear()
                self._render_background()
            except Exception as exc:
                self._status.configure(text=f"Error rendering page: {exc}")
                return
        else:
            # PNG fallback: load pre-rendered image from DB
            self._reopen_store()
            run_id = getattr(self, "_selected_run_id", None)
            png_info = self._store.get_page_image(
                self._doc_id, self._page, run_id=run_id
            )
            if png_info and Path(png_info["path"]).exists():
                from PIL import Image as _Image

                self._bg_image = _Image.open(png_info["path"]).convert("RGB")
                self._scale = png_info["dpi"] / 72.0
                self._png_fallback_mode = True
                self._zoom_image_cache.clear()
                self._render_background()
            else:
                self._canvas.delete("all")
                self._canvas.create_text(
                    10,
                    10,
                    text="No page image available — PDF not found"
                    " and no exported PNG exists.",
                    anchor="nw",
                    fill="#888888",
                    font=("TkDefaultFont", 11),
                )
                self._status.configure(text="No image available for this page")
                return

        # Always load detections from the database — they persist across
        # sessions and are available immediately after any pipeline run.
        self._reopen_store()
        if self._pdf_path and self._pdf_path.exists():
            self._doc_id = self._store.register_document(self._pdf_path)
        self._canvas_boxes.clear()
        self._selected_box = None
        self._multi_selected.clear()

        # Refresh to pick up data committed by the pipeline worker thread
        self._store.refresh()

        # Load detections: use run-specific if a run is selected, else latest
        run_id = getattr(self, "_selected_run_id", None)
        if run_id:
            dets = self._store.get_detections_for_run(self._doc_id, self._page, run_id)
        else:
            dets = self._store.get_latest_detections_for_page(self._doc_id, self._page)

        # Filter out dismissed detections so they don't reappear
        dismissed_ids = self._store.get_dismissed_ids_for_page(self._doc_id, self._page)
        dets = [d for d in dets if d["detection_id"] not in dismissed_ids]

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
        elif self._doc_has_detections:
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
        self._active_drift_text = ""

        cfg = getattr(getattr(self, "state", None), "config", None)
        if not getattr(cfg, "ml_drift_enabled", False):
            if hasattr(self, "_update_annotation_runtime_summary"):
                self._update_annotation_runtime_summary()
            return

        if not self._canvas_boxes:
            if hasattr(self, "_update_annotation_runtime_summary"):
                self._update_annotation_runtime_summary()
            return

        try:
            from pathlib import Path as _Path

            import numpy as np

            drift_stats_raw = getattr(cfg, "ml_drift_stats_path", "")
            drift_stats_path = _Path(drift_stats_raw) if drift_stats_raw else None
            if drift_stats_path is None:
                return
            if not drift_stats_path.is_absolute():
                drift_stats_path = _Path.cwd() / drift_stats_path
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
                self._active_drift_text = (
                    f"Drift: active on {drifted_count}/{total} detections"
                )
                self._drift_indicator.configure(text=f"⚠ {self._active_drift_text}")
        except Exception:  # noqa: BLE001 — drift check is best-effort
            # Silently ignore drift check failures
            pass
        finally:
            if hasattr(self, "_update_annotation_runtime_summary"):
                self._update_annotation_runtime_summary()

    # ── Page element summary ───────────────────────────────────────

    def _update_page_summary(self) -> None:
        """Refresh the per-page element type summary in the sidebar."""
        if not self._canvas_boxes:
            if self._doc_has_detections:
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
