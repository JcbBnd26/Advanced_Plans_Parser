"""Tests for page navigation and annotation tab logic."""

from __future__ import annotations

import pytest

from plancheck.ingest.ingest import point_in_polygon

from scripts.gui.tab_annotation import _reshape_bbox_from_handle, _scale_polygon_to_bbox

# ---------------------------------------------------------------------------
# Helpers — lightweight stand-in for AnnotationTab state logic
# ---------------------------------------------------------------------------


class _PageNavState:
    """Extracts the pure page-navigation state machine from AnnotationTab.

    This avoids needing a real tkinter window or notebook while still
    testing the clamping, tracking, and status-message logic.
    """

    def __init__(self, page_count: int = 0) -> None:
        self.page: int = 0
        self.page_count: int = page_count
        self.pipeline_pages: set[int] = set()
        self.status: str = ""

    def set_page(self, val: int) -> None:
        """Clamp and set the current page."""
        if self.page_count > 0 and val >= self.page_count:
            val = self.page_count - 1
        if val < 0:
            val = 0
        self.page = val

    def prev_page(self) -> bool:
        """Move to previous page; return True if moved."""
        if self.page > 0:
            self.set_page(self.page - 1)
            return True
        return False

    def next_page(self) -> bool:
        """Move to next page; return True if moved."""
        upper = self.page_count - 1 if self.page_count > 0 else 0
        if self.page < upper:
            self.set_page(self.page + 1)
            return True
        return False

    def build_status(self, n_detections: int, n_corrected: int = 0) -> str:
        """Build the status message for the current page."""
        page_label = f"Page {self.page}"
        if self.page_count > 0:
            page_label += f" of {self.page_count}"

        if n_detections > 0:
            status = f"{page_label} — {n_detections} detections"
            if n_corrected:
                status += f" ({n_corrected} corrected)"
        elif self.page in self.pipeline_pages:
            status = f"{page_label} — pipeline ran, no detections found"
        else:
            status = f"{page_label} — click 'Run Pipeline' to detect boxes"
        return status


# ---------------------------------------------------------------------------
# Tests — Page clamping
# ---------------------------------------------------------------------------


class TestPageClamping:
    """Verify page clamping logic."""

    def test_clamp_negative(self) -> None:
        nav = _PageNavState(page_count=5)
        nav.set_page(-3)
        assert nav.page == 0

    def test_clamp_over_max(self) -> None:
        nav = _PageNavState(page_count=5)
        nav.set_page(10)
        assert nav.page == 4  # 0-based: max is count-1

    def test_clamp_exact_max(self) -> None:
        nav = _PageNavState(page_count=5)
        nav.set_page(5)
        assert nav.page == 4

    def test_valid_page(self) -> None:
        nav = _PageNavState(page_count=5)
        nav.set_page(3)
        assert nav.page == 3

    def test_zero_page_count(self) -> None:
        nav = _PageNavState(page_count=0)
        nav.set_page(5)
        # No page_count constraint
        assert nav.page == 5


# ---------------------------------------------------------------------------
# Tests — Arrow navigation
# ---------------------------------------------------------------------------


class TestArrowNavigation:
    """Verify prev/next page navigation."""

    def test_next_page_increments(self) -> None:
        nav = _PageNavState(page_count=5)
        assert nav.next_page()
        assert nav.page == 1

    def test_next_page_stops_at_end(self) -> None:
        nav = _PageNavState(page_count=3)
        nav.set_page(2)
        assert not nav.next_page()
        assert nav.page == 2

    def test_prev_page_decrements(self) -> None:
        nav = _PageNavState(page_count=5)
        nav.set_page(3)
        assert nav.prev_page()
        assert nav.page == 2

    def test_prev_page_stops_at_zero(self) -> None:
        nav = _PageNavState(page_count=5)
        nav.set_page(0)
        assert not nav.prev_page()
        assert nav.page == 0

    def test_full_traversal(self) -> None:
        nav = _PageNavState(page_count=4)
        pages_visited = [nav.page]
        while nav.next_page():
            pages_visited.append(nav.page)
        assert pages_visited == [0, 1, 2, 3]

    def test_zero_page_count_no_nav(self) -> None:
        """With 0 pages, can't navigate."""
        nav = _PageNavState(page_count=0)
        assert not nav.next_page()
        assert not nav.prev_page()


# ---------------------------------------------------------------------------
# Tests — Pipeline page tracking
# ---------------------------------------------------------------------------


class TestPipelinePageTracking:
    """Verify that pipeline-run pages are tracked."""

    def test_initial_empty(self) -> None:
        nav = _PageNavState(page_count=5)
        assert len(nav.pipeline_pages) == 0

    def test_add_pages(self) -> None:
        nav = _PageNavState(page_count=5)
        nav.pipeline_pages.add(0)
        nav.pipeline_pages.add(2)
        assert 0 in nav.pipeline_pages
        assert 2 in nav.pipeline_pages
        assert 1 not in nav.pipeline_pages

    def test_clear_on_new_pdf(self) -> None:
        nav = _PageNavState(page_count=5)
        nav.pipeline_pages.add(0)
        nav.pipeline_pages.add(1)
        # Simulate new PDF load
        nav.pipeline_pages.clear()
        nav.page_count = 3
        assert len(nav.pipeline_pages) == 0


# ---------------------------------------------------------------------------
# Tests — Status messages
# ---------------------------------------------------------------------------


class TestStatusMessages:
    """Verify status bar messages for different page states."""

    def test_no_pipeline_no_detections(self) -> None:
        nav = _PageNavState(page_count=5)
        status = nav.build_status(n_detections=0)
        assert "Run Pipeline" in status
        assert "Page 0 of 5" in status

    def test_pipeline_ran_no_detections(self) -> None:
        nav = _PageNavState(page_count=5)
        nav.pipeline_pages.add(0)
        status = nav.build_status(n_detections=0)
        assert "pipeline ran" in status
        assert "no detections" in status

    def test_with_detections(self) -> None:
        nav = _PageNavState(page_count=5)
        nav.pipeline_pages.add(0)
        status = nav.build_status(n_detections=7)
        assert "7 detections" in status
        assert "Page 0 of 5" in status

    def test_with_corrections(self) -> None:
        nav = _PageNavState(page_count=5)
        status = nav.build_status(n_detections=10, n_corrected=3)
        assert "10 detections" in status
        assert "3 corrected" in status

    def test_page_label_no_count(self) -> None:
        nav = _PageNavState(page_count=0)
        status = nav.build_status(n_detections=0)
        assert "Page 0" in status
        assert "of" not in status

    def test_different_page(self) -> None:
        nav = _PageNavState(page_count=10)
        nav.set_page(7)
        status = nav.build_status(n_detections=5)
        assert "Page 7 of 10" in status

    def test_unprocessed_page_after_nav(self) -> None:
        """Navigating to a page where pipeline hasn't run shows prompt."""
        nav = _PageNavState(page_count=5)
        nav.pipeline_pages.add(0)  # Only page 0 processed
        nav.set_page(2)
        status = nav.build_status(n_detections=0)
        assert "Run Pipeline" in status
        assert "Page 2 of 5" in status


# ---------------------------------------------------------------------------
# Tests — All-pages pipeline tracking
# ---------------------------------------------------------------------------


class TestRunAllPages:
    """Verify that running all pages marks them all as processed."""

    def test_all_pages_marked(self) -> None:
        nav = _PageNavState(page_count=4)
        for pg in range(nav.page_count):
            nav.pipeline_pages.add(pg)
        assert nav.pipeline_pages == {0, 1, 2, 3}

    def test_status_after_all_pages(self) -> None:
        nav = _PageNavState(page_count=3)
        for pg in range(nav.page_count):
            nav.pipeline_pages.add(pg)
        # Each page should show "pipeline ran" not "Run Pipeline"
        for pg in range(nav.page_count):
            nav.set_page(pg)
            status = nav.build_status(n_detections=0)
            assert "pipeline ran" in status
            assert "Run Pipeline" not in status


# ---------------------------------------------------------------------------
# Tests — Element type registry
# ---------------------------------------------------------------------------


class _ElementTypeRegistry:
    """Mirrors the dynamic element type logic in AnnotationTab."""

    LABEL_COLORS: dict[str, str] = {
        "notes_column": "#1ea01e",
        "notes_block": "#2ecc40",
        "header": "#dc1e1e",
        "abbreviations": "#e05096",
        "legend": "#009682",
        "revision": "#dcc800",
        "standard_detail": "#0090dc",
        "title_block": "#8c00c8",
        "misc_title": "#ff8c00",
    }
    ELEMENT_TYPES: list[str] = list(LABEL_COLORS.keys())

    def __init__(self) -> None:
        # Copy class-level dicts so tests are isolated
        self.label_colors = dict(self.LABEL_COLORS)
        self.element_types = list(self.ELEMENT_TYPES)

    def register(self, name: str) -> bool:
        """Register a new element type. Returns True if added."""
        name = name.strip().lower().replace(" ", "_")
        if not name or name in self.label_colors:
            return False
        _palette = [
            "#e6194b",
            "#3cb44b",
            "#4363d8",
            "#f58231",
            "#911eb4",
            "#42d4f4",
            "#f032e6",
            "#bfef45",
            "#fabebe",
            "#469990",
        ]
        idx = len(self.label_colors) % len(_palette)
        self.label_colors[name] = _palette[idx]
        self.element_types.append(name)
        return True


class TestElementTypeRegistry:
    """Verify dynamic element type registration."""

    def test_default_types(self) -> None:
        reg = _ElementTypeRegistry()
        assert "notes_column" in reg.element_types
        assert "notes_block" in reg.element_types
        assert "header" in reg.element_types

    def test_register_new_type(self) -> None:
        reg = _ElementTypeRegistry()
        assert reg.register("zoning_box")
        assert "zoning_box" in reg.element_types
        assert "zoning_box" in reg.label_colors

    def test_register_duplicate_rejected(self) -> None:
        reg = _ElementTypeRegistry()
        assert not reg.register("header")

    def test_register_normalizes_name(self) -> None:
        reg = _ElementTypeRegistry()
        assert reg.register("My Custom Type")
        assert "my_custom_type" in reg.element_types

    def test_register_empty_rejected(self) -> None:
        reg = _ElementTypeRegistry()
        assert not reg.register("")
        assert not reg.register("   ")

    def test_register_assigns_color(self) -> None:
        reg = _ElementTypeRegistry()
        reg.register("new_type_a")
        assert reg.label_colors["new_type_a"].startswith("#")

    def test_notes_block_distinct_from_column(self) -> None:
        reg = _ElementTypeRegistry()
        assert reg.label_colors["notes_block"] != reg.label_colors["notes_column"]


# ---------------------------------------------------------------------------
# Tests — Notes column vs notes block labeling
# ---------------------------------------------------------------------------


class TestNotesLabeling:
    """Verify that notes_column and notes_block are distinct labels."""

    def test_notes_block_in_defaults(self) -> None:
        """notes_block must be a recognized element type."""
        reg = _ElementTypeRegistry()
        assert "notes_block" in reg.element_types

    def test_notes_column_in_defaults(self) -> None:
        """notes_column must be a recognized element type."""
        reg = _ElementTypeRegistry()
        assert "notes_column" in reg.element_types

    def test_both_have_colors(self) -> None:
        reg = _ElementTypeRegistry()
        assert reg.label_colors["notes_column"]
        assert reg.label_colors["notes_block"]


# ---------------------------------------------------------------------------
# Point-in-polygon
# ---------------------------------------------------------------------------


class TestPointInPolygon:
    """Verify ray-casting point-in-polygon used for merged box click."""

    def test_inside_square(self) -> None:
        sq = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        assert point_in_polygon(5, 5, sq)

    def test_outside_square(self) -> None:
        sq = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        assert not point_in_polygon(15, 5, sq)

    def test_l_shape_inside_arm(self) -> None:
        # L-shape: bottom-left block + top-right arm
        l_shape = [
            (0, 0),
            (10, 0),
            (10, 5),
            (20, 5),
            (20, 10),
            (0, 10),
            (0, 0),
        ]
        # Inside the right arm
        assert point_in_polygon(15, 7, l_shape)
        # Inside the left body
        assert point_in_polygon(5, 3, l_shape)

    def test_l_shape_outside_notch(self) -> None:
        l_shape = [
            (0, 0),
            (10, 0),
            (10, 5),
            (20, 5),
            (20, 10),
            (0, 10),
            (0, 0),
        ]
        # Outside in the notch area (top-right of left side)
        assert not point_in_polygon(15, 2, l_shape)

    def test_inside_triangle(self) -> None:
        tri = [(0, 0), (10, 0), (5, 10), (0, 0)]
        assert point_in_polygon(5, 3, tri)

    def test_outside_triangle(self) -> None:
        tri = [(0, 0), (10, 0), (5, 10), (0, 0)]
        assert not point_in_polygon(0, 10, tri)

    def test_empty_polygon(self) -> None:
        assert not point_in_polygon(5, 5, [])

    def test_point_far_outside(self) -> None:
        sq = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        assert not point_in_polygon(-100, -100, sq)


class TestPolygonFillDefault:
    """Verify that merged polygon boxes should NOT fill with black."""

    def test_fill_empty_when_not_selected(self) -> None:
        """When not selected, fill_kw must include fill='' for polygons."""
        fill_kw: dict = {}
        is_polygon = True
        is_selected = False

        if is_selected:
            fill_kw = {"fill": "#1ea01e", "stipple": "gray25"}
        if is_polygon and "fill" not in fill_kw:
            fill_kw["fill"] = ""

        assert fill_kw["fill"] == ""

    def test_fill_color_when_selected(self) -> None:
        """When selected, fill_kw should have the color with stipple."""
        fill_kw: dict = {}
        is_polygon = True
        is_selected = True

        if is_selected:
            fill_kw = {"fill": "#1ea01e", "stipple": "gray25"}
        if is_polygon and "fill" not in fill_kw:
            fill_kw["fill"] = ""

        assert fill_kw["fill"] == "#1ea01e"
        assert fill_kw["stipple"] == "gray25"


# ---------------------------------------------------------------------------
# Fit-to-window logic
# ---------------------------------------------------------------------------


class _FitState:
    """Extracts the pure fit-to-window zoom logic from AnnotationTab."""

    def __init__(self, img_w: int, img_h: int) -> None:
        self.img_w = img_w
        self.img_h = img_h
        self.zoom: float = 1.0

    def fit(self, canvas_w: int, canvas_h: int) -> float | None:
        """Calculate and set zoom to fit image in canvas. Returns new zoom."""
        if self.img_w <= 0 or self.img_h <= 0:
            return None
        if canvas_w <= 1 or canvas_h <= 1:
            return None
        zoom_w = canvas_w / self.img_w
        zoom_h = canvas_h / self.img_h
        new_zoom = min(zoom_w, zoom_h)
        new_zoom = max(0.25, min(new_zoom, 5.0))
        self.zoom = new_zoom
        return new_zoom


class TestFitToWindow:
    """Fit-to-window zoom tests."""

    def test_wide_image_fits_horizontally(self) -> None:
        s = _FitState(1000, 500)
        z = s.fit(500, 500)
        assert z == pytest.approx(0.5)

    def test_tall_image_fits_vertically(self) -> None:
        s = _FitState(500, 1000)
        z = s.fit(500, 500)
        assert z == pytest.approx(0.5)

    def test_square_exact_fit(self) -> None:
        s = _FitState(500, 500)
        z = s.fit(500, 500)
        assert z == pytest.approx(1.0)

    def test_zoom_clamped_high(self) -> None:
        """Canvas much larger than image — zoom capped at 5.0."""
        s = _FitState(100, 100)
        z = s.fit(2000, 2000)
        assert z == 5.0

    def test_zoom_clamped_low(self) -> None:
        """Canvas much smaller than image — zoom floored at 0.25."""
        s = _FitState(10000, 10000)
        z = s.fit(100, 100)
        assert z == 0.25

    def test_canvas_too_small_returns_none(self) -> None:
        s = _FitState(500, 500)
        assert s.fit(1, 1) is None
        assert s.zoom == 1.0  # unchanged

    def test_no_image_returns_none(self) -> None:
        s = _FitState(0, 0)
        assert s.fit(500, 500) is None


# ---------------------------------------------------------------------------
# Box copy / paste logic
# ---------------------------------------------------------------------------


class _CopyPasteState:
    """Extracts the copy/paste box logic from AnnotationTab.

    Uses an internal clipboard dict and a simple box list, modelling the
    core copy/paste behaviour without tkinter.
    """

    def __init__(self) -> None:
        self.doc_id: str = "doc1"
        self.clipboard: dict | None = None
        self.boxes: list[dict] = []
        self.status: str = ""
        self._next_id: int = 1

    def copy_box(self, box: dict) -> None:
        x0, y0, x1, y1 = box["pdf_bbox"]
        self.clipboard = {
            "element_type": box["element_type"],
            "width": x1 - x0,
            "height": y1 - y0,
        }
        self.status = (
            f"Copied {box['element_type']} box "
            f"({x1 - x0:.0f}\u00d7{y1 - y0:.0f} pt)"
        )

    def paste_box(self, pdf_x: float, pdf_y: float) -> dict | None:
        if not self.clipboard or not self.doc_id:
            return None

        w = self.clipboard["width"]
        h = self.clipboard["height"]
        chosen = self.clipboard["element_type"]

        x0 = pdf_x - w / 2
        y0 = pdf_y - h / 2
        x1 = pdf_x + w / 2
        y1 = pdf_y + h / 2

        if x0 < 0:
            x1 -= x0
            x0 = 0
        if y0 < 0:
            y1 -= y0
            y0 = 0

        box = {
            "detection_id": f"det_{self._next_id}",
            "element_type": chosen,
            "pdf_bbox": (x0, y0, x1, y1),
        }
        self._next_id += 1
        self.boxes.append(box)
        self.status = f"Pasted {chosen} detection"
        return box


class TestBoxCopyPaste:
    """Copy/paste box clipboard tests."""

    @pytest.fixture()
    def state(self) -> _CopyPasteState:
        return _CopyPasteState()

    @pytest.fixture()
    def sample_box(self) -> dict:
        return {
            "detection_id": "d1",
            "element_type": "header",
            "pdf_bbox": (100.0, 200.0, 300.0, 400.0),
        }

    def test_copy_stores_template(
        self, state: _CopyPasteState, sample_box: dict
    ) -> None:
        state.copy_box(sample_box)
        assert state.clipboard is not None
        assert state.clipboard["element_type"] == "header"
        assert state.clipboard["width"] == pytest.approx(200.0)
        assert state.clipboard["height"] == pytest.approx(200.0)

    def test_copy_status_message(
        self, state: _CopyPasteState, sample_box: dict
    ) -> None:
        state.copy_box(sample_box)
        assert "Copied" in state.status
        assert "header" in state.status

    def test_paste_without_copy_returns_none(self, state: _CopyPasteState) -> None:
        assert state.paste_box(400.0, 400.0) is None

    def test_paste_creates_box(self, state: _CopyPasteState, sample_box: dict) -> None:
        state.copy_box(sample_box)
        pasted = state.paste_box(500.0, 500.0)
        assert pasted is not None
        assert pasted["element_type"] == "header"
        assert len(state.boxes) == 1

    def test_paste_centres_at_location(
        self, state: _CopyPasteState, sample_box: dict
    ) -> None:
        state.copy_box(sample_box)
        pasted = state.paste_box(500.0, 500.0)
        assert pasted is not None
        x0, y0, x1, y1 = pasted["pdf_bbox"]
        assert (x0 + x1) / 2 == pytest.approx(500.0)
        assert (y0 + y1) / 2 == pytest.approx(500.0)

    def test_paste_preserves_dimensions(
        self, state: _CopyPasteState, sample_box: dict
    ) -> None:
        state.copy_box(sample_box)
        pasted = state.paste_box(500.0, 500.0)
        assert pasted is not None
        x0, y0, x1, y1 = pasted["pdf_bbox"]
        assert (x1 - x0) == pytest.approx(200.0)
        assert (y1 - y0) == pytest.approx(200.0)

    def test_paste_clamps_negative_x(
        self, state: _CopyPasteState, sample_box: dict
    ) -> None:
        """Pasting near left edge should shift the box to non-negative coordinates."""
        state.copy_box(sample_box)
        pasted = state.paste_box(50.0, 500.0)  # half-width is 100, so x0 would be -50
        assert pasted is not None
        x0, y0, x1, y1 = pasted["pdf_bbox"]
        assert x0 == pytest.approx(0.0)
        assert (x1 - x0) == pytest.approx(200.0)

    def test_paste_clamps_negative_y(
        self, state: _CopyPasteState, sample_box: dict
    ) -> None:
        """Pasting near top edge should shift the box to non-negative coordinates."""
        state.copy_box(sample_box)
        pasted = state.paste_box(500.0, 50.0)
        assert pasted is not None
        x0, y0, x1, y1 = pasted["pdf_bbox"]
        assert y0 == pytest.approx(0.0)
        assert (y1 - y0) == pytest.approx(200.0)

    def test_paste_multiple_increments_ids(
        self, state: _CopyPasteState, sample_box: dict
    ) -> None:
        state.copy_box(sample_box)
        b1 = state.paste_box(400.0, 400.0)
        b2 = state.paste_box(600.0, 600.0)
        assert b1 is not None and b2 is not None
        assert b1["detection_id"] != b2["detection_id"]
        assert len(state.boxes) == 2

    def test_paste_no_doc_returns_none(
        self, state: _CopyPasteState, sample_box: dict
    ) -> None:
        state.copy_box(sample_box)
        state.doc_id = ""
        assert state.paste_box(400.0, 400.0) is None

    def test_copy_overwrites_previous(self, state: _CopyPasteState) -> None:
        box1 = {
            "detection_id": "d1",
            "element_type": "header",
            "pdf_bbox": (0.0, 0.0, 100.0, 50.0),
        }
        box2 = {
            "detection_id": "d2",
            "element_type": "legend",
            "pdf_bbox": (0.0, 0.0, 300.0, 200.0),
        }
        state.copy_box(box1)
        state.copy_box(box2)
        assert state.clipboard is not None
        assert state.clipboard["element_type"] == "legend"
        assert state.clipboard["width"] == pytest.approx(300.0)
        assert state.clipboard["height"] == pytest.approx(200.0)

    def test_paste_uses_latest_copy(self, state: _CopyPasteState) -> None:
        box1 = {
            "detection_id": "d1",
            "element_type": "header",
            "pdf_bbox": (0.0, 0.0, 100.0, 50.0),
        }
        box2 = {
            "detection_id": "d2",
            "element_type": "legend",
            "pdf_bbox": (0.0, 0.0, 300.0, 200.0),
        }
        state.copy_box(box1)
        state.copy_box(box2)
        pasted = state.paste_box(400.0, 400.0)
        assert pasted is not None
        assert pasted["element_type"] == "legend"
        x0, y0, x1, y1 = pasted["pdf_bbox"]
        assert (x1 - x0) == pytest.approx(300.0)
        assert (y1 - y0) == pytest.approx(200.0)

    def test_paste_status_message(
        self, state: _CopyPasteState, sample_box: dict
    ) -> None:
        state.copy_box(sample_box)
        state.paste_box(500.0, 500.0)
        assert "Pasted" in state.status
        assert "header" in state.status


# ---------------------------------------------------------------------------
# Box grouping (WBS hierarchy) logic
# ---------------------------------------------------------------------------


class _GroupState:
    """Extracts the box-grouping logic from AnnotationTab.

    Models the WBS-style parent-child hierarchy without tkinter.
    """

    def __init__(self) -> None:
        self.groups: dict[str, dict] = {}
        self.boxes: list[dict] = []
        self.status: str = ""
        self._next_grp: int = 1

    def add_box(
        self,
        detection_id: str,
        element_type: str = "header",
    ) -> dict:
        box: dict = {
            "detection_id": detection_id,
            "element_type": element_type,
            "group_id": None,
            "is_group_root": False,
        }
        self.boxes.append(box)
        return box

    def _find_box(self, detection_id: str) -> dict | None:
        for b in self.boxes:
            if b["detection_id"] == detection_id:
                return b
        return None

    def create_group(
        self,
        root_detection_id: str,
        label: str,
    ) -> str | None:
        box = self._find_box(root_detection_id)
        if not box or box["group_id"]:
            return None
        gid = f"grp_{self._next_grp}"
        self._next_grp += 1
        box["group_id"] = gid
        box["is_group_root"] = True
        self.groups[gid] = {
            "label": label,
            "root_detection_id": root_detection_id,
            "members": [root_detection_id],
        }
        self.status = f"Created group \u2039{label}\u203a"
        return gid

    def add_to_group(
        self,
        group_id: str,
        detection_id: str,
    ) -> bool:
        grp = self.groups.get(group_id)
        if not grp:
            return False
        box = self._find_box(detection_id)
        if not box or box["group_id"]:
            return False
        box["group_id"] = group_id
        box["is_group_root"] = False
        grp["members"].append(detection_id)
        return True

    def remove_from_group(self, detection_id: str) -> None:
        box = self._find_box(detection_id)
        if not box or not box["group_id"]:
            return
        gid = box["group_id"]
        grp = self.groups.get(gid)
        if not grp:
            return

        if box["is_group_root"]:
            # Remove entire group
            for member_id in grp["members"]:
                mb = self._find_box(member_id)
                if mb:
                    mb["group_id"] = None
                    mb["is_group_root"] = False
            del self.groups[gid]
            self.status = f"Deleted group (parent removed)"
        else:
            box["group_id"] = None
            box["is_group_root"] = False
            if detection_id in grp["members"]:
                grp["members"].remove(detection_id)
            self.status = f"Removed from group"

    def get_group_for_detection(self, detection_id: str) -> dict | None:
        box = self._find_box(detection_id)
        if not box or not box["group_id"]:
            return None
        gid = box["group_id"]
        grp = self.groups.get(gid)
        if not grp:
            return None
        return {"group_id": gid, **grp}

    def group_display_text(self, detection_id: str) -> str:
        box = self._find_box(detection_id)
        if not box or not box["group_id"]:
            return "\u2014"
        grp = self.groups.get(box["group_id"])
        if not grp:
            return "\u2014"
        label = grp["label"]
        if box["is_group_root"]:
            n_children = len(grp["members"]) - 1
            return f"\u25cf {label} (parent \u2014 {n_children} children)"
        return f"\u2192 {label}"


class TestBoxGrouping:
    """WBS-style box grouping tests."""

    @pytest.fixture()
    def state(self) -> _GroupState:
        s = _GroupState()
        s.add_box("d1", "notes_column")
        s.add_box("d2", "header")
        s.add_box("d3", "abbreviations")
        s.add_box("d4", "abbreviations")
        return s

    def test_create_group(self, state: _GroupState) -> None:
        gid = state.create_group("d1", "Abbreviations Column")
        assert gid is not None
        assert gid.startswith("grp_")
        assert state.groups[gid]["label"] == "Abbreviations Column"

    def test_root_box_marked(self, state: _GroupState) -> None:
        gid = state.create_group("d1", "Notes")
        box = state._find_box("d1")
        assert box is not None
        assert box["group_id"] == gid
        assert box["is_group_root"] is True

    def test_add_member(self, state: _GroupState) -> None:
        gid = state.create_group("d1", "Notes")
        assert gid is not None
        ok = state.add_to_group(gid, "d2")
        assert ok is True
        box = state._find_box("d2")
        assert box is not None
        assert box["group_id"] == gid
        assert box["is_group_root"] is False

    def test_add_multiple_members(self, state: _GroupState) -> None:
        gid = state.create_group("d1", "Notes")
        assert gid is not None
        state.add_to_group(gid, "d2")
        state.add_to_group(gid, "d3")
        assert len(state.groups[gid]["members"]) == 3  # root + 2 children

    def test_remove_child(self, state: _GroupState) -> None:
        gid = state.create_group("d1", "Notes")
        assert gid is not None
        state.add_to_group(gid, "d2")
        state.remove_from_group("d2")
        box = state._find_box("d2")
        assert box is not None
        assert box["group_id"] is None
        assert gid in state.groups  # group still exists

    def test_remove_root_deletes_group(self, state: _GroupState) -> None:
        gid = state.create_group("d1", "Notes")
        assert gid is not None
        state.add_to_group(gid, "d2")
        state.add_to_group(gid, "d3")
        state.remove_from_group("d1")
        assert gid not in state.groups
        # All members should be ungrouped
        for did in ("d1", "d2", "d3"):
            box = state._find_box(did)
            assert box is not None
            assert box["group_id"] is None

    def test_get_group_for_detection(self, state: _GroupState) -> None:
        gid = state.create_group("d1", "Notes")
        assert gid is not None
        state.add_to_group(gid, "d2")
        result = state.get_group_for_detection("d2")
        assert result is not None
        assert result["group_id"] == gid
        assert result["label"] == "Notes"

    def test_ungrouped_returns_none(self, state: _GroupState) -> None:
        assert state.get_group_for_detection("d4") is None

    def test_duplicate_add_rejected(self, state: _GroupState) -> None:
        gid = state.create_group("d1", "Notes")
        assert gid is not None
        state.add_to_group(gid, "d2")
        ok = state.add_to_group(gid, "d2")  # already in group
        assert ok is False

    def test_cannot_create_on_grouped_box(self, state: _GroupState) -> None:
        gid = state.create_group("d1", "Notes")
        assert gid is not None
        result = state.create_group("d1", "Another")
        assert result is None

    def test_display_text_root(self, state: _GroupState) -> None:
        gid = state.create_group("d1", "Notes")
        assert gid is not None
        state.add_to_group(gid, "d2")
        text = state.group_display_text("d1")
        assert "Notes" in text
        assert "parent" in text
        assert "1 children" in text

    def test_display_text_child(self, state: _GroupState) -> None:
        gid = state.create_group("d1", "Notes")
        assert gid is not None
        state.add_to_group(gid, "d2")
        text = state.group_display_text("d2")
        assert "\u2192 Notes" == text

    def test_display_text_ungrouped(self, state: _GroupState) -> None:
        assert state.group_display_text("d4") == "\u2014"

    def test_create_group_status(self, state: _GroupState) -> None:
        state.create_group("d1", "Abbreviations Column")
        assert "Created" in state.status
        assert "Abbreviations Column" in state.status

    # ── Edge-case / integration-level group tests ──────────────────

    def test_add_to_nonexistent_group(self, state: _GroupState) -> None:
        """Adding to a group ID that doesn't exist should be a no-op."""
        ok = state.add_to_group("grp_bogus", "d1")
        assert ok is False
        assert state._find_box("d1")["group_id"] is None

    def test_remove_from_ungrouped_box(self, state: _GroupState) -> None:
        """Removing a box that's not in any group is a safe no-op."""
        state.remove_from_group("d3")  # should not raise
        assert state._find_box("d3")["group_id"] is None

    def test_remove_nonexistent_detection(self, state: _GroupState) -> None:
        """Removing a detection_id that doesn't exist does nothing."""
        state.remove_from_group("no_such_id")  # should not raise

    def test_multiple_groups_independent(self, state: _GroupState) -> None:
        """Two separate groups on different roots stay independent."""
        gid1 = state.create_group("d1", "GroupA")
        gid2 = state.create_group("d2", "GroupB")
        assert gid1 != gid2
        state.add_to_group(gid1, "d3")
        state.add_to_group(gid2, "d4")
        assert state._find_box("d3")["group_id"] == gid1
        assert state._find_box("d4")["group_id"] == gid2
        assert len(state.groups[gid1]["members"]) == 2
        assert len(state.groups[gid2]["members"]) == 2

    def test_cross_group_add_rejected(self, state: _GroupState) -> None:
        """A box already in groupA cannot be added to groupB."""
        gid1 = state.create_group("d1", "GroupA")
        gid2 = state.create_group("d2", "GroupB")
        state.add_to_group(gid1, "d3")
        ok = state.add_to_group(gid2, "d3")  # already in GroupA
        assert ok is False
        assert state._find_box("d3")["group_id"] == gid1

    def test_delete_root_frees_children_for_regroup(self, state: _GroupState) -> None:
        """After the root is removed, children can join a new group."""
        gid = state.create_group("d1", "OldGroup")
        state.add_to_group(gid, "d2")
        state.remove_from_group("d1")  # deletes entire group
        # d2 should now be free
        gid2 = state.create_group("d2", "NewGroup")
        assert gid2 is not None
        assert state._find_box("d2")["group_id"] == gid2

    def test_display_text_multiple_children(self, state: _GroupState) -> None:
        """Parent display shows correct child count when multiple added."""
        state.create_group("d1", "Big")
        state.add_to_group(state._find_box("d1")["group_id"], "d2")
        state.add_to_group(state._find_box("d1")["group_id"], "d3")
        state.add_to_group(state._find_box("d1")["group_id"], "d4")
        display = state.group_display_text("d1")
        assert "3 children" in display

    def test_display_text_after_child_removal(self, state: _GroupState) -> None:
        """After removing a child, parent count decreases."""
        gid = state.create_group("d1", "Shrink")
        state.add_to_group(gid, "d2")
        state.add_to_group(gid, "d3")
        state.remove_from_group("d2")
        display = state.group_display_text("d1")
        assert "1 children" in display

    def test_group_lookup_after_child_swap(self, state: _GroupState) -> None:
        """get_group_for_detection reflects remove + re-add correctly."""
        gid = state.create_group("d1", "G")
        state.add_to_group(gid, "d2")
        state.remove_from_group("d2")
        assert state.get_group_for_detection("d2") is None
        state.add_to_group(gid, "d2")  # re-add
        result = state.get_group_for_detection("d2")
        assert result is not None
        assert result["group_id"] == gid

    def test_status_messages_on_operations(self, state: _GroupState) -> None:
        """Verify status text is set for each group operation."""
        gid = state.create_group("d1", "StatusGroup")
        assert "Created" in state.status

        state.add_to_group(gid, "d2")
        # No explicit status for add in _GroupState, test the fields instead
        assert state._find_box("d2")["group_id"] == gid

        state.remove_from_group("d2")
        assert "Removed" in state.status

        state.remove_from_group("d1")  # root → delete
        assert "Deleted" in state.status


# ---------------------------------------------------------------------------
# Box move-drag logic
# ---------------------------------------------------------------------------


class _MoveDragState:
    """Extracts the move-drag state machine from AnnotationTab.

    Models: click on a box selects it AND prepares for move-drag.
    If the user drags, the box moves. If they just release, it's
    a plain select.
    """

    def __init__(self) -> None:
        self.boxes: list[dict] = []
        self.selected_box: dict | None = None
        self.move_dragging: bool = False
        self.move_start_pdf: tuple[float, float] | None = None
        self.move_orig_bbox: tuple[float, float, float, float] | None = None
        self.move_orig_polygon: list[tuple[float, float]] | None = None
        self.status: str = ""

    def add_box(
        self,
        detection_id: str,
        bbox: tuple[float, float, float, float],
        polygon: list[tuple[float, float]] | None = None,
    ) -> dict:
        box = {
            "detection_id": detection_id,
            "bbox": bbox,
            "polygon": polygon,
            "corrected": False,
        }
        self.boxes.append(box)
        return box

    def _find_box(self, detection_id: str) -> dict | None:
        for b in self.boxes:
            if b["detection_id"] == detection_id:
                return b
        return None

    def click(self, detection_id: str) -> None:
        """Simulate pressing mouse on a box (selects + prepares move)."""
        box = self._find_box(detection_id)
        if not box:
            return
        # Select if not already selected
        if self.selected_box is not box:
            self.selected_box = box
        # Always prepare for move on press
        cx = (box["bbox"][0] + box["bbox"][2]) / 2
        cy = (box["bbox"][1] + box["bbox"][3]) / 2
        self.move_dragging = True
        self.move_start_pdf = (cx, cy)
        self.move_orig_bbox = box["bbox"]
        self.move_orig_polygon = list(box["polygon"]) if box["polygon"] else None

    def drag(self, pdf_x: float, pdf_y: float) -> None:
        """Simulate dragging to a point."""
        if not self.move_dragging or not self.selected_box or not self.move_start_pdf:
            return
        dx = pdf_x - self.move_start_pdf[0]
        dy = pdf_y - self.move_start_pdf[1]
        ox0, oy0, ox1, oy1 = self.move_orig_bbox
        nx0 = max(ox0 + dx, 0)
        ny0 = max(oy0 + dy, 0)
        nx1 = nx0 + (ox1 - ox0)
        ny1 = ny0 + (oy1 - oy0)
        self.selected_box["bbox"] = (nx0, ny0, nx1, ny1)
        if self.move_orig_polygon:
            self.selected_box["polygon"] = [
                (px + dx, py + dy) for px, py in self.move_orig_polygon
            ]

    def release(self) -> bool:
        """Simulate mouse-up. Returns True if box actually moved."""
        if not self.move_dragging or not self.selected_box:
            self.move_dragging = False
            return False
        moved = self.move_orig_bbox != self.selected_box["bbox"]
        if moved:
            self.selected_box["corrected"] = True
            self.status = "Moved box to new position"
        self.move_dragging = False
        self.move_start_pdf = None
        self.move_orig_bbox = None
        self.move_orig_polygon = None
        return moved


class TestBoxMoveDrag:
    """Tests for click-drag-to-move behaviour."""

    @pytest.fixture()
    def state(self) -> _MoveDragState:
        s = _MoveDragState()
        s.add_box("b1", (100, 100, 200, 200))
        s.add_box("b2", (300, 300, 400, 400))
        return s

    def test_click_selects_and_prepares_move(self, state: _MoveDragState) -> None:
        state.click("b1")
        assert state.selected_box is not None
        assert state.selected_box["detection_id"] == "b1"
        assert state.move_dragging is True
        assert state.move_orig_bbox == (100, 100, 200, 200)

    def test_release_without_drag_is_plain_select(self, state: _MoveDragState) -> None:
        state.click("b1")
        moved = state.release()
        assert moved is False
        assert state.selected_box["detection_id"] == "b1"
        assert state.selected_box["corrected"] is False

    def test_drag_moves_box(self, state: _MoveDragState) -> None:
        state.click("b1")
        state.drag(200, 200)  # move to the right and down
        x0, y0, x1, y1 = state.selected_box["bbox"]
        # Box center started at (150, 150); moved to (200, 200) → delta +50, +50
        assert x0 == pytest.approx(150)
        assert y0 == pytest.approx(150)
        assert x1 == pytest.approx(250)
        assert y1 == pytest.approx(250)

    def test_release_marks_corrected(self, state: _MoveDragState) -> None:
        state.click("b1")
        state.drag(200, 200)
        moved = state.release()
        assert moved is True
        assert state.selected_box["corrected"] is True
        assert "Moved" in state.status

    def test_no_drag_no_correction(self, state: _MoveDragState) -> None:
        state.click("b1")
        # Release without dragging
        moved = state.release()
        assert moved is False
        assert state.selected_box["corrected"] is False

    def test_box_stays_non_negative(self, state: _MoveDragState) -> None:
        state.click("b1")
        state.drag(-1000, -1000)  # try to drag way off-screen
        x0, y0, _, _ = state.selected_box["bbox"]
        assert x0 >= 0
        assert y0 >= 0

    def test_clicking_different_box_selects_new(self, state: _MoveDragState) -> None:
        state.click("b1")
        state.release()  # plain select
        state.click("b2")  # different box
        assert state.selected_box["detection_id"] == "b2"
        assert state.move_dragging is True  # prepared for drag

    def test_polygon_moves_with_box(self, state: _MoveDragState) -> None:
        poly = [(100, 100), (200, 100), (200, 200), (100, 200), (100, 100)]
        state.add_box("bp", (100, 100, 200, 200), polygon=poly)
        state.click("bp")
        state.drag(200, 200)
        new_poly = state.selected_box["polygon"]
        # Each vertex should shift by +50, +50
        assert new_poly[0] == pytest.approx((150, 150), abs=1)
        assert new_poly[1] == pytest.approx((250, 150), abs=1)

    def test_move_preserves_box_size(self, state: _MoveDragState) -> None:
        state.click("b1")
        state.drag(500, 500)
        x0, y0, x1, y1 = state.selected_box["bbox"]
        assert (x1 - x0) == pytest.approx(100)
        assert (y1 - y0) == pytest.approx(100)

    def test_release_clears_move_state(self, state: _MoveDragState) -> None:
        state.click("b1")
        state.drag(200, 200)
        state.release()
        assert state.move_dragging is False
        assert state.move_start_pdf is None
        assert state.move_orig_bbox is None
        assert state.move_orig_polygon is None


# ---------------------------------------------------------------------------
# Rescan text button — state-machine tests
# ---------------------------------------------------------------------------


class _RescanState:
    """Lightweight model of the rescan-text logic in AnnotationTab."""

    def __init__(self) -> None:
        self.selected_box: dict | None = None
        self.pdf_path: str | None = None
        self.page: int = 0
        self.text_widget_state: str = "disabled"
        self.text_widget_content: str = ""
        self.status: str = ""
        # Simulate extract_text_in_bbox returning canned text
        self.extract_result: str = "rescanned content"
        self.extract_error: Exception | None = None

    # -- helpers that mirror the widget state toggling pattern -----------
    def _set_text(self, content: str) -> None:
        """Mirrors _insp_text state toggling: normal → write → disabled."""
        self.text_widget_state = "normal"
        self.text_widget_content = content
        self.text_widget_state = "disabled"

    def select_box(self, box: dict) -> None:
        self.selected_box = box
        self._set_text(box.get("text_content", ""))

    def deselect(self) -> None:
        self.selected_box = None
        self._set_text("")

    def on_rescan_text(self) -> None:
        if not self.selected_box:
            self.status = "No box selected"
            return
        if not self.pdf_path:
            self.status = "No PDF loaded"
            return
        if self.extract_error:
            self.status = f"Rescan failed: {self.extract_error}"
            return

        new_text = self.extract_result
        self.selected_box["text_content"] = new_text
        self._set_text(new_text)
        n_chars = len(new_text)
        etype = self.selected_box.get("element_type", "unknown")
        self.status = f"Rescanned text for {etype} — {n_chars} chars"


class TestRescanText:
    @pytest.fixture()
    def state(self) -> _RescanState:
        s = _RescanState()
        s.pdf_path = "/tmp/test.pdf"
        return s

    def test_rescan_no_selection(self, state: _RescanState) -> None:
        state.on_rescan_text()
        assert state.status == "No box selected"

    def test_rescan_no_pdf(self, state: _RescanState) -> None:
        state.select_box({"element_type": "header", "text_content": "old"})
        state.pdf_path = None
        state.on_rescan_text()
        assert state.status == "No PDF loaded"

    def test_rescan_updates_text(self, state: _RescanState) -> None:
        state.select_box({"element_type": "header", "text_content": "old"})
        state.on_rescan_text()
        assert state.text_widget_content == "rescanned content"
        assert state.selected_box["text_content"] == "rescanned content"
        assert "17 chars" in state.status

    def test_rescan_state_toggling(self, state: _RescanState) -> None:
        """Widget state must be 'disabled' after rescan (read-only)."""
        state.select_box({"element_type": "note", "text_content": ""})
        state.on_rescan_text()
        assert state.text_widget_state == "disabled"

    def test_rescan_then_reselect(self, state: _RescanState) -> None:
        """After rescan, selecting another box must still update text."""
        state.select_box({"element_type": "a", "text_content": "aaa"})
        state.on_rescan_text()
        state.deselect()
        state.select_box({"element_type": "b", "text_content": "bbb"})
        assert state.text_widget_content == "bbb"
        assert state.text_widget_state == "disabled"

    def test_rescan_error_shows_status(self, state: _RescanState) -> None:
        state.select_box({"element_type": "x", "text_content": ""})
        state.extract_error = RuntimeError("pdfplumber crashed")
        state.on_rescan_text()
        assert "Rescan failed" in state.status
        assert state.text_widget_content == ""  # unchanged


# ---------------------------------------------------------------------------
# Polygon-aware rescan — state-machine tests
# ---------------------------------------------------------------------------


class _PolygonRescanState(_RescanState):
    """Extends _RescanState to model polygon-vs-rect dispatch."""

    def __init__(self) -> None:
        super().__init__()
        self.polygon_extract_result: str = "polygon text"
        self.last_mode: str = ""

    def select_box(self, box: dict) -> None:  # type: ignore[override]
        self.selected_box = box
        self._set_text(box.get("text_content", ""))

    def on_rescan_text(self) -> None:
        if not self.selected_box:
            self.status = "No box selected"
            return
        if not self.pdf_path:
            self.status = "No PDF loaded"
            return
        if self.extract_error:
            self.status = f"Rescan failed: {self.extract_error}"
            return

        polygon = self.selected_box.get("polygon")
        if polygon:
            new_text = self.polygon_extract_result
            self.last_mode = "polygon"
        else:
            new_text = self.extract_result
            self.last_mode = "rect"

        self.selected_box["text_content"] = new_text
        self._set_text(new_text)
        n_chars = len(new_text)
        etype = self.selected_box.get("element_type", "unknown")
        self.status = f"Rescanned text for {etype} — {n_chars} chars ({self.last_mode})"


class TestPolygonRescan:
    @pytest.fixture()
    def state(self) -> _PolygonRescanState:
        s = _PolygonRescanState()
        s.pdf_path = "/tmp/test.pdf"
        return s

    def test_rect_fallback(self, state: _PolygonRescanState) -> None:
        """Without polygon, should fall back to rect extraction."""
        state.select_box({"element_type": "header", "text_content": "old"})
        state.on_rescan_text()
        assert state.last_mode == "rect"
        assert state.text_widget_content == "rescanned content"

    def test_polygon_used(self, state: _PolygonRescanState) -> None:
        """With polygon, should use polygon extraction."""
        poly = [(0, 0), (100, 0), (100, 50), (0, 50)]
        state.select_box(
            {
                "element_type": "notes_column",
                "text_content": "old",
                "polygon": poly,
            }
        )
        state.on_rescan_text()
        assert state.last_mode == "polygon"
        assert state.text_widget_content == "polygon text"
        assert "(polygon)" in state.status

    def test_polygon_none_uses_rect(self, state: _PolygonRescanState) -> None:
        state.select_box(
            {
                "element_type": "legend",
                "text_content": "",
                "polygon": None,
            }
        )
        state.on_rescan_text()
        assert state.last_mode == "rect"


# ---------------------------------------------------------------------------
# Word overlay — state-machine tests
# ---------------------------------------------------------------------------


class _WordOverlayState:
    """Lightweight model of the word-overlay toggle logic."""

    def __init__(self) -> None:
        self.pdf_path: str | None = None
        self.page: int = 0
        self.overlay_on: bool = False
        self.overlay_ids: list[int] = []
        self.status: str = ""
        self.zoom: float = 1.0
        self.scale: float = 150 / 72.0  # default DPI

        # Canned word data (simulates pdfplumber output)
        self.words: list[dict] = [
            {"x0": 10, "top": 20, "x1": 50, "bottom": 35, "text": "hello"},
            {"x0": 60, "top": 20, "x1": 100, "bottom": 35, "text": "world"},
            {"x0": 10, "top": 40, "x1": 80, "bottom": 55, "text": "testing"},
        ]
        self.extract_error: Exception | None = None

    def _effective_scale(self) -> float:
        return self.scale * self.zoom

    def toggle_word_overlay(self) -> None:
        self.overlay_on = not self.overlay_on
        if self.overlay_on:
            self.draw_word_overlay()
        else:
            self.clear_word_overlay()

    def draw_word_overlay(self) -> None:
        self.clear_word_overlay()
        if not self.pdf_path:
            return
        if self.extract_error:
            self.status = f"Word overlay failed: {self.extract_error}"
            return
        eff = self._effective_scale()
        for i, w in enumerate(self.words):
            # Simulate canvas rectangle IDs
            self.overlay_ids.append(1000 + i)
        n = len(self.words)
        self.status = f"Word overlay: {n} words on page {self.page}"

    def clear_word_overlay(self) -> None:
        self.overlay_ids.clear()

    def draw_all_boxes(self) -> None:
        """Simulate full redraw — must refresh word overlay if active."""
        if self.overlay_on:
            self.draw_word_overlay()


class TestWordOverlay:
    @pytest.fixture()
    def state(self) -> _WordOverlayState:
        s = _WordOverlayState()
        s.pdf_path = "/tmp/test.pdf"
        return s

    def test_toggle_on(self, state: _WordOverlayState) -> None:
        state.toggle_word_overlay()
        assert state.overlay_on is True
        assert len(state.overlay_ids) == 3
        assert "3 words" in state.status

    def test_toggle_off(self, state: _WordOverlayState) -> None:
        state.toggle_word_overlay()  # on
        state.toggle_word_overlay()  # off
        assert state.overlay_on is False
        assert len(state.overlay_ids) == 0

    def test_no_pdf(self, state: _WordOverlayState) -> None:
        state.pdf_path = None
        state.toggle_word_overlay()
        assert state.overlay_on is True
        assert len(state.overlay_ids) == 0

    def test_overlay_survives_redraw(self, state: _WordOverlayState) -> None:
        """draw_all_boxes must re-draw word overlay when active."""
        state.toggle_word_overlay()  # on
        state.draw_all_boxes()
        assert len(state.overlay_ids) == 3

    def test_overlay_not_drawn_when_off(self, state: _WordOverlayState) -> None:
        state.draw_all_boxes()
        assert len(state.overlay_ids) == 0

    def test_toggle_twice_no_duplicates(self, state: _WordOverlayState) -> None:
        state.toggle_word_overlay()  # on
        state.draw_word_overlay()  # manual second draw
        assert len(state.overlay_ids) == 3  # should clear first

    def test_error_shows_status(self, state: _WordOverlayState) -> None:
        state.extract_error = RuntimeError("bad pdf")
        state.toggle_word_overlay()
        assert "Word overlay failed" in state.status
        assert len(state.overlay_ids) == 0

    def test_page_shown_in_status(self, state: _WordOverlayState) -> None:
        state.page = 5
        state.toggle_word_overlay()
        assert "page 5" in state.status


# ---------------------------------------------------------------------------
# Box reshape-handle logic
# ---------------------------------------------------------------------------


def test_reshape_bbox_from_handle_clamps_min_size() -> None:
    orig = (10.0, 10.0, 20.0, 20.0)
    # Drag the west handle past the east edge → should clamp to (ox1 - min_size)
    new_bbox = _reshape_bbox_from_handle(orig, "w", px=100.0, py=0.0, min_size=1.0)
    assert new_bbox == (19.0, 10.0, 20.0, 20.0)


def test_scale_polygon_to_bbox_scales_points() -> None:
    orig_bbox = (5.0, 10.0, 15.0, 30.0)
    polygon = [(5.0, 10.0), (15.0, 10.0), (15.0, 30.0), (5.0, 30.0)]
    new_bbox = (5.0, 10.0, 25.0, 50.0)  # 2x width, 2x height
    scaled = _scale_polygon_to_bbox(orig_bbox, polygon, new_bbox)
    assert scaled == [(5.0, 10.0), (25.0, 10.0), (25.0, 50.0), (5.0, 50.0)]
