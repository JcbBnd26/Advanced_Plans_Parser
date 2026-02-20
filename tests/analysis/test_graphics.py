"""Unit tests for plancheck.analysis.graphics – PDF graphics extraction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from plancheck.models import GraphicElement

# ── helpers ────────────────────────────────────────────────────────────


def _make_line(x0=0, top=0, x1=100, bottom=0, stroking_color=None, linewidth=1.0):
    return {
        "x0": x0,
        "top": top,
        "x1": x1,
        "bottom": bottom,
        "stroking_color": stroking_color,
        "linewidth": linewidth,
    }


def _make_rect(
    x0=0,
    top=0,
    x1=100,
    bottom=50,
    stroking_color=None,
    non_stroking_color=None,
    linewidth=1.0,
):
    return {
        "x0": x0,
        "top": top,
        "x1": x1,
        "bottom": bottom,
        "stroking_color": stroking_color,
        "non_stroking_color": non_stroking_color,
        "linewidth": linewidth,
    }


def _make_curve(pts, stroking_color=None, non_stroking_color=None, linewidth=1.0):
    return {
        "pts": pts,
        "stroking_color": stroking_color,
        "non_stroking_color": non_stroking_color,
        "linewidth": linewidth,
    }


def _mock_pdf(lines=None, rects=None, curves=None):
    """Build a mock pdfplumber.open() context that yields one page."""
    page = MagicMock()
    page.lines = lines or []
    page.rects = rects or []
    page.curves = curves or []

    pdf_obj = MagicMock()
    pdf_obj.pages = [page]
    pdf_obj.__enter__ = MagicMock(return_value=pdf_obj)
    pdf_obj.__exit__ = MagicMock(return_value=False)
    return pdf_obj


# ── Tests ──────────────────────────────────────────────────────────────


class TestExtractGraphicsLines:
    """Line extraction from PDF pages."""

    @patch("plancheck.analysis.graphics.pdfplumber")
    def test_single_horizontal_line(self, mock_pdfplumber):
        from plancheck.analysis.graphics import extract_graphics

        mock_pdfplumber.open.return_value = _mock_pdf(
            lines=[_make_line(x0=10, top=50, x1=200, bottom=50)]
        )
        result = extract_graphics("test.pdf", 0)

        assert len(result) == 1
        g = result[0]
        assert g.element_type == "line"
        assert g.x0 == 10.0
        assert g.x1 == 200.0
        assert g.page == 0

    @patch("plancheck.analysis.graphics.pdfplumber")
    def test_line_coords_normalized(self, mock_pdfplumber):
        """x0/y0 should be min, x1/y1 should be max regardless of input order."""
        from plancheck.analysis.graphics import extract_graphics

        mock_pdfplumber.open.return_value = _mock_pdf(
            lines=[_make_line(x0=200, top=100, x1=10, bottom=5)]
        )
        result = extract_graphics("test.pdf", 0)

        g = result[0]
        assert g.x0 <= g.x1
        assert g.y0 <= g.y1

    @patch("plancheck.analysis.graphics.pdfplumber")
    def test_line_stroke_color(self, mock_pdfplumber):
        from plancheck.analysis.graphics import extract_graphics

        mock_pdfplumber.open.return_value = _mock_pdf(
            lines=[_make_line(stroking_color=(0, 0, 0))]
        )
        result = extract_graphics("test.pdf", 0)
        assert result[0].stroke_color == (0, 0, 0)

    @patch("plancheck.analysis.graphics.pdfplumber")
    def test_line_linewidth(self, mock_pdfplumber):
        from plancheck.analysis.graphics import extract_graphics

        mock_pdfplumber.open.return_value = _mock_pdf(lines=[_make_line(linewidth=2.5)])
        result = extract_graphics("test.pdf", 0)
        assert result[0].linewidth == 2.5


class TestExtractGraphicsRects:
    """Rectangle extraction from PDF pages."""

    @patch("plancheck.analysis.graphics.pdfplumber")
    def test_single_rect(self, mock_pdfplumber):
        from plancheck.analysis.graphics import extract_graphics

        mock_pdfplumber.open.return_value = _mock_pdf(
            rects=[_make_rect(x0=0, top=0, x1=100, bottom=50)]
        )
        result = extract_graphics("test.pdf", 0)

        assert len(result) == 1
        g = result[0]
        assert g.element_type == "rect"
        assert g.x0 == 0.0
        assert g.y1 == 50.0

    @patch("plancheck.analysis.graphics.pdfplumber")
    def test_rect_fill_color(self, mock_pdfplumber):
        from plancheck.analysis.graphics import extract_graphics

        mock_pdfplumber.open.return_value = _mock_pdf(
            rects=[_make_rect(non_stroking_color=(1, 0, 0))]
        )
        result = extract_graphics("test.pdf", 0)
        assert result[0].fill_color == (1, 0, 0)

    @patch("plancheck.analysis.graphics.pdfplumber")
    def test_rect_stroke_and_fill(self, mock_pdfplumber):
        from plancheck.analysis.graphics import extract_graphics

        mock_pdfplumber.open.return_value = _mock_pdf(
            rects=[_make_rect(stroking_color=(0, 0, 0), non_stroking_color=(1, 1, 1))]
        )
        result = extract_graphics("test.pdf", 0)
        g = result[0]
        assert g.stroke_color == (0, 0, 0)
        assert g.fill_color == (1, 1, 1)


class TestExtractGraphicsCurves:
    """Curve extraction from PDF pages."""

    @patch("plancheck.analysis.graphics.pdfplumber")
    def test_single_curve(self, mock_pdfplumber):
        from plancheck.analysis.graphics import extract_graphics

        pts = [(10, 20), (30, 40), (50, 10)]
        mock_pdfplumber.open.return_value = _mock_pdf(curves=[_make_curve(pts)])
        result = extract_graphics("test.pdf", 0)

        assert len(result) == 1
        g = result[0]
        assert g.element_type == "curve"
        assert g.x0 == 10.0
        assert g.x1 == 50.0
        assert g.y0 == 10.0
        assert g.y1 == 40.0
        assert g.pts == pts

    @patch("plancheck.analysis.graphics.pdfplumber")
    def test_curve_empty_pts_skipped(self, mock_pdfplumber):
        """Curves with no points should be skipped."""
        from plancheck.analysis.graphics import extract_graphics

        mock_pdfplumber.open.return_value = _mock_pdf(curves=[_make_curve(pts=[])])
        result = extract_graphics("test.pdf", 0)
        assert len(result) == 0

    @patch("plancheck.analysis.graphics.pdfplumber")
    def test_curve_colors(self, mock_pdfplumber):
        from plancheck.analysis.graphics import extract_graphics

        mock_pdfplumber.open.return_value = _mock_pdf(
            curves=[
                _make_curve(
                    pts=[(0, 0), (10, 10)],
                    stroking_color=(0, 0, 1),
                    non_stroking_color=(0, 1, 0),
                )
            ]
        )
        result = extract_graphics("test.pdf", 0)
        g = result[0]
        assert g.stroke_color == (0, 0, 1)
        assert g.fill_color == (0, 1, 0)


class TestExtractGraphicsMixed:
    """Edge cases and mixed element extraction."""

    @patch("plancheck.analysis.graphics.pdfplumber")
    def test_empty_page(self, mock_pdfplumber):
        """Page with no graphics should return empty list."""
        from plancheck.analysis.graphics import extract_graphics

        mock_pdfplumber.open.return_value = _mock_pdf()
        result = extract_graphics("test.pdf", 0)
        assert result == []

    @patch("plancheck.analysis.graphics.pdfplumber")
    def test_mixed_elements(self, mock_pdfplumber):
        """Lines, rects, and curves on the same page."""
        from plancheck.analysis.graphics import extract_graphics

        mock_pdfplumber.open.return_value = _mock_pdf(
            lines=[_make_line()],
            rects=[_make_rect()],
            curves=[_make_curve(pts=[(0, 0), (10, 10)])],
        )
        result = extract_graphics("test.pdf", 0)
        assert len(result) == 3
        types = {g.element_type for g in result}
        assert types == {"line", "rect", "curve"}

    @patch("plancheck.analysis.graphics.pdfplumber")
    def test_multiple_lines(self, mock_pdfplumber):
        from plancheck.analysis.graphics import extract_graphics

        mock_pdfplumber.open.return_value = _mock_pdf(
            lines=[
                _make_line(x0=0, top=0, x1=50, bottom=0),
                _make_line(x0=0, top=100, x1=50, bottom=100),
                _make_line(x0=0, top=200, x1=50, bottom=200),
            ]
        )
        result = extract_graphics("test.pdf", 0)
        assert len(result) == 3

    @patch("plancheck.analysis.graphics.pdfplumber")
    def test_missing_optional_fields(self, mock_pdfplumber):
        """Graphics with missing optional fields should use defaults."""
        from plancheck.analysis.graphics import extract_graphics

        mock_pdfplumber.open.return_value = _mock_pdf(
            lines=[{"x0": 10, "x1": 20}],  # missing top, bottom, stroking_color
        )
        result = extract_graphics("test.pdf", 0)
        assert len(result) == 1
        g = result[0]
        # Defaults for missing keys
        assert g.y0 == 0.0
        assert g.y1 == 0.0
        assert g.stroke_color is None
        assert g.linewidth == 1.0

    @patch("plancheck.analysis.graphics.pdfplumber")
    def test_all_elements_have_correct_page(self, mock_pdfplumber):
        from plancheck.analysis.graphics import extract_graphics

        mock_pdfplumber.open.return_value = _mock_pdf(
            lines=[_make_line()],
            rects=[_make_rect()],
        )
        result = extract_graphics("test.pdf", 0)
        for g in result:
            assert g.page == 0

    @patch("plancheck.analysis.graphics.pdfplumber")
    def test_result_types(self, mock_pdfplumber):
        """All returned items should be GraphicElement instances."""
        from plancheck.analysis.graphics import extract_graphics

        mock_pdfplumber.open.return_value = _mock_pdf(
            lines=[_make_line()],
            rects=[_make_rect()],
            curves=[_make_curve(pts=[(0, 0), (1, 1)])],
        )
        result = extract_graphics("test.pdf", 0)
        for g in result:
            assert isinstance(g, GraphicElement)
