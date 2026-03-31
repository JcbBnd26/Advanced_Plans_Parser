"""PDF graphics extraction (lines, rectangles, curves)."""

from __future__ import annotations

import logging
from typing import List

from ..models import GraphicElement

logger = logging.getLogger("plancheck.graphics")


def _parse_graphics_dicts(
    page_num: int,
    lines: list[dict],
    rects: list[dict],
    curves: list[dict],
) -> List[GraphicElement]:
    """Convert raw pdfplumber dicts into :class:`GraphicElement` objects.

    This is the shared core used by both :func:`extract_graphics` (which
    opens the PDF itself) and :func:`extract_graphics_from_data` (which
    accepts pre-extracted dicts from :class:`PageContext`).
    """
    graphics: List[GraphicElement] = []

    for line in lines:
        x0 = float(line.get("x0", 0))
        y0 = float(line.get("top", 0))
        x1 = float(line.get("x1", 0))
        y1 = float(line.get("bottom", 0))
        stroke = line.get("stroking_color")
        linewidth = float(line.get("linewidth", 1.0))
        graphics.append(
            GraphicElement(
                page=page_num,
                element_type="line",
                x0=min(x0, x1),
                y0=min(y0, y1),
                x1=max(x0, x1),
                y1=max(y0, y1),
                stroke_color=stroke,
                linewidth=linewidth,
            )
        )

    for rect in rects:
        x0 = float(rect.get("x0", 0))
        y0 = float(rect.get("top", 0))
        x1 = float(rect.get("x1", 0))
        y1 = float(rect.get("bottom", 0))
        stroke = rect.get("stroking_color")
        fill = rect.get("non_stroking_color")
        linewidth = float(rect.get("linewidth", 1.0))
        graphics.append(
            GraphicElement(
                page=page_num,
                element_type="rect",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                stroke_color=stroke,
                fill_color=fill,
                linewidth=linewidth,
            )
        )

    for curve in curves:
        pts = curve.get("pts", [])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        stroke = curve.get("stroking_color")
        fill = curve.get("non_stroking_color")
        linewidth = float(curve.get("linewidth", 1.0))
        graphics.append(
            GraphicElement(
                page=page_num,
                element_type="curve",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                stroke_color=stroke,
                fill_color=fill,
                linewidth=linewidth,
                pts=pts,
            )
        )

    return graphics


def extract_graphics_from_data(
    page_num: int,
    lines: list[dict],
    rects: list[dict],
    curves: list[dict],
) -> List[GraphicElement]:
    """Build :class:`GraphicElement` list from pre-extracted pdfplumber dicts.

    This is the **single-open** variant — callers supply raw dicts already
    obtained via ``page.lines``, ``page.rects``, ``page.curves`` (held in
    a :class:`PageContext`), so no PDF open is needed.

    Parameters
    ----------
    page_num : int
        Zero-based page index.
    lines, rects, curves : list[dict]
        Raw pdfplumber graphical-element dicts.

    Returns
    -------
    list[GraphicElement]
    """
    return _parse_graphics_dicts(page_num, lines, rects, curves)


def extract_graphics(pdf_path: str, page_num: int) -> List[GraphicElement]:
    """
    Extract graphical elements (lines, rects, curves) from a PDF page.

    Returns a list of GraphicElement objects with their bounding boxes and colors.

    .. note:: For pipeline use prefer :func:`extract_graphics_from_data`
       with a :class:`PageContext` to avoid an additional ``pdfplumber.open()``.
    """
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        return _parse_graphics_dicts(
            page_num,
            list(page.lines),
            list(page.rects),
            list(page.curves),
        )
