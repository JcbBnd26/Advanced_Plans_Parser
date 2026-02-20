"""PDF graphics extraction (lines, rectangles, curves)."""

from __future__ import annotations

import logging
from typing import List

import pdfplumber

from ..models import GraphicElement

logger = logging.getLogger("plancheck.graphics")


def extract_graphics(pdf_path: str, page_num: int) -> List[GraphicElement]:
    """
    Extract graphical elements (lines, rects, curves) from a PDF page.

    Returns a list of GraphicElement objects with their bounding boxes and colors.
    """
    graphics: List[GraphicElement] = []

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]

        # Extract lines
        for line in page.lines:
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

        # Extract rectangles
        for rect in page.rects:
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

        # Extract curves
        for curve in page.curves:
            pts = curve.get("pts", [])
            if not pts:
                continue

            # Get bounding box from points
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
