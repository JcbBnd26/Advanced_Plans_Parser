"""Graphical element model for PDF vector graphics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class GraphicElement:
    """A graphical element extracted from the PDF (line, rect, curve)."""

    page: int
    element_type: str  # "line", "rect", "curve"
    x0: float
    y0: float
    x1: float
    y1: float
    stroke_color: Optional[Tuple] = None  # RGB or CMYK tuple
    fill_color: Optional[Tuple] = None  # RGB or CMYK tuple (for rects/curves)
    linewidth: float = 1.0
    pts: Optional[List[Tuple[float, float]]] = None  # For curves

    def width(self) -> float:
        """Horizontal extent in points."""
        return self.x1 - self.x0

    def height(self) -> float:
        """Vertical extent in points."""
        return self.y1 - self.y0

    def area(self) -> float:
        """Area in square points, clamped to zero."""
        return max(0.0, self.width()) * max(0.0, self.height())

    def bbox(self) -> Tuple[float, float, float, float]:
        """Bounding box as ``(x0, y0, x1, y1)``."""
        return (self.x0, self.y0, self.x1, self.y1)

    def is_small_symbol(self, max_size: float = 50.0) -> bool:
        """Check if this element is small enough to be a legend symbol."""
        return self.width() <= max_size and self.height() <= max_size

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        d: dict = {
            "page": self.page,
            "element_type": self.element_type,
            "x0": round(self.x0, 3),
            "y0": round(self.y0, 3),
            "x1": round(self.x1, 3),
            "y1": round(self.y1, 3),
            "linewidth": round(self.linewidth, 3),
        }
        if self.stroke_color is not None:
            d["stroke_color"] = list(self.stroke_color)
        if self.fill_color is not None:
            d["fill_color"] = list(self.fill_color)
        if self.pts is not None:
            d["pts"] = [[round(x, 3), round(y, 3)] for x, y in self.pts]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GraphicElement":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        return cls(
            page=d["page"],
            element_type=d["element_type"],
            x0=d["x0"],
            y0=d["y0"],
            x1=d["x1"],
            y1=d["y1"],
            stroke_color=tuple(d["stroke_color"]) if d.get("stroke_color") else None,
            fill_color=tuple(d["fill_color"]) if d.get("fill_color") else None,
            linewidth=d.get("linewidth", 1.0),
            pts=[(p[0], p[1]) for p in d["pts"]] if d.get("pts") else None,
        )
