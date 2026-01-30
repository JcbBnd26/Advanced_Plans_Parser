from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Region:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    tag: str = "page"

    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


def whole_page(page: int, width: float, height: float) -> List[Region]:
    return [Region(page=page, x0=0.0, y0=0.0, x1=width, y1=height, tag="page")]
