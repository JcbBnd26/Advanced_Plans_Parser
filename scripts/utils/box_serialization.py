"""Serialization helpers for runner scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from plancheck import GlyphBox


def save_boxes_json(boxes: list["GlyphBox"], out_path: Path) -> None:
    serializable = [
        {
            "page": b.page,
            "x0": b.x0,
            "y0": b.y0,
            "x1": b.x1,
            "y1": b.y1,
            "text": b.text,
            "origin": b.origin,
        }
        for b in boxes
    ]
    out_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
