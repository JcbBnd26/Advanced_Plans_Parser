"""Shared utilities for the corrections store modules.

These functions are extracted from the main CorrectionStore to enable
mixin composition without circular imports.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _utcnow_iso() -> str:
    """Return current UTC time as ISO-8601 string with 'Z' suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _gen_id(prefix: str) -> str:
    """Generate a short prefixed ID — e.g. ``det_a1b2c3d4``."""
    return f"{prefix}{uuid4().hex[:8]}"


def _deterministic_sort_key(detection_id: str) -> str:
    """Return a deterministic sort key for a detection_id.

    Uses ``hashlib.md5`` to produce a stable hex digest that is
    consistent across ``PYTHONHASHSEED`` values, Python versions,
    and machines.
    """
    return hashlib.md5(detection_id.encode()).hexdigest()


def _rows_to_detection_dicts(rows) -> list[dict[str, Any]]:
    """Convert SQLite rows to detection dicts."""
    results: list[dict[str, Any]] = []
    for r in rows:
        poly_raw = r["polygon_json"]
        polygon = json.loads(poly_raw) if poly_raw else None
        if polygon:
            polygon = [tuple(p) for p in polygon]
        results.append(
            {
                "detection_id": r["detection_id"],
                "element_type": r["element_type"],
                "confidence": r["confidence"],
                "bbox": (
                    r["bbox_x0"],
                    r["bbox_y0"],
                    r["bbox_x1"],
                    r["bbox_y1"],
                ),
                "text_content": r["text_content"],
                "features": json.loads(r["features_json"]),
                "polygon": polygon,
                "created_at": r["created_at"],
            }
        )
    return results


__all__ = [
    "_utcnow_iso",
    "_gen_id",
    "_deterministic_sort_key",
    "_rows_to_detection_dicts",
]
