"""Shared utilities for grouping submodules.

This module consolidates common patterns and helpers used across the
clustering decomposition (lines, spatial, labeling, notes_columns).
Internal module — not exported from the grouping package.
"""

from __future__ import annotations

import io
import re
from contextlib import contextmanager
from typing import IO

# ── Note-number detection patterns ─────────────────────────────────────
# Used by spatial, labeling, notes_columns, and clustering modules.

NOTE_SIMPLE_RE = re.compile(r"^\d+\.")
"""Matches simple numeric note markers like '1.', '12.'"""

NOTE_BROAD_RE = re.compile(r"^(?:\d+\.|[A-Z]\.|[a-z]\.|\(\d+\)|\([A-Za-z]\))")
"""Matches broader note markers including letters and parenthetical forms."""


# ── Debug output helper ────────────────────────────────────────────────


@contextmanager
def _open_debug(path: str | None) -> IO[str]:  # type: ignore[type-arg]
    """Yield a writable text stream for debug output.

    When *path* is ``None``, yields an in-memory no-op sink so callers
    can unconditionally call ``dbg.write()`` without touching the filesystem.
    """
    if path is None:
        yield io.StringIO()
    else:
        f = open(path, "a", encoding="utf-8")
        try:
            yield f
        finally:
            f.close()
