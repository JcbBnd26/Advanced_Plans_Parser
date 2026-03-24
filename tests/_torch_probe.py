"""Helpers for probing whether torch is usable in this test environment."""

from __future__ import annotations

import importlib.util


def torch_import_usable() -> bool:
    """Return True when torch is installed."""
    return importlib.util.find_spec("torch") is not None
