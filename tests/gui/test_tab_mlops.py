"""Tests for tab_mlops module (Phase 4 GUI)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# Add the gui scripts directory to path (same as gui.py does)
_project = Path(__file__).resolve().parent.parent.parent
_gui_dir = _project / "scripts" / "gui"
if str(_gui_dir) not in sys.path:
    sys.path.insert(0, str(_gui_dir))


class TestMLOpsTabImport:
    """Verify tab_mlops can be imported (catches syntax errors)."""

    def test_import_succeeds(self):
        # This validates that the module parses and the class is defined.
        # We can't instantiate it without tkinter, but import is safe.
        mod = importlib.import_module("tab_mlops")
        assert hasattr(mod, "MLOpsTab")
