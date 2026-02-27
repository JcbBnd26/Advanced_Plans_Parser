"""Tests for tab_query module (Query GUI tab)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_project = Path(__file__).resolve().parent.parent.parent
_gui_dir = _project / "scripts" / "gui"
if str(_gui_dir) not in sys.path:
    sys.path.insert(0, str(_gui_dir))


class TestQueryTabImport:
    """Verify tab_query can be imported (catches syntax errors)."""

    def test_import_succeeds(self):
        mod = importlib.import_module("tab_query")
        assert hasattr(mod, "QueryTab")

    def test_has_expected_methods(self):
        mod = importlib.import_module("tab_query")
        cls = mod.QueryTab
        for method in (
            "_load_run",
            "_on_send",
            "_on_search",
            "_clear_chat",
            "_export_chat",
            "_show_cost",
            "_init_engine",
        ):
            assert hasattr(cls, method), f"Missing method: {method}"
