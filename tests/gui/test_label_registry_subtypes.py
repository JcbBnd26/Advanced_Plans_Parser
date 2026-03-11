import tkinter as tk

import pytest

from scripts.gui.mixins.label_registry import LabelRegistryMixin


class _DummyVar:
    def __init__(self, value=None) -> None:
        self._value = value

    def get(self):
        return self._value

    def set(self, value) -> None:
        self._value = value


class _DummyWidget:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def configure(self, **kwargs) -> None:
        self.kwargs.update(kwargs)


class _Host(LabelRegistryMixin):
    def __init__(self) -> None:
        self.LABEL_COLORS = {"title_block": "#8c00c8", "header": "#dc1e1e"}
        self.ELEMENT_TYPES = ["title_block", "header"]
        self._type_combo = _DummyWidget()
        self._subtype_combo = _DummyWidget()
        self._type_var = _DummyVar("")
        self._subtype_var = _DummyVar("")
        self._filter_label_vars: dict[str, object] = {}
        self._status = _DummyWidget()
        self.root = object()
        self.rebuild_calls = 0

    def _rebuild_filter_controls(self) -> None:
        self.rebuild_calls += 1


def test_ensure_title_subtypes_registers_gui_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tk, "BooleanVar", lambda value=True: _DummyVar(value))
    host = _Host()

    host._ensure_title_subtype_labels()

    assert "page_title" in host.ELEMENT_TYPES
    assert "plan_title" in host.ELEMENT_TYPES
    assert host._subtype_combo.kwargs["values"] == host._title_subtypes()
    assert "page_title" in host._filter_label_vars


def test_title_subtype_controls_sync_with_active_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tk, "BooleanVar", lambda value=True: _DummyVar(value))
    host = _Host()
    host._ensure_title_subtype_labels()

    host._set_active_element_type("title_block")

    assert host._type_var.get() == "title_block"
    assert host._subtype_var.get() == ""
    assert host._subtype_combo.kwargs["state"] == "readonly"

    host._subtype_var.set("page_title")
    host._on_subtype_selected()

    assert host._type_var.get() == "page_title"
    assert host._subtype_var.get() == "page_title"

    host._set_active_element_type("header")

    assert host._subtype_var.get() == ""
    assert host._subtype_combo.kwargs["state"] == "disabled"
