import logging
from pathlib import Path

import pytest

from plancheck.config import GroupingConfig
from scripts.gui.gui import GuiState


def test_gui_state_notify_logs_and_continues(caplog: pytest.LogCaptureFixture) -> None:
    state = GuiState()

    calls: list[str] = []

    def bad_subscriber() -> None:
        calls.append("bad")
        raise RuntimeError("boom")

    def good_subscriber() -> None:
        calls.append("good")

    state.subscribe("pdf_changed", bad_subscriber)
    state.subscribe("pdf_changed", good_subscriber)

    with caplog.at_level(logging.ERROR):
        state.notify("pdf_changed")

    assert calls == ["bad", "good"]
    assert any(
        "GuiState subscriber failed" in record.getMessage() for record in caplog.records
    )


def test_gui_state_set_config_tracks_path() -> None:
    state = GuiState()

    cfg = GroupingConfig(ml_hierarchical_enabled=True)
    path = Path("configs/hierarchical.yaml")

    state.set_config(cfg, config_file_path=path)

    assert state.config.ml_hierarchical_enabled is True
    assert state.config_file_path == path


def test_gui_state_queue_config_load_notifies_subscribers() -> None:
    state = GuiState()
    calls: list[dict] = []

    def on_load() -> None:
        calls.append(state.pending_config or {})

    state.subscribe("load_config", on_load)
    state.queue_config_load({"ml_hierarchical_enabled": True})

    assert calls == [{"ml_hierarchical_enabled": True}]
    assert state.pending_config == {"ml_hierarchical_enabled": True}
