import logging

import pytest

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
