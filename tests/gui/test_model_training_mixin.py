from types import SimpleNamespace

from scripts.gui.mixins.model_training import ModelTrainingMixin


class _MixinHost(ModelTrainingMixin):
    pass


def test_format_retrain_status_reports_stage2_training() -> None:
    host = _MixinHost()

    result = SimpleNamespace(
        error="",
        rolled_back=False,
        accepted=True,
        metrics={"f1_weighted": 0.82},
        stage2_trained=True,
        stage2_metrics={"f1_weighted": 0.71},
        stage2_error="",
        stage2_skipped_reason="",
    )

    text, color = host._format_retrain_status(result, action_label="Model trained")

    assert text == "Model trained — S1 F1 82.0%; S2 F1 71.0%"
    assert color == "green"


def test_format_retrain_status_reports_stage2_skip() -> None:
    host = _MixinHost()

    result = SimpleNamespace(
        error="",
        rolled_back=False,
        accepted=True,
        metrics={"f1_weighted": 0.82},
        stage2_trained=False,
        stage2_metrics={},
        stage2_error="",
        stage2_skipped_reason="only 4 subtype examples (need >= 10)",
    )

    text, color = host._format_retrain_status(
        result,
        action_label="Bootstrapped",
        bootstrap_examples=12,
    )

    assert (
        text
        == "Bootstrapped (12 examples) — S1 F1 82.0%; Stage 2 skipped: only 4 subtype examples (need >= 10)"
    )
    assert color == "green"


def test_format_model_readiness_status_when_retrain_is_due() -> None:
    host = _MixinHost()

    text, color = host._format_model_readiness_status(
        model_exists=True,
        pending_corrections=60,
        threshold=50,
    )

    assert text == "Model loaded ✓ | Pending corrections: 60/50 — retrain recommended"
    assert color == "orange"


def test_format_model_readiness_status_when_model_missing() -> None:
    host = _MixinHost()

    text, color = host._format_model_readiness_status(
        model_exists=False,
        pending_corrections=12,
        threshold=50,
    )

    assert text == "No model trained | Pending corrections: 12/50"
    assert color == "gray"


def test_format_annotation_runtime_summary_reports_drift_and_retrain() -> None:
    host = _MixinHost()
    host.state = SimpleNamespace(
        config=SimpleNamespace(
            ml_hierarchical_enabled=True,
            ml_drift_enabled=True,
            ml_drift_stats_path="data/drift_stats.json",
        )
    )

    text = host._format_annotation_runtime_summary(
        pending_corrections=60,
        threshold=50,
        active_drift_text="Drift: active on 2/9 detections",
    )

    assert (
        text
        == "Routing: hierarchical | Drift: active on 2/9 detections | Retrain: 60/50 pending (recommended)"
    )


def test_format_annotation_runtime_summary_reflects_disabled_drift() -> None:
    host = _MixinHost()
    host.state = SimpleNamespace(
        config=SimpleNamespace(
            ml_hierarchical_enabled=False,
            ml_drift_enabled=False,
            ml_drift_stats_path="data/custom_drift.json",
        )
    )

    text = host._format_annotation_runtime_summary(
        pending_corrections=4,
        threshold=50,
        active_drift_text="",
    )

    assert text == "Routing: Stage 1 only | Drift: disabled | Retrain: 4/50 pending"


# ---------------------------------------------------------------------------
# Thread-path tests for _on_train_model
# ---------------------------------------------------------------------------

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_thread_host(tmp_path: Path) -> "_ThreadTestHost":
    """Build a minimal ModelTrainingMixin host suitable for thread tests."""

    class _ThreadTestHost(ModelTrainingMixin):
        def __init__(self) -> None:
            self._closing = False
            self._train_gen = 0
            self._train_cancel_event = threading.Event()
            self._last_metrics = None
            self._last_stage2_metrics = None

            # root.after() calls its callback immediately so _safe_after works
            # synchronously inside the daemon thread.
            self.root = MagicMock()
            self.root.winfo_exists.return_value = True
            self.root.after.side_effect = lambda delay, cb: cb()

            self._model_status_label = MagicMock()
            self.state = SimpleNamespace(
                config=SimpleNamespace(
                    ml_model_path=str(tmp_path / "model.pkl"),
                    ml_stage2_model_path=str(tmp_path / "model_stage2.pkl"),
                )
            )

        def _reload_classifiers(self) -> None:
            """No-op — avoids touching the filesystem in unit tests."""

    return _ThreadTestHost()


def _wait_for_final_configure(host: "object", *, timeout: float = 10.0) -> list[dict]:
    """Block until the label is configured with a non-orange foreground.

    Returns the list of all kwargs dicts passed to ``configure``.
    """
    done = threading.Event()
    collected: list[dict] = []

    def _side_effect(**kwargs: object) -> None:
        collected.append(dict(kwargs))
        if kwargs.get("foreground") not in (None, "orange"):
            done.set()

    host._model_status_label.configure.side_effect = _side_effect
    return done, collected


def test_on_train_model_updates_label_on_success(tmp_path: Path) -> None:
    """Training thread sets the status label to green with an F1 score."""
    host = _make_thread_host(tmp_path)
    done, collected = _wait_for_final_configure(host)

    mock_result = SimpleNamespace(
        error="",
        rolled_back=False,
        accepted=True,
        metrics={"f1_weighted": 0.82},
        stage2_trained=False,
        stage2_metrics={},
        stage2_error="",
        stage2_skipped_reason="",
    )

    mock_store = MagicMock()
    mock_store.close = MagicMock()

    with patch(
        "plancheck.corrections.retrain_trigger.auto_retrain",
        return_value=mock_result,
    ):
        with patch(
            "plancheck.corrections.store.CorrectionStore",
            return_value=mock_store,
        ):
            host._on_train_model()
            assert done.wait(timeout=10.0), "Training thread did not complete"

    final = collected[-1]
    assert "S1 F1" in final.get("text", ""), f"Unexpected label text: {final}"
    assert final.get("foreground") == "green"


def test_on_train_model_surfaces_exception(tmp_path: Path) -> None:
    """When training raises, the status label shows 'Train failed:' in red."""
    host = _make_thread_host(tmp_path)
    done, collected = _wait_for_final_configure(host)

    with patch(
        "plancheck.corrections.store.CorrectionStore",
        side_effect=RuntimeError("db connection failed"),
    ):
        host._on_train_model()
        assert done.wait(timeout=10.0), "Error path did not complete"

    final = collected[-1]
    assert "Train failed:" in final.get("text", ""), f"Unexpected label text: {final}"
    assert final.get("foreground") == "red"
