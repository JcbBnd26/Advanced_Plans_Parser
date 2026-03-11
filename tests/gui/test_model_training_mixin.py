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
