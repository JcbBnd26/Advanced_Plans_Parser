from types import SimpleNamespace

from plancheck.config import GroupingConfig
from plancheck.corrections.hierarchical_classifier import ClassificationResult
from scripts.gui.event_handler import EventHandlerMixin
from scripts.gui.gui import GuiState


class _FakeClassifier:
    def __init__(self, label: str, confidence: float, model_path: str) -> None:
        self.label = label
        self.confidence = confidence
        self.model_path = model_path

    def model_exists(self) -> bool:
        return True

    def predict(self, features):
        return self.label, self.confidence


class _Handler(EventHandlerMixin):
    pass


def test_predict_model_suggestion_uses_direct_classifier_when_hierarchy_off() -> None:
    handler = _Handler()
    handler.state = GuiState()
    handler.state.set_config(GroupingConfig(ml_hierarchical_enabled=False))
    handler._classifier = _FakeClassifier(
        label="title_block",
        confidence=0.83,
        model_path="data/element_classifier.pkl",
    )

    prediction = handler._predict_model_suggestion({"x0": 1.0}, text="A1")

    assert prediction == (
        "title_block",
        0.83,
        "Model suggests: title_block (83%)",
    )


def test_predict_model_suggestion_uses_hierarchical_subtype(monkeypatch) -> None:
    handler = _Handler()
    handler.state = GuiState()
    handler.state.set_config(GroupingConfig(ml_hierarchical_enabled=True))
    handler._classifier = _FakeClassifier(
        label="title_block",
        confidence=0.88,
        model_path="data/element_classifier.pkl",
    )

    monkeypatch.setattr(
        "plancheck.corrections.hierarchical_classifier.classify_element",
        lambda *args, **kwargs: ClassificationResult(
            label="page_title",
            confidence=0.74,
            family="title",
            family_confidence=0.88,
            subtype="page_title",
            subtype_confidence=0.74,
        ),
    )

    prediction = handler._predict_model_suggestion({"x0": 1.0}, text="SHEET 1")

    assert prediction == (
        "page_title",
        0.74,
        "Model suggests: page_title (74%) [Stage 2]",
    )


def test_predict_model_suggestion_falls_back_to_gui_label_for_title_family(
    monkeypatch,
) -> None:
    handler = _Handler()
    handler.state = GuiState()
    handler.state.set_config(GroupingConfig(ml_hierarchical_enabled=True))
    handler._classifier = _FakeClassifier(
        label="title_block",
        confidence=0.91,
        model_path="data/element_classifier.pkl",
    )

    monkeypatch.setattr(
        "plancheck.corrections.hierarchical_classifier.classify_element",
        lambda *args, **kwargs: ClassificationResult(
            label="title",
            confidence=0.91,
            family="title",
            family_confidence=0.91,
            stage2_skipped=True,
        ),
    )

    prediction = handler._predict_model_suggestion({"x0": 1.0}, text="TITLE")

    assert prediction == (
        "title_block",
        0.91,
        "Model suggests: title_block (91%) [Stage 1]",
    )


def test_predict_model_suggestion_details_surface_low_confidence_candidates(
    monkeypatch,
) -> None:
    handler = _Handler()
    handler.state = GuiState()
    handler.state.set_config(GroupingConfig(ml_hierarchical_enabled=True))
    handler._classifier = _FakeClassifier(
        label="title_block",
        confidence=0.87,
        model_path="data/element_classifier.pkl",
    )

    monkeypatch.setattr(
        "plancheck.corrections.hierarchical_classifier.classify_element",
        lambda *args, **kwargs: ClassificationResult(
            label="page_title",
            confidence=0.45,
            family="title",
            family_confidence=0.87,
            subtype="page_title",
            subtype_confidence=0.45,
            low_confidence=True,
            stage2_candidates=[("page_title", 0.45), ("plan_title", 0.40)],
        ),
    )

    prediction = handler._predict_model_suggestion_details({"x0": 1.0}, text="S1")

    assert prediction == {
        "label": "page_title",
        "confidence": 0.45,
        "text": "Model suggests: page_title (45%) [Stage 2, low confidence]",
        "detail_text": "Review: low-confidence title subtype. Alternatives: page_title 45%, plan_title 40%.",
    }


def test_predict_model_suggestion_details_note_when_stage2_unavailable(
    monkeypatch,
) -> None:
    handler = _Handler()
    handler.state = GuiState()
    handler.state.set_config(GroupingConfig(ml_hierarchical_enabled=True))
    handler._classifier = _FakeClassifier(
        label="title_block",
        confidence=0.91,
        model_path="data/element_classifier.pkl",
    )

    monkeypatch.setattr(
        "plancheck.corrections.hierarchical_classifier.classify_element",
        lambda *args, **kwargs: ClassificationResult(
            label="title",
            confidence=0.91,
            family="title",
            family_confidence=0.91,
            stage2_skipped=True,
        ),
    )

    prediction = handler._predict_model_suggestion_details({"x0": 1.0}, text="T")

    assert prediction == {
        "label": "title_block",
        "confidence": 0.91,
        "text": "Model suggests: title_block (91%) [Stage 1]",
        "detail_text": "Review: Stage 2 title refinement is unavailable, so this remains a Stage 1 title_block suggestion.",
    }
