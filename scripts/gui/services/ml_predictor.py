"""ML prediction service for the annotation tab.

Wraps classifier access, hierarchical classification, and stage-2
candidate formatting.  Consumed by SelectTool (deferred prediction)
and DrawTool (auto-label on new boxes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class MLPredictor:
    """Encapsulates ML-assisted label suggestions for the annotation UI."""

    def __init__(self, tab: Any) -> None:
        self._tab = tab

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def format_stage2_candidates(candidates: list[tuple[str, float]]) -> str:
        """Format Stage-2 alternatives for compact inspector display."""
        if not candidates:
            return ""
        return ", ".join(
            f"{label} {confidence:.0%}" for label, confidence in candidates
        )

    def get_configured_classifier(self):
        """Return the Stage-1 classifier selected by the current GUI config."""
        from plancheck.corrections.classifier import ElementClassifier

        tab = self._tab
        cfg = getattr(getattr(tab, "state", None), "config", None)
        model_path = Path(getattr(cfg, "ml_model_path", tab._classifier.model_path))
        current_model_path = Path(getattr(tab._classifier, "model_path", model_path))
        if current_model_path == model_path:
            return tab._classifier
        return ElementClassifier(model_path=model_path)

    # ── Public API ──────────────────────────────────────────────

    def predict(
        self,
        features: dict[str, Any],
        *,
        text: str = "",
    ) -> tuple[str, float, str] | None:
        """Predict a GUI-facing label suggestion using current ML settings.

        Returns ``(label, confidence, display_text)`` or ``None``.
        """
        suggestion = self.predict_details(features, text=text)
        if suggestion is None:
            return None
        return suggestion["label"], suggestion["confidence"], suggestion["text"]

    def predict_details(
        self,
        features: dict[str, Any],
        *,
        text: str = "",
    ) -> dict[str, Any] | None:
        """Return the suggestion label plus review metadata for the inspector."""
        classifier = self.get_configured_classifier()
        if not classifier.model_exists():
            return None

        raw_label, raw_conf = classifier.predict(features)
        tab = self._tab
        cfg = getattr(getattr(tab, "state", None), "config", None)
        if not getattr(cfg, "ml_hierarchical_enabled", False):
            return {
                "label": raw_label,
                "confidence": raw_conf,
                "text": f"Model suggests: {raw_label} ({raw_conf:.0%})",
                "detail_text": "",
            }

        from plancheck.corrections.hierarchical_classifier import classify_element

        result = classify_element(
            features,
            text=text,
            stage1_model_path=Path(cfg.ml_model_path),
            stage2_model_path=Path(cfg.ml_stage2_model_path),
            enable_llm=False,
        )

        display_label = result.label
        display_conf = result.confidence
        stage_suffix = "Stage 1"
        detail_parts: list[str] = []

        if result.stage2_skipped and result.label == "title":
            display_label = raw_label
            display_conf = raw_conf
            detail_parts.append(
                f"Routing: Stage 1 {raw_label} ({raw_conf:.0%}) -> Stage 2 skipped."
            )
            detail_parts.append(
                "Review: Stage 2 title refinement is unavailable, so this remains a "
                f"Stage 1 {raw_label} suggestion."
            )
        elif result.subtype:
            stage_suffix = "Stage 2"
            detail_parts.append(
                f"Routing: Stage 1 {raw_label} ({raw_conf:.0%}) -> "
                f"Stage 2 {display_label} ({display_conf:.0%})."
            )

        if result.low_confidence and stage_suffix == "Stage 2":
            stage_suffix = "Stage 2, low confidence"
            alternatives = self.format_stage2_candidates(result.stage2_candidates)
            detail_parts.append("Review: low-confidence title subtype.")
            if alternatives:
                detail_parts.append(f"Alternatives: {alternatives}.")
        elif result.llm_used:
            detail_parts.append(
                f"Routing: Stage 1 {raw_label} ({raw_conf:.0%}) -> "
                f"LLM tiebreak {display_label} ({display_conf:.0%})."
            )
            detail_parts.append(
                "Review: resolved by LLM tiebreaker after close Stage 2 candidates."
            )

        detail_text = " ".join(detail_parts).strip()

        return {
            "label": display_label,
            "confidence": display_conf,
            "text": f"Model suggests: {display_label} ({display_conf:.0%}) [{stage_suffix}]",
            "detail_text": detail_text,
        }
