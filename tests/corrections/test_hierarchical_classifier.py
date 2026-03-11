"""Tests for the hierarchical two-stage classification system.

Covers:
- TitleSubtypeClassifier (subtype_classifier.py)
  - encode_subtype_features adds the three extra relational keys
  - featurize_title_subtype helper
  - predict / predict_top_k against a trained model
  - model_exists returns False when file absent
- HierarchicalClassifier (hierarchical_classifier.py)
  - Stage-1 confidence < threshold → low_confidence result
  - Stage-1 non-title family → Stage-2 skipped
  - Stage-1 model absent → empty label, low_confidence
  - Stage-1 title + Stage-2 absent → title fallback
  - Stage-1 title + Stage-2 confident → subtype label
  - Stage-1 title + Stage-2 uncertain + LLM disabled → low_confidence
  - Stage-1 title + Stage-2 uncertain + LLM enabled but unavailable → fallback
- LLM tiebreaker (llm_checks.py)
  - llm_classify_title_subtype returns ("", 0.0) when provider unavailable
  - returns ("", 0.0) when text is empty
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from plancheck.corrections.hierarchical_classifier import (
    STAGE1_CONFIDENCE_THRESHOLD,
    STAGE2_CONFIDENCE_THRESHOLD,
    TITLE_FAMILY_LABEL,
    ClassificationResult,
    classify_element,
)
from plancheck.corrections.subtype_classifier import (
    TITLE_SUBTYPES,
    TitleSubtypeClassifier,
    encode_subtype_features,
    featurize_title_subtype,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_features() -> dict:
    """Minimal feature dict as produced by featurize()."""
    return {
        "font_size_pt": 12.0,
        "font_size_max_pt": 14.0,
        "font_size_min_pt": 10.0,
        "is_all_caps": 1,
        "is_bold": 1,
        "token_count": 3,
        "row_count": 1,
        "x_frac": 0.1,
        "y_frac": 0.05,
        "x_center_frac": 0.25,
        "y_center_frac": 0.07,
        "width_frac": 0.5,
        "height_frac": 0.03,
        "aspect_ratio": 12.0,
        "contains_digit": 0,
        "starts_with_digit": 0,
        "has_colon": 0,
        "has_period_after_num": 0,
        "text_length": 15,
        "avg_chars_per_token": 5.0,
        "zone": "page",
        "neighbor_count": 1,
        "unique_word_ratio": 1.0,
        "uppercase_word_frac": 1.0,
        "avg_word_length": 4.0,
        "kw_notes_pattern": 0,
        "kw_header_pattern": 1,
        "kw_legend_pattern": 0,
        "kw_abbreviation_pattern": 0,
        "kw_revision_pattern": 0,
        "kw_title_block_pattern": 0,
        "kw_detail_pattern": 0,
        "text_density": 0.001,
        "x_dist_to_right_margin": 0.4,
        "line_width_variance": 0.0,
        "mean_token_confidence": 0.95,
        "min_token_confidence": 0.90,
        "is_below_header": 0,
        "header_distance_pts": 500.0,
        "sibling_count": 0,
        "column_position_index": 1,
        "starts_with_note_number": 0,
    }


@pytest.fixture
def training_jsonl_subtype(tmp_path: Path, base_features: dict) -> Path:
    """Create a small JSONL file with title subtype labels for training."""
    path = tmp_path / "subtype_train.jsonl"
    lines: list[str] = []
    labels = TITLE_SUBTYPES  # 7 labels
    for i in range(70):  # 10 examples per label
        feats = dict(base_features)
        feats["near_north_arrow"] = 0
        feats["near_detail_bubble"] = 0
        feats["near_section_arrow"] = 0
        label = labels[i % len(labels)]
        split = "train" if i < 56 else "val"
        feats["x_frac"] = 0.05 * (i % 10)
        feats["y_frac"] = 0.03 * (i % 7)
        lines.append(json.dumps({"features": feats, "label": label, "split": split}))
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# TitleSubtypeClassifier — feature encoding
# ---------------------------------------------------------------------------


class TestEncodeSubtypeFeatures:
    """encode_subtype_features appends extra relational keys."""

    def test_output_longer_than_base(self, base_features: dict) -> None:
        from plancheck.corrections.classifier import encode_features, _NUMERIC_KEYS, ZONE_VALUES

        base_dim = len(_NUMERIC_KEYS) + len(ZONE_VALUES)
        extended = featurize_title_subtype(
            base_features, near_north_arrow=True
        )
        out = encode_subtype_features(extended)
        assert len(out) == base_dim + 3

    def test_extra_keys_default_zero(self, base_features: dict) -> None:
        extended = featurize_title_subtype(base_features)
        out = encode_subtype_features(extended)
        # Last 3 values are the extra keys; should be 0
        assert out[-3] == 0.0  # near_north_arrow
        assert out[-2] == 0.0  # near_detail_bubble
        assert out[-1] == 0.0  # near_section_arrow

    def test_near_north_arrow_flag(self, base_features: dict) -> None:
        extended = featurize_title_subtype(
            base_features, near_north_arrow=True
        )
        out = encode_subtype_features(extended)
        assert out[-3] == 1.0

    def test_near_detail_bubble_flag(self, base_features: dict) -> None:
        extended = featurize_title_subtype(
            base_features, near_detail_bubble=True
        )
        out = encode_subtype_features(extended)
        assert out[-2] == 1.0

    def test_near_section_arrow_flag(self, base_features: dict) -> None:
        extended = featurize_title_subtype(
            base_features, near_section_arrow=True
        )
        out = encode_subtype_features(extended)
        assert out[-1] == 1.0

    def test_featurize_title_subtype_preserves_base(
        self, base_features: dict
    ) -> None:
        extended = featurize_title_subtype(base_features, near_north_arrow=True)
        assert extended["font_size_pt"] == base_features["font_size_pt"]
        assert extended["near_north_arrow"] == 1
        assert extended["near_detail_bubble"] == 0
        assert extended["near_section_arrow"] == 0

    def test_does_not_mutate_input(self, base_features: dict) -> None:
        orig = dict(base_features)
        featurize_title_subtype(base_features, near_north_arrow=True)
        assert base_features == orig


# ---------------------------------------------------------------------------
# TitleSubtypeClassifier — model_exists / predict guards
# ---------------------------------------------------------------------------


class TestTitleSubtypeClassifier:
    def test_model_exists_false_when_file_absent(self, tmp_path: Path) -> None:
        clf = TitleSubtypeClassifier(model_path=tmp_path / "nonexistent.pkl")
        assert clf.model_exists() is False

    def test_model_exists_true_after_training(
        self,
        tmp_path: Path,
        base_features: dict,
        training_jsonl_subtype: Path,
    ) -> None:
        model_path = tmp_path / "subtype.pkl"
        clf = TitleSubtypeClassifier(model_path=model_path)
        clf.train(training_jsonl_subtype)
        assert clf.model_exists() is True

    def test_predict_returns_known_subtype(
        self,
        tmp_path: Path,
        base_features: dict,
        training_jsonl_subtype: Path,
    ) -> None:
        model_path = tmp_path / "subtype.pkl"
        clf = TitleSubtypeClassifier(model_path=model_path)
        clf.train(training_jsonl_subtype)

        extended = featurize_title_subtype(base_features)
        label, conf = clf.predict(extended)

        assert label in TITLE_SUBTYPES
        assert 0.0 <= conf <= 1.0

    def test_predict_top_k_returns_k_results(
        self,
        tmp_path: Path,
        base_features: dict,
        training_jsonl_subtype: Path,
    ) -> None:
        model_path = tmp_path / "subtype.pkl"
        clf = TitleSubtypeClassifier(model_path=model_path)
        clf.train(training_jsonl_subtype)

        extended = featurize_title_subtype(base_features)
        top2 = clf.predict_top_k(extended, k=2)

        assert len(top2) == 2
        assert top2[0][1] >= top2[1][1]  # Ordered by confidence descending
        assert all(lbl in TITLE_SUBTYPES for lbl, _ in top2)

    def test_train_returns_metrics_dict(
        self,
        tmp_path: Path,
        training_jsonl_subtype: Path,
    ) -> None:
        model_path = tmp_path / "subtype.pkl"
        clf = TitleSubtypeClassifier(model_path=model_path)
        metrics = clf.train(training_jsonl_subtype)

        assert "accuracy" in metrics
        assert "n_train" in metrics
        assert metrics["n_train"] > 0

    def test_get_feature_importance_returns_dict(
        self,
        tmp_path: Path,
        training_jsonl_subtype: Path,
    ) -> None:
        model_path = tmp_path / "subtype.pkl"
        clf = TitleSubtypeClassifier(model_path=model_path)
        clf.train(training_jsonl_subtype)
        importance = clf.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) > 0


# ---------------------------------------------------------------------------
# ClassificationResult dataclass
# ---------------------------------------------------------------------------


class TestClassificationResult:
    def test_default_values(self) -> None:
        result = ClassificationResult(
            label="title",
            confidence=0.9,
            family="title",
            family_confidence=0.9,
        )
        assert result.subtype is None
        assert result.low_confidence is False
        assert result.stage2_skipped is False
        assert result.llm_used is False
        assert result.stage2_candidates == []

    def test_full_construction(self) -> None:
        result = ClassificationResult(
            label="page_title",
            confidence=0.85,
            family="title",
            family_confidence=0.92,
            subtype="page_title",
            subtype_confidence=0.85,
            stage2_candidates=[("page_title", 0.85), ("plan_title", 0.10)],
        )
        assert result.label == "page_title"
        assert result.family == "title"
        assert result.llm_used is False


# ---------------------------------------------------------------------------
# classify_element routing logic
# ---------------------------------------------------------------------------


def _make_s1_mock(label: str, conf: float) -> MagicMock:
    """Build a mock Stage-1 classifier that returns (label, conf)."""
    mock = MagicMock()
    mock.model_exists.return_value = True
    mock.predict.return_value = (label, conf)
    return mock


def _make_s2_mock(
    top_k_results: list[tuple[str, float]],
) -> MagicMock:
    """Build a mock Stage-2 classifier."""
    mock = MagicMock()
    mock.model_exists.return_value = True
    mock.predict_top_k.return_value = top_k_results
    return mock


class TestClassifyElement:
    """Unit tests for classify_element routing contract."""

    def test_stage1_model_absent_returns_empty_label(
        self, base_features: dict, tmp_path: Path
    ) -> None:
        result = classify_element(
            base_features,
            stage1_model_path=tmp_path / "missing.pkl",
        )
        assert result.label == ""
        assert result.low_confidence is True
        assert result.stage2_skipped is True

    def test_stage1_low_confidence_flagged(
        self, base_features: dict, tmp_path: Path
    ) -> None:
        s1_path = tmp_path / "s1.pkl"
        s2_path = tmp_path / "s2.pkl"
        mock_s1 = _make_s1_mock("notes", 0.4)  # below threshold 0.7

        with patch(
            "plancheck.corrections.hierarchical_classifier._get_stage1",
            return_value=mock_s1,
        ):
            result = classify_element(
                base_features,
                stage1_model_path=s1_path,
                stage2_model_path=s2_path,
            )

        assert result.label == "notes"
        assert result.low_confidence is True

    def test_stage1_non_title_skips_stage2(
        self, base_features: dict, tmp_path: Path
    ) -> None:
        s1_path = tmp_path / "s1.pkl"
        s2_path = tmp_path / "s2.pkl"
        mock_s1 = _make_s1_mock("notes", 0.95)

        with patch(
            "plancheck.corrections.hierarchical_classifier._get_stage1",
            return_value=mock_s1,
        ):
            result = classify_element(
                base_features,
                stage1_model_path=s1_path,
                stage2_model_path=s2_path,
            )

        assert result.label == "notes"
        assert result.family == "notes"
        assert result.stage2_skipped is True
        assert result.subtype is None

    def test_stage1_title_stage2_absent_returns_title(
        self, base_features: dict, tmp_path: Path
    ) -> None:
        s1_path = tmp_path / "s1.pkl"
        s2_path = tmp_path / "s2_missing.pkl"
        mock_s1 = _make_s1_mock(TITLE_FAMILY_LABEL, 0.88)
        mock_s2 = MagicMock()
        mock_s2.model_exists.return_value = False

        with (
            patch(
                "plancheck.corrections.hierarchical_classifier._get_stage1",
                return_value=mock_s1,
            ),
            patch(
                "plancheck.corrections.hierarchical_classifier._get_stage2",
                return_value=mock_s2,
            ),
        ):
            result = classify_element(
                base_features,
                stage1_model_path=s1_path,
                stage2_model_path=s2_path,
            )

        assert result.label == TITLE_FAMILY_LABEL
        assert result.stage2_skipped is True

    def test_stage1_title_stage2_confident_returns_subtype(
        self, base_features: dict, tmp_path: Path
    ) -> None:
        s1_path = tmp_path / "s1.pkl"
        s2_path = tmp_path / "s2.pkl"
        mock_s1 = _make_s1_mock(TITLE_FAMILY_LABEL, 0.88)
        mock_s2 = _make_s2_mock([("page_title", 0.85), ("plan_title", 0.10)])

        with (
            patch(
                "plancheck.corrections.hierarchical_classifier._get_stage1",
                return_value=mock_s1,
            ),
            patch(
                "plancheck.corrections.hierarchical_classifier._get_stage2",
                return_value=mock_s2,
            ),
        ):
            result = classify_element(
                base_features,
                stage1_model_path=s1_path,
                stage2_model_path=s2_path,
            )

        assert result.label == "page_title"
        assert result.family == TITLE_FAMILY_LABEL
        assert result.subtype == "page_title"
        assert result.low_confidence is False
        assert result.llm_used is False
        assert len(result.stage2_candidates) == 2

    def test_stage2_uncertain_llm_disabled_returns_low_confidence(
        self, base_features: dict, tmp_path: Path
    ) -> None:
        s1_path = tmp_path / "s1.pkl"
        s2_path = tmp_path / "s2.pkl"
        mock_s1 = _make_s1_mock(TITLE_FAMILY_LABEL, 0.88)
        mock_s2 = _make_s2_mock(
            [("page_title", 0.45), ("plan_title", 0.40)]  # both below threshold 0.6
        )

        with (
            patch(
                "plancheck.corrections.hierarchical_classifier._get_stage1",
                return_value=mock_s1,
            ),
            patch(
                "plancheck.corrections.hierarchical_classifier._get_stage2",
                return_value=mock_s2,
            ),
        ):
            result = classify_element(
                base_features,
                stage1_model_path=s1_path,
                stage2_model_path=s2_path,
                enable_llm=False,
            )

        assert result.low_confidence is True
        assert result.llm_used is False
        assert result.subtype == "page_title"  # best guess, but low confidence

    def test_stage2_uncertain_llm_unavailable_falls_back(
        self, base_features: dict, tmp_path: Path
    ) -> None:
        s1_path = tmp_path / "s1.pkl"
        s2_path = tmp_path / "s2.pkl"
        mock_s1 = _make_s1_mock(TITLE_FAMILY_LABEL, 0.90)
        mock_s2 = _make_s2_mock([("map_title", 0.50), ("page_title", 0.35)])

        with (
            patch(
                "plancheck.corrections.hierarchical_classifier._get_stage1",
                return_value=mock_s1,
            ),
            patch(
                "plancheck.corrections.hierarchical_classifier._get_stage2",
                return_value=mock_s2,
            ),
            patch(
                "plancheck.corrections.hierarchical_classifier._run_llm_tiebreaker",
                return_value=("", 0.0),
            ),
        ):
            result = classify_element(
                base_features,
                stage1_model_path=s1_path,
                stage2_model_path=s2_path,
                enable_llm=True,
            )

        # Falls back to Stage-2 argmax with low_confidence
        assert result.label == "map_title"
        assert result.low_confidence is True
        assert result.llm_used is False

    def test_stage2_uncertain_llm_returns_result(
        self, base_features: dict, tmp_path: Path
    ) -> None:
        s1_path = tmp_path / "s1.pkl"
        s2_path = tmp_path / "s2.pkl"
        mock_s1 = _make_s1_mock(TITLE_FAMILY_LABEL, 0.88)
        mock_s2 = _make_s2_mock([("map_title", 0.50), ("page_title", 0.35)])

        with (
            patch(
                "plancheck.corrections.hierarchical_classifier._get_stage1",
                return_value=mock_s1,
            ),
            patch(
                "plancheck.corrections.hierarchical_classifier._get_stage2",
                return_value=mock_s2,
            ),
            patch(
                "plancheck.corrections.hierarchical_classifier._run_llm_tiebreaker",
                return_value=("map_title", 0.82),
            ),
        ):
            result = classify_element(
                base_features,
                "VICINITY MAP",
                stage1_model_path=s1_path,
                stage2_model_path=s2_path,
                enable_llm=True,
            )

        assert result.label == "map_title"
        assert result.llm_used is True
        assert result.low_confidence is False


# ---------------------------------------------------------------------------
# LLM tiebreaker
# ---------------------------------------------------------------------------


class TestLlmClassifyTitleSubtype:
    def test_returns_empty_when_provider_unavailable(
        self, base_features: dict
    ) -> None:
        from plancheck.checks.llm_checks import llm_classify_title_subtype

        label, conf = llm_classify_title_subtype(
            text="FLOOR PLAN",
            features=base_features,
            candidates=[("page_title", 0.45), ("plan_title", 0.40)],
            provider="nonexistent_provider",
        )
        assert label == ""
        assert conf == 0.0

    def test_returns_empty_for_blank_text(self, base_features: dict) -> None:
        from plancheck.checks.llm_checks import llm_classify_title_subtype

        label, conf = llm_classify_title_subtype(
            text="   ",
            features=base_features,
            candidates=[("page_title", 0.45), ("plan_title", 0.40)],
        )
        assert label == ""
        assert conf == 0.0

    def test_returns_empty_for_empty_text(self, base_features: dict) -> None:
        from plancheck.checks.llm_checks import llm_classify_title_subtype

        label, conf = llm_classify_title_subtype(
            text="",
            features=base_features,
            candidates=[("page_title", 0.45), ("plan_title", 0.40)],
        )
        assert label == ""
        assert conf == 0.0

    def test_mocked_llm_returns_valid_subtype(self, base_features: dict) -> None:
        from plancheck.checks.llm_checks import llm_classify_title_subtype

        mock_client = MagicMock()
        mock_client.chat_structured.return_value = (
            {"subtype": "page_title", "confidence": 0.91},
            MagicMock(),
        )

        with (
            patch(
                "plancheck.checks.llm_checks.is_llm_available", return_value=True
            ),
            patch(
                "plancheck.checks.llm_checks.LLMClient",
                return_value=mock_client,
            ),
        ):
            label, conf = llm_classify_title_subtype(
                text="FLOOR PLAN",
                features=base_features,
                candidates=[("page_title", 0.45), ("plan_title", 0.40)],
                provider="ollama",
            )

        assert label == "page_title"
        assert conf == pytest.approx(0.91)

    def test_unknown_subtype_from_llm_ignored(self, base_features: dict) -> None:
        from plancheck.checks.llm_checks import llm_classify_title_subtype

        mock_client = MagicMock()
        mock_client.chat_structured.return_value = (
            {"subtype": "made_up_type", "confidence": 0.99},
            MagicMock(),
        )

        with (
            patch(
                "plancheck.checks.llm_checks.is_llm_available", return_value=True
            ),
            patch(
                "plancheck.checks.llm_checks.LLMClient",
                return_value=mock_client,
            ),
        ):
            label, conf = llm_classify_title_subtype(
                text="FLOOR PLAN",
                features=base_features,
                candidates=[("page_title", 0.45)],
                provider="ollama",
            )

        assert label == ""
        assert conf == 0.0

    def test_llm_exception_returns_empty(self, base_features: dict) -> None:
        from plancheck.checks.llm_checks import llm_classify_title_subtype

        mock_client = MagicMock()
        mock_client.chat_structured.side_effect = RuntimeError("LLM timeout")

        with (
            patch(
                "plancheck.checks.llm_checks.is_llm_available", return_value=True
            ),
            patch(
                "plancheck.checks.llm_checks.LLMClient",
                return_value=mock_client,
            ),
        ):
            label, conf = llm_classify_title_subtype(
                text="FLOOR PLAN",
                features=base_features,
                candidates=[("page_title", 0.45)],
                provider="ollama",
            )

        assert label == ""
        assert conf == 0.0
