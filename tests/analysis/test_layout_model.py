"""Tests for Phase 2.2 — LayoutLMv3 Layout Model.

Covers:
- LayoutPrediction dataclass (fields, to_dict)
- LayoutModel graceful degradation when torch/transformers missing
- is_layout_available() returns bool
- Config fields (ml_layout_enabled, ml_layout_model_path defaults)
- _aggregate_predictions logic with synthetic data
- predict_layout convenience wrapper fallback
- PageResult.layout_predictions field
- LAYOUT_LABELS constant

Note: Tests mock torch/transformers to avoid needing heavy deps in CI.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from plancheck.analysis.layout_model import (
    _MAX_SEQ_LENGTH,
    LAYOUT_LABELS,
    LayoutModel,
    LayoutPrediction,
    is_layout_available,
    predict_layout,
)
from plancheck.config import GroupingConfig

# ── LayoutPrediction dataclass ─────────────────────────────────────────


class TestLayoutPrediction:
    """LayoutPrediction dataclass fields and serialization."""

    def test_fields(self):
        pred = LayoutPrediction(
            label="notes",
            confidence=0.92,
            bbox=(10.0, 20.0, 300.0, 400.0),
            token_indices=[0, 1, 2],
        )
        assert pred.label == "notes"
        assert pred.confidence == 0.92
        assert pred.bbox == (10.0, 20.0, 300.0, 400.0)
        assert pred.token_indices == [0, 1, 2]

    def test_default_token_indices(self):
        pred = LayoutPrediction(label="border", confidence=0.5, bbox=(0, 0, 1, 1))
        assert pred.token_indices == []

    def test_to_dict(self):
        pred = LayoutPrediction(
            label="title_block",
            confidence=0.87654321,
            bbox=(5.0, 10.0, 200.0, 150.0),
            token_indices=[3, 4, 5],
        )
        d = pred.to_dict()
        assert d["label"] == "title_block"
        assert d["confidence"] == 0.8765  # rounded to 4 dp
        assert d["bbox"] == [5.0, 10.0, 200.0, 150.0]
        # token_indices not in to_dict (by design)
        assert "token_indices" not in d

    def test_to_dict_bbox_is_list(self):
        pred = LayoutPrediction(
            label="drawing",
            confidence=1.0,
            bbox=(0, 0, 100, 100),
        )
        assert isinstance(pred.to_dict()["bbox"], list)


# ── Config fields ──────────────────────────────────────────────────────


class TestLayoutConfig:
    """Config has ml_layout_enabled and ml_layout_model_path."""

    def test_defaults(self):
        cfg = GroupingConfig()
        assert cfg.ml_layout_enabled is False
        assert cfg.ml_layout_model_path == "microsoft/layoutlmv3-base"

    def test_custom_model_path(self):
        cfg = GroupingConfig(ml_layout_model_path="/my/finetuned/model")
        assert cfg.ml_layout_model_path == "/my/finetuned/model"

    def test_enabled_flag(self):
        cfg = GroupingConfig(ml_layout_enabled=True)
        assert cfg.ml_layout_enabled is True


# ── LAYOUT_LABELS constant ────────────────────────────────────────────


class TestLayoutLabels:
    """LAYOUT_LABELS vocabulary matches expectations."""

    def test_contains_expected_labels(self):
        for label in [
            "border",
            "drawing",
            "notes",
            "title_block",
            "legend",
            "abbreviations",
            "revisions",
            "details",
            "unknown",
        ]:
            assert label in LAYOUT_LABELS

    def test_unknown_is_last(self):
        assert LAYOUT_LABELS[-1] == "unknown"

    def test_nine_labels(self):
        assert len(LAYOUT_LABELS) == 9


# ── is_layout_available ───────────────────────────────────────────────


class TestIsLayoutAvailable:
    """Availability probe returns bool."""

    def test_returns_bool(self):
        result = is_layout_available()
        assert isinstance(result, bool)


# ── LayoutModel basic API ─────────────────────────────────────────────


class TestLayoutModelBasic:
    """LayoutModel instantiation and graceful degradation."""

    def test_init_defaults(self):
        model = LayoutModel()
        assert model.model_name == "microsoft/layoutlmv3-base"
        assert model.available is False  # not loaded yet

    def test_init_custom_path(self):
        model = LayoutModel(model_name_or_path="my/custom-model")
        assert model.model_name == "my/custom-model"

    def test_init_custom_num_labels(self):
        model = LayoutModel(num_labels=5)
        assert model._num_labels == 5

    def test_label_names(self):
        model = LayoutModel()
        names = model.label_names
        assert len(names) == 9
        assert names[0] == "border"

    def test_label_names_custom_num_labels(self):
        model = LayoutModel(num_labels=3)
        assert len(model.label_names) == 3
        assert model.label_names == ["border", "drawing", "notes"]

    def test_predict_returns_empty_when_unavailable(self):
        """When torch/transformers missing, predict() returns []."""
        model = LayoutModel()
        model._ensure_model = lambda: False  # Simulate unavailable

        from PIL import Image

        img = Image.new("RGB", (100, 100), "white")
        result = model.predict(img, [], 100.0, 100.0)
        assert result == []

    def test_predict_empty_tokens_returns_empty(self):
        """With no tokens, returns [] even if model is 'loaded'."""
        model = LayoutModel()
        model._ensure_model = lambda: True

        from PIL import Image

        img = Image.new("RGB", (100, 100), "white")
        result = model.predict(img, [], 100.0, 100.0)
        assert result == []


# ── _aggregate_predictions ─────────────────────────────────────────────


def _make_tokens(coords: list[tuple[float, float, float, float]]):
    """Create fake token objects from (x0, y0, x1, y1) tuples."""
    return [
        SimpleNamespace(x0=c[0], y0=c[1], x1=c[2], y1=c[3], text=f"tok{i}")
        for i, c in enumerate(coords)
    ]


class TestAggregateePredictions:
    """_aggregate_predictions groups contiguous same-label tokens."""

    def _run(self, pred_ids, pred_confs, token_coords):
        """Helper: call _aggregate_predictions on a LayoutModel instance."""
        model = LayoutModel()
        tokens = _make_tokens(token_coords)
        boxes = [[int(c[0]), int(c[1]), int(c[2]), int(c[3])] for c in token_coords]
        return model._aggregate_predictions(
            np.array(pred_ids),
            np.array(pred_confs),
            tokens,
            boxes,
            page_width=1000.0,
            page_height=1000.0,
        )

    def test_single_token_single_group(self):
        # pred_ids[0] is CLS → skip; pred_ids[1] maps to token 0
        results = self._run(
            pred_ids=[0, 2],  # CLS=border(ignored positionally), tok0=notes
            pred_confs=[0.9, 0.85],
            token_coords=[(10, 20, 50, 60)],
        )
        assert len(results) == 1
        assert results[0].label == "notes"
        assert results[0].confidence == pytest.approx(0.85)
        assert results[0].token_indices == [0]

    def test_two_same_label_tokens_merge(self):
        results = self._run(
            pred_ids=[0, 2, 2],  # CLS, notes, notes
            pred_confs=[0.0, 0.8, 0.9],
            token_coords=[(10, 10, 50, 30), (55, 10, 100, 30)],
        )
        assert len(results) == 1
        assert results[0].label == "notes"
        # bbox merges: x0=10, y0=10, x1=100, y1=30
        assert results[0].bbox == (10, 10, 100, 30)
        assert results[0].confidence == pytest.approx(0.85)
        assert results[0].token_indices == [0, 1]

    def test_two_different_labels_split(self):
        results = self._run(
            pred_ids=[0, 2, 3],  # CLS, notes, title_block
            pred_confs=[0.0, 0.8, 0.7],
            token_coords=[(10, 10, 50, 30), (200, 200, 300, 250)],
        )
        assert len(results) == 2
        assert results[0].label == "notes"
        assert results[1].label == "title_block"

    def test_three_groups(self):
        results = self._run(
            pred_ids=[0, 0, 0, 2, 2],  # CLS, border, border, notes, notes
            pred_confs=[0.0, 0.9, 0.8, 0.7, 0.6],
            token_coords=[
                (0, 0, 10, 10),
                (15, 0, 30, 10),
                (100, 100, 200, 200),
                (210, 100, 300, 200),
            ],
        )
        assert len(results) == 2
        assert results[0].label == "border"
        assert results[0].token_indices == [0, 1]
        assert results[1].label == "notes"
        assert results[1].token_indices == [2, 3]

    def test_out_of_range_label_id_maps_to_unknown(self):
        results = self._run(
            pred_ids=[0, 99],  # CLS, out-of-range
            pred_confs=[0.0, 0.5],
            token_coords=[(10, 10, 50, 50)],
        )
        assert len(results) == 1
        assert results[0].label == "unknown"

    def test_empty_tokens_returns_empty(self):
        results = self._run(
            pred_ids=[0],  # just CLS
            pred_confs=[0.0],
            token_coords=[],
        )
        assert results == []


# ── predict_layout convenience wrapper ─────────────────────────────────


class TestPredictLayoutConvenience:
    """predict_layout() convenience function."""

    def test_returns_empty_when_unavailable(self):
        """Without torch/transformers, returns an empty list."""
        from PIL import Image

        img = Image.new("RGB", (100, 100), "white")
        # predict_layout will try to load the model; if torch/transformers
        # aren't installed it gracefully returns []
        with patch(
            "plancheck.analysis.layout_model.LayoutModel._ensure_model",
            return_value=False,
        ):
            result = predict_layout(img, [], 100.0, 100.0)
        assert result == []


# ── PageResult integration ─────────────────────────────────────────────


class TestPageResultLayoutPredictions:
    """PageResult has layout_predictions field."""

    def test_field_exists_and_defaults_empty(self):
        from plancheck.pipeline import PageResult

        pr = PageResult()
        assert hasattr(pr, "layout_predictions")
        assert pr.layout_predictions == []

    def test_field_stores_predictions(self):
        from plancheck.pipeline import PageResult

        pred = LayoutPrediction(
            label="notes",
            confidence=0.9,
            bbox=(0, 0, 100, 100),
        )
        pr = PageResult()
        pr.layout_predictions = [pred]
        assert len(pr.layout_predictions) == 1
        assert pr.layout_predictions[0].label == "notes"


# ── LayoutModel.save / load stubs ─────────────────────────────────────


class TestLayoutModelSaveLoad:
    """save() raises when no model loaded; load() returns LayoutModel."""

    def test_save_raises_without_model(self):
        model = LayoutModel()
        with pytest.raises(RuntimeError, match="No model to save"):
            model.save("/tmp/nonexistent")

    def test_load_returns_layout_model(self):
        """load() creates a LayoutModel pointed at the given path."""
        with patch.object(LayoutModel, "_ensure_model", return_value=False):
            m = LayoutModel.load("/some/path")
        assert isinstance(m, LayoutModel)
        assert m.model_name == "/some/path"
