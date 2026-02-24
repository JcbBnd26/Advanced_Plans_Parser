"""Tests for plancheck.corrections.metrics."""

from __future__ import annotations

import pytest

from plancheck.corrections.metrics import compute_metrics, format_metrics_table

# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_perfect_predictions(self) -> None:
        y_true = ["a", "b", "c", "a", "b", "c"]
        y_pred = ["a", "b", "c", "a", "b", "c"]
        m = compute_metrics(y_true, y_pred)

        assert m["accuracy"] == 1.0
        for label in ["a", "b", "c"]:
            assert m["per_class"][label]["precision"] == 1.0
            assert m["per_class"][label]["recall"] == 1.0
            assert m["per_class"][label]["f1"] == 1.0

    def test_all_wrong(self) -> None:
        y_true = ["a", "a", "a"]
        y_pred = ["b", "b", "b"]
        m = compute_metrics(y_true, y_pred)

        assert m["accuracy"] == 0.0
        assert m["per_class"]["a"]["recall"] == 0.0
        assert m["per_class"]["b"]["precision"] == 0.0

    def test_partial_accuracy(self) -> None:
        y_true = ["a", "a", "b", "b"]
        y_pred = ["a", "b", "b", "a"]
        m = compute_metrics(y_true, y_pred)

        assert m["accuracy"] == 0.5

    def test_confusion_matrix_shape(self) -> None:
        y_true = ["a", "b", "c"]
        y_pred = ["a", "c", "b"]
        m = compute_metrics(y_true, y_pred, labels=["a", "b", "c"])

        cm = m["confusion_matrix"]
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)
        # "a" predicted correctly
        assert cm[0][0] == 1
        # "b" predicted as "c"
        assert cm[1][2] == 1
        # "c" predicted as "b"
        assert cm[2][1] == 1

    def test_labels_ordering(self) -> None:
        y_true = ["cat", "dog"]
        y_pred = ["cat", "cat"]
        m = compute_metrics(y_true, y_pred, labels=["cat", "dog"])

        assert m["labels"] == ["cat", "dog"]

    def test_labels_auto_derived(self) -> None:
        y_true = ["z", "a"]
        y_pred = ["z", "a"]
        m = compute_metrics(y_true, y_pred)
        assert m["labels"] == ["a", "z"]  # sorted

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            compute_metrics(["a"], ["a", "b"])

    def test_empty_lists(self) -> None:
        m = compute_metrics([], [])
        assert m["accuracy"] == 0.0

    def test_single_class(self) -> None:
        y_true = ["x", "x", "x"]
        y_pred = ["x", "x", "x"]
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == 1.0
        assert m["per_class"]["x"]["support"] == 3

    def test_support_sums(self) -> None:
        y_true = ["a", "a", "b", "c", "c", "c"]
        y_pred = ["a", "b", "b", "c", "a", "c"]
        m = compute_metrics(y_true, y_pred)
        assert m["per_class"]["a"]["support"] == 2
        assert m["per_class"]["b"]["support"] == 1
        assert m["per_class"]["c"]["support"] == 3

    def test_precision_recall_f1_values(self) -> None:
        """Binary case: TP=2, FP=1, FN=1 for class 'pos'."""
        y_true = ["pos", "pos", "pos", "neg"]
        y_pred = ["pos", "pos", "neg", "pos"]
        m = compute_metrics(y_true, y_pred, labels=["pos", "neg"])

        p = m["per_class"]["pos"]
        assert p["precision"] == pytest.approx(2 / 3, abs=1e-3)
        assert p["recall"] == pytest.approx(2 / 3, abs=1e-3)


# ---------------------------------------------------------------------------
# format_metrics_table
# ---------------------------------------------------------------------------


class TestFormatMetricsTable:
    def test_returns_string(self) -> None:
        m = compute_metrics(["a", "b"], ["a", "b"])
        text = format_metrics_table(m)
        assert isinstance(text, str)

    def test_contains_accuracy(self) -> None:
        m = compute_metrics(["a", "b"], ["a", "b"])
        text = format_metrics_table(m)
        assert "Accuracy" in text
        assert "100.00%" in text

    def test_contains_labels(self) -> None:
        m = compute_metrics(["cat", "dog"], ["cat", "dog"])
        text = format_metrics_table(m)
        assert "cat" in text
        assert "dog" in text

    def test_contains_confusion_matrix(self) -> None:
        m = compute_metrics(["a", "b", "a"], ["a", "b", "b"])
        text = format_metrics_table(m)
        assert "Confusion matrix" in text
