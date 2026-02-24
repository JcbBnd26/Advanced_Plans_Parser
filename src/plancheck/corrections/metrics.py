"""Lightweight classification metrics — no sklearn dependency.

Computes precision, recall, F1, accuracy, and confusion matrix from
lists of true / predicted string labels.  Keeps the GUI startup fast
by avoiding a heavy ``sklearn.metrics`` import.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence


def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: List[str] | None = None,
) -> dict:
    """Compute multi-class classification metrics.

    Parameters
    ----------
    y_true, y_pred : sequence of str
        Ground-truth and predicted labels (same length).
    labels : list[str] | None
        Ordered label list.  If *None*, derived from the union of
        *y_true* and *y_pred* (sorted alphabetically).

    Returns
    -------
    dict
        ``accuracy``          – float 0–1
        ``per_class``         – ``{label: {precision, recall, f1, support}}``
        ``confusion_matrix``  – list-of-lists ``[true_idx][pred_idx]``
        ``labels``            – label list used for the matrix axes
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    n_labels = len(labels)

    # ── Confusion matrix ───────────────────────────────────────────
    cm: list[list[int]] = [[0] * n_labels for _ in range(n_labels)]
    for yt, yp in zip(y_true, y_pred):
        ti = label_to_idx.get(yt)
        pi = label_to_idx.get(yp)
        if ti is not None and pi is not None:
            cm[ti][pi] += 1

    # ── Per-class metrics ──────────────────────────────────────────
    per_class: Dict[str, dict] = {}
    correct = 0
    total = len(y_true)

    for i, label in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n_labels)) - tp
        fn = sum(cm[i][c] for c in range(n_labels)) - tp
        support = sum(cm[i])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    correct = sum(cm[i][i] for i in range(n_labels))
    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": labels,
    }


def format_metrics_table(metrics: dict) -> str:
    """Format metrics as a human-readable ASCII table.

    Parameters
    ----------
    metrics : dict
        Output of :func:`compute_metrics`.

    Returns
    -------
    str
        Multi-line string suitable for printing or display in a log panel.
    """
    lines: list[str] = []
    lines.append(
        f"Accuracy: {metrics['accuracy']:.2%}  (n={sum(m['support'] for m in metrics['per_class'].values())})"
    )
    lines.append("")

    # Header
    header = f"{'Label':<20s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Sup':>5s}"
    lines.append(header)
    lines.append("-" * len(header))

    for label in metrics["labels"]:
        m = metrics["per_class"][label]
        lines.append(
            f"{label:<20s} {m['precision']:>6.2%} {m['recall']:>6.2%} "
            f"{m['f1']:>6.2%} {m['support']:>5d}"
        )

    lines.append("-" * len(header))

    # Confusion matrix
    lines.append("")
    lines.append("Confusion matrix (rows=true, cols=pred):")
    labels = metrics["labels"]
    cm = metrics["confusion_matrix"]
    col_w = max(len(l) for l in labels) + 1
    col_w = max(col_w, 5)

    # Header row
    hdr = " " * (col_w + 2) + "".join(l[:col_w].rjust(col_w) for l in labels)
    lines.append(hdr)

    for i, label in enumerate(labels):
        row_str = label[:col_w].ljust(col_w) + "  "
        row_str += "".join(str(cm[i][j]).rjust(col_w) for j in range(len(labels)))
        lines.append(row_str)

    return "\n".join(lines)
