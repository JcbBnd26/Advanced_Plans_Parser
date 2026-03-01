"""Feature extraction for VOCR candidate hit/miss classification (Level 2).

Produces a fixed-length numeric feature vector from a
:class:`~plancheck.models.VocrCandidate` plus optional page-level context.

Feature schema (31 dimensions)::

    ┌─────────────────────────────┬───────┐
    │ Group                       │ Dims  │
    ├─────────────────────────────┼───────┤
    │ Trigger-method one-hot      │ 18    │
    │ Confidence (base)           │  1    │
    │ Bbox position & size        │  6    │
    │ Predicted symbol one-hot    │  4    │
    │ Context numeric features    │  2    │
    │                             │ ----  │
    │ Total                       │ 31    │
    └─────────────────────────────┴───────┘
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

import numpy as np

from ..models import VOCR_TRIGGER_METHODS

if TYPE_CHECKING:
    from ..models import VocrCandidate

log = logging.getLogger(__name__)

CANDIDATE_FEATURE_VERSION = 1
CANDIDATE_FEATURE_DIM = 31

# Ordered list of trigger methods for one-hot encoding (must stay in sync
# with VOCR_TRIGGER_METHODS — we assert the length at import time).
_METHOD_INDEX = {m: i for i, m in enumerate(VOCR_TRIGGER_METHODS)}
assert len(_METHOD_INDEX) == 18, f"Expected 18 methods, got {len(_METHOD_INDEX)}"

# Common predicted symbols → index.  Anything else → "other" slot.
_SYMBOL_INDEX = {"%": 0, "°": 1, "±": 2}
_SYMBOL_OTHER = 3  # index for symbols not in the map
_N_SYMBOL_SLOTS = 4


def featurize_candidate(
    candidate: "VocrCandidate",
    page_width: float = 612.0,
    page_height: float = 792.0,
) -> np.ndarray:
    """Return a 31-dimensional float32 feature vector for *candidate*.

    Parameters
    ----------
    candidate : VocrCandidate
        The candidate to featurize.
    page_width, page_height : float
        Page dimensions for bbox normalisation (defaults = US Letter).

    Returns
    -------
    np.ndarray
        Shape ``(31,)`` float32.
    """
    vec = np.zeros(CANDIDATE_FEATURE_DIM, dtype=np.float32)
    offset = 0

    # ── 1. Trigger-method one-hot (18 dims) ────────────────────────
    for m in candidate.trigger_methods:
        idx = _METHOD_INDEX.get(m)
        if idx is not None:
            vec[offset + idx] = 1.0
    offset += len(_METHOD_INDEX)  # 18

    # ── 2. Base confidence (1 dim) ─────────────────────────────────
    vec[offset] = candidate.confidence
    offset += 1  # 19

    # ── 3. Bbox position & size normalised (6 dims) ────────────────
    pw = max(page_width, 1.0)
    ph = max(page_height, 1.0)
    vec[offset + 0] = candidate.x0 / pw  # x0_frac
    vec[offset + 1] = candidate.y0 / ph  # y0_frac
    vec[offset + 2] = (candidate.x1 - candidate.x0) / pw  # width_frac
    vec[offset + 3] = (candidate.y1 - candidate.y0) / ph  # height_frac
    # center position
    vec[offset + 4] = ((candidate.x0 + candidate.x1) / 2.0) / pw  # cx_frac
    vec[offset + 5] = ((candidate.y0 + candidate.y1) / 2.0) / ph  # cy_frac
    offset += 6  # 25

    # ── 4. Predicted symbol one-hot (4 dims) ───────────────────────
    sym = candidate.predicted_symbol
    sym_idx = _SYMBOL_INDEX.get(sym, _SYMBOL_OTHER)
    vec[offset + sym_idx] = 1.0
    offset += _N_SYMBOL_SLOTS  # 29

    # ── 5. Context numeric features (2 dims) ───────────────────────
    ctx = candidate.context
    vec[offset + 0] = float(ctx.get("gap_pts", 0.0))  # gap size
    vec[offset + 1] = float(len(ctx.get("neighbor_text", "")))  # neighbor text len
    offset += 2  # 31

    assert offset == CANDIDATE_FEATURE_DIM
    return vec


def featurize_candidates_batch(
    candidates: List["VocrCandidate"],
    page_width: float = 612.0,
    page_height: float = 792.0,
) -> np.ndarray:
    """Featurize a list of candidates into an ``(N, 31)`` array.

    Returns an empty ``(0, 31)`` array when *candidates* is empty.
    """
    if not candidates:
        return np.empty((0, CANDIDATE_FEATURE_DIM), dtype=np.float32)
    return np.stack(
        [featurize_candidate(c, page_width, page_height) for c in candidates]
    )


def featurize_outcome_row(
    row: dict,
    method_list: tuple = VOCR_TRIGGER_METHODS,
) -> np.ndarray:
    """Featurize a ``candidate_outcomes`` DB row into a 31-d vector.

    This mirrors :func:`featurize_candidate` but operates on the flat
    dict returned by ``CorrectionStore.get_candidate_outcomes()``.
    """
    vec = np.zeros(CANDIDATE_FEATURE_DIM, dtype=np.float32)
    offset = 0

    # 1. Trigger-method one-hot
    methods = (row.get("trigger_methods") or "").split(",")
    for m in methods:
        m = m.strip()
        idx = _METHOD_INDEX.get(m)
        if idx is not None:
            vec[offset + idx] = 1.0
    offset += len(_METHOD_INDEX)

    # 2. Confidence
    vec[offset] = float(row.get("confidence", 0.5))
    offset += 1

    # 3. Bbox
    pw = max(float(row.get("page_width", 612.0)), 1.0)
    ph = max(float(row.get("page_height", 792.0)), 1.0)
    x0 = float(row.get("bbox_x0", 0))
    y0 = float(row.get("bbox_y0", 0))
    x1 = float(row.get("bbox_x1", 0))
    y1 = float(row.get("bbox_y1", 0))
    vec[offset + 0] = x0 / pw
    vec[offset + 1] = y0 / ph
    vec[offset + 2] = (x1 - x0) / pw
    vec[offset + 3] = (y1 - y0) / ph
    vec[offset + 4] = ((x0 + x1) / 2.0) / pw
    vec[offset + 5] = ((y0 + y1) / 2.0) / ph
    offset += 6

    # 4. Predicted symbol
    sym = row.get("predicted_symbol", "")
    sym_idx = _SYMBOL_INDEX.get(sym, _SYMBOL_OTHER)
    vec[offset + sym_idx] = 1.0
    offset += _N_SYMBOL_SLOTS

    # 5. Context (from features_json if available, else zero)
    import json

    try:
        feat = json.loads(row.get("features_json", "{}") or "{}")
    except (json.JSONDecodeError, TypeError):
        feat = {}
    vec[offset + 0] = float(feat.get("gap_pts", 0.0))
    vec[offset + 1] = float(feat.get("neighbor_text_len", 0))
    offset += 2

    return vec
