"""VOCR candidate detection subpackage.

This module provides backward-compatible re-exports of the public API
and internal methods (for testing).
"""

from .api import compute_candidate_stats, detect_vocr_candidates
from .helpers import _group_by_baseline, _iou_bbox, _pad_bbox
from .merge import _merge_overlapping_candidates
from .methods import (
    _detect_baseline_style_gaps,
    _detect_char_encoding_failures,
    _detect_cross_ref_phrases,
    _detect_dense_cluster_holes,
    _detect_dimension_geometry,
    _detect_font_subset_correlation,
    _detect_impossible_sequences,
    _detect_intraline_gaps,
    _detect_keyword_cooccurrence,
    _detect_near_duplicate_lines,
    _detect_placeholder_tokens,
    _detect_regex_digit_patterns,
    _detect_semantic_no_units,
    _detect_template_adjacency,
    _detect_token_width_anomaly,
    _detect_vector_circles,
    _detect_vocab_triggers,
)

__all__ = [
    # Public API
    "detect_vocr_candidates",
    "compute_candidate_stats",
    # Internal (exported for tests)
    "_detect_baseline_style_gaps",
    "_detect_char_encoding_failures",
    "_detect_cross_ref_phrases",
    "_detect_dense_cluster_holes",
    "_detect_dimension_geometry",
    "_detect_font_subset_correlation",
    "_detect_impossible_sequences",
    "_detect_intraline_gaps",
    "_detect_keyword_cooccurrence",
    "_detect_near_duplicate_lines",
    "_detect_placeholder_tokens",
    "_detect_regex_digit_patterns",
    "_detect_semantic_no_units",
    "_detect_template_adjacency",
    "_detect_token_width_anomaly",
    "_detect_vector_circles",
    "_detect_vocab_triggers",
    "_group_by_baseline",
    "_iou_bbox",
    "_merge_overlapping_candidates",
    "_pad_bbox",
]
