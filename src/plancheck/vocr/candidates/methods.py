"""Individual VOCR candidate detection methods.

Each method returns a list of VocrCandidate with populated trigger_methods.
This module re-exports 17 detection methods from focused submodules.

Submodules:
- encoding_quality: OCR/rendering corruption detection
- gap_patterns: Gap-based symbol prediction
- density_grid: Grid-based blank cell detection
- keyword_inference: Keyword→symbol mapping
- cross_line: Line fingerprinting & comparison
- rendering_analysis: Font/width/vector checks
- dimension_analysis: Unit/geometry checks
"""

from __future__ import annotations

# Cross-line comparison
from .cross_line import _detect_cross_ref_phrases, _detect_near_duplicate_lines

# Grid density detection
from .density_grid import _detect_dense_cluster_holes

# Dimension/unit analysis
from .dimension_analysis import _detect_dimension_geometry, _detect_semantic_no_units

# Encoding/placeholder detection
from .encoding_quality import _detect_char_encoding_failures, _detect_placeholder_tokens

# Gap-based detection
from .gap_patterns import (
    _detect_baseline_style_gaps,
    _detect_impossible_sequences,
    _detect_intraline_gaps,
    _detect_regex_digit_patterns,
    _detect_template_adjacency,
)

# Keyword inference
from .keyword_inference import _detect_keyword_cooccurrence, _detect_vocab_triggers

# Rendering analysis
from .rendering_analysis import (
    _detect_font_subset_correlation,
    _detect_token_width_anomaly,
    _detect_vector_circles,
)

__all__ = [
    "_detect_char_encoding_failures",
    "_detect_placeholder_tokens",
    "_detect_intraline_gaps",
    "_detect_dense_cluster_holes",
    "_detect_baseline_style_gaps",
    "_detect_template_adjacency",
    "_detect_regex_digit_patterns",
    "_detect_impossible_sequences",
    "_detect_vocab_triggers",
    "_detect_keyword_cooccurrence",
    "_detect_cross_ref_phrases",
    "_detect_near_duplicate_lines",
    "_detect_font_subset_correlation",
    "_detect_token_width_anomaly",
    "_detect_vector_circles",
    "_detect_semantic_no_units",
    "_detect_dimension_geometry",
]
