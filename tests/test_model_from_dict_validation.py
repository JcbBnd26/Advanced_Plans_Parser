"""Tests for Pydantic-backed from_dict() validation.

These ensure malformed serialized payloads fail fast with clear exceptions.
"""

import pytest

from plancheck.models import GlyphBox, Span, VocrCandidate


def test_glyphbox_from_dict_rejects_missing_fields() -> None:
    with pytest.raises(ValueError):
        GlyphBox.from_dict({"page": 0, "x0": 1.0})


def test_span_from_dict_rejects_wrong_type() -> None:
    with pytest.raises(ValueError):
        Span.from_dict({"token_indices": "not-a-list"})


def test_vocr_candidate_from_dict_rejects_bad_bbox() -> None:
    with pytest.raises(ValueError):
        VocrCandidate.from_dict({"page": 0, "bbox": [1.0, 2.0, 3.0]})
