"""Shared fixtures for corrections tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from plancheck.corrections.store import CorrectionStore


@pytest.fixture
def tmp_store(tmp_path: Path) -> CorrectionStore:
    """Return a CorrectionStore backed by a temporary database."""
    return CorrectionStore(db_path=tmp_path / "test.db")


@pytest.fixture
def sample_features() -> dict:
    """Minimal valid feature dict for testing."""
    return {
        "font_size_pt": 8.0,
        "font_size_max_pt": 10.0,
        "font_size_min_pt": 6.0,
        "is_all_caps": 0,
        "is_bold": 0,
        "token_count": 5,
        "row_count": 2,
        "x_frac": 0.7,
        "y_frac": 0.1,
        "x_center_frac": 0.8,
        "y_center_frac": 0.3,
        "width_frac": 0.2,
        "height_frac": 0.4,
        "aspect_ratio": 2.5,
        "contains_digit": 1,
        "starts_with_digit": 1,
        "has_colon": 0,
        "has_period_after_num": 1,
        "text_length": 40,
        "avg_chars_per_token": 8.0,
        "zone": "right_margin",
        "neighbor_count": 3,
        # Text-content features (v2)
        "unique_word_ratio": 0.8,
        "uppercase_word_frac": 0.4,
        "avg_word_length": 5.0,
        "kw_notes_pattern": 0,
        "kw_header_pattern": 0,
        "kw_legend_pattern": 0,
        "kw_abbreviation_pattern": 0,
        "kw_revision_pattern": 0,
        "kw_title_block_pattern": 0,
        "kw_detail_pattern": 0,
    }
