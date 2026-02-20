"""Tests for _split_wide_blocks and flag_suspect_header_words in clustering.py."""

import re
import tempfile
from pathlib import Path
from statistics import median

import pytest
from conftest import make_block, make_box

from plancheck.config import GroupingConfig
from plancheck.grouping.clustering import (
    _block_first_row_text,
    _extract_note_numbers,
    _get_first_block_text,
    _get_last_block_text,
    _split_wide_blocks,
    flag_suspect_header_words,
)
from plancheck.models import BlockCluster, GlyphBox, Line, RowBand

# ── helpers ────────────────────────────────────────────────────────────


def _make_wide_block_data(
    left_texts: list[str],
    right_texts: list[str],
    left_x0: float = 50.0,
    right_x0: float = 400.0,
    col_width: float = 150.0,
    y_start: float = 100.0,
    line_h: float = 14.0,
    page: int = 0,
):
    """Build tokens and a Line-based BlockCluster spanning two visual columns.

    Returns (tokens, block) where block.lines reference indices into tokens.
    """
    tokens: list[GlyphBox] = []
    lines: list[Line] = []
    n_rows = max(len(left_texts), len(right_texts))

    for row_idx in range(n_rows):
        y0 = y_start + row_idx * (line_h + 4)
        y1 = y0 + line_h
        indices = []

        if row_idx < len(left_texts):
            idx = len(tokens)
            tokens.append(
                make_box(
                    left_x0, y0, left_x0 + col_width, y1, left_texts[row_idx], page=page
                )
            )
            indices.append(idx)

        if row_idx < len(right_texts):
            idx = len(tokens)
            tokens.append(
                make_box(
                    right_x0,
                    y0,
                    right_x0 + col_width,
                    y1,
                    right_texts[row_idx],
                    page=page,
                )
            )
            indices.append(idx)

        if indices:
            yc = [(tokens[i].y0 + tokens[i].y1) / 2.0 for i in indices]
            lines.append(
                Line(
                    line_id=row_idx,
                    page=page,
                    token_indices=indices,
                    baseline_y=median(yc),
                )
            )

    block = BlockCluster(page=page, rows=[], lines=lines, _tokens=tokens)
    block.populate_rows_from_lines()
    return tokens, block


def _make_narrow_block(
    text: str, x0: float, y0: float, w: float = 100.0, h: float = 14.0, page: int = 0
):
    """Return a single-line narrow block (to pad the median width)."""
    return make_block([(x0, y0, x0 + w, y0 + h, text)], page=page)


# ── _split_wide_blocks ────────────────────────────────────────────────


class TestSplitWideBlocks:
    """Tests for the _split_wide_blocks function."""

    def test_no_split_when_few_blocks(self):
        """< 3 blocks → returned unchanged."""
        cfg = GroupingConfig()
        b1 = _make_narrow_block("A", 50, 100)
        b2 = _make_narrow_block("B", 50, 120)
        result = _split_wide_blocks([b1, b2], [], cfg, 10.0)
        assert result == [b1, b2]

    def test_narrow_blocks_unchanged(self):
        """Blocks narrower than the width threshold are returned unchanged."""
        cfg = GroupingConfig()
        blocks = [_make_narrow_block(f"B{i}", 50, 100 + i * 20) for i in range(5)]
        result = _split_wide_blocks(blocks, [], cfg, 10.0)
        assert len(result) == 5

    def test_splits_two_column_block(self):
        """A block spanning two visual columns (large interior gap) is split."""
        cfg = GroupingConfig()
        tokens, wide_block = _make_wide_block_data(
            left_texts=["NOTE 1", "NOTE 2"],
            right_texts=["NOTE 3", "NOTE 4"],
            left_x0=50,
            right_x0=400,
            col_width=120,
        )

        # Pad with narrow blocks to set median width low
        narrow = [
            _make_narrow_block(f"N{i}", 50, 300 + i * 20, w=120) for i in range(5)
        ]
        all_blocks = narrow + [wide_block]

        result = _split_wide_blocks(all_blocks, tokens, cfg, 10.0)
        # The 5 narrow blocks + the wide block split into ≥2
        assert len(result) > len(all_blocks)

    def test_no_split_when_gap_too_small(self):
        """Adjacent tokens with small gap → no split."""
        cfg = GroupingConfig()
        tokens, wide_block = _make_wide_block_data(
            left_texts=["HELLO"],
            right_texts=["WORLD"],
            left_x0=50,
            right_x0=210,  # Small gap (210 - 200 = 10)
            col_width=150,
        )
        narrow = [
            _make_narrow_block(f"N{i}", 50, 300 + i * 20, w=100) for i in range(5)
        ]
        all_blocks = narrow + [wide_block]
        result = _split_wide_blocks(all_blocks, tokens, cfg, 10.0)
        # Gap is too small at default threshold → no split
        assert len(result) == len(all_blocks)

    def test_single_line_wide_block_not_split(self):
        """Wide block with only 1 line is not split (needs ≥2 lines)."""
        cfg = GroupingConfig()
        tokens, wide_block = _make_wide_block_data(
            left_texts=["A"],
            right_texts=["B"],
            left_x0=50,
            right_x0=400,
            col_width=120,
        )
        # Only 1 line → skip
        narrow = [_make_narrow_block(f"N{i}", 50, 200 + i * 20, w=50) for i in range(5)]
        all_blocks = narrow + [wide_block]
        result = _split_wide_blocks(all_blocks, tokens, cfg, 10.0)
        assert len(result) == len(all_blocks)


# ── flag_suspect_header_words ─────────────────────────────────────────


class TestFlagSuspectHeaderWords:
    """Tests for the flag_suspect_header_words function."""

    def _run(self, blocks, min_word_len=10):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            debug_path = f.name
        try:
            return flag_suspect_header_words(
                blocks, min_word_len=min_word_len, debug_path=debug_path
            )
        finally:
            Path(debug_path).unlink(missing_ok=True)

    def test_flags_long_allcaps_word(self):
        """A fused compound like SITEWORK should be flagged."""
        blk = make_block([(10, 100, 200, 112, "CLEARINGGRUBBING")], is_header=True)
        suspects = self._run([blk])
        assert len(suspects) == 1
        assert suspects[0].word_text == "CLEARINGGRUBBING"
        assert "long_allcaps" in suspects[0].reason

    def test_ignores_allowed_words(self):
        """Words in the allow-list (CONSTRUCTION etc.) are not flagged."""
        blk = make_block([(10, 100, 200, 112, "CONSTRUCTION")], is_header=True)
        suspects = self._run([blk])
        assert len(suspects) == 0

    def test_ignores_short_words(self):
        """Words shorter than min_word_len are not flagged."""
        blk = make_block([(10, 100, 80, 112, "NOTES")], is_header=True)
        suspects = self._run([blk])
        assert len(suspects) == 0

    def test_ignores_non_header_blocks(self):
        """Non-header blocks are skipped entirely."""
        blk = make_block([(10, 100, 200, 112, "CLEARINGGRUBBING")], is_header=False)
        suspects = self._run([blk])
        assert len(suspects) == 0

    def test_ignores_mixed_case(self):
        """Mixed-case words are not flagged (not all-caps)."""
        blk = make_block([(10, 100, 200, 112, "ClearingGrub")], is_header=True)
        suspects = self._run([blk])
        assert len(suspects) == 0

    def test_multiple_words_mixed(self):
        """Only the suspect word is flagged, not the allowed one."""
        blk = make_block(
            [
                (10, 100, 100, 112, "CONSTRUCTION"),
                (110, 100, 300, 112, "SCHEDULETABLE"),
            ],
            is_header=True,
        )
        suspects = self._run([blk])
        assert len(suspects) == 1
        assert suspects[0].word_text == "SCHEDULETABLE"

    def test_custom_min_word_len(self):
        """Lowering min_word_len catches shorter fused words."""
        blk = make_block([(10, 100, 100, 112, "SITEPREP")], is_header=True)
        assert len(self._run([blk], min_word_len=10)) == 0
        assert len(self._run([blk], min_word_len=5)) == 1

    def test_digits_in_word_still_flagged(self):
        """Alphanumeric all-caps words are still flagged (e.g. SECTION12DETAILS)."""
        blk = make_block([(10, 100, 200, 112, "SECTION12DETAILS")], is_header=True)
        suspects = self._run([blk])
        assert len(suspects) == 1

    def test_empty_block_no_crash(self):
        """Block with no rows doesn't crash."""
        blk = BlockCluster(page=0, rows=[], is_header=True)
        suspects = self._run([blk])
        assert len(suspects) == 0


# ── _block_first_row_text ─────────────────────────────────────────────


class TestBlockFirstRowText:
    def test_basic(self):
        blk = make_block([(10, 100, 60, 112, "general"), (70, 100, 120, 112, "notes")])
        assert _block_first_row_text(blk) == "GENERAL NOTES"

    def test_empty(self):
        blk = BlockCluster(page=0, rows=[])
        assert _block_first_row_text(blk) == ""

    def test_whitespace_collapse(self):
        blk = make_block(
            [(10, 100, 60, 112, "  HELLO  "), (70, 100, 120, 112, "  WORLD  ")]
        )
        assert _block_first_row_text(blk) == "HELLO WORLD"


# ── _extract_note_numbers ─────────────────────────────────────────────


class TestExtractNoteNumbers:
    def _make_notes_col(self, texts):
        """Make a NotesColumn-like object with notes_blocks containing given texts."""
        from plancheck.models import NotesColumn

        blocks = []
        for i, text in enumerate(texts):
            blocks.append(make_block([(10, 100 + i * 20, 200, 112 + i * 20, text)]))
        col = NotesColumn(page=0, notes_blocks=blocks)
        return col

    def test_extracts_numbers(self):
        col = self._make_notes_col(["1. First note", "2. Second note", "3. Third note"])
        nums = _extract_note_numbers(col)
        assert nums == [1, 2, 3]

    def test_no_numbers(self):
        col = self._make_notes_col(["General text", "More text"])
        nums = _extract_note_numbers(col)
        assert nums == []

    def test_mixed(self):
        col = self._make_notes_col(
            ["preamble", "1. Note one", "continuation", "2. Note two"]
        )
        nums = _extract_note_numbers(col)
        assert 1 in nums
        assert 2 in nums
