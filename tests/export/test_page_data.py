"""Tests for plancheck.export.page_data — serialize/deserialize round-trips."""

import pytest
from conftest import make_block, make_box

from plancheck.export.page_data import deserialize_page, serialize_page
from plancheck.models import BlockCluster, Line, NotesColumn, RowBand, Span


class TestRoundTrip:
    """serialize_page → deserialize_page should reproduce the inputs."""

    def _make_tokens_blocks_columns(self):
        """Build a realistic set of tokens, blocks, and notes columns."""
        tokens = [
            make_box(10, 100, 60, 112, "GENERAL", page=0),
            make_box(70, 100, 130, 112, "NOTES:", page=0),
            make_box(10, 120, 25, 132, "1.", page=0),
            make_box(30, 120, 150, 132, "SEE PLANS", page=0),
        ]
        # Build block with lines
        line0 = Line(line_id=0, page=0, token_indices=[0, 1], baseline_y=106.0)
        line0.spans = [Span(token_indices=[0, 1])]
        line1 = Line(line_id=1, page=0, token_indices=[2, 3], baseline_y=126.0)
        line1.spans = [Span(token_indices=[2, 3])]

        header_block = BlockCluster(
            page=0, lines=[line0], _tokens=tokens, is_header=True
        )
        header_block.populate_rows_from_lines()
        notes_block = BlockCluster(page=0, lines=[line1], _tokens=tokens, is_notes=True)
        notes_block.populate_rows_from_lines()

        blocks = [header_block, notes_block]
        col = NotesColumn(page=0, header=header_block, notes_blocks=[notes_block])
        return tokens, blocks, [col]

    def test_basic_round_trip(self):
        tokens, blocks, columns = self._make_tokens_blocks_columns()
        data = serialize_page(
            page=0,
            page_width=612.0,
            page_height=792.0,
            tokens=tokens,
            blocks=blocks,
            notes_columns=columns,
        )
        rt_tokens, rt_blocks, rt_columns, pw, ph = deserialize_page(data)

        assert pw == 612.0
        assert ph == 792.0
        assert len(rt_tokens) == len(tokens)
        assert len(rt_blocks) == len(blocks)
        assert len(rt_columns) == len(columns)

    def test_token_text_preserved(self):
        tokens, blocks, columns = self._make_tokens_blocks_columns()
        data = serialize_page(0, 612.0, 792.0, tokens, blocks, columns)
        rt_tokens, *_ = deserialize_page(data)

        original_texts = sorted(t.text for t in tokens)
        restored_texts = sorted(t.text for t in rt_tokens)
        assert original_texts == restored_texts

    def test_block_bbox_preserved(self):
        tokens, blocks, columns = self._make_tokens_blocks_columns()
        data = serialize_page(0, 612.0, 792.0, tokens, blocks, columns)
        _, rt_blocks, *_ = deserialize_page(data)

        for orig, restored in zip(blocks, rt_blocks):
            assert orig.bbox() == restored.bbox()

    def test_notes_column_header_text(self):
        tokens, blocks, columns = self._make_tokens_blocks_columns()
        data = serialize_page(0, 612.0, 792.0, tokens, blocks, columns)
        _, _, rt_columns, *_ = deserialize_page(data)

        assert rt_columns[0].header_text() == columns[0].header_text()

    def test_empty_page(self):
        data = serialize_page(0, 612.0, 792.0, [], [], [])
        rt_tokens, rt_blocks, rt_columns, pw, ph = deserialize_page(data)
        assert rt_tokens == []
        assert rt_blocks == []
        assert rt_columns == []

    def test_version_key(self):
        data = serialize_page(0, 612.0, 792.0, [], [], [])
        assert data["version"] == 1

    def test_block_lines_preserved(self):
        tokens, blocks, columns = self._make_tokens_blocks_columns()
        data = serialize_page(0, 612.0, 792.0, tokens, blocks, columns)
        _, rt_blocks, *_ = deserialize_page(data)

        for orig, restored in zip(blocks, rt_blocks):
            assert len(restored.lines or []) == len(orig.lines or [])

    def test_rows_populated_after_deserialize(self):
        tokens, blocks, columns = self._make_tokens_blocks_columns()
        data = serialize_page(0, 612.0, 792.0, tokens, blocks, columns)
        _, rt_blocks, *_ = deserialize_page(data)

        for blk in rt_blocks:
            if blk.lines:
                assert len(blk.rows) > 0, "rows should be populated from lines"


class TestSerializeOnly:
    def test_page_metadata(self):
        data = serialize_page(
            page=3,
            page_width=1224.0,
            page_height=792.0,
            tokens=[],
            blocks=[],
            notes_columns=[],
        )
        assert data["page"] == 3
        assert data["page_width"] == 1224.0
        assert data["page_height"] == 792.0

    def test_tokens_serialized(self):
        tokens = [make_box(10, 20, 30, 40, "TEST")]
        data = serialize_page(0, 612.0, 792.0, tokens, [], [])
        assert len(data["tokens"]) == 1
        assert data["tokens"][0]["text"] == "TEST"
