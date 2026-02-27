"""Tests for plancheck.llm.index — chunking and document index."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from plancheck.llm.index import (
    _CHROMA_AVAILABLE,
    Chunk,
    SearchResult,
    chunks_from_document_result,
    chunks_from_page_result,
)

# ── Helpers to build minimal PageResult-like objects ──────────────────


@dataclass
class _FakeBox:
    text: str = ""


@dataclass
class _FakeBlock:
    text: str = ""
    _boxes: list = field(default_factory=list)

    def get_all_boxes(self):
        return self._boxes


@dataclass
class _FakeCol:
    blocks: list = field(default_factory=list)
    _full_text: str = ""

    def full_text(self):
        return self._full_text


@dataclass
class _FakeLegendEntry:
    symbol: str = ""
    description: str = ""


@dataclass
class _FakeLegendRegion:
    entries: list = field(default_factory=list)


@dataclass
class _FakeAbbrEntry:
    abbreviation: str = ""
    full_text: str = ""


@dataclass
class _FakeAbbrRegion:
    entries: list = field(default_factory=list)


@dataclass
class _FakeRevEntry:
    number: str = ""
    date: str = ""
    description: str = ""


@dataclass
class _FakeRevRegion:
    entries: list = field(default_factory=list)


@dataclass
class _FakeTitleBlock:
    project_name: str = ""
    sheet_title: str = ""
    sheet_number: str = ""
    drawn_by: str = ""
    checked_by: str = ""
    date: str = ""
    scale: str = ""


@dataclass
class _FakePageResult:
    page_number: int = 1
    notes_columns: list = field(default_factory=list)
    legends: list = field(default_factory=list)
    abbreviations: list = field(default_factory=list)
    revisions: list = field(default_factory=list)
    title_block: Any = None
    standard_details: list = field(default_factory=list)
    misc_titles: list = field(default_factory=list)


@dataclass
class _FakeDocResult:
    pages: list = field(default_factory=list)


# ── Chunk tests ───────────────────────────────────────────────────────


class TestChunk:
    def test_content_hash_deterministic(self):
        c = Chunk(text="hello", page=1)
        assert c.content_hash() == c.content_hash()

    def test_content_hash_varies(self):
        c1 = Chunk(text="hello", page=1)
        c2 = Chunk(text="world", page=1)
        assert c1.content_hash() != c2.content_hash()


# ── Chunking from PageResult ─────────────────────────────────────────


class TestChunksFromPageResult:
    def test_empty_page(self):
        pr = _FakePageResult()
        chunks = chunks_from_page_result(pr)
        assert chunks == []

    def test_notes_columns(self):
        block = _FakeBlock(_boxes=[_FakeBox("GENERAL NOTE 1"), _FakeBox("text here")])
        col = _FakeCol(blocks=[block], _full_text="GENERAL NOTE 1 text here")
        pr = _FakePageResult(notes_columns=[col])
        chunks = chunks_from_page_result(pr)
        # Should have block-level + full-column chunks
        assert any(c.region_type == "notes" for c in chunks)
        assert any(c.region_type == "notes_full" for c in chunks)

    def test_legend_entries(self):
        entry = _FakeLegendEntry(symbol="▲", description="Fire extinguisher")
        region = _FakeLegendRegion(entries=[entry])
        pr = _FakePageResult(legends=[region])
        chunks = chunks_from_page_result(pr)
        assert len(chunks) == 1
        assert "Fire extinguisher" in chunks[0].text
        assert chunks[0].region_type == "legend"

    def test_abbreviation_entries(self):
        entry = _FakeAbbrEntry(
            abbreviation="HVAC", full_text="Heating Ventilation and Air Conditioning"
        )
        region = _FakeAbbrRegion(entries=[entry])
        pr = _FakePageResult(abbreviations=[region])
        chunks = chunks_from_page_result(pr)
        assert len(chunks) == 1
        assert "HVAC" in chunks[0].text

    def test_revision_entries(self):
        entry = _FakeRevEntry(
            number="1", date="2024-01-15", description="Initial release"
        )
        region = _FakeRevRegion(entries=[entry])
        pr = _FakePageResult(revisions=[region])
        chunks = chunks_from_page_result(pr)
        assert len(chunks) == 1
        assert "Initial release" in chunks[0].text

    def test_title_block(self):
        tb = _FakeTitleBlock(project_name="Test Building", sheet_number="A-101")
        pr = _FakePageResult(title_block=tb)
        chunks = chunks_from_page_result(pr)
        assert any("Test Building" in c.text for c in chunks)
        assert any("A-101" in c.text for c in chunks)

    def test_page_number_propagated(self):
        entry = _FakeLegendEntry(symbol="X", description="Item")
        region = _FakeLegendRegion(entries=[entry])
        pr = _FakePageResult(page_number=5, legends=[region])
        chunks = chunks_from_page_result(pr)
        assert all(c.page == 5 for c in chunks)


class TestChunksFromDocumentResult:
    def test_multi_page(self):
        pr1 = _FakePageResult(
            page_number=1, title_block=_FakeTitleBlock(project_name="P1")
        )
        pr2 = _FakePageResult(
            page_number=2, title_block=_FakeTitleBlock(project_name="P2")
        )
        dr = _FakeDocResult(pages=[pr1, pr2])
        chunks = chunks_from_document_result(dr)
        pages = {c.page for c in chunks}
        assert 1 in pages
        assert 2 in pages


# ── DocumentIndex tests (require ChromaDB) ────────────────────────────


@pytest.mark.skipif(not _CHROMA_AVAILABLE, reason="chromadb not installed")
class TestDocumentIndex:
    def test_add_and_count(self):
        from plancheck.llm.index import DocumentIndex

        idx = DocumentIndex(collection_name="test_add")
        chunks = [
            Chunk(
                text="Exit corridor width 44 inches",
                page=1,
                region_type="notes",
                source_id="c1",
            ),
            Chunk(
                text="Seismic design category D",
                page=1,
                region_type="notes",
                source_id="c2",
            ),
            Chunk(
                text="Roof live load 20 PSF",
                page=2,
                region_type="notes",
                source_id="c3",
            ),
        ]
        added = idx.add_chunks(chunks)
        assert added == 3
        assert idx.count == 3

    def test_dedup(self):
        from plancheck.llm.index import DocumentIndex

        idx = DocumentIndex(collection_name="test_dedup")
        c = Chunk(text="Same text", page=1, source_id="dup")
        idx.add_chunks([c, c, c])
        assert idx.count == 1

    def test_search_returns_results(self):
        from plancheck.llm.index import DocumentIndex

        idx = DocumentIndex(collection_name="test_search")
        chunks = [
            Chunk(
                text="Exit corridor minimum width is 44 inches per IBC 1005.1",
                page=1,
                region_type="notes",
                source_id="a",
            ),
            Chunk(
                text="Concrete strength 4000 PSI at 28 days",
                page=1,
                region_type="notes",
                source_id="b",
            ),
            Chunk(
                text="Steel ASTM A992 wide flanges",
                page=2,
                region_type="notes",
                source_id="c",
            ),
        ]
        idx.add_chunks(chunks)
        results = idx.search("What is the corridor width?", n_results=2)
        assert len(results) <= 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].score > 0

    def test_page_filter(self):
        from plancheck.llm.index import DocumentIndex

        idx = DocumentIndex(collection_name="test_page_filter")
        chunks = [
            Chunk(text="Note on page 1", page=1, source_id="p1"),
            Chunk(text="Note on page 2", page=2, source_id="p2"),
        ]
        idx.add_chunks(chunks)
        results = idx.search("Note", n_results=5, page_filter=1)
        assert all(r.chunk.page == 1 for r in results)

    def test_region_filter(self):
        from plancheck.llm.index import DocumentIndex

        idx = DocumentIndex(collection_name="test_region_filter")
        chunks = [
            Chunk(text="Legend item", page=1, region_type="legend", source_id="l1"),
            Chunk(text="Notes item", page=1, region_type="notes", source_id="n1"),
        ]
        idx.add_chunks(chunks)
        results = idx.search("item", n_results=5, region_filter="legend")
        assert all(r.chunk.region_type == "legend" for r in results)

    def test_index_page_result(self):
        from plancheck.llm.index import DocumentIndex

        idx = DocumentIndex(collection_name="test_idx_pr")
        entry = _FakeLegendEntry(symbol="▲", description="Fire alarm pull station")
        region = _FakeLegendRegion(entries=[entry])
        pr = _FakePageResult(page_number=3, legends=[region])
        added = idx.index_page_result(pr)
        assert added >= 1
        assert idx.count >= 1

    def test_clear(self):
        from plancheck.llm.index import DocumentIndex

        idx = DocumentIndex()
        idx.add_chunks([Chunk(text="test", page=1, source_id="t1")])
        assert idx.count >= 1
        idx.clear()
        assert idx.count == 0
