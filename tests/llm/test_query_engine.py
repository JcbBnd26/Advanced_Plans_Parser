"""Tests for plancheck.llm.query_engine — RAG query engine."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from plancheck.llm.client import ChatMeta, LLMClient
from plancheck.llm.cost import CostTracker
from plancheck.llm.index import _CHROMA_AVAILABLE, Chunk, DocumentIndex, SearchResult

# ── Helpers ───────────────────────────────────────────────────────────


@dataclass
class _FakeLegendEntry:
    symbol: str = ""
    description: str = ""


@dataclass
class _FakeLegendRegion:
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


def _build_test_page() -> _FakePageResult:
    """Build a page with diverse content for testing."""
    block1 = _FakeBlock(
        _boxes=[
            _FakeBox("EXIT CORRIDORS SHALL MAINTAIN A MINIMUM WIDTH OF 44 INCHES"),
            _FakeBox("PER IBC 1005.1"),
        ]
    )
    block2 = _FakeBlock(
        _boxes=[
            _FakeBox("CONCRETE: f'c = 4000 PSI @ 28 DAYS"),
        ]
    )
    block3 = _FakeBlock(
        _boxes=[
            _FakeBox("SEISMIC DESIGN CATEGORY: D"),
        ]
    )
    col = _FakeCol(
        blocks=[block1, block2, block3],
        _full_text=(
            "EXIT CORRIDORS SHALL MAINTAIN A MINIMUM WIDTH OF 44 INCHES PER IBC 1005.1\n"
            "CONCRETE: f'c = 4000 PSI @ 28 DAYS\n"
            "SEISMIC DESIGN CATEGORY: D"
        ),
    )
    legend = _FakeLegendRegion(
        entries=[
            _FakeLegendEntry(symbol="▲", description="Fire alarm pull station"),
            _FakeLegendEntry(symbol="●", description="Smoke detector"),
        ]
    )
    tb = _FakeTitleBlock(
        project_name="Test Building",
        sheet_number="A-101",
        sheet_title="Floor Plan",
    )
    return _FakePageResult(
        page_number=1,
        notes_columns=[col],
        legends=[legend],
        title_block=tb,
    )


# ── QueryResult tests ────────────────────────────────────────────────


class TestQueryResult:
    def test_to_dict(self):
        from plancheck.llm.query_engine import QueryResult

        r = QueryResult(text="answer", confidence=0.9, sources=[{"page": 1}])
        d = r.to_dict()
        assert d["text"] == "answer"
        assert d["confidence"] == 0.9


# ── ResponseCache tests ──────────────────────────────────────────────


class TestResponseCache:
    def test_put_and_get(self):
        from plancheck.llm.query_engine import QueryResult, _ResponseCache

        cache = _ResponseCache(max_size=10)
        r = QueryResult(text="cached answer")
        cache.put("q1", "ctx1", r)
        assert cache.get("q1", "ctx1") is r

    def test_miss(self):
        from plancheck.llm.query_engine import _ResponseCache

        cache = _ResponseCache()
        assert cache.get("q", "c") is None

    def test_eviction(self):
        from plancheck.llm.query_engine import QueryResult, _ResponseCache

        cache = _ResponseCache(max_size=2)
        cache.put("q1", "c1", QueryResult(text="a1"))
        cache.put("q2", "c2", QueryResult(text="a2"))
        cache.put("q3", "c3", QueryResult(text="a3"))
        assert cache.size == 2

    def test_clear(self):
        from plancheck.llm.query_engine import QueryResult, _ResponseCache

        cache = _ResponseCache()
        cache.put("q", "c", QueryResult(text="x"))
        cache.clear()
        assert cache.size == 0


# ── DocumentQueryEngine tests (require ChromaDB) ─────────────────────


@pytest.mark.skipif(not _CHROMA_AVAILABLE, reason="chromadb not installed")
class TestDocumentQueryEngine:
    def _make_engine(self, collection_name: str = "test_qe"):
        from plancheck.llm.query_engine import DocumentQueryEngine

        index = DocumentIndex(collection_name=collection_name)
        pr = _build_test_page()
        index.index_page_result(pr)

        llm = LLMClient(provider="ollama", policy="local_only")
        return DocumentQueryEngine(
            index=index,
            llm=llm,
            n_context_chunks=3,
            enable_cache=True,
        )

    def test_search_only(self):
        engine = self._make_engine("test_search_only")
        results = engine.search_only("corridor width")
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_query_with_mocked_llm(self):
        engine = self._make_engine("test_query_mock")
        with patch.object(
            engine.llm,
            "chat_with_metadata",
            return_value=(
                "The minimum corridor width is 44 inches (Page 1, notes).",
                ChatMeta(
                    provider="ollama",
                    model="llama3.1:8b",
                    input_tokens=100,
                    output_tokens=20,
                ),
            ),
        ):
            result = engine.query("What is the corridor width?")
            assert "44" in result.text
            assert result.meta is not None

    def test_query_caching(self):
        engine = self._make_engine("test_cache")
        call_count = 0

        def mock_chat(sys, usr):
            nonlocal call_count
            call_count += 1
            return (
                "Answer",
                ChatMeta(
                    provider="ollama", model="m", input_tokens=10, output_tokens=5
                ),
            )

        with patch.object(engine.llm, "chat_with_metadata", side_effect=mock_chat):
            r1 = engine.query("test question")
            r2 = engine.query("test question")

        assert call_count == 1  # second call was cached
        assert r2.cached is True

    def test_structured_query(self):
        engine = self._make_engine("test_structured")
        payload = {
            "answer": "44 inches per IBC 1005.1",
            "confidence": 0.95,
            "sources": [
                {"page": 1, "region_type": "notes", "excerpt": "EXIT CORRIDORS"}
            ],
        }
        with patch.object(
            engine.llm,
            "chat_structured",
            return_value=(
                payload,
                ChatMeta(
                    provider="ollama", model="m", input_tokens=100, output_tokens=50
                ),
            ),
        ):
            result = engine.query("corridor width?", structured=True)
            assert result.confidence == 0.95
            assert len(result.sources) == 1

    def test_history_tracking(self):
        engine = self._make_engine("test_history")
        with patch.object(
            engine.llm,
            "chat_with_metadata",
            return_value=(
                "answer",
                ChatMeta(
                    provider="ollama", model="m", input_tokens=10, output_tokens=5
                ),
            ),
        ):
            engine.query("question 1")
            engine.query("question 2")

        assert len(engine.history) == 2
        assert engine.history[0]["question"] == "question 1"

    def test_llm_error_handled(self):
        engine = self._make_engine("test_error")
        with patch.object(
            engine.llm,
            "chat_with_metadata",
            side_effect=RuntimeError("Connection failed"),
        ):
            result = engine.query("test")
            assert "Error" in result.text

    def test_from_page_result(self):
        from plancheck.llm.query_engine import DocumentQueryEngine

        pr = _build_test_page()
        with patch("plancheck.llm.client._OLLAMA_AVAILABLE", True):
            engine = DocumentQueryEngine.from_page_result(pr)
            assert engine.index.count > 0

    def test_page_filter(self):
        engine = self._make_engine("test_pf")
        # All content is page 1, so filtering to page 99 should return empty context
        results = engine.search_only("corridor", page_filter=99)
        assert len(results) == 0

    def test_clear_cache(self):
        engine = self._make_engine("test_cc")
        with patch.object(
            engine.llm,
            "chat_with_metadata",
            return_value=(
                "answer",
                ChatMeta(
                    provider="ollama", model="m", input_tokens=10, output_tokens=5
                ),
            ),
        ):
            engine.query("q")
        assert engine._cache.size == 1
        engine.clear_cache()
        assert engine._cache.size == 0
