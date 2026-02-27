"""End-to-end integration test: Index → QueryEngine → EntityExtractor.

Wires the full stack with a mocked LLM to verify that data flows
correctly through all layers without real API calls.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from plancheck.llm.client import ChatMeta, LLMClient
from plancheck.llm.cost import CostTracker
from plancheck.llm.entity_extraction import EntityExtractor, ExtractionResult
from plancheck.llm.index import DocumentIndex, chunks_from_page_result
from plancheck.llm.query_engine import DocumentQueryEngine, QueryResult

# ── Fixtures ──────────────────────────────────────────────────────


def _make_page_result(page: int = 1):
    """Minimal PageResult-like object with realistic construction notes."""
    notes = [
        SimpleNamespace(
            text=(
                "All structural steel shall be ASTM A992 Grade 50. "
                "Provide 4000 psi normal weight concrete at 28 days. "
                "Minimum corridor width: 3'-6\" clear."
            )
        ),
        SimpleNamespace(
            text=(
                "Install VAV boxes for all interior zones. "
                "Fire extinguisher cabinets per code at all corridors. "
                "Emergency lighting at exit pathways."
            )
        ),
    ]
    col = SimpleNamespace(blocks=notes, full_text=" ".join(n.text for n in notes))
    legend_entry = SimpleNamespace(symbol="FE", description="Fire Extinguisher")
    legend = SimpleNamespace(entries=[legend_entry])
    abbr_entry = SimpleNamespace(
        abbreviation="HVAC", full_text="Heating Ventilation and Air Conditioning"
    )
    abbr = SimpleNamespace(entries=[abbr_entry])
    tb = SimpleNamespace(
        project_name="Test Building Project",
        sheet_title="Floor Plan",
        sheet_number="A-101",
        drawn_by="JD",
        checked_by="RK",
        date="2026-01-15",
        scale='1/4" = 1\'-0"',
    )
    return SimpleNamespace(
        page_number=page,
        notes_columns=[col],
        legends=[legend],
        abbreviations=[abbr],
        revisions=[],
        title_block=tb,
        standard_details=[],
        misc_titles=[],
    )


@pytest.fixture()
def page_result():
    return _make_page_result(page=1)


@pytest.fixture()
def doc_result():
    return SimpleNamespace(pages=[_make_page_result(1), _make_page_result(2)])


@pytest.fixture()
def collection_name():
    return f"e2e_test_{uuid.uuid4().hex[:8]}"


# ── Integration tests ─────────────────────────────────────────────


class TestEndToEnd:
    """Full stack: Index → QueryEngine → results, with mocked LLM."""

    def test_index_to_query_engine(self, page_result, collection_name):
        """Index a page, run a query, get a cited answer."""
        idx = DocumentIndex(collection_name=collection_name)
        n = idx.index_page_result(page_result)
        assert n > 0, "Should index at least one chunk"

        mock_llm = MagicMock(spec=LLMClient)
        mock_meta = ChatMeta(
            provider="ollama", model="test", input_tokens=100, output_tokens=50
        )
        mock_llm.chat_with_metadata.return_value = (
            "The minimum corridor width is 3'-6\" clear (Page 1, notes).",
            mock_meta,
        )
        mock_llm.cost_tracker = CostTracker()

        engine = DocumentQueryEngine(index=idx, llm=mock_llm)
        result = engine.query("What is the minimum corridor width?")

        assert isinstance(result, QueryResult)
        assert "3'-6\"" in result.text or "corridor" in result.text.lower()
        assert result.search_results, "Should have search results"
        assert result.meta is not None
        # The LLM was actually called
        assert mock_llm.chat_with_metadata.called

    def test_query_caching_end_to_end(self, page_result, collection_name):
        """Second identical query returns cached result without LLM call."""
        idx = DocumentIndex(collection_name=collection_name)
        idx.index_page_result(page_result)

        mock_llm = MagicMock(spec=LLMClient)
        mock_meta = ChatMeta(provider="ollama", model="test")
        mock_llm.chat_with_metadata.return_value = ("Answer.", mock_meta)
        mock_llm.cost_tracker = CostTracker()

        engine = DocumentQueryEngine(index=idx, llm=mock_llm)
        r1 = engine.query("What steel grade is specified?")
        r2 = engine.query("What steel grade is specified?")

        assert mock_llm.chat_with_metadata.call_count == 1, "Cached — only 1 LLM call"
        assert r2.cached is True

    def test_search_only_no_llm(self, page_result, collection_name):
        """search_only returns results without calling the LLM."""
        idx = DocumentIndex(collection_name=collection_name)
        idx.index_page_result(page_result)

        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.cost_tracker = CostTracker()
        engine = DocumentQueryEngine(index=idx, llm=mock_llm)

        results = engine.search_only("fire extinguisher", n_results=3)
        assert len(results) > 0
        assert not mock_llm.chat_with_metadata.called

    def test_entity_extraction_end_to_end(self, page_result, collection_name):
        """Extract entities through the full stack with mocked LLM."""
        idx = DocumentIndex(collection_name=collection_name)
        idx.index_page_result(page_result)

        mock_llm = MagicMock(spec=LLMClient)
        mock_meta = ChatMeta(provider="ollama", model="test")
        mock_llm.chat_structured.return_value = (
            [
                {
                    "entity_type": "material",
                    "value": "ASTM A992 Grade 50 steel",
                    "context": "structural steel",
                    "confidence": 0.95,
                },
                {
                    "entity_type": "material",
                    "value": "4000 psi concrete",
                    "context": "normal weight concrete",
                    "confidence": 0.90,
                },
            ],
            mock_meta,
        )
        mock_llm.cost_tracker = CostTracker()

        ext = EntityExtractor(idx, mock_llm, batch_size=5)
        result = ext.extract(["material"])

        assert isinstance(result, ExtractionResult)
        assert len(result.entities) >= 1
        assert all(e.entity_type == "material" for e in result.entities)
        # Export round-trip
        json_text = result.to_json()
        loaded = ExtractionResult.from_json(json_text)
        assert len(loaded.entities) == len(result.entities)

    def test_multi_page_document(self, doc_result):
        """Full flow with a multi-page DocumentResult."""
        cname = f"e2e_multi_{uuid.uuid4().hex[:8]}"
        idx = DocumentIndex(collection_name=cname)
        n = idx.index_document_result(doc_result)
        assert n > 0

        mock_llm = MagicMock(spec=LLMClient)
        mock_meta = ChatMeta(provider="ollama", model="test")
        mock_llm.chat_with_metadata.return_value = (
            "The project is 'Test Building Project' (Page 1, title_block).",
            mock_meta,
        )
        mock_llm.cost_tracker = CostTracker()

        engine = DocumentQueryEngine(index=idx, llm=mock_llm)
        result = engine.query("What is the project name?")
        assert result.text
        assert len(engine.history) == 1


class TestRetryLogic:
    """Verify that LLMClient retries transient failures."""

    @patch("plancheck.llm.client._OLLAMA_AVAILABLE", True)
    def test_retry_succeeds_on_second_attempt(self):
        """Transient error on first call, success on second."""
        client = LLMClient(provider="ollama", max_retries=3)
        # Clear any cached ollama client
        LLMClient._client_cache.clear()

        mock_ollama_client = MagicMock()
        mock_ollama_client.chat.side_effect = [
            ConnectionError("timeout"),
            {"message": {"content": "Success after retry"}},
        ]

        with patch.object(
            client, "_get_or_create_client", return_value=mock_ollama_client
        ):
            result = client.chat("sys", "user")

        assert result == "Success after retry"
        assert mock_ollama_client.chat.call_count == 2

    @patch("plancheck.llm.client._OLLAMA_AVAILABLE", True)
    def test_retry_exhausted(self):
        """All retries fail — raises RuntimeError."""
        client = LLMClient(provider="ollama", max_retries=2)
        LLMClient._client_cache.clear()

        mock_ollama_client = MagicMock()
        mock_ollama_client.chat.side_effect = ConnectionError("down")

        with patch.object(
            client, "_get_or_create_client", return_value=mock_ollama_client
        ):
            with pytest.raises(RuntimeError, match="failed after 2 attempts"):
                client.chat("sys", "user")

        assert mock_ollama_client.chat.call_count == 2

    def test_non_retryable_error_raises_immediately(self):
        """RuntimeError (e.g. missing lib) is not retried."""
        client = LLMClient(provider="ollama", max_retries=3)
        with pytest.raises(RuntimeError, match="not installed"):
            client.chat("sys", "user")


class TestClientCache:
    """Verify that API client instances are reused."""

    @patch("plancheck.llm.client._OLLAMA_AVAILABLE", True)
    def test_same_config_reuses_client(self):
        LLMClient._client_cache.clear()
        c1 = LLMClient(provider="ollama", api_base="http://localhost:11434")
        c2 = LLMClient(provider="ollama", api_base="http://localhost:11434")

        mock_ollama = MagicMock()
        mock_ollama.Client.return_value = MagicMock()
        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            client1 = c1._get_or_create_client("ollama")
            client2 = c2._get_or_create_client("ollama")

        assert client1 is client2, "Same config should reuse the client"

    @patch("plancheck.llm.client._OLLAMA_AVAILABLE", True)
    def test_different_config_creates_new_client(self):
        LLMClient._client_cache.clear()
        c1 = LLMClient(provider="ollama", api_base="http://host-a:11434")
        c2 = LLMClient(provider="ollama", api_base="http://host-b:11434")

        mock_ollama = MagicMock()
        mock_ollama.Client.side_effect = [MagicMock(), MagicMock()]
        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            client1 = c1._get_or_create_client("ollama")
            client2 = c2._get_or_create_client("ollama")

        assert client1 is not client2, "Different hosts should get different clients"


class TestLRUCacheOrdering:
    """Verify the LRU cache evicts least-recently-used, not oldest."""

    def test_access_refreshes_entry(self, page_result, collection_name):
        """Accessing A, then B, then A again → B should be evicted first."""
        from plancheck.llm.query_engine import QueryResult, _ResponseCache

        cache = _ResponseCache(max_size=2)
        cache.put("q1", "ctx", QueryResult(text="answer1"))
        cache.put("q2", "ctx", QueryResult(text="answer2"))

        # Access q1 to refresh it
        cache.get("q1", "ctx")

        # Insert q3 — should evict q2 (LRU), NOT q1
        cache.put("q3", "ctx", QueryResult(text="answer3"))

        assert (
            cache.get("q1", "ctx") is not None
        ), "q1 should still exist (recently used)"
        assert (
            cache.get("q2", "ctx") is None
        ), "q2 should be evicted (least recently used)"
        assert cache.get("q3", "ctx") is not None, "q3 should exist (just added)"
