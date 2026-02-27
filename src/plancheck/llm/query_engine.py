"""Document Query Engine — RAG over plan-page content.

Combines:
- :class:`~plancheck.llm.index.DocumentIndex` for semantic search
- :class:`~plancheck.llm.client.LLMClient` for answer synthesis

Usage::

    from plancheck.llm.query_engine import DocumentQueryEngine

    engine = DocumentQueryEngine.from_document_result(doc_result)
    answer = engine.query("What is the minimum corridor width?")
    print(answer.text)
    print(answer.sources)  # cited chunks

Public API
----------
DocumentQueryEngine  - Build index, search, synthesize answers.
QueryResult          - Answer text + source citations + metadata.
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from plancheck.llm.client import ChatMeta, LLMClient
from plancheck.llm.cost import CostTracker
from plancheck.llm.index import (
    Chunk,
    DocumentIndex,
    SearchResult,
    chunks_from_document_result,
    chunks_from_page_result,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a technical assistant for reviewing construction plan documents.
You answer questions based ONLY on the provided context excerpts from the plan.

Rules:
1. Answer based strictly on the provided context. Do not make assumptions.
2. If the context does not contain enough information to answer, say so clearly.
3. Cite the page number and region type for each fact you reference, e.g. "(Page 1, notes)".
4. Keep answers concise and factual.
5. Use the exact values, measurements, and terminology from the plan text.
"""

_USER_TEMPLATE = """\
CONTEXT (excerpts from the construction plan):
{context}

QUESTION: {question}

Answer the question using ONLY the context above. Cite page numbers.\
"""

_STRUCTURED_TEMPLATE = """\
CONTEXT (excerpts from the construction plan):
{context}

QUESTION: {question}

Return a JSON object with:
- "answer": string — your answer citing page numbers
- "confidence": float — 0.0 to 1.0 how confident the answer is supported by context
- "sources": list of {{"page": int, "region_type": string, "excerpt": string}}

Return ONLY valid JSON.\
"""


# ---------------------------------------------------------------------------
# QueryResult
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    """Result of a query against the document index."""

    text: str = ""
    confidence: float = 0.0
    sources: list[dict] = field(default_factory=list)
    search_results: list[SearchResult] = field(default_factory=list)
    meta: ChatMeta | None = None
    cached: bool = False
    query_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "sources": self.sources,
            "cached": self.cached,
            "meta": self.meta.to_dict() if self.meta else None,
        }


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------


class _ResponseCache:
    """True LRU cache keyed by (query_hash, context_hash).

    Uses :class:`collections.OrderedDict` — accessing an entry moves it
    to the end, so the *first* entry is always the least-recently-used.
    """

    def __init__(self, max_size: int = 200) -> None:
        self._store: collections.OrderedDict[str, QueryResult] = (
            collections.OrderedDict()
        )
        self._max_size = max_size

    def _key(self, query: str, context: str) -> str:
        raw = f"{query}||{context}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def get(self, query: str, context: str) -> QueryResult | None:
        key = self._key(query, context)
        result = self._store.get(key)
        if result is not None:
            self._store.move_to_end(key)  # mark as recently used
        return result

    def put(self, query: str, context: str, result: QueryResult) -> None:
        key = self._key(query, context)
        if key in self._store:
            self._store.move_to_end(key)
        elif len(self._store) >= self._max_size:
            self._store.popitem(last=False)  # evict least-recently-used
        self._store[key] = result

    def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# DocumentQueryEngine
# ---------------------------------------------------------------------------


class DocumentQueryEngine:
    """RAG engine: semantic search → LLM synthesis → cited answer.

    Parameters
    ----------
    index : DocumentIndex
        Pre-built document index.
    llm : LLMClient
        LLM client for answer generation.
    n_context_chunks : int
        Number of chunks to retrieve as context (default 5).
    enable_cache : bool
        Cache responses to avoid repeated LLM calls.
    """

    def __init__(
        self,
        index: DocumentIndex,
        llm: LLMClient,
        *,
        n_context_chunks: int = 5,
        enable_cache: bool = True,
    ) -> None:
        self.index = index
        self.llm = llm
        self.n_context_chunks = n_context_chunks
        self._cache = _ResponseCache() if enable_cache else None
        self._history: list[dict] = []

    # ── Factory methods ────────────────────────────────────────────

    @classmethod
    def from_document_result(
        cls,
        doc_result: Any,
        *,
        provider: str = "ollama",
        model: str = "llama3.1:8b",
        api_key: str = "",
        api_base: str = "http://localhost:11434",
        temperature: float = 0.1,
        policy: str = "local_only",
        embedding_model: str = "all-MiniLM-L6-v2",
        n_context_chunks: int = 5,
        enable_cache: bool = True,
    ) -> "DocumentQueryEngine":
        """Build an engine from a ``DocumentResult``."""
        index = DocumentIndex(embedding_model=embedding_model)
        index.index_document_result(doc_result)

        cost_tracker = CostTracker()
        llm = LLMClient(
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            policy=policy,
            cost_tracker=cost_tracker,
        )

        return cls(
            index=index,
            llm=llm,
            n_context_chunks=n_context_chunks,
            enable_cache=enable_cache,
        )

    @classmethod
    def from_page_result(
        cls,
        page_result: Any,
        *,
        provider: str = "ollama",
        model: str = "llama3.1:8b",
        api_key: str = "",
        api_base: str = "http://localhost:11434",
        temperature: float = 0.1,
        policy: str = "local_only",
        embedding_model: str = "all-MiniLM-L6-v2",
        n_context_chunks: int = 5,
        enable_cache: bool = True,
    ) -> "DocumentQueryEngine":
        """Build an engine from a single ``PageResult``."""
        index = DocumentIndex(embedding_model=embedding_model)
        index.index_page_result(page_result)

        cost_tracker = CostTracker()
        llm = LLMClient(
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            policy=policy,
            cost_tracker=cost_tracker,
        )

        return cls(
            index=index,
            llm=llm,
            n_context_chunks=n_context_chunks,
            enable_cache=enable_cache,
        )

    # ── Query ──────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        *,
        page_filter: int | None = None,
        region_filter: str | None = None,
        structured: bool = False,
    ) -> QueryResult:
        """Ask a question about the indexed plan documents.

        Parameters
        ----------
        question : str
            Natural-language question.
        page_filter : int, optional
            Restrict context to a specific page.
        region_filter : str, optional
            Restrict context to a specific region type.
        structured : bool
            If True, request a JSON response with confidence and sources.

        Returns
        -------
        QueryResult
        """
        # 1. Retrieve relevant chunks
        search_results = self.index.search(
            question,
            n_results=self.n_context_chunks,
            page_filter=page_filter,
            region_filter=region_filter,
        )

        # 2. Build context string
        context = self._format_context(search_results)

        # 3. Check cache
        if self._cache:
            cached = self._cache.get(question, context)
            if cached is not None:
                cached.cached = True
                return cached

        # 4. Build prompt and call LLM
        if structured:
            user_prompt = _STRUCTURED_TEMPLATE.format(
                context=context, question=question
            )
        else:
            user_prompt = _USER_TEMPLATE.format(context=context, question=question)

        try:
            if structured:
                parsed, meta = self.llm.chat_structured(
                    _SYSTEM_PROMPT, user_prompt, expect_json=False
                )
                result = self._parse_structured(parsed, meta, search_results)
            else:
                text, meta = self.llm.chat_with_metadata(_SYSTEM_PROMPT, user_prompt)
                result = QueryResult(
                    text=text.strip(),
                    search_results=search_results,
                    meta=meta,
                    sources=[
                        {
                            "page": sr.chunk.page,
                            "region_type": sr.chunk.region_type,
                            "excerpt": sr.chunk.text[:100],
                        }
                        for sr in search_results
                    ],
                )
        except Exception as exc:
            log.warning("Query engine LLM call failed: %s", exc)
            result = QueryResult(
                text=f"Error: {exc}",
                search_results=search_results,
            )

        # 5. Cache and record
        if self._cache:
            self._cache.put(question, context, result)

        self._history.append(
            {
                "question": question,
                "answer_preview": result.text[:100],
                "n_sources": len(result.sources),
                "cached": result.cached,
            }
        )

        return result

    def search_only(
        self,
        question: str,
        n_results: int = 5,
        *,
        page_filter: int | None = None,
        region_filter: str | None = None,
    ) -> list[SearchResult]:
        """Semantic search without LLM synthesis."""
        return self.index.search(
            question,
            n_results=n_results,
            page_filter=page_filter,
            region_filter=region_filter,
        )

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _format_context(results: list[SearchResult]) -> str:
        """Format search results into a context string for the LLM."""
        parts: list[str] = []
        for r in results:
            c = r.chunk
            header = f"[Page {c.page}, {c.region_type}]"
            parts.append(f"{header}\n{c.text}")
        return "\n\n---\n\n".join(parts) if parts else "(no relevant context found)"

    @staticmethod
    def _parse_structured(
        parsed: Any,
        meta: ChatMeta,
        search_results: list[SearchResult],
    ) -> QueryResult:
        """Parse a structured LLM response into a QueryResult."""
        if isinstance(parsed, dict):
            return QueryResult(
                text=parsed.get("answer", str(parsed)),
                confidence=float(parsed.get("confidence", 0.0)),
                sources=parsed.get("sources", []),
                search_results=search_results,
                meta=meta,
            )
        # Fallback — treat as plain text
        return QueryResult(
            text=str(parsed),
            search_results=search_results,
            meta=meta,
        )

    # ── Properties ─────────────────────────────────────────────────

    @property
    def history(self) -> list[dict]:
        """Query history for this session."""
        return list(self._history)

    @property
    def cost_summary(self) -> dict:
        """Return cost tracker summary."""
        return self.llm.cost_tracker.summary()

    def clear_cache(self) -> None:
        if self._cache:
            self._cache.clear()
