"""Document indexing and semantic search over plan-page content.

This module builds a ChromaDB vector index from pipeline results
(``PageResult`` / ``DocumentResult``) and provides a search interface
that returns ranked chunks with metadata (page number, region type,
bounding-box extents, etc.).

The index can be:
- **In-memory** — fast, ephemeral (default)
- **Persistent** — saved to disk for reuse across sessions

Embedding is handled by ``sentence-transformers`` via ChromaDB's built-in
embedding functions, or by a lightweight character-n-gram fallback when
``sentence-transformers`` is not installed.

Public API
----------
DocumentIndex       - Build, persist, and query a plan-document index.
Chunk               - A single indexed text chunk with metadata.
SearchResult        - Ranked search result with score.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Availability probes
# ---------------------------------------------------------------------------

_CHROMA_AVAILABLE = False
_ST_AVAILABLE = False

try:
    import chromadb  # noqa: F401

    _CHROMA_AVAILABLE = True
except ImportError:
    pass

try:
    import sentence_transformers  # noqa: F401

    _ST_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A discrete piece of text extracted from a plan page."""

    text: str
    page: int = 0
    region_type: str = ""  # e.g. "notes", "legend", "title_block"
    source_id: str = ""  # unique id for dedup
    metadata: dict = field(default_factory=dict)

    def content_hash(self) -> str:
        """Deterministic hash of text + page for cache keys."""
        raw = f"{self.page}:{self.text}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class SearchResult:
    """A ranked search hit."""

    chunk: Chunk
    score: float = 0.0  # similarity (higher = better)
    rank: int = 0


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------


def _safe_text(obj: Any, attr: str = "text") -> str:
    """Extract text from an object, handling missing attributes gracefully."""
    if hasattr(obj, attr):
        val = getattr(obj, attr)
        return val() if callable(val) else (val or "")
    return ""


def _chunk_blocks(blocks: Sequence[Any], page: int, region_type: str) -> list[Chunk]:
    """Turn a sequence of text blocks into Chunks."""
    chunks: list[Chunk] = []
    for i, block in enumerate(blocks):
        text = ""
        if hasattr(block, "get_all_boxes"):
            boxes = block.get_all_boxes()
            text = " ".join(getattr(b, "text", "") for b in boxes)
        elif hasattr(block, "text"):
            text = block.text or ""
        text = text.strip()
        if not text:
            continue
        chunks.append(
            Chunk(
                text=text,
                page=page,
                region_type=region_type,
                source_id=f"p{page}_{region_type}_{i}",
                metadata={"block_index": i},
            )
        )
    return chunks


def chunks_from_page_result(pr: Any) -> list[Chunk]:
    """Extract indexable chunks from a ``PageResult``.

    Chunks are created for:
    - Notes columns (block-level)
    - Legend entries
    - Abbreviation entries
    - Title block fields
    - Revision entries
    - General plan notes (full text of each notes column)
    """
    chunks: list[Chunk] = []
    page = getattr(pr, "page_number", 0)

    # ── Notes columns ─────────────────────────────────────────────
    if pr.notes_columns:
        for ci, col in enumerate(pr.notes_columns):
            blocks = getattr(col, "blocks", [])
            col_chunks = _chunk_blocks(blocks, page, "notes")
            chunks.extend(col_chunks)
            # Also add full-column text as a single chunk (for broader context)
            full = _safe_text(col, "full_text")
            if not full and blocks:
                parts: list[str] = []
                for b in blocks:
                    t = _safe_text(b, "text")
                    if not t and hasattr(b, "get_all_boxes"):
                        t = " ".join(
                            getattr(box, "text", "") for box in b.get_all_boxes()
                        )
                    if t:
                        parts.append(t)
                full = " ".join(parts)
            if full.strip():
                chunks.append(
                    Chunk(
                        text=full.strip(),
                        page=page,
                        region_type="notes_full",
                        source_id=f"p{page}_notes_full_{ci}",
                    )
                )

    # ── Legend entries ─────────────────────────────────────────────
    if pr.legends:
        for li, leg in enumerate(pr.legends):
            entries = getattr(leg, "entries", [])
            for ei, entry in enumerate(entries):
                symbol = getattr(entry, "symbol", "")
                desc = getattr(entry, "description", "")
                text = f"{symbol}: {desc}".strip(": ")
                if text:
                    chunks.append(
                        Chunk(
                            text=text,
                            page=page,
                            region_type="legend",
                            source_id=f"p{page}_legend_{li}_{ei}",
                        )
                    )

    # ── Abbreviations ─────────────────────────────────────────────
    if pr.abbreviations:
        for ai, abbr_region in enumerate(pr.abbreviations):
            entries = getattr(abbr_region, "entries", [])
            for ei, entry in enumerate(entries):
                short = getattr(entry, "abbreviation", "")
                full = getattr(entry, "full_text", "")
                text = f"{short} = {full}".strip(" =")
                if text:
                    chunks.append(
                        Chunk(
                            text=text,
                            page=page,
                            region_type="abbreviation",
                            source_id=f"p{page}_abbr_{ai}_{ei}",
                        )
                    )

    # ── Revisions ─────────────────────────────────────────────────
    if pr.revisions:
        for ri, rev_region in enumerate(pr.revisions):
            entries = getattr(rev_region, "entries", [])
            for ei, entry in enumerate(entries):
                desc = getattr(entry, "description", "")
                date = getattr(entry, "date", "")
                num = getattr(entry, "number", "")
                text = f"Rev {num} ({date}): {desc}".strip()
                if text and text != "Rev  ():":
                    chunks.append(
                        Chunk(
                            text=text,
                            page=page,
                            region_type="revision",
                            source_id=f"p{page}_rev_{ri}_{ei}",
                        )
                    )

    # ── Title block ───────────────────────────────────────────────
    if pr.title_block:
        tb = pr.title_block
        for fld in (
            "project_name",
            "sheet_title",
            "sheet_number",
            "drawn_by",
            "checked_by",
            "date",
            "scale",
        ):
            val = getattr(tb, fld, "")
            if val:
                chunks.append(
                    Chunk(
                        text=f"{fld}: {val}",
                        page=page,
                        region_type="title_block",
                        source_id=f"p{page}_tb_{fld}",
                    )
                )

    # ── Standard detail entries ───────────────────────────────────
    if pr.standard_details:
        for si, sd_region in enumerate(pr.standard_details):
            entries = getattr(sd_region, "entries", [])
            for ei, entry in enumerate(entries):
                name = getattr(entry, "name", "")
                ref = getattr(entry, "reference", "")
                text = f"{name} ({ref})".strip(" ()")
                if text:
                    chunks.append(
                        Chunk(
                            text=text,
                            page=page,
                            region_type="standard_detail",
                            source_id=f"p{page}_sd_{si}_{ei}",
                        )
                    )

    # ── Misc title regions ────────────────────────────────────────
    if pr.misc_titles:
        for mi, mt in enumerate(pr.misc_titles):
            title = getattr(mt, "title", "")
            if title:
                chunks.append(
                    Chunk(
                        text=title,
                        page=page,
                        region_type="misc_title",
                        source_id=f"p{page}_mt_{mi}",
                    )
                )

    return chunks


def chunks_from_document_result(dr: Any) -> list[Chunk]:
    """Extract chunks from every page in a ``DocumentResult``."""
    all_chunks: list[Chunk] = []
    for pr in getattr(dr, "pages", []):
        all_chunks.extend(chunks_from_page_result(pr))
    return all_chunks


# ---------------------------------------------------------------------------
# Document index
# ---------------------------------------------------------------------------


class DocumentIndex:
    """Semantic search index backed by ChromaDB.

    Parameters
    ----------
    persist_dir : Path | str | None
        If provided, the index is persisted to this directory.
        If None, an ephemeral in-memory index is used.
    collection_name : str
        Name of the ChromaDB collection.
    embedding_model : str
        Sentence-transformer model name for embeddings.
    """

    def __init__(
        self,
        persist_dir: Path | str | None = None,
        collection_name: str = "plan_documents",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        if not _CHROMA_AVAILABLE:
            raise RuntimeError(
                "chromadb is not installed. Install with: "
                "pip install 'plancheck[query]'"
            )

        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Initialise ChromaDB client
        import chromadb

        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        else:
            self._client = chromadb.Client()  # in-memory

        # Embedding function
        ef = self._get_embedding_function()
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

    def _get_embedding_function(self):
        """Return a ChromaDB-compatible embedding function."""
        import chromadb.utils.embedding_functions as ef

        if _ST_AVAILABLE:
            return ef.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model
            )
        else:
            log.warning(
                "sentence-transformers not installed — using default embeddings. "
                "Install with: pip install 'plancheck[query]'"
            )
            return ef.DefaultEmbeddingFunction()

    # ── Indexing ───────────────────────────────────────────────────

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Add chunks to the index.  Returns the number added (after dedup)."""
        if not chunks:
            return 0

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        seen: set[str] = set()
        for c in chunks:
            cid = c.source_id or c.content_hash()
            if cid in seen:
                continue
            seen.add(cid)
            ids.append(cid)
            documents.append(c.text)
            meta = {
                "page": c.page,
                "region_type": c.region_type,
                "content_hash": c.content_hash(),
            }
            meta.update({k: str(v) for k, v in c.metadata.items()})
            metadatas.append(meta)

        # ChromaDB upsert handles duplicates gracefully
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        return len(ids)

    def index_page_result(self, pr: Any) -> int:
        """Extract chunks from a PageResult and add to the index."""
        chunks = chunks_from_page_result(pr)
        return self.add_chunks(chunks)

    def index_document_result(self, dr: Any) -> int:
        """Extract chunks from a DocumentResult and add to the index."""
        chunks = chunks_from_document_result(dr)
        return self.add_chunks(chunks)

    # ── Search ─────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n_results: int = 5,
        *,
        page_filter: int | None = None,
        region_filter: str | None = None,
    ) -> list[SearchResult]:
        """Semantic search over indexed chunks.

        Parameters
        ----------
        query : str
            Natural-language query text.
        n_results : int
            Maximum number of results to return.
        page_filter : int, optional
            Restrict results to a specific page number.
        region_filter : str, optional
            Restrict results to a specific region type.

        Returns
        -------
        list[SearchResult]
            Ranked results (highest similarity first).
        """
        where: dict | None = None
        conditions = []
        if page_filter is not None:
            conditions.append({"page": page_filter})
        if region_filter:
            conditions.append({"region_type": region_filter})

        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(n_results, self._collection.count() or 1),
        }
        if where:
            kwargs["where"] = where

        try:
            results = self._collection.query(**kwargs)
        except Exception as exc:
            log.warning("ChromaDB query failed: %s", exc)
            return []

        search_results: list[SearchResult] = []
        if not results or not results.get("documents"):
            return search_results

        docs = results["documents"][0]
        metas = (
            results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        )
        dists = (
            results["distances"][0] if results.get("distances") else [0.0] * len(docs)
        )
        ids = results["ids"][0] if results.get("ids") else [""] * len(docs)

        for rank, (doc, meta, dist, cid) in enumerate(zip(docs, metas, dists, ids)):
            # ChromaDB returns distance; convert to similarity (cosine: sim = 1 - dist)
            similarity = max(0.0, 1.0 - dist)
            chunk = Chunk(
                text=doc,
                page=int(meta.get("page", 0)),
                region_type=str(meta.get("region_type", "")),
                source_id=cid,
                metadata=meta,
            )
            search_results.append(
                SearchResult(
                    chunk=chunk,
                    score=round(similarity, 4),
                    rank=rank + 1,
                )
            )

        return search_results

    # ── Utilities ──────────────────────────────────────────────────

    @property
    def count(self) -> int:
        """Number of chunks in the index."""
        return self._collection.count()

    def clear(self) -> None:
        """Remove all documents from the collection."""
        self._client.delete_collection(self.collection_name)
        ef = self._get_embedding_function()
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
