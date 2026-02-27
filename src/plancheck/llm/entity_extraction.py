"""Structured entity extraction from plan documents.

Uses the document index + LLM to extract typed entities (Materials,
Dimensions, Equipment) from construction-plan content with structured
JSON output enforcement.

Extraction is *chunk-based*: the extractor pulls relevant chunks from
the index and asks the LLM to identify entities within each chunk.
This keeps prompts short and results traceable to source text.

Public API
----------
EntityExtractor     - Extract typed entities from indexed plan content.
Entity              - A single extracted entity with type, value, and provenance.
ExtractionResult    - Collection of entities with metadata and export helpers.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

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
# Entity types
# ---------------------------------------------------------------------------

ENTITY_TYPES = ("material", "dimension", "equipment")


@dataclass
class Entity:
    """A single extracted entity."""

    entity_type: str  # "material", "dimension", "equipment"
    value: str  # the extracted value, e.g. "W14×22 steel beam"
    context: str = ""  # surrounding text that mentions this entity
    page: int = 0
    region_type: str = ""  # source region, e.g. "notes", "legend"
    confidence: float = 0.0  # 0.0–1.0 LLM-reported confidence
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "entity_type": self.entity_type,
            "value": self.value,
            "context": self.context,
            "page": self.page,
            "region_type": self.region_type,
            "confidence": round(self.confidence, 3),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Entity:
        return cls(
            entity_type=d.get("entity_type", ""),
            value=d.get("value", ""),
            context=d.get("context", ""),
            page=d.get("page", 0),
            region_type=d.get("region_type", ""),
            confidence=d.get("confidence", 0.0),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Extraction result
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    """Collection of extracted entities with export helpers."""

    entities: list[Entity] = field(default_factory=list)
    meta: list[ChatMeta] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # ── Filtering ──────────────────────────────────────────────

    def by_type(self, entity_type: str) -> list[Entity]:
        """Return entities of a given type."""
        return [e for e in self.entities if e.entity_type == entity_type]

    @property
    def materials(self) -> list[Entity]:
        return self.by_type("material")

    @property
    def dimensions(self) -> list[Entity]:
        return self.by_type("dimension")

    @property
    def equipment(self) -> list[Entity]:
        return self.by_type("equipment")

    # ── Summaries ──────────────────────────────────────────────

    @property
    def summary(self) -> dict:
        return {
            "total": len(self.entities),
            "materials": len(self.materials),
            "dimensions": len(self.dimensions),
            "equipment": len(self.equipment),
            "errors": len(self.errors),
            "llm_calls": len(self.meta),
        }

    # ── Export ─────────────────────────────────────────────────

    def to_json(self, indent: int = 2) -> str:
        """Serialize all entities to a JSON string."""
        return json.dumps(
            {
                "entities": [e.to_dict() for e in self.entities],
                "summary": self.summary,
            },
            indent=indent,
        )

    def save_json(self, path: str | Path) -> None:
        """Write entities to a JSON file."""
        Path(path).write_text(self.to_json(), encoding="utf-8")

    def to_csv(self) -> str:
        """Serialize entities to CSV string."""
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=[
                "entity_type",
                "value",
                "context",
                "page",
                "region_type",
                "confidence",
            ],
        )
        writer.writeheader()
        for e in self.entities:
            writer.writerow(
                {
                    "entity_type": e.entity_type,
                    "value": e.value,
                    "context": e.context[:200],
                    "page": e.page,
                    "region_type": e.region_type,
                    "confidence": round(e.confidence, 3),
                }
            )
        return buf.getvalue()

    def save_csv(self, path: str | Path) -> None:
        """Write entities to a CSV file."""
        Path(path).write_text(self.to_csv(), encoding="utf-8")

    @classmethod
    def from_json(cls, text: str) -> ExtractionResult:
        """Load from a JSON string."""
        data = json.loads(text)
        entities = [Entity.from_dict(e) for e in data.get("entities", [])]
        return cls(entities=entities)

    @classmethod
    def load_json(cls, path: str | Path) -> ExtractionResult:
        """Load from a JSON file."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a construction-document entity extractor.
Given an excerpt from architectural/engineering plan notes, extract
all entities of the requested type(s).

Return ONLY a JSON array of objects.  Each object must have:
  - "entity_type": one of "material", "dimension", "equipment"
  - "value": the extracted entity value (be precise, include units)
  - "context": a short phrase from the source text showing where this entity appears
  - "confidence": a float 0.0 to 1.0 indicating your confidence

Rules:
1. Extract ONLY entities present in the given text — do not invent.
2. For materials: include specification grades, thicknesses, coatings
   (e.g., "5/8\" Type X gypsum board", "A992 steel", "4000 psi concrete").
3. For dimensions: include all measurements with units
   (e.g., "3'-6\"", "12\" o.c.", "2'-0\" min clear", "10'-0\" ceiling height").
4. For equipment: include mechanical, electrical, plumbing, fire-protection
   items (e.g., "VAV box", "fire extinguisher", "emergency light", "panel board").
5. If no entities of the requested type exist in the text, return an empty array: []
6. Do NOT wrap the JSON in markdown fences.
"""

_USER_TEMPLATE = """\
Extract all {entity_types} from this plan excerpt (page {page}, {region_type}):

---
{text}
---

Return a JSON array of entity objects.
"""

_BATCH_USER_TEMPLATE = """\
Extract all entities of types [{entity_types}] from these plan excerpts:

{chunks_text}

Return a JSON array of entity objects.
"""


# ---------------------------------------------------------------------------
# Entity extractor
# ---------------------------------------------------------------------------


class EntityExtractor:
    """Extract structured entities from plan documents using LLM.

    Parameters
    ----------
    index : DocumentIndex
        The semantic index to search for relevant chunks.
    llm : LLMClient
        The LLM client for entity extraction.
    batch_size : int
        Number of chunks to combine per LLM call (reduces API calls).
    max_chunks : int
        Maximum chunks to process per entity type.
    """

    def __init__(
        self,
        index: DocumentIndex,
        llm: LLMClient,
        *,
        batch_size: int = 3,
        max_chunks: int = 30,
    ) -> None:
        self.index = index
        self.llm = llm
        self.batch_size = max(1, batch_size)
        self.max_chunks = max(1, max_chunks)

    # ── Factory methods ────────────────────────────────────────

    @classmethod
    def from_page_result(
        cls,
        pr: Any,
        *,
        provider: str = "ollama",
        model: str = "llama3.1:8b",
        api_key: str = "",
        api_base: str = "http://localhost:11434",
        temperature: float = 0.1,
        policy: str = "local_only",
        **kwargs: Any,
    ) -> EntityExtractor:
        """Build an extractor from a single PageResult."""
        idx = DocumentIndex(collection_name="entity_extraction_page")
        idx.index_page_result(pr)
        llm = LLMClient(
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            policy=policy,
        )
        return cls(idx, llm, **kwargs)

    @classmethod
    def from_document_result(
        cls,
        dr: Any,
        *,
        provider: str = "ollama",
        model: str = "llama3.1:8b",
        api_key: str = "",
        api_base: str = "http://localhost:11434",
        temperature: float = 0.1,
        policy: str = "local_only",
        **kwargs: Any,
    ) -> EntityExtractor:
        """Build an extractor from a full DocumentResult."""
        idx = DocumentIndex(collection_name="entity_extraction_doc")
        idx.index_document_result(dr)
        llm = LLMClient(
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            policy=policy,
        )
        return cls(idx, llm, **kwargs)

    # ── Core extraction ────────────────────────────────────────

    def _gather_chunks(
        self,
        entity_types: Sequence[str],
        *,
        page_filter: int | None = None,
        region_filter: str | None = None,
    ) -> list[Chunk]:
        """Search the index for chunks likely to contain entities."""
        # Use targeted search queries for each entity type
        search_queries = {
            "material": [
                "material specification concrete steel",
                "gypsum board insulation coating finish",
                "wood lumber plywood membrane",
            ],
            "dimension": [
                "width height length clearance spacing",
                "feet inches minimum maximum",
                "dimension measurement size",
            ],
            "equipment": [
                "HVAC mechanical electrical equipment",
                "fire extinguisher sprinkler alarm panel",
                "plumbing fixture pump valve",
            ],
        }

        seen_ids: set[str] = set()
        chunks: list[Chunk] = []

        for etype in entity_types:
            queries = search_queries.get(etype, [f"{etype} specification"])
            for q in queries:
                results = self.index.search(
                    q,
                    n_results=self.max_chunks,
                    page_filter=page_filter,
                    region_filter=region_filter,
                )
                for r in results:
                    if r.chunk.source_id not in seen_ids:
                        seen_ids.add(r.chunk.source_id)
                        chunks.append(r.chunk)
                if len(chunks) >= self.max_chunks:
                    break

        return chunks[: self.max_chunks]

    def _extract_from_chunks(
        self,
        chunks: list[Chunk],
        entity_types: Sequence[str],
    ) -> ExtractionResult:
        """Send chunks to LLM in batches and parse entity results."""
        all_entities: list[Entity] = []
        all_meta: list[ChatMeta] = []
        errors: list[str] = []

        type_label = ", ".join(entity_types)

        # Process in batches
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]

            if len(batch) == 1:
                c = batch[0]
                user_prompt = _USER_TEMPLATE.format(
                    entity_types=type_label,
                    page=c.page,
                    region_type=c.region_type or "general",
                    text=c.text[:2000],
                )
            else:
                chunks_text = "\n\n".join(
                    f"[Page {c.page}, {c.region_type or 'general'}]:\n{c.text[:1500]}"
                    for c in batch
                )
                user_prompt = _BATCH_USER_TEMPLATE.format(
                    entity_types=type_label,
                    chunks_text=chunks_text,
                )

            try:
                parsed, meta = self.llm.chat_structured(
                    _SYSTEM_PROMPT,
                    user_prompt,
                    expect_json=True,
                )
                all_meta.append(meta)

                entities = self._parse_entities(parsed, batch, entity_types)
                all_entities.extend(entities)

            except (ValueError, RuntimeError) as exc:
                msg = f"Extraction failed for batch starting at chunk {i}: {exc}"
                log.warning(msg)
                errors.append(msg)

        # Deduplicate
        all_entities = self._deduplicate(all_entities)

        return ExtractionResult(
            entities=all_entities,
            meta=all_meta,
            errors=errors,
        )

    def _parse_entities(
        self,
        parsed: Any,
        source_chunks: list[Chunk],
        allowed_types: Sequence[str],
    ) -> list[Entity]:
        """Convert LLM JSON output to Entity objects."""
        # Accept both a raw list and a dict with an "entities" key
        if isinstance(parsed, dict):
            parsed = parsed.get("entities", [])

        if not isinstance(parsed, list):
            log.warning("Unexpected LLM output type: %s", type(parsed))
            return []

        entities: list[Entity] = []
        # Default provenance from first source chunk
        default_page = source_chunks[0].page if source_chunks else 0
        default_region = source_chunks[0].region_type if source_chunks else ""

        for item in parsed:
            if not isinstance(item, dict):
                continue

            etype = str(item.get("entity_type", "")).lower().strip()
            value = str(item.get("value", "")).strip()

            if not value:
                continue
            # Keep entities whose type is in the allowed set OR is at
            # least a recognised type (e.g. LLM returned "dimension"
            # when we only asked for "material").  Discard truly
            # unknown types.
            if etype not in allowed_types and etype not in ENTITY_TYPES:
                continue

            entities.append(
                Entity(
                    entity_type=etype,
                    value=value,
                    context=str(item.get("context", ""))[:300],
                    page=int(item.get("page", default_page)),
                    region_type=str(item.get("region_type", default_region)),
                    confidence=float(item.get("confidence", 0.0)),
                    metadata={},
                )
            )

        return entities

    def _deduplicate(self, entities: list[Entity]) -> list[Entity]:
        """Remove duplicate entities (same type + value + page)."""
        seen: set[str] = set()
        unique: list[Entity] = []
        for e in entities:
            key = f"{e.entity_type}|{e.value.lower().strip()}|{e.page}"
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique

    # ── Public API ─────────────────────────────────────────────

    def extract(
        self,
        entity_types: Sequence[str] | None = None,
        *,
        page_filter: int | None = None,
        region_filter: str | None = None,
    ) -> ExtractionResult:
        """Extract entities of the specified types.

        Parameters
        ----------
        entity_types : sequence of str, optional
            Types to extract.  Defaults to all three:
            ``("material", "dimension", "equipment")``.
        page_filter : int, optional
            Restrict to a specific page.
        region_filter : str, optional
            Restrict to a specific region type (e.g. ``"notes"``).

        Returns
        -------
        ExtractionResult
            Extracted entities with export helpers.
        """
        if entity_types is None:
            entity_types = list(ENTITY_TYPES)
        else:
            entity_types = [t.lower().strip() for t in entity_types]

        chunks = self._gather_chunks(
            entity_types,
            page_filter=page_filter,
            region_filter=region_filter,
        )

        if not chunks:
            log.info("No chunks found for entity types: %s", entity_types)
            return ExtractionResult()

        return self._extract_from_chunks(chunks, entity_types)

    def extract_materials(self, **kwargs: Any) -> ExtractionResult:
        """Convenience: extract material entities only."""
        return self.extract(["material"], **kwargs)

    def extract_dimensions(self, **kwargs: Any) -> ExtractionResult:
        """Convenience: extract dimension entities only."""
        return self.extract(["dimension"], **kwargs)

    def extract_equipment(self, **kwargs: Any) -> ExtractionResult:
        """Convenience: extract equipment entities only."""
        return self.extract(["equipment"], **kwargs)

    @property
    def cost_summary(self) -> dict:
        """Return cost tracker summary."""
        return self.llm.cost_tracker.summary()
