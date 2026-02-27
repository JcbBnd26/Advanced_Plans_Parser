"""Tests for plancheck.llm.entity_extraction."""

from __future__ import annotations

import csv
import io
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from plancheck.llm.entity_extraction import (
    _SYSTEM_PROMPT,
    _USER_TEMPLATE,
    ENTITY_TYPES,
    Entity,
    EntityExtractor,
    ExtractionResult,
)

# ═══════════════════════════════════════════════════════════════════
# Entity dataclass
# ═══════════════════════════════════════════════════════════════════


class TestEntity:
    def test_to_dict(self):
        e = Entity(
            entity_type="material",
            value="A992 steel",
            context="use A992 steel for all columns",
            page=2,
            region_type="notes",
            confidence=0.95,
        )
        d = e.to_dict()
        assert d["entity_type"] == "material"
        assert d["value"] == "A992 steel"
        assert d["page"] == 2
        assert d["confidence"] == 0.95

    def test_round_trip(self):
        original = Entity(
            entity_type="dimension",
            value="3'-6\"",
            context="minimum clear width of 3'-6\"",
            page=5,
            region_type="notes",
            confidence=0.88,
            metadata={"source": "note_12"},
        )
        restored = Entity.from_dict(original.to_dict())
        assert restored.entity_type == original.entity_type
        assert restored.value == original.value
        assert restored.page == original.page
        assert restored.confidence == original.confidence

    def test_from_dict_defaults(self):
        e = Entity.from_dict({})
        assert e.entity_type == ""
        assert e.value == ""
        assert e.page == 0
        assert e.confidence == 0.0


# ═══════════════════════════════════════════════════════════════════
# ExtractionResult
# ═══════════════════════════════════════════════════════════════════


class TestExtractionResult:
    @pytest.fixture()
    def result(self):
        return ExtractionResult(
            entities=[
                Entity("material", "4000 psi concrete", page=1),
                Entity("material", "A992 steel", page=1),
                Entity("dimension", "3'-6\"", page=2),
                Entity("equipment", "VAV box", page=3),
            ]
        )

    def test_by_type(self, result):
        assert len(result.by_type("material")) == 2
        assert len(result.by_type("dimension")) == 1
        assert len(result.by_type("equipment")) == 1
        assert len(result.by_type("unknown")) == 0

    def test_properties(self, result):
        assert len(result.materials) == 2
        assert len(result.dimensions) == 1
        assert len(result.equipment) == 1

    def test_summary(self, result):
        s = result.summary
        assert s["total"] == 4
        assert s["materials"] == 2
        assert s["dimensions"] == 1
        assert s["equipment"] == 1
        assert s["errors"] == 0

    def test_to_json(self, result):
        j = result.to_json()
        data = json.loads(j)
        assert len(data["entities"]) == 4
        assert data["summary"]["total"] == 4

    def test_from_json_round_trip(self, result):
        j = result.to_json()
        restored = ExtractionResult.from_json(j)
        assert len(restored.entities) == 4
        assert restored.entities[0].value == "4000 psi concrete"

    def test_to_csv(self, result):
        csv_text = result.to_csv()
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        assert len(rows) == 4
        assert rows[0]["entity_type"] == "material"
        assert rows[0]["value"] == "4000 psi concrete"

    def test_save_and_load_json(self, result, tmp_path):
        p = tmp_path / "entities.json"
        result.save_json(p)
        loaded = ExtractionResult.load_json(p)
        assert len(loaded.entities) == 4

    def test_save_csv(self, result, tmp_path):
        p = tmp_path / "entities.csv"
        result.save_csv(p)
        text = p.read_text()
        assert "material" in text
        assert "VAV box" in text

    def test_empty_result(self):
        r = ExtractionResult()
        assert r.summary["total"] == 0
        assert r.to_json()


# ═══════════════════════════════════════════════════════════════════
# EntityExtractor (with mocked LLM + ChromaDB index)
# ═══════════════════════════════════════════════════════════════════


# Helpers to build a minimal PageResult-like object
def _make_page_result(page: int = 1, notes_text: str = ""):
    """Return a minimal object that chunks_from_page_result can process."""
    block = SimpleNamespace(text=notes_text)
    col = SimpleNamespace(blocks=[block], full_text=notes_text)
    return SimpleNamespace(
        page_number=page,
        notes_columns=[col],
        legends=[],
        abbreviations=[],
        revisions=[],
        title_block=None,
        standard_details=[],
        misc_titles=[],
    )


@pytest.fixture()
def _index_with_notes():
    """Build an in-memory DocumentIndex with sample construction notes."""
    import uuid

    from plancheck.llm.index import DocumentIndex

    idx = DocumentIndex(collection_name=f"test_entity_{uuid.uuid4().hex[:8]}")

    notes = [
        "All structural steel shall be ASTM A992 Grade 50. "
        "Concrete shall be 4000 psi normal weight at 28 days.",
        "Minimum corridor width: 3'-6\" clear. "
        "Door openings: 3'-0\" wide × 7'-0\" high minimum.",
        "Provide VAV boxes for all interior zones. "
        "Install fire extinguisher cabinets per code. "
        "Emergency lights at all exit corridors.",
        '5/8" Type X gypsum board on all rated assemblies. '
        "Roof insulation: R-30 minimum. "
        "Waterproof membrane at all below-grade walls.",
    ]

    for i, text in enumerate(notes):
        pr = _make_page_result(page=i + 1, notes_text=text)
        idx.index_page_result(pr)

    return idx


class TestEntityExtractor:
    def test_gather_chunks_materials(self, _index_with_notes):
        """Verify semantic search finds material-related chunks."""
        llm = MagicMock()
        ext = EntityExtractor(_index_with_notes, llm, max_chunks=10)
        chunks = ext._gather_chunks(["material"])
        assert len(chunks) > 0
        # At least one chunk should mention steel or concrete
        texts = " ".join(c.text for c in chunks).lower()
        assert "steel" in texts or "concrete" in texts or "gypsum" in texts

    def test_gather_chunks_dimensions(self, _index_with_notes):
        llm = MagicMock()
        ext = EntityExtractor(_index_with_notes, llm, max_chunks=10)
        chunks = ext._gather_chunks(["dimension"])
        assert len(chunks) > 0

    def test_gather_chunks_equipment(self, _index_with_notes):
        llm = MagicMock()
        ext = EntityExtractor(_index_with_notes, llm, max_chunks=10)
        chunks = ext._gather_chunks(["equipment"])
        assert len(chunks) > 0
        texts = " ".join(c.text for c in chunks).lower()
        assert "vav" in texts or "extinguisher" in texts or "emergency" in texts

    def test_parse_entities_list(self):
        """_parse_entities handles a JSON list correctly."""
        from plancheck.llm.index import Chunk

        llm = MagicMock()
        idx = MagicMock()
        ext = EntityExtractor(idx, llm)

        parsed = [
            {
                "entity_type": "material",
                "value": "A992 steel",
                "context": "structural steel",
                "confidence": 0.9,
            },
            {
                "entity_type": "material",
                "value": "4000 psi concrete",
                "context": "concrete spec",
                "confidence": 0.85,
            },
        ]
        source = [Chunk(text="test", page=1, region_type="notes")]
        entities = ext._parse_entities(parsed, source, ["material"])
        assert len(entities) == 2
        assert entities[0].value == "A992 steel"
        assert entities[1].confidence == 0.85

    def test_parse_entities_dict_wrapper(self):
        """_parse_entities handles a dict with 'entities' key."""
        from plancheck.llm.index import Chunk

        llm = MagicMock()
        idx = MagicMock()
        ext = EntityExtractor(idx, llm)

        parsed = {
            "entities": [
                {"entity_type": "dimension", "value": "3'-6\"", "confidence": 0.9}
            ]
        }
        source = [Chunk(text="test", page=2)]
        entities = ext._parse_entities(parsed, source, ["dimension"])
        assert len(entities) == 1
        assert entities[0].value == "3'-6\""

    def test_parse_entities_empty_value_skipped(self):
        from plancheck.llm.index import Chunk

        llm = MagicMock()
        idx = MagicMock()
        ext = EntityExtractor(idx, llm)

        parsed = [{"entity_type": "material", "value": "", "confidence": 0.5}]
        entities = ext._parse_entities(parsed, [Chunk(text="x")], ["material"])
        assert len(entities) == 0

    def test_deduplicate(self):
        llm = MagicMock()
        idx = MagicMock()
        ext = EntityExtractor(idx, llm)

        entities = [
            Entity("material", "A992 steel", page=1),
            Entity("material", "a992 steel", page=1),  # duplicate (case diff)
            Entity("material", "A992 steel", page=2),  # different page → keep
        ]
        unique = ext._deduplicate(entities)
        assert len(unique) == 2

    def test_extract_with_mocked_llm(self, _index_with_notes):
        """Full extract() flow with mocked LLM returning valid JSON."""
        from plancheck.llm.client import ChatMeta

        mock_llm = MagicMock()
        mock_meta = ChatMeta(provider="ollama", model="test")

        # LLM returns a list of entities
        mock_response = [
            {
                "entity_type": "material",
                "value": "A992 Grade 50 steel",
                "context": "structural steel",
                "confidence": 0.92,
            },
            {
                "entity_type": "material",
                "value": "4000 psi concrete",
                "context": "concrete spec",
                "confidence": 0.88,
            },
        ]
        mock_llm.chat_structured.return_value = (mock_response, mock_meta)

        ext = EntityExtractor(_index_with_notes, mock_llm, batch_size=5)
        result = ext.extract(["material"])

        assert mock_llm.chat_structured.called
        assert len(result.entities) > 0
        assert all(e.entity_type == "material" for e in result.entities)

    def test_extract_handles_llm_error(self, _index_with_notes):
        """Extract gracefully handles LLM failures."""
        mock_llm = MagicMock()
        mock_llm.chat_structured.side_effect = RuntimeError("LLM unavailable")

        ext = EntityExtractor(_index_with_notes, mock_llm, batch_size=5)
        result = ext.extract(["material"])

        assert len(result.errors) > 0
        assert len(result.entities) == 0

    def test_extract_all_types(self, _index_with_notes):
        """Extract with default types (all three)."""
        from plancheck.llm.client import ChatMeta

        mock_llm = MagicMock()
        mock_meta = ChatMeta(provider="ollama", model="test")
        mock_llm.chat_structured.return_value = (
            [
                {"entity_type": "material", "value": "steel", "confidence": 0.9},
                {"entity_type": "dimension", "value": "3'-0\"", "confidence": 0.8},
                {"entity_type": "equipment", "value": "VAV box", "confidence": 0.7},
            ],
            mock_meta,
        )

        ext = EntityExtractor(_index_with_notes, mock_llm, batch_size=10)
        result = ext.extract()  # default = all types

        assert len(result.entities) >= 1
        types_found = {e.entity_type for e in result.entities}
        # At least one type should be present
        assert types_found & {"material", "dimension", "equipment"}

    def test_convenience_methods(self, _index_with_notes):
        """extract_materials/dimensions/equipment call extract with correct type."""
        from plancheck.llm.client import ChatMeta

        mock_llm = MagicMock()
        mock_meta = ChatMeta(provider="ollama", model="test")
        mock_llm.chat_structured.return_value = ([], mock_meta)

        ext = EntityExtractor(_index_with_notes, mock_llm)

        ext.extract_materials()
        ext.extract_dimensions()
        ext.extract_equipment()

        assert mock_llm.chat_structured.call_count >= 3

    def test_cost_summary(self, _index_with_notes):
        mock_llm = MagicMock()
        mock_llm.cost_tracker = MagicMock()
        mock_llm.cost_tracker.summary.return_value = {
            "call_count": 5,
            "total_input_tokens": 1000,
            "total_output_tokens": 200,
            "total_cost_usd": 0.01,
        }
        ext = EntityExtractor(_index_with_notes, mock_llm)
        s = ext.cost_summary
        assert s["call_count"] == 5


# ═══════════════════════════════════════════════════════════════════
# Prompt coverage
# ═══════════════════════════════════════════════════════════════════


class TestPrompts:
    def test_system_prompt_has_entity_types(self):
        for t in ENTITY_TYPES:
            assert t in _SYSTEM_PROMPT

    def test_user_template_placeholders(self):
        filled = _USER_TEMPLATE.format(
            entity_types="material",
            page=1,
            region_type="notes",
            text="sample text",
        )
        assert "material" in filled
        assert "sample text" in filled
