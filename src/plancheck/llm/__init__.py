"""LLM integration layer for Advanced Plan Parser.

This package provides:

* :class:`~plancheck.llm.client.LLMClient` — unified interface to Ollama,
  OpenAI, and Anthropic with policy enforcement, token tracking, and
  structured-output helpers.
* :mod:`~plancheck.llm.cost` — token counting and cost estimation.

All Phase 1 components (query engine, compliance assistant, entity extraction)
import the LLM client from this package rather than from
``plancheck.checks.llm_checks``.
"""

from plancheck.llm.client import LLMClient, is_llm_available
from plancheck.llm.entity_extraction import Entity, EntityExtractor, ExtractionResult

__all__ = [
    "LLMClient",
    "is_llm_available",
    "EntityExtractor",
    "ExtractionResult",
    "Entity",
]
