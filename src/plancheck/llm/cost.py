"""Token counting and cost estimation utilities.

This module provides helpers to count tokens for the various LLM back-ends
and to estimate API costs.  It is used by :class:`~plancheck.llm.client.LLMClient`
to populate the ``metadata`` dict returned by :meth:`chat_with_metadata`, and by
the benchmark script (``scripts/diagnostics/run_llm_benchmark.py``).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing tables — USD per 1 K tokens (mid-2025)
# ---------------------------------------------------------------------------

#: Known model pricing.  Keys are ``"provider/model"`` strings.
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Local — free
    "ollama/llama3.1:8b": {"input": 0.0, "output": 0.0},
    "ollama/llama3:8b": {"input": 0.0, "output": 0.0},
    "ollama/mistral:7b": {"input": 0.0, "output": 0.0},
    # OpenAI
    "openai/gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "openai/gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "openai/gpt-4o": {"input": 0.005, "output": 0.015},
    # Anthropic
    "anthropic/claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    "anthropic/claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
}


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def count_tokens(
    text: str, provider: str = "openai", model: str = "gpt-4o-mini"
) -> int:
    """Estimate the number of tokens in *text*.

    Uses ``tiktoken`` when available (accurate for OpenAI models).
    Falls back to a character-based heuristic (~4 chars/token).
    """
    if provider == "openai":
        try:
            import tiktoken  # type: ignore[import-untyped]

            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            pass

    # Heuristic fallback
    return max(1, len(text) // 4)


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    provider: str,
    model: str,
) -> float:
    """Return estimated USD cost for a single LLM call."""
    key = f"{provider}/{model}"
    pricing = MODEL_PRICING.get(key)
    if pricing is None:
        log.debug("No pricing data for %s — returning $0", key)
        return 0.0
    return (input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing[
        "output"
    ]


# ---------------------------------------------------------------------------
# Cost accumulator (session-level)
# ---------------------------------------------------------------------------


@dataclass
class CostTracker:
    """Accumulates LLM usage statistics across multiple calls.

    Thread-safety note: this class is *not* thread-safe.  Wrap access in a
    lock if shared across threads.
    """

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    call_count: int = 0
    #: Per-call records for auditing / export.
    history: list[dict] = field(default_factory=list)

    def record(
        self,
        *,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Record a single LLM call and return its estimated cost."""
        cost = estimate_cost(input_tokens, output_tokens, provider, model)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost
        self.call_count += 1
        self.history.append(
            {
                "provider": provider,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
            }
        )
        return cost

    def summary(self) -> dict:
        """Return a summary dict suitable for JSON export."""
        return {
            "call_count": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
        }
