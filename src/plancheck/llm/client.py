"""Unified LLM client with policy enforcement, token tracking, and
structured-output support.

This module is the single entry-point for all LLM interactions in the
project.  It is consumed by:

* ``plancheck.checks.llm_checks`` (semantic checks)
* ``plancheck.llm.query_engine`` (Phase 1.1)
* ``plancheck.llm.entity_extraction`` (Phase 1.3)

The legacy ``LLMClient`` in ``checks/llm_checks.py`` is now a thin
re-export of this class for backwards compatibility.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from plancheck.llm.cost import CostTracker, count_tokens, estimate_cost

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Availability probes (lazy — evaluated once on first import)
# ---------------------------------------------------------------------------

_OLLAMA_AVAILABLE = False
_OPENAI_AVAILABLE = False
_ANTHROPIC_AVAILABLE = False

try:
    import ollama as _ollama_mod  # noqa: F401

    _OLLAMA_AVAILABLE = True
except ImportError:
    pass

try:
    import openai as _openai_mod  # noqa: F401

    _OPENAI_AVAILABLE = True
except ImportError:
    pass

try:
    import anthropic as _anthropic_mod  # noqa: F401

    _ANTHROPIC_AVAILABLE = True
except ImportError:
    pass


def is_llm_available(provider: str = "ollama") -> bool:
    """Return True if the given provider's client library is installed."""
    return {
        "ollama": _OLLAMA_AVAILABLE,
        "openai": _OPENAI_AVAILABLE,
        "anthropic": _ANTHROPIC_AVAILABLE,
    }.get(provider, False)


# ---------------------------------------------------------------------------
# Chat response metadata
# ---------------------------------------------------------------------------


@dataclass
class ChatMeta:
    """Metadata returned alongside LLM responses."""

    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    cached: bool = False

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "latency_s": round(self.latency_s, 3),
            "cached": self.cached,
        }


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


class LLMClient:
    """Unified LLM client with policy enforcement and cost tracking.

    Parameters
    ----------
    provider : str
        ``"ollama"``, ``"openai"``, or ``"anthropic"``.
    model : str
        Model name (e.g. ``"llama3.1:8b"``, ``"gpt-4o-mini"``).
    api_key : str
        API key (not needed for ollama).
    api_base : str
        Base URL (for Ollama; default ``http://localhost:11434``).
    temperature : float
        LLM temperature (0.0–2.0).
    policy : str
        Data-privacy policy — ``"local_only"`` (default), ``"cloud_allowed"``,
        or ``"cloud_with_consent"``.
    cost_tracker : CostTracker | None
        Optional shared tracker to accumulate cost across calls.
    max_retries : int
        Maximum number of attempts for transient failures (default 3).
        Uses exponential backoff (1s, 2s, 4s, …).
    """

    _CLOUD_PROVIDERS = {"openai", "anthropic"}

    #: Cached provider client instances (class-level, shared across instances
    #: with the same config).  Keyed by ``(provider, api_key, api_base)``.
    _client_cache: dict[tuple[str, str, str], Any] = {}

    #: Lock for thread-safe cache access
    _cache_lock: "threading.Lock | None" = None

    @classmethod
    def _get_cache_lock(cls) -> "threading.Lock":
        """Lazily create and return the cache lock."""
        if cls._cache_lock is None:
            import threading

            cls._cache_lock = threading.Lock()
        return cls._cache_lock

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "llama3.1:8b",
        api_key: str = "",
        api_base: str = "http://localhost:11434",
        temperature: float = 0.1,
        policy: str = "local_only",
        cost_tracker: CostTracker | None = None,
        max_retries: int = 3,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.policy = policy
        self.cost_tracker = cost_tracker or CostTracker()
        self.max_retries = max(1, max_retries)

    # ── Resource management ────────────────────────────────────────

    def close(self) -> None:
        """Close this client's cached SDK connection.

        Removes the SDK client from the cache and calls its close() method
        if available. Safe to call multiple times.
        """
        cache_key = (self.provider, self.api_key, self.api_base)
        with self._get_cache_lock():
            client = self._client_cache.pop(cache_key, None)
        if client is not None and hasattr(client, "close"):
            try:
                client.close()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass

    @classmethod
    def close_all(cls) -> None:
        """Close all cached SDK connections.

        Call this at program shutdown to release HTTP connections held
        by ollama, openai, and anthropic SDK clients.
        """
        with cls._get_cache_lock():
            clients = list(cls._client_cache.values())
            cls._client_cache.clear()
        for client in clients:
            if hasattr(client, "close"):
                try:
                    client.close()
                except Exception:  # noqa: BLE001 — best-effort cleanup
                    pass

    def __enter__(self) -> "LLMClient":
        """Context manager entry — returns self."""
        return self

    def __exit__(self, *exc: Any) -> None:
        """Context manager exit — closes this client's cached connection."""
        self.close()

    # ── Policy helpers ─────────────────────────────────────────────

    def _is_cloud(self) -> bool:
        return self.provider in self._CLOUD_PROVIDERS

    def _enforce_policy(self) -> None:
        """Raise if the current provider violates the data-privacy policy."""
        if not self._is_cloud():
            return
        if self.policy == "local_only":
            raise RuntimeError(
                f"LLM provider {self.provider!r} requires cloud access, "
                f"but llm_policy is 'local_only'.  Change the policy to "
                f"'cloud_allowed' or 'cloud_with_consent' in your config."
            )

    # ── Public API ─────────────────────────────────────────────────

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat message and return the assistant's response text.

        Raises
        ------
        RuntimeError
            If the provider library is not installed, the API call fails,
            or the data-privacy policy blocks the call.
        """
        text, _meta = self.chat_with_metadata(system_prompt, user_prompt)
        return text

    def chat_with_metadata(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[str, ChatMeta]:
        """Like :meth:`chat` but also returns a :class:`ChatMeta` object
        with token counts, cost, and latency."""
        self._enforce_policy()

        full_input = system_prompt + "\n" + user_prompt
        input_tokens = count_tokens(full_input, self.provider, self.model)

        t0 = time.perf_counter()
        text = self._dispatch(system_prompt, user_prompt)
        latency = time.perf_counter() - t0

        output_tokens = count_tokens(text, self.provider, self.model)
        cost = estimate_cost(input_tokens, output_tokens, self.provider, self.model)
        self.cost_tracker.record(
            provider=self.provider,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        meta = ChatMeta(
            provider=self.provider,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_s=latency,
        )
        return text, meta

    def chat_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        expect_json: bool = True,
    ) -> tuple[Any, ChatMeta]:
        """Send a chat and attempt to parse the response as JSON.

        Returns ``(parsed_object, meta)`` where ``parsed_object`` is the
        decoded JSON (usually a ``dict`` or ``list``).  If parsing fails and
        *expect_json* is True, raises ``ValueError``.
        """
        text, meta = self.chat_with_metadata(system_prompt, user_prompt)

        # Strip markdown code fences
        cleaned = text.strip()
        if cleaned.startswith("```"):
            first_nl = cleaned.index("\n") if "\n" in cleaned else 3
            cleaned = cleaned[first_nl + 1 :]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            if expect_json:
                raise ValueError(
                    f"LLM response was not valid JSON: {exc}\n"
                    f"Response text (first 200 chars): {text[:200]}"
                ) from exc
            parsed = text  # Return raw text if JSON not required

        return parsed, meta

    # ── Provider dispatch ──────────────────────────────────────────

    def _dispatch(self, system_prompt: str, user_prompt: str) -> str:
        """Dispatch to the provider with retry + exponential backoff."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                if self.provider == "ollama":
                    return self._chat_ollama(system_prompt, user_prompt)
                elif self.provider == "openai":
                    return self._chat_openai(system_prompt, user_prompt)
                elif self.provider == "anthropic":
                    return self._chat_anthropic(system_prompt, user_prompt)
                else:
                    raise ValueError(f"Unknown LLM provider: {self.provider!r}")
            except (ValueError, RuntimeError):
                # Non-retryable errors (missing lib, bad provider, policy)
                raise
            except Exception as exc:
                last_exc = exc
                delay = 2**attempt  # 1s, 2s, 4s …
                log.warning(
                    "LLM call attempt %d/%d failed (%s), retrying in %ds…",
                    attempt + 1,
                    self.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise RuntimeError(
            f"LLM call failed after {self.max_retries} attempts: {last_exc}"
        ) from last_exc

    # ── Client cache helpers ───────────────────────────────────────

    def _get_or_create_client(self, provider: str) -> Any:
        """Return a cached SDK client, creating one if needed (thread-safe)."""
        cache_key = (provider, self.api_key, self.api_base)

        # Fast path: check cache without lock
        client = self._client_cache.get(cache_key)
        if client is not None:
            return client

        # Slow path: acquire lock and create client
        with self._get_cache_lock():
            # Double-check after acquiring lock
            client = self._client_cache.get(cache_key)
            if client is not None:
                return client

            if provider == "ollama":
                import ollama

                client = ollama.Client(host=self.api_base)
            elif provider == "openai":
                import openai

                client = openai.OpenAI(
                    api_key=self.api_key or None,
                    base_url=(
                        self.api_base
                        if self.api_base != "http://localhost:11434"
                        else None
                    ),
                )
            elif provider == "anthropic":
                import anthropic

                client = anthropic.Anthropic(api_key=self.api_key or None)
            else:
                raise ValueError(f"Unknown provider: {provider!r}")

            self._client_cache[cache_key] = client
            return client

    # ── Provider implementations ───────────────────────────────────

    def _chat_ollama(self, system_prompt: str, user_prompt: str) -> str:
        if not _OLLAMA_AVAILABLE:
            raise RuntimeError("ollama package not installed")

        client = self._get_or_create_client("ollama")
        response = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": self.temperature},
        )
        return response["message"]["content"]

    def _chat_openai(self, system_prompt: str, user_prompt: str) -> str:
        if not _OPENAI_AVAILABLE:
            raise RuntimeError("openai package not installed")

        client = self._get_or_create_client("openai")
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    def _chat_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        if not _ANTHROPIC_AVAILABLE:
            raise RuntimeError("anthropic package not installed")

        client = self._get_or_create_client("anthropic")
        response = client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=self.temperature,
        )
        return "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
