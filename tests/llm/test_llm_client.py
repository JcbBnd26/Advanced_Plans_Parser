"""Tests for the plancheck.llm package — client, cost tracker, and structured output."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from plancheck.llm.client import ChatMeta, LLMClient, is_llm_available
from plancheck.llm.cost import CostTracker, count_tokens, estimate_cost


# ── is_llm_available ──────────────────────────────────────────────────


class TestIsLLMAvailable:
    def test_returns_bool(self):
        for p in ("ollama", "openai", "anthropic"):
            assert isinstance(is_llm_available(p), bool)

    def test_unknown(self):
        assert is_llm_available("xyz") is False


# ── CostTracker ───────────────────────────────────────────────────────


class TestCostTracker:
    def test_empty(self):
        t = CostTracker()
        assert t.call_count == 0
        assert t.total_cost_usd == 0.0

    def test_record(self):
        t = CostTracker()
        cost = t.record(
            provider="openai", model="gpt-4o-mini",
            input_tokens=1000, output_tokens=500,
        )
        assert cost > 0
        assert t.call_count == 1
        assert t.total_input_tokens == 1000
        assert t.total_output_tokens == 500
        assert len(t.history) == 1

    def test_summary(self):
        t = CostTracker()
        t.record(provider="ollama", model="llama3.1:8b",
                 input_tokens=200, output_tokens=100)
        s = t.summary()
        assert s["call_count"] == 1
        assert s["total_cost_usd"] == 0.0  # local is free


# ── count_tokens ──────────────────────────────────────────────────────


class TestCountTokens:
    def test_heuristic(self):
        text = "A" * 400
        tokens = count_tokens(text, provider="ollama", model="llama3.1:8b")
        assert tokens == 100  # 400/4

    def test_empty(self):
        assert count_tokens("", provider="ollama") >= 1  # min 1


# ── estimate_cost ─────────────────────────────────────────────────────


class TestEstimateCost:
    def test_local_is_free(self):
        assert estimate_cost(5000, 1000, "ollama", "llama3.1:8b") == 0.0

    def test_openai_costs(self):
        cost = estimate_cost(1000, 500, "openai", "gpt-4o-mini")
        assert cost > 0

    def test_unknown_model(self):
        assert estimate_cost(1000, 500, "foo", "bar") == 0.0


# ── ChatMeta ──────────────────────────────────────────────────────────


class TestChatMeta:
    def test_to_dict(self):
        m = ChatMeta(
            provider="ollama", model="llama3.1:8b",
            input_tokens=100, output_tokens=50,
            cost_usd=0.0, latency_s=1.234,
        )
        d = m.to_dict()
        assert d["provider"] == "ollama"
        assert d["latency_s"] == 1.234
        assert d["cached"] is False


# ── LLMClient policy ─────────────────────────────────────────────────


class TestLLMClientPolicy:
    def test_local_only_blocks_openai(self):
        c = LLMClient(provider="openai", policy="local_only")
        with pytest.raises(RuntimeError, match="local_only"):
            c.chat("s", "u")

    def test_local_only_allows_ollama(self):
        with patch("plancheck.llm.client._OLLAMA_AVAILABLE", False):
            c = LLMClient(provider="ollama", policy="local_only")
            with pytest.raises(RuntimeError, match="ollama package not installed"):
                c.chat("s", "u")

    def test_cloud_allowed(self):
        with patch("plancheck.llm.client._OPENAI_AVAILABLE", False):
            c = LLMClient(provider="openai", policy="cloud_allowed")
            with pytest.raises(RuntimeError, match="openai package not installed"):
                c.chat("s", "u")


# ── LLMClient.chat_with_metadata ─────────────────────────────────────


class TestChatWithMetadata:
    def test_returns_meta(self):
        with patch("plancheck.llm.client._OLLAMA_AVAILABLE", True):
            c = LLMClient(provider="ollama")
            with patch.object(c, "_chat_ollama", return_value="hello"):
                text, meta = c.chat_with_metadata("system", "user")
                assert text == "hello"
                assert isinstance(meta, ChatMeta)
                assert meta.provider == "ollama"
                assert meta.input_tokens > 0
                assert meta.latency_s >= 0

    def test_cost_tracker_accumulates(self):
        tracker = CostTracker()
        with patch("plancheck.llm.client._OLLAMA_AVAILABLE", True):
            c = LLMClient(provider="ollama", cost_tracker=tracker)
            with patch.object(c, "_chat_ollama", return_value="result"):
                c.chat("s", "u")
                c.chat("s", "u")
        assert tracker.call_count == 2


# ── LLMClient.chat_structured ────────────────────────────────────────


class TestChatStructured:
    def test_parses_json(self):
        payload = [{"key": "value"}]
        with patch("plancheck.llm.client._OLLAMA_AVAILABLE", True):
            c = LLMClient(provider="ollama")
            with patch.object(c, "_chat_ollama", return_value=json.dumps(payload)):
                parsed, meta = c.chat_structured("sys", "usr")
                assert parsed == payload

    def test_strips_markdown_fence(self):
        payload = {"answer": 42}
        fenced = "```json\n" + json.dumps(payload) + "\n```"
        with patch("plancheck.llm.client._OLLAMA_AVAILABLE", True):
            c = LLMClient(provider="ollama")
            with patch.object(c, "_chat_ollama", return_value=fenced):
                parsed, _ = c.chat_structured("sys", "usr")
                assert parsed == payload

    def test_invalid_json_raises(self):
        with patch("plancheck.llm.client._OLLAMA_AVAILABLE", True):
            c = LLMClient(provider="ollama")
            with patch.object(c, "_chat_ollama", return_value="not json"):
                with pytest.raises(ValueError, match="not valid JSON"):
                    c.chat_structured("sys", "usr")

    def test_invalid_json_returns_text_if_not_required(self):
        with patch("plancheck.llm.client._OLLAMA_AVAILABLE", True):
            c = LLMClient(provider="ollama")
            with patch.object(c, "_chat_ollama", return_value="plain text"):
                parsed, _ = c.chat_structured("sys", "usr", expect_json=False)
                assert parsed == "plain text"
