"""Tests for Phase 3.2 — LLM-Assisted Semantic Checks.

Covers:
- Graceful degradation when LLM libraries are not installed
- LLMClient provider routing
- Response parsing (JSON, markdown fences, invalid JSON)
- run_llm_checks() integration (mocked LLM calls)
- Config fields (enable_llm_checks, llm_provider, etc.)
- CheckResult output format

Note: Tests mock LLM client libraries to avoid needing API keys in CI.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from plancheck.checks.llm_checks import (
    LLMClient,
    _parse_llm_response,
    is_llm_available,
    run_llm_checks,
)
from plancheck.checks.semantic_checks import CheckResult
from plancheck.config import GroupingConfig

# ── Config fields ──────────────────────────────────────────────────────


class TestLLMConfig:
    """Config has all LLM-related fields."""

    def test_defaults(self):
        cfg = GroupingConfig()
        assert cfg.enable_llm_checks is False
        assert cfg.llm_provider == "ollama"
        assert cfg.llm_model == "llama3.1:8b"
        assert cfg.llm_api_key == ""
        assert cfg.llm_api_base == "http://localhost:11434"
        assert cfg.llm_temperature == 0.1

    def test_temperature_validation(self):
        """Temperature must be in [0, 2]."""
        with pytest.raises(ValueError):
            GroupingConfig(llm_temperature=-0.1)
        with pytest.raises(ValueError):
            GroupingConfig(llm_temperature=2.1)

    def test_valid_temperature(self):
        cfg = GroupingConfig(llm_temperature=0.0)
        assert cfg.llm_temperature == 0.0
        cfg = GroupingConfig(llm_temperature=2.0)
        assert cfg.llm_temperature == 2.0


# ── Availability probe ────────────────────────────────────────────────


class TestAvailability:
    """is_llm_available() reflects import status."""

    def test_returns_bool(self):
        for provider in ("ollama", "openai", "anthropic"):
            result = is_llm_available(provider)
            assert isinstance(result, bool)

    def test_unknown_provider(self):
        assert is_llm_available("unknown_provider") is False


# ── Response parsing ──────────────────────────────────────────────────


class TestParseLLMResponse:
    """_parse_llm_response handles various LLM output formats."""

    def test_valid_json_array(self):
        response = json.dumps(
            [
                {
                    "issue_type": "vague",
                    "severity": "warning",
                    "message": "Note is too vague",
                    "note_text": "per local codes",
                }
            ]
        )
        findings = _parse_llm_response(response)
        assert len(findings) == 1
        assert findings[0]["issue_type"] == "vague"
        assert findings[0]["severity"] == "warning"

    def test_empty_array(self):
        findings = _parse_llm_response("[]")
        assert findings == []

    def test_markdown_fenced_json(self):
        response = '```json\n[{"issue_type": "contradiction", "severity": "info", "message": "Two notes conflict", "note_text": "test"}]\n```'
        findings = _parse_llm_response(response)
        assert len(findings) == 1
        assert findings[0]["issue_type"] == "contradiction"

    def test_markdown_fenced_plain(self):
        response = '```\n[{"issue_type": "missing_ref", "severity": "warning", "message": "Missing ref", "note_text": "see detail 5"}]\n```'
        findings = _parse_llm_response(response)
        assert len(findings) == 1

    def test_invalid_json_returns_empty(self):
        findings = _parse_llm_response("This is not JSON at all")
        assert findings == []

    def test_severity_clamped_to_info_or_warning(self):
        """LLM findings should never be 'error' — severity gets clamped."""
        response = json.dumps(
            [
                {
                    "issue_type": "vague",
                    "severity": "error",
                    "message": "Something",
                    "note_text": "x",
                }
            ]
        )
        findings = _parse_llm_response(response)
        assert findings[0]["severity"] == "info"

    def test_unknown_issue_type_falls_back(self):
        response = json.dumps(
            [
                {
                    "issue_type": "unknown_type",
                    "severity": "warning",
                    "message": "Something",
                    "note_text": "x",
                }
            ]
        )
        findings = _parse_llm_response(response)
        assert findings[0]["issue_type"] == "vague"  # fallback

    def test_missing_message_skipped(self):
        response = json.dumps(
            [
                {
                    "issue_type": "vague",
                    "severity": "info",
                    "message": "",
                    "note_text": "x",
                }
            ]
        )
        findings = _parse_llm_response(response)
        assert findings == []

    def test_single_object_wrapped_in_list(self):
        response = json.dumps(
            {
                "issue_type": "incomplete",
                "severity": "info",
                "message": "Truncated",
                "note_text": "x",
            }
        )
        findings = _parse_llm_response(response)
        assert len(findings) == 1

    def test_embedded_json_extraction(self):
        response = 'Here are my findings:\n[{"issue_type": "vague", "severity": "warning", "message": "Vague note", "note_text": "per applicable codes"}]\nHope this helps!'
        findings = _parse_llm_response(response)
        assert len(findings) == 1


# ── LLMClient ─────────────────────────────────────────────────────────


class TestLLMClient:
    """LLMClient routes to the correct provider."""

    def test_unknown_provider_raises(self):
        client = LLMClient(provider="nonexistent")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            client.chat("system", "user")

    def test_ollama_not_installed_raises(self):
        with patch("plancheck.llm.client._OLLAMA_AVAILABLE", False):
            client = LLMClient(provider="ollama")
            with pytest.raises(RuntimeError, match="ollama package not installed"):
                client.chat("system", "user")

    def test_openai_not_installed_raises(self):
        with patch("plancheck.llm.client._OPENAI_AVAILABLE", False):
            client = LLMClient(provider="openai", policy="cloud_allowed")
            with pytest.raises(RuntimeError, match="openai package not installed"):
                client.chat("system", "user")

    def test_anthropic_not_installed_raises(self):
        with patch("plancheck.llm.client._ANTHROPIC_AVAILABLE", False):
            client = LLMClient(provider="anthropic", policy="cloud_allowed")
            with pytest.raises(RuntimeError, match="anthropic package not installed"):
                client.chat("system", "user")

    def test_local_only_blocks_cloud_openai(self):
        client = LLMClient(provider="openai", policy="local_only")
        with pytest.raises(RuntimeError, match="llm_policy is 'local_only'"):
            client.chat("system", "user")

    def test_local_only_blocks_cloud_anthropic(self):
        client = LLMClient(provider="anthropic", policy="local_only")
        with pytest.raises(RuntimeError, match="llm_policy is 'local_only'"):
            client.chat("system", "user")

    def test_local_only_allows_ollama(self):
        """local_only should not block ollama — it's a local provider."""
        with patch("plancheck.llm.client._OLLAMA_AVAILABLE", False):
            client = LLMClient(provider="ollama", policy="local_only")
            # Should fail with 'not installed', not with policy error
            with pytest.raises(RuntimeError, match="ollama package not installed"):
                client.chat("system", "user")

    def test_cloud_allowed_permits_openai(self):
        with patch("plancheck.llm.client._OPENAI_AVAILABLE", False):
            client = LLMClient(provider="openai", policy="cloud_allowed")
            # Should fail with 'not installed', not with policy error
            with pytest.raises(RuntimeError, match="openai package not installed"):
                client.chat("system", "user")


# ── run_llm_checks ────────────────────────────────────────────────────


class TestRunLLMChecks:
    """Integration tests for run_llm_checks (with mocked LLM calls)."""

    def test_empty_notes_returns_empty(self):
        results = run_llm_checks(notes_columns=None, provider="ollama")
        assert results == []

    def test_no_provider_available_returns_empty(self):
        with patch("plancheck.llm.client._OLLAMA_AVAILABLE", False):
            notes = MagicMock()
            notes.text = "Some construction note"
            results = run_llm_checks(notes_columns=[notes], provider="ollama")
            assert results == []

    def test_successful_check_returns_check_results(self):
        """Mock a successful LLM call and verify CheckResult output."""
        mock_response = json.dumps(
            [
                {
                    "issue_type": "vague",
                    "severity": "warning",
                    "message": "Note 3 says 'per applicable codes' without specifying which codes",
                    "note_text": "per applicable codes",
                },
                {
                    "issue_type": "missing_ref",
                    "severity": "info",
                    "message": "Reference to Detail 5/A-101 not found",
                    "note_text": "See Detail 5/A-101",
                },
            ]
        )

        mock_client = MagicMock()
        mock_client.chat.return_value = mock_response

        notes = MagicMock()
        notes.text = (
            "1. All work per applicable codes.\n2. See Detail 5/A-101 for dimensions."
        )
        notes.full_text = MagicMock(return_value=notes.text)

        with (
            patch("plancheck.checks.llm_checks.LLMClient", return_value=mock_client),
            patch("plancheck.llm.client._OLLAMA_AVAILABLE", True),
        ):
            results = run_llm_checks(
                notes_columns=[notes],
                page=1,
                provider="ollama",
            )

        assert len(results) == 2
        assert all(isinstance(r, CheckResult) for r in results)
        assert results[0].check_id == "LLM_VAGUE_1"
        assert results[0].severity == "warning"
        assert results[0].page == 1
        assert results[1].check_id == "LLM_MISSING_REF_2"

    def test_llm_error_returns_error_result(self):
        """When LLM call raises, return an info-level error CheckResult."""
        mock_client = MagicMock()
        mock_client.chat.side_effect = RuntimeError("Connection refused")

        notes = MagicMock()
        notes.text = "Some note"
        notes.full_text = MagicMock(return_value=notes.text)

        with (
            patch("plancheck.checks.llm_checks.LLMClient", return_value=mock_client),
            patch("plancheck.llm.client._OLLAMA_AVAILABLE", True),
        ):
            results = run_llm_checks(
                notes_columns=[notes],
                provider="ollama",
            )

        assert len(results) == 1
        assert results[0].check_id == "LLM_ERROR"
        assert "Connection refused" in results[0].message

    def test_check_result_has_details(self):
        """CheckResult.details should contain LLM metadata."""
        mock_response = json.dumps(
            [
                {
                    "issue_type": "code_compliance",
                    "severity": "warning",
                    "message": "May not comply with IBC 2021",
                    "note_text": "test note",
                }
            ]
        )

        mock_client = MagicMock()
        mock_client.chat.return_value = mock_response

        notes = MagicMock()
        notes.text = "test note"
        notes.full_text = MagicMock(return_value=notes.text)

        with (
            patch("plancheck.checks.llm_checks.LLMClient", return_value=mock_client),
            patch("plancheck.llm.client._OLLAMA_AVAILABLE", True),
        ):
            results = run_llm_checks(
                notes_columns=[notes],
                provider="ollama",
                model="llama3.1:8b",
            )

        assert results[0].details["llm_provider"] == "ollama"
        assert results[0].details["llm_model"] == "llama3.1:8b"
        assert results[0].details["issue_type"] == "code_compliance"
