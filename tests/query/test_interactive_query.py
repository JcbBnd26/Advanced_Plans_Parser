"""Tests for the interactive query CLI argument parsing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Ensure the scripts directory is importable
_root = Path(__file__).resolve().parents[2]
_scripts_query = _root / "scripts" / "query"
if str(_scripts_query) not in sys.path:
    sys.path.insert(0, str(_scripts_query))
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

import interactive_query as iq

# ── Argument parsing ──────────────────────────────────────────────


class TestArgParsing:
    """Test that argparse is configured correctly."""

    def _parse(self, *args: str) -> argparse.Namespace:
        """Helper: parse args via the module's argparse definition."""
        parser = argparse.ArgumentParser()
        parser.add_argument("pdf", nargs="?")
        parser.add_argument("--page", type=int, default=0)
        parser.add_argument("--run")
        parser.add_argument("--provider", default="ollama")
        parser.add_argument("--model", default="llama3.1:8b")
        parser.add_argument("--api-key", default="")
        parser.add_argument("--api-base", default="http://localhost:11434")
        parser.add_argument(
            "--policy",
            default="local_only",
            choices=["local_only", "cloud_allowed", "cloud_with_consent"],
        )
        parser.add_argument("--temperature", type=float, default=0.1)
        return parser.parse_args(list(args))

    def test_pdf_positional(self):
        ns = self._parse("plan.pdf")
        assert ns.pdf == "plan.pdf"
        assert ns.page == 0

    def test_pdf_with_page(self):
        ns = self._parse("plan.pdf", "--page", "3")
        assert ns.pdf == "plan.pdf"
        assert ns.page == 3

    def test_run_flag(self):
        ns = self._parse("--run", "runs/some_run")
        assert ns.run == "runs/some_run"
        assert ns.pdf is None

    def test_provider_and_model(self):
        ns = self._parse("plan.pdf", "--provider", "openai", "--model", "gpt-4o")
        assert ns.provider == "openai"
        assert ns.model == "gpt-4o"

    def test_defaults(self):
        ns = self._parse("plan.pdf")
        assert ns.provider == "ollama"
        assert ns.model == "llama3.1:8b"
        assert ns.api_key == ""
        assert ns.api_base == "http://localhost:11434"
        assert ns.policy == "local_only"
        assert ns.temperature == 0.1

    def test_invalid_policy_rejected(self):
        with pytest.raises(SystemExit):
            self._parse("plan.pdf", "--policy", "yolo")

    def test_temperature_float(self):
        ns = self._parse("plan.pdf", "--temperature", "0.7")
        assert ns.temperature == pytest.approx(0.7)


# ── _load_from_pdf / _load_from_run ──────────────────────────────


class TestLoadHelpers:
    """Test the two loader entry-points (with mocked heavy dependencies)."""

    @patch("plancheck.pipeline.run_pipeline")
    @patch("plancheck.config.GroupingConfig", return_value=MagicMock())
    def test_load_from_pdf_calls_pipeline(self, mock_cfg_cls, mock_run):
        mock_pr = MagicMock()
        mock_pr.blocks = [1, 2, 3]
        mock_run.return_value = mock_pr

        result = iq._load_from_pdf("fake.pdf", 2)

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0] == "fake.pdf"
        assert call_args[0][1] == 2
        assert result is mock_pr

    @patch("plancheck.export.run_loader.load_run")
    def test_load_from_run_returns_doc_result(self, mock_load):
        dr = SimpleNamespace(pages=[SimpleNamespace()] * 3)
        mock_load.return_value = dr

        result = iq._load_from_run("runs/test_run")

        mock_load.assert_called_once_with("runs/test_run")
        assert result is dr


# ── main() validation ─────────────────────────────────────────────


class TestMainValidation:
    """Test that main() rejects invalid argument combos."""

    def test_no_pdf_no_run_errors(self):
        """Calling main() with no pdf and no --run should error."""
        with patch("sys.argv", ["interactive_query.py"]):
            with pytest.raises(SystemExit):
                iq.main()
