"""Tests for the query eval harness scoring functions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure scripts dir is importable
_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "scripts" / "diagnostics"))

from run_query_eval import compute_summary, keyword_score  # noqa: E402

# ── keyword_score ─────────────────────────────────────────────


class TestKeywordScore:
    def test_all_keywords_found(self):
        assert (
            keyword_score("The steel grade is A992 per ASTM.", ["steel", "A992"]) == 1.0
        )

    def test_no_keywords_found(self):
        assert keyword_score("Nothing relevant here.", ["steel", "concrete"]) == 0.0

    def test_partial_keywords(self):
        assert keyword_score(
            "The concrete is 4000 PSI.", ["concrete", "psi", "rebar"]
        ) == pytest.approx(2 / 3)

    def test_empty_keywords(self):
        """Empty keyword list should auto-pass."""
        assert keyword_score("Any text", []) == 1.0

    def test_case_insensitive(self):
        assert keyword_score("FIRE RATED assembly", ["fire", "rat"]) == 1.0

    def test_partial_match_via_regex(self):
        """Keywords like 'revis' should match 'revision'."""
        assert keyword_score("Revision 3 dated 2024-01-15", ["revis", "date"]) == 1.0


# ── compute_summary ───────────────────────────────────────────


class TestComputeSummary:
    def _make_results(self, n: int = 5) -> list[dict]:
        return [
            {
                "id": f"q{i:02d}",
                "question": f"Question {i}",
                "topic": "notes" if i % 2 == 0 else "materials",
                "difficulty": ["easy", "medium", "hard"][i % 3],
                "answer": f"Answer {i}",
                "answer_length": 50,
                "keyword_score": 0.2 * i,
                "has_content": True,
                "num_sources": 2,
                "latency_s": 1.0 + i * 0.5,
                "cached": False,
                "error": None,
            }
            for i in range(1, n + 1)
        ]

    def test_total_count(self):
        results = self._make_results(5)
        s = compute_summary(results)
        assert s["total_questions"] == 5

    def test_no_errors(self):
        results = self._make_results(3)
        assert compute_summary(results)["errors"] == 0

    def test_has_difficulty_breakdown(self):
        results = self._make_results(6)
        s = compute_summary(results)
        assert "easy_count" in s
        assert "medium_count" in s
        assert "hard_count" in s

    def test_empty_results(self):
        assert compute_summary([]) == {}

    def test_avg_keyword_score(self):
        results = self._make_results(5)
        s = compute_summary(results)
        expected = (0.2 + 0.4 + 0.6 + 0.8 + 1.0) / 5
        assert s["avg_keyword_score"] == pytest.approx(expected, abs=0.01)


# ── question_bank.json validity ───────────────────────────────


class TestQuestionBank:
    @pytest.fixture
    def bank(self):
        path = _root / "tests" / "query" / "question_bank.json"
        return json.loads(path.read_text())

    def test_bank_has_questions(self, bank):
        assert len(bank["questions"]) == 50

    def test_all_questions_have_required_fields(self, bank):
        for q in bank["questions"]:
            assert "id" in q
            assert "question" in q
            assert "expected_answer" in q
            assert "required_keywords" in q
            assert "difficulty" in q

    def test_ids_are_unique(self, bank):
        ids = [q["id"] for q in bank["questions"]]
        assert len(ids) == len(set(ids))

    def test_difficulties_are_valid(self, bank):
        valid = {"easy", "medium", "hard"}
        for q in bank["questions"]:
            assert q["difficulty"] in valid, f"{q['id']} has invalid difficulty"
