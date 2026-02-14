"""Tests for plancheck.ocr_reconcile — symbol detection and injection helpers."""

import pytest
from conftest import make_box

from plancheck.config import GroupingConfig
from plancheck.ocr_reconcile import (
    _extra_symbols,
    _find_line_neighbours,
    _has_allowed_symbol,
    _has_digit_neighbour_left,
    _has_numeric_symbol_context,
    _is_digit_group,
)


class TestHasAllowedSymbol:
    def test_slash_present(self):
        assert _has_allowed_symbol("09/15/25", "%/°±") is True

    def test_percent_present(self):
        assert _has_allowed_symbol("85%", "%/°±") is True

    def test_no_symbols(self):
        assert _has_allowed_symbol("HELLO", "%/°±") is False

    def test_empty_text(self):
        assert _has_allowed_symbol("", "%/°±") is False


class TestHasNumericSymbolContext:
    def test_slash_between_digits(self):
        assert _has_numeric_symbol_context("09/15/25", "%/°±") is True

    def test_slash_in_words(self):
        # "SURFACING/MILLINGS" — no digit context
        assert _has_numeric_symbol_context("SURFACING/MILLINGS", "%/°±") is False

    def test_percent_after_digit(self):
        assert _has_numeric_symbol_context("85%", "%/°±") is True

    def test_percent_no_digit(self):
        assert _has_numeric_symbol_context("ABC%", "%/°±") is False

    def test_degree_after_digit(self):
        assert _has_numeric_symbol_context("45°", "%/°±") is True

    def test_no_allowed_symbols(self):
        assert _has_numeric_symbol_context("HELLO", "%/°±") is False

    def test_partial_date(self):
        assert _has_numeric_symbol_context("1/2", "%/°±") is True


class TestExtraSymbols:
    def test_slash_missing_from_pdf(self):
        result = _extra_symbols("09/15", "0915", "%/°±")
        assert "/" in result

    def test_no_extras(self):
        result = _extra_symbols("HELLO", "HELLO", "%/°±")
        assert result == ""

    def test_percent_missing(self):
        result = _extra_symbols("85%", "85", "%/°±")
        assert "%" in result


class TestIsDigitGroup:
    def test_starts_with_digit(self):
        assert _is_digit_group("09") is True
        assert _is_digit_group("8.33") is True

    def test_pure_text(self):
        assert _is_digit_group("SECTION") is False

    def test_mixed_needs_ratio(self):
        cfg = GroupingConfig()
        # "2A" starts with digit → True
        assert _is_digit_group("2A", cfg) is True
        # "AB2" does not start with digit; need ratio check
        # 1 digit out of 3 = 33% < 50% default ratio
        assert _is_digit_group("AB2", cfg) is False

    def test_empty_string(self):
        assert _is_digit_group("") is False


class TestHasDigitNeighbourLeft:
    def test_digit_to_left(self):
        digit_token = make_box(10, 100, 30, 112, "85", origin="text")
        candidate = make_box(32, 100, 40, 112, "%")
        cfg = GroupingConfig()
        result = _has_digit_neighbour_left(
            candidate, [digit_token], proximity_pts=10.0, cfg=cfg
        )
        assert result is True

    def test_no_digit_nearby(self):
        text_token = make_box(10, 100, 30, 112, "ABC", origin="text")
        candidate = make_box(32, 100, 40, 112, "%")
        cfg = GroupingConfig()
        result = _has_digit_neighbour_left(
            candidate, [text_token], proximity_pts=10.0, cfg=cfg
        )
        assert result is False

    def test_digit_too_far(self):
        digit_token = make_box(10, 100, 30, 112, "85", origin="text")
        candidate = make_box(100, 100, 110, 112, "%")
        cfg = GroupingConfig()
        result = _has_digit_neighbour_left(
            candidate, [digit_token], proximity_pts=10.0, cfg=cfg
        )
        assert result is False


class TestFindLineNeighbours:
    def test_finds_same_line(self):
        tokens = [
            make_box(10, 100, 30, 112, "85", origin="text"),
            make_box(40, 100, 80, 112, "TOTAL", origin="text"),
        ]
        ocr_box = make_box(32, 100, 38, 112, "%")
        cfg = GroupingConfig()
        result = _find_line_neighbours(ocr_box, tokens, anchor_margin=25.0, cfg=cfg)
        assert len(result) == 2

    def test_ignores_far_line(self):
        tokens = [
            make_box(10, 200, 30, 212, "99", origin="text"),
        ]
        ocr_box = make_box(10, 100, 20, 112, "%")
        cfg = GroupingConfig()
        result = _find_line_neighbours(ocr_box, tokens, anchor_margin=25.0, cfg=cfg)
        assert len(result) == 0
