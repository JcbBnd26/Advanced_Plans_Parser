"""Tests for plancheck.ocr_reconcile — symbol detection and injection helpers."""

import pytest
from conftest import make_box

from plancheck.config import GroupingConfig
from plancheck.reconcile.reconcile import (
    SymbolCandidate,
    _accept_candidates,
    _build_match_index,
    _center,
    _estimate_char_width,
    _extra_symbols,
    _find_line_neighbours,
    _generate_symbol_candidates,
    _has_allowed_symbol,
    _has_digit_neighbour_left,
    _has_numeric_symbol_context,
    _is_digit_group,
    _overlap_ratio,
    _overlaps_existing,
)
from plancheck.vocr.extract import _dedup_tiles, _iou


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


# =====================================================================
# Deeper geometry & pipeline coverage
# =====================================================================


class TestIoU:
    def test_identical_boxes(self):
        a = make_box(0, 0, 10, 10, "")
        assert _iou(a, a) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = make_box(0, 0, 10, 10, "")
        b = make_box(20, 20, 30, 30, "")
        assert _iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = make_box(0, 0, 10, 10, "")
        b = make_box(5, 0, 15, 10, "")
        # Inter = 5*10 = 50; union = 100+100-50 = 150
        assert _iou(a, b) == pytest.approx(50.0 / 150.0)

    def test_contained(self):
        outer = make_box(0, 0, 20, 20, "")
        inner = make_box(5, 5, 10, 10, "")
        # Inter = 5*5 = 25; union = 400+25-25 = 400
        assert _iou(outer, inner) == pytest.approx(25.0 / 400.0)


class TestOverlapRatio:
    def test_full_coverage(self):
        a = make_box(5, 5, 10, 10, "")
        b = make_box(0, 0, 20, 20, "")
        # b fully covers a → ratio = a.area / a.area = 1.0
        assert _overlap_ratio(a, b) == pytest.approx(1.0)

    def test_no_coverage(self):
        a = make_box(0, 0, 10, 10, "")
        b = make_box(20, 20, 30, 30, "")
        assert _overlap_ratio(a, b) == 0.0

    def test_zero_area(self):
        a = make_box(5, 5, 5, 5, "")  # zero-area
        b = make_box(0, 0, 10, 10, "")
        assert _overlap_ratio(a, b) == 0.0


class TestCenter:
    def test_simple(self):
        b = make_box(10, 20, 30, 40, "")
        cx, cy = _center(b)
        assert cx == 20.0
        assert cy == 30.0


class TestOverlapsExisting:
    def test_no_overlap(self):
        cand = make_box(100, 100, 110, 110, "%")
        tokens = [make_box(0, 0, 10, 10, "A")]
        assert _overlaps_existing(cand, tokens) is False

    def test_high_iou_overlap(self):
        cand = make_box(0, 0, 10, 10, "%")
        tokens = [make_box(1, 1, 11, 11, "A")]
        assert _overlaps_existing(cand, tokens) is True

    def test_empty_tokens(self):
        cand = make_box(0, 0, 10, 10, "%")
        assert _overlaps_existing(cand, []) is False


class TestEstimateCharWidth:
    def test_normal(self):
        tokens = [
            make_box(0, 0, 50, 10, "HELLO", origin="text"),  # 50/5 = 10
            make_box(60, 0, 90, 10, "HI", origin="text"),  # 30/2 = 15
        ]
        w = _estimate_char_width(tokens)
        # median of [10, 15] = 12.5
        assert w == pytest.approx(12.5)

    def test_empty_fallback(self):
        cfg = GroupingConfig()
        w = _estimate_char_width([], cfg)
        assert w == cfg.ocr_reconcile_char_width_fallback

    def test_zero_width_excluded(self):
        tokens = [make_box(5, 0, 5, 10, "A", origin="text")]  # zero width
        cfg = GroupingConfig()
        w = _estimate_char_width(tokens, cfg)
        assert w == cfg.ocr_reconcile_char_width_fallback


class TestDedupTiles:
    def test_no_dups(self):
        a = make_box(0, 0, 10, 10, "A")
        b = make_box(50, 50, 60, 60, "B")
        out, confs = _dedup_tiles([a, b], [0.9, 0.8])
        assert len(out) == 2

    def test_duplicate_keeps_higher_conf(self):
        a = make_box(0, 0, 10, 10, "A")
        b = make_box(1, 1, 11, 11, "B")  # heavy overlap
        out, confs = _dedup_tiles([a, b], [0.7, 0.95])
        assert len(out) == 1
        assert out[0].text == "B"
        assert confs[0] == 0.95

    def test_single_token(self):
        a = make_box(0, 0, 10, 10, "X")
        out, confs = _dedup_tiles([a], [0.5])
        assert len(out) == 1

    def test_empty(self):
        out, confs = _dedup_tiles([], [])
        assert out == []
        assert confs == []


class TestBuildMatchIndex:
    def test_matched_tokens(self):
        ocr_tokens = [make_box(0, 0, 10, 10, "85%", origin="ocr_full")]
        pdf_tokens = [make_box(0, 0, 10, 10, "85", origin="text")]
        cfg = GroupingConfig()
        matches = _build_match_index(ocr_tokens, [0.9], pdf_tokens, cfg)
        assert len(matches) == 1
        assert matches[0].match_type == "iou"
        assert matches[0].ocr_confidence == 0.9

    def test_unmatched_token(self):
        ocr_tokens = [make_box(500, 500, 510, 510, "XYZ", origin="ocr_full")]
        pdf_tokens = [make_box(0, 0, 10, 10, "85", origin="text")]
        cfg = GroupingConfig()
        matches = _build_match_index(ocr_tokens, [0.8], pdf_tokens, cfg)
        assert len(matches) == 1
        assert matches[0].match_type == "unmatched"


class TestGenerateSymbolCandidates:
    def test_slash_between_digits(self):
        """OCR sees '09/15' → should generate a slash candidate between two digit anchors."""
        ocr_box = make_box(10, 100, 60, 112, "09/15", origin="ocr_full")
        pdf_tokens = [
            make_box(10, 100, 28, 112, "09", origin="text"),
            make_box(35, 100, 60, 112, "15", origin="text"),
        ]
        cfg = GroupingConfig()
        candidates = _generate_symbol_candidates(ocr_box, pdf_tokens, cfg)
        slash_cands = [c for c in candidates if c.symbol == "/"]
        assert len(slash_cands) >= 1
        assert slash_cands[0].slot_type == "between_digits"

    def test_percent_after_digit(self):
        ocr_box = make_box(10, 100, 50, 112, "85%", origin="ocr_full")
        pdf_tokens = [make_box(10, 100, 35, 112, "85", origin="text")]
        cfg = GroupingConfig()
        candidates = _generate_symbol_candidates(ocr_box, pdf_tokens, cfg)
        pct_cands = [c for c in candidates if c.symbol == "%"]
        assert len(pct_cands) >= 1
        assert pct_cands[0].slot_type == "after_digit"

    def test_no_digit_anchors(self):
        ocr_box = make_box(10, 100, 60, 112, "ABC/DEF", origin="ocr_full")
        pdf_tokens = [make_box(10, 100, 60, 112, "ABCDEF", origin="text")]
        cfg = GroupingConfig()
        candidates = _generate_symbol_candidates(ocr_box, pdf_tokens, cfg)
        assert candidates == []


class TestAcceptCandidates:
    def _make_candidate(self, symbol, x0, y0, x1, y1, ocr_box, anchor_right=None):
        return SymbolCandidate(
            symbol=symbol,
            slot_type="after_digit",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            ocr_source=ocr_box,
            anchor_right=anchor_right,
        )

    def test_accepts_in_clear_area(self):
        ocr = make_box(10, 100, 50, 112, "85%", origin="ocr_full")
        anchor = make_box(10, 100, 35, 112, "85", origin="text")
        cand = self._make_candidate("%", 36, 100, 44, 112, ocr, anchor_right=anchor)
        result = _accept_candidates([cand], [anchor], page_width=600.0)
        assert result[0].status == "accepted"

    def test_rejects_out_of_bounds(self):
        ocr = make_box(10, 100, 50, 112, "85%", origin="ocr_full")
        cand = self._make_candidate("%", -5, 100, 3, 112, ocr)
        result = _accept_candidates([cand], [], page_width=600.0)
        assert result[0].status == "rejected"
        assert result[0].reject_reason == "out_of_bounds"

    def test_rejects_already_in_pdf(self):
        ocr = make_box(10, 100, 50, 112, "85%", origin="ocr_full")
        existing = make_box(38, 100, 44, 112, "%", origin="text")
        cand = self._make_candidate("%", 38, 100, 44, 112, ocr)
        cfg = GroupingConfig()
        result = _accept_candidates([cand], [existing], page_width=600.0, cfg=cfg)
        assert result[0].status == "rejected"
        assert result[0].reject_reason == "already_in_pdf"
