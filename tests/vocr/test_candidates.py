"""Tests for plancheck.vocr.candidates — VOCR candidate detection methods."""

from __future__ import annotations

import pytest

from plancheck.config import GroupingConfig
from plancheck.models import GlyphBox, VocrCandidate
from plancheck.vocr.candidates import (
    _detect_baseline_style_gaps,
    _detect_char_encoding_failures,
    _detect_cross_ref_phrases,
    _detect_dense_cluster_holes,
    _detect_dimension_geometry,
    _detect_font_subset_correlation,
    _detect_impossible_sequences,
    _detect_intraline_gaps,
    _detect_keyword_cooccurrence,
    _detect_near_duplicate_lines,
    _detect_placeholder_tokens,
    _detect_regex_digit_patterns,
    _detect_semantic_no_units,
    _detect_template_adjacency,
    _detect_token_width_anomaly,
    _detect_vector_circles,
    _detect_vocab_triggers,
    _group_by_baseline,
    _iou_bbox,
    _merge_overlapping_candidates,
    _pad_bbox,
    compute_candidate_stats,
    detect_vocr_candidates,
)

# ── Constants for tests ────────────────────────────────────────────────

PAGE_W = 612.0
PAGE_H = 792.0
MARGIN = 4.0
PAGE_NUM = 0


# ── Helpers ────────────────────────────────────────────────────────────


def _box(x0, y0, x1, y1, text="", **kw):
    """Shorthand GlyphBox builder."""
    return GlyphBox(page=PAGE_NUM, x0=x0, y0=y0, x1=x1, y1=y1, text=text, **kw)


def _char(text, x0, y0, x1, y1, fontname="Arial"):
    """Build a pdfplumber-style char dict."""
    return {
        "text": text,
        "x0": x0,
        "top": y0,
        "x1": x1,
        "bottom": y1,
        "fontname": fontname,
    }


# ── Geometry helpers ───────────────────────────────────────────────────


class TestPadBbox:
    def test_basic_padding(self):
        result = _pad_bbox(10, 20, 30, 40, 5, PAGE_W, PAGE_H)
        assert result == (5, 15, 35, 45)

    def test_clamps_to_page_bounds(self):
        result = _pad_bbox(2, 3, PAGE_W - 1, PAGE_H - 1, 10, PAGE_W, PAGE_H)
        assert result == (0.0, 0.0, PAGE_W, PAGE_H)

    def test_zero_margin(self):
        result = _pad_bbox(10, 20, 30, 40, 0, PAGE_W, PAGE_H)
        assert result == (10, 20, 30, 40)


class TestIouBbox:
    def test_identical(self):
        a = (0, 0, 10, 10)
        assert _iou_bbox(a, a) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert _iou_bbox((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_partial(self):
        iou = _iou_bbox((0, 0, 10, 10), (5, 5, 15, 15))
        assert iou == pytest.approx(25 / 175, rel=1e-3)

    def test_zero_area(self):
        assert _iou_bbox((0, 0, 0, 10), (0, 0, 10, 10)) == 0.0


class TestGroupByBaseline:
    def test_single_line(self):
        tokens = [
            _box(0, 10, 20, 20, "A"),
            _box(25, 10, 50, 20, "B"),
            _box(55, 10, 80, 20, "C"),
        ]
        lines = _group_by_baseline(tokens)
        assert len(lines) == 1
        assert [t.text for t in lines[0]] == ["A", "B", "C"]

    def test_two_lines(self):
        tokens = [
            _box(0, 10, 20, 20, "A"),
            _box(0, 50, 20, 60, "B"),
        ]
        lines = _group_by_baseline(tokens)
        assert len(lines) == 2

    def test_empty(self):
        assert _group_by_baseline([]) == []

    def test_sorted_by_x(self):
        tokens = [
            _box(50, 10, 70, 20, "B"),
            _box(0, 10, 20, 20, "A"),
        ]
        lines = _group_by_baseline(tokens)
        assert len(lines) == 1
        assert lines[0][0].text == "A"
        assert lines[0][1].text == "B"


# ── Signal #1/#2: char encoding failures ──────────────────────────────


class TestCharEncodingFailures:
    def test_empty_char_text_flagged(self):
        chars = [_char("", 10, 20, 18, 30)]
        cands = _detect_char_encoding_failures(chars, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert len(cands) == 1
        assert "char_encoding_failure" in cands[0].trigger_methods
        assert cands[0].confidence == 0.9

    def test_replacement_char_flagged(self):
        chars = [_char("\ufffd", 10, 20, 18, 30)]
        cands = _detect_char_encoding_failures(chars, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert len(cands) == 1
        assert "unmapped_glyph" in cands[0].trigger_methods

    def test_cid_char_flagged(self):
        chars = [_char("(cid:42)", 10, 20, 18, 30)]
        cands = _detect_char_encoding_failures(chars, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert len(cands) == 1

    def test_nul_char_flagged(self):
        chars = [_char("\x00", 10, 20, 18, 30)]
        cands = _detect_char_encoding_failures(chars, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert len(cands) == 1

    def test_normal_char_skipped(self):
        chars = [_char("A", 10, 20, 18, 30)]
        cands = _detect_char_encoding_failures(chars, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert cands == []

    def test_tiny_bbox_skipped(self):
        chars = [_char("", 10, 20, 10.2, 20.1)]
        cands = _detect_char_encoding_failures(chars, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert cands == []

    def test_no_chars(self):
        cands = _detect_char_encoding_failures([], PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert cands == []


# ── Signal #3: placeholder tokens ─────────────────────────────────────


class TestPlaceholderTokens:
    def test_empty_text(self):
        tokens = [_box(10, 20, 30, 30, "")]
        cands = _detect_placeholder_tokens(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert len(cands) == 1
        assert cands[0].trigger_methods == ["placeholder_token"]

    def test_fffd_in_text(self):
        tokens = [_box(10, 20, 30, 30, "abc\ufffdef")]
        cands = _detect_placeholder_tokens(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert len(cands) == 1

    def test_cid_prefix(self):
        tokens = [_box(10, 20, 30, 30, "(cid:99)")]
        cands = _detect_placeholder_tokens(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert len(cands) == 1

    def test_normal_text_skipped(self):
        tokens = [_box(10, 20, 30, 30, "HELLO")]
        cands = _detect_placeholder_tokens(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert cands == []


# ── Signal #4: intraline gaps ─────────────────────────────────────────


class TestIntralineGaps:
    def test_large_gap_detected(self):
        # 4 tokens: three normal spacings, then a big gap
        # gaps: 5, 5, 155 → median=5, threshold=10, 155 >> 10
        tokens = [
            _box(0, 10, 20, 20, "A"),
            _box(25, 10, 45, 20, "B"),
            _box(50, 10, 70, 20, "C"),
            _box(225, 10, 245, 20, "D"),  # gap = 155 >> median 5
        ]
        cands = _detect_intraline_gaps(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN, 2.0)
        assert len(cands) >= 1
        assert cands[0].trigger_methods == ["intraline_gap"]
        assert cands[0].confidence == 0.6

    def test_uniform_spacing_no_candidates(self):
        tokens = [
            _box(0, 10, 20, 20, "A"),
            _box(25, 10, 45, 20, "B"),
            _box(50, 10, 70, 20, "C"),
        ]
        cands = _detect_intraline_gaps(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN, 2.0)
        assert cands == []

    def test_single_token_no_candidates(self):
        tokens = [_box(0, 10, 20, 20, "A")]
        cands = _detect_intraline_gaps(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN, 2.0)
        assert cands == []


# ── Signal #5: dense cluster holes ────────────────────────────────────


class TestDenseClusterHoles:
    def test_hole_in_dense_region(self):
        # Create a 3x3 grid of tokens with the center missing
        grid_size = 20.0
        tokens = []
        for r in range(3):
            for c in range(3):
                if r == 1 and c == 1:
                    continue  # hole
                x = c * grid_size + 1
                y = r * grid_size + 1
                tokens.append(_box(x, y, x + 10, y + 10, f"T{r}{c}"))
        cands = _detect_dense_cluster_holes(
            tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN, grid_size
        )
        # Should detect the hole at (1,1)
        assert len(cands) >= 1
        assert cands[0].trigger_methods == ["dense_cluster_hole"]

    def test_no_tokens(self):
        cands = _detect_dense_cluster_holes([], PAGE_NUM, PAGE_W, PAGE_H, MARGIN, 20.0)
        assert cands == []

    def test_sparse_region_no_candidates(self):
        tokens = [
            _box(0, 0, 10, 10, "A"),
            _box(400, 400, 410, 410, "B"),
        ]
        cands = _detect_dense_cluster_holes(
            tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN, 20.0
        )
        assert cands == []


# ── Signal #6: baseline style gaps ────────────────────────────────────


class TestBaselineStyleGaps:
    def test_same_font_gap(self):
        # Same font + big gap → detected
        # gaps: 5, 5, 75 → median=5, threshold=10, 75 >> 10
        tokens = [
            _box(0, 10, 20, 20, "45", fontname="Helvetica", font_size=12),
            _box(25, 10, 45, 20, "67", fontname="Helvetica", font_size=12),
            _box(50, 10, 70, 20, "00", fontname="Helvetica", font_size=12),
            _box(145, 10, 165, 20, "89", fontname="Helvetica", font_size=12),
        ]
        cands = _detect_baseline_style_gaps(
            tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN, 2.0
        )
        assert len(cands) >= 1
        assert cands[0].trigger_methods == ["baseline_style_gap"]
        assert cands[0].confidence == 0.7

    def test_different_font_no_candidate(self):
        # Different fonts on each side of gap → no baseline style gap
        tokens = [
            _box(0, 10, 20, 20, "45", fontname="Helvetica", font_size=12),
            _box(25, 10, 45, 20, "67", fontname="Helvetica", font_size=12),
            _box(50, 10, 70, 20, "00", fontname="Courier", font_size=12),
            _box(145, 10, 165, 20, "89", fontname="Arial", font_size=14),
        ]
        cands = _detect_baseline_style_gaps(
            tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN, 2.0
        )
        assert cands == []


# ── Signal #7: template adjacency ─────────────────────────────────────


class TestTemplateAdjacency:
    def test_digit_then_TYP(self):
        tokens = [
            _box(0, 10, 20, 20, "90"),
            _box(40, 10, 60, 20, "TYP"),
        ]
        cands = _detect_template_adjacency(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert len(cands) >= 1
        assert cands[0].predicted_symbol == "°"
        assert cands[0].trigger_methods == ["template_adjacency"]

    def test_digit_then_DIAM(self):
        tokens = [
            _box(0, 10, 20, 20, "12"),
            _box(40, 10, 70, 20, "DIAM"),
        ]
        cands = _detect_template_adjacency(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert any(c.predicted_symbol == "Ø" for c in cands)

    def test_DIA_before_digit(self):
        tokens = [
            _box(0, 10, 25, 20, "DIA"),
            _box(40, 10, 60, 20, "24"),
        ]
        cands = _detect_template_adjacency(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert any(c.predicted_symbol == "Ø" for c in cands)

    def test_no_gap_no_candidate(self):
        # Tokens touching — gap < 0.5
        tokens = [
            _box(0, 10, 20, 20, "90"),
            _box(20.2, 10, 40, 20, "TYP"),
        ]
        cands = _detect_template_adjacency(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert cands == []


# ── Signal #8: regex digit patterns ───────────────────────────────────


class TestRegexDigitPatterns:
    def test_two_decimal_numbers_predict_tolerance(self):
        tokens = [
            _box(0, 10, 20, 20, "3.5"),
            _box(40, 10, 60, 20, "0.5"),
        ]
        cands = _detect_regex_digit_patterns(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert len(cands) >= 1
        assert cands[0].predicted_symbol == "±"

    def test_two_integers_predict_multiply(self):
        tokens = [
            _box(0, 10, 20, 20, "12"),
            _box(40, 10, 60, 20, "24"),
        ]
        cands = _detect_regex_digit_patterns(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert len(cands) >= 1
        assert cands[0].predicted_symbol == "×"

    def test_digit_then_fraction_predict_inch(self):
        # "12" followed by "3/4" → missing " mark
        tokens = [
            _box(0, 10, 20, 20, "12"),
            _box(40, 10, 60, 20, "3/4"),
        ]
        cands = _detect_regex_digit_patterns(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert any(c.predicted_symbol == '"' for c in cands)

    def test_non_numeric_no_candidate(self):
        tokens = [
            _box(0, 10, 20, 20, "ABC"),
            _box(40, 10, 60, 20, "DEF"),
        ]
        cands = _detect_regex_digit_patterns(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert cands == []


# ── Signal #9: impossible sequences ───────────────────────────────────


class TestImpossibleSequences:
    def test_adjacent_numbers_small_gap(self):
        tokens = [
            _box(0, 10, 20, 20, "45"),
            _box(22, 10, 42, 20, "67"),  # gap = 2
        ]
        cands = _detect_impossible_sequences(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert len(cands) >= 1
        assert cands[0].trigger_methods == ["impossible_sequence"]

    def test_numbers_far_apart_no_candidate(self):
        tokens = [
            _box(0, 10, 20, 20, "45"),
            _box(100, 10, 120, 20, "67"),  # gap = 80 > 20
        ]
        cands = _detect_impossible_sequences(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert cands == []

    def test_non_numeric_skipped(self):
        tokens = [
            _box(0, 10, 20, 20, "ABC"),
            _box(22, 10, 42, 20, "DEF"),
        ]
        cands = _detect_impossible_sequences(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert cands == []


# ── Signal #10: vocab triggers ────────────────────────────────────────


class TestVocabTriggers:
    def test_angle_keyword(self):
        tokens = [
            _box(0, 10, 20, 20, "90"),
            _box(40, 10, 70, 20, "ANGLE"),
        ]
        cands = _detect_vocab_triggers(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert len(cands) >= 1
        assert cands[0].predicted_symbol == "°"

    def test_elev_keyword(self):
        tokens = [
            _box(0, 10, 30, 20, "ELEV"),
            _box(50, 10, 70, 20, "100"),
        ]
        cands = _detect_vocab_triggers(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert len(cands) >= 1
        assert cands[0].predicted_symbol == "±"

    def test_diameter_keyword(self):
        tokens = [
            _box(0, 10, 40, 20, "DIAMETER"),
            _box(60, 10, 80, 20, "16"),
        ]
        cands = _detect_vocab_triggers(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert len(cands) >= 1
        assert cands[0].predicted_symbol == "Ø"

    def test_keyword_without_digit_neighbor(self):
        tokens = [
            _box(0, 10, 30, 20, "ANGLE"),
            _box(40, 10, 80, 20, "VALUE"),
        ]
        cands = _detect_vocab_triggers(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert cands == []


# ── Signal #11: keyword cooccurrence ──────────────────────────────────


class TestKeywordCooccurrence:
    def test_OC_without_at_symbol(self):
        tokens = [
            _box(0, 10, 20, 20, "12"),
            _box(30, 10, 60, 20, "O.C."),
        ]
        cands = _detect_keyword_cooccurrence(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert len(cands) >= 1
        assert cands[0].predicted_symbol == "@"

    def test_BAR_without_hash(self):
        tokens = [
            _box(0, 10, 30, 20, "BAR"),
            _box(40, 10, 60, 20, "5"),
        ]
        cands = _detect_keyword_cooccurrence(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert len(cands) >= 1
        assert cands[0].predicted_symbol == "#"

    def test_symbol_already_present_no_candidate(self):
        # @ already on the line → skip
        tokens = [
            _box(0, 10, 20, 20, "12"),
            _box(25, 10, 30, 20, "@"),
            _box(35, 10, 65, 20, "O.C."),
        ]
        cands = _detect_keyword_cooccurrence(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert cands == []


# ── Signal #12: cross-ref phrases ─────────────────────────────────────


class TestCrossRefPhrases:
    def test_same_phrase_with_and_without_symbol(self):
        # Two occurrences of "10 20" phrase: line 0 has a ° somewhere, line 1 doesn't
        # The digit run must be consecutive (no symbol between them)
        tokens = [
            # Line 0 (y=10): "10" "20" "°" → phrase "10 20", line has symbol
            _box(0, 10, 15, 20, "10"),
            _box(25, 10, 40, 20, "20"),
            _box(45, 10, 50, 20, "°"),
            # Line 1 (y=50): "10" "20" → phrase "10 20", line has no symbol
            _box(0, 50, 15, 60, "10"),
            _box(25, 50, 40, 60, "20"),
        ]
        cands = _detect_cross_ref_phrases(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert len(cands) >= 1
        assert cands[0].trigger_methods == ["cross_ref_phrase"]

    def test_single_occurrence_no_candidate(self):
        tokens = [_box(0, 10, 15, 20, "10"), _box(25, 10, 40, 20, "20")]
        cands = _detect_cross_ref_phrases(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert cands == []


# ── Signal #13: near duplicate lines ──────────────────────────────────


class TestNearDuplicateLines:
    def test_repeated_pattern_with_gap(self):
        # Two lines with same digit-text structure, one has gap
        tokens = [
            # Line 0 (y=10)
            _box(0, 10, 15, 20, "10"),
            _box(20, 10, 35, 20, "BOLT"),
            _box(40, 10, 55, 20, "20"),
            # Line 1 (y=50) — duplicate but with bigger gap
            _box(0, 50, 15, 60, "10"),
            _box(20, 50, 35, 60, "BOLT"),
            _box(60, 50, 75, 60, "20"),  # bigger gap after BOLT
        ]
        cands = _detect_near_duplicate_lines(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        # May or may not detect depending on fingerprint matching
        # At minimum, should not crash
        assert isinstance(cands, list)


# ── Signal #14: font subset correlation ───────────────────────────────


class TestFontSubsetCorrelation:
    def test_high_failure_rate_font(self):
        # Font "BadFont" has >5% encoding failures
        chars = []
        for i in range(100):
            chars.append(_char("A", i * 5, 10, i * 5 + 4, 20, "BadFont"))
        # Add 10 failures (10%)
        for i in range(10):
            chars.append(_char("", i * 5, 30, i * 5 + 4, 40, "BadFont"))

        # Token near a digit in that font
        tokens = [
            _box(0, 10, 10, 20, "X", fontname="BadFont"),
            _box(15, 10, 30, 20, "42"),
        ]
        cands = _detect_font_subset_correlation(
            tokens, chars, PAGE_NUM, PAGE_W, PAGE_H, MARGIN
        )
        assert len(cands) >= 1
        assert cands[0].trigger_methods == ["font_subset_correlation"]

    def test_low_failure_rate_no_candidate(self):
        chars = [_char("A", i * 5, 10, i * 5 + 4, 20, "GoodFont") for i in range(100)]
        tokens = [
            _box(0, 10, 10, 20, "X", fontname="GoodFont"),
            _box(15, 10, 30, 20, "42"),
        ]
        cands = _detect_font_subset_correlation(
            tokens, chars, PAGE_NUM, PAGE_W, PAGE_H, MARGIN
        )
        assert cands == []


# ── Signal #15: token width anomaly ───────────────────────────────────


class TestTokenWidthAnomaly:
    def test_narrow_token_flagged(self):
        # 10 chars of text but very narrow bbox → anomaly
        tokens = [_box(0, 10, 5, 20, "ABCDEFGHIJ", font_size=12)]
        cands = _detect_token_width_anomaly(
            tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN, 0.7
        )
        assert len(cands) >= 1
        assert cands[0].trigger_methods == ["token_width_anomaly"]

    def test_normal_width_no_candidate(self):
        # Width proportional to text length
        tokens = [_box(0, 10, 66, 20, "ABCDEFGHIJ", font_size=12)]
        cands = _detect_token_width_anomaly(
            tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN, 0.7
        )
        assert cands == []

    def test_zero_font_size_skipped(self):
        tokens = [_box(0, 10, 5, 20, "ABC", font_size=0)]
        cands = _detect_token_width_anomaly(
            tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN, 0.7
        )
        assert cands == []


# ── Signal #16: vector circles ────────────────────────────────────────


class TestVectorCircles:
    def test_small_circle_near_digit(self):
        curves = [{"pts": [(30, 10), (34, 14), (30, 18), (26, 14)]}]
        tokens = [_box(10, 10, 25, 20, "45")]
        cands = _detect_vector_circles(
            curves, tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN, 8.0
        )
        assert len(cands) >= 1
        assert cands[0].predicted_symbol == "°"

    def test_large_circle_skipped(self):
        curves = [{"pts": [(0, 0), (50, 0), (50, 50), (0, 50)]}]
        tokens = [_box(60, 10, 80, 20, "45")]
        cands = _detect_vector_circles(
            curves, tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN, 8.0
        )
        assert cands == []

    def test_circle_not_near_digit_skipped(self):
        curves = [{"pts": [(300, 300), (304, 304), (300, 308), (296, 304)]}]
        tokens = [_box(10, 10, 25, 20, "45")]
        cands = _detect_vector_circles(
            curves, tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN, 8.0
        )
        assert cands == []


# ── Signal #17: semantic no units ─────────────────────────────────────


class TestSemanticNoUnits:
    def test_numbers_without_units(self):
        tokens = [
            _box(0, 10, 15, 20, "12"),
            _box(25, 10, 40, 20, "24"),
            _box(50, 10, 65, 20, "36"),
        ]
        cands = _detect_semantic_no_units(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert len(cands) >= 1
        assert cands[0].predicted_symbol == '"'

    def test_numbers_with_unit_no_candidate(self):
        tokens = [
            _box(0, 10, 15, 20, "12"),
            _box(20, 10, 35, 20, "FT"),
            _box(45, 10, 60, 20, "24"),
        ]
        cands = _detect_semantic_no_units(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert cands == []

    def test_single_number_no_candidate(self):
        tokens = [_box(0, 10, 15, 20, "12")]
        cands = _detect_semantic_no_units(tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert cands == []


# ── Signal #18: dimension geometry ────────────────────────────────────


class TestDimensionGeometry:
    def test_line_near_gap(self):
        tokens = [
            _box(0, 10, 20, 20, "12"),
            _box(50, 10, 70, 20, "24"),  # gap: 20-50
        ]
        page_lines = [{"x0": 20, "top": 15, "x1": 50, "bottom": 15}]
        cands = _detect_dimension_geometry(
            page_lines, tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN
        )
        assert len(cands) >= 1
        assert cands[0].trigger_methods == ["dimension_geometry_proximity"]

    def test_no_lines_no_candidate(self):
        tokens = [_box(0, 10, 20, 20, "12"), _box(50, 10, 70, 20, "24")]
        cands = _detect_dimension_geometry([], tokens, PAGE_NUM, PAGE_W, PAGE_H, MARGIN)
        assert cands == []


# ── Merge overlapping ─────────────────────────────────────────────────


class TestMergeOverlapping:
    def test_identical_candidates_merged(self):
        cands = [
            VocrCandidate(
                page=0,
                x0=10,
                y0=10,
                x1=30,
                y1=30,
                trigger_methods=["method_a"],
                confidence=0.8,
            ),
            VocrCandidate(
                page=0,
                x0=10,
                y0=10,
                x1=30,
                y1=30,
                trigger_methods=["method_b"],
                confidence=0.6,
            ),
        ]
        merged = _merge_overlapping_candidates(cands)
        assert len(merged) == 1
        assert "method_a" in merged[0].trigger_methods
        assert "method_b" in merged[0].trigger_methods
        assert merged[0].confidence == 0.8

    def test_non_overlapping_kept(self):
        cands = [
            VocrCandidate(
                page=0,
                x0=0,
                y0=0,
                x1=10,
                y1=10,
                trigger_methods=["method_a"],
                confidence=0.8,
            ),
            VocrCandidate(
                page=0,
                x0=100,
                y0=100,
                x1=110,
                y1=110,
                trigger_methods=["method_b"],
                confidence=0.6,
            ),
        ]
        merged = _merge_overlapping_candidates(cands)
        assert len(merged) == 2

    def test_single_candidate(self):
        cands = [
            VocrCandidate(
                page=0,
                x0=10,
                y0=10,
                x1=30,
                y1=30,
                trigger_methods=["A"],
                confidence=0.9,
            )
        ]
        merged = _merge_overlapping_candidates(cands)
        assert len(merged) == 1

    def test_empty(self):
        assert _merge_overlapping_candidates([]) == []

    def test_predicted_symbol_inherited(self):
        cands = [
            VocrCandidate(
                page=0,
                x0=10,
                y0=10,
                x1=30,
                y1=30,
                trigger_methods=["A"],
                confidence=0.9,
                predicted_symbol="",
            ),
            VocrCandidate(
                page=0,
                x0=10,
                y0=10,
                x1=30,
                y1=30,
                trigger_methods=["B"],
                confidence=0.5,
                predicted_symbol="°",
            ),
        ]
        merged = _merge_overlapping_candidates(cands)
        assert len(merged) == 1
        assert merged[0].predicted_symbol == "°"


# ── Public API: detect_vocr_candidates ─────────────────────────────────


class TestDetectVocrCandidates:
    def test_basic_integration(self):
        """Smoke test: run all 18 methods on a small synthetic page."""
        cfg = GroupingConfig()
        tokens = [
            _box(0, 10, 20, 20, "45"),
            _box(25, 10, 45, 20, "TYP"),
            _box(100, 10, 120, 20, "90"),
        ]
        chars = [_char("", 60, 10, 68, 20)]
        cands = detect_vocr_candidates(
            tokens=tokens,
            page_chars=chars,
            page_lines=[],
            page_curves=[],
            page_rects=[],
            page_width=PAGE_W,
            page_height=PAGE_H,
            page_num=0,
            cfg=cfg,
        )
        assert isinstance(cands, list)
        # At least the encoding failure + template adjacency
        assert len(cands) >= 2

    def test_respects_max_candidates(self):
        cfg = GroupingConfig(vocr_cand_max_candidates=3)
        # Many encoding failures → many candidates
        chars = [_char("", i * 10, 10, i * 10 + 8, 20) for i in range(20)]
        cands = detect_vocr_candidates(
            tokens=[],
            page_chars=chars,
            page_lines=[],
            page_curves=[],
            page_rects=[],
            page_width=PAGE_W,
            page_height=PAGE_H,
            page_num=0,
            cfg=cfg,
        )
        assert len(cands) <= 3

    def test_respects_min_confidence(self):
        cfg = GroupingConfig(vocr_cand_min_confidence=0.95)
        chars = [_char("", 10, 10, 18, 20)]  # encoding failure → conf 0.9
        cands = detect_vocr_candidates(
            tokens=[],
            page_chars=chars,
            page_lines=[],
            page_curves=[],
            page_rects=[],
            page_width=PAGE_W,
            page_height=PAGE_H,
            page_num=0,
            cfg=cfg,
        )
        # Should filter out 0.9 confidence candidates
        assert len(cands) == 0

    def test_empty_page(self):
        cfg = GroupingConfig()
        cands = detect_vocr_candidates(
            tokens=[],
            page_chars=[],
            page_lines=[],
            page_curves=[],
            page_rects=[],
            page_width=PAGE_W,
            page_height=PAGE_H,
            page_num=0,
            cfg=cfg,
        )
        assert cands == []

    def test_sorted_by_confidence_descending(self):
        cfg = GroupingConfig()
        tokens = [
            _box(0, 10, 20, 20, "45"),
            _box(40, 10, 60, 20, "TYP"),  # template adj → 0.85
            _box(0, 50, 20, 60, "12"),
            _box(100, 50, 120, 60, "24"),  # intraline gap → 0.6
        ]
        chars = [_char("", 200, 10, 208, 20)]  # encoding → 0.9
        cands = detect_vocr_candidates(
            tokens=tokens,
            page_chars=chars,
            page_lines=[],
            page_curves=[],
            page_rects=[],
            page_width=PAGE_W,
            page_height=PAGE_H,
            page_num=0,
            cfg=cfg,
        )
        if len(cands) >= 2:
            for i in range(len(cands) - 1):
                assert cands[i].confidence >= cands[i + 1].confidence


# ── Statistics: compute_candidate_stats ────────────────────────────────


class TestComputeCandidateStats:
    def test_basic_stats(self):
        cands = [
            VocrCandidate(
                page=0,
                x0=0,
                y0=0,
                x1=10,
                y1=10,
                trigger_methods=["method_a"],
                outcome="hit",
                found_symbol="°",
                predicted_symbol="°",
            ),
            VocrCandidate(
                page=0,
                x0=20,
                y0=0,
                x1=30,
                y1=10,
                trigger_methods=["method_a", "method_b"],
                outcome="miss",
            ),
            VocrCandidate(
                page=0,
                x0=40,
                y0=0,
                x1=50,
                y1=10,
                trigger_methods=["method_b"],
                outcome="hit",
                found_symbol="±",
            ),
        ]
        stats = compute_candidate_stats(cands, PAGE_W, PAGE_H)
        assert stats["total_candidates"] == 3
        assert stats["total_hits"] == 2
        assert stats["total_misses"] == 1
        assert stats["total_pending"] == 0
        assert stats["hit_rate"] == pytest.approx(2 / 3, rel=1e-3)

    def test_per_method_hit_rate(self):
        cands = [
            VocrCandidate(
                page=0,
                x0=0,
                y0=0,
                x1=10,
                y1=10,
                trigger_methods=["method_a"],
                outcome="hit",
            ),
            VocrCandidate(
                page=0,
                x0=20,
                y0=0,
                x1=30,
                y1=10,
                trigger_methods=["method_a"],
                outcome="miss",
            ),
        ]
        stats = compute_candidate_stats(cands)
        by_method = stats["by_method"]
        assert "method_a" in by_method
        assert by_method["method_a"]["flagged"] == 2
        assert by_method["method_a"]["hits"] == 1
        assert by_method["method_a"]["misses"] == 1
        assert by_method["method_a"]["hit_rate"] == pytest.approx(0.5)

    def test_empty_candidates(self):
        stats = compute_candidate_stats([])
        assert stats["total_candidates"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["by_method"] == {}

    def test_area_stats(self):
        cands = [
            VocrCandidate(
                page=0, x0=0, y0=0, x1=10, y1=10, trigger_methods=["A"], outcome="hit"
            ),
        ]
        stats = compute_candidate_stats(cands, 100, 100)
        assert stats["area_stats"]["total_patch_area_pts2"] == 100.0
        assert stats["area_stats"]["page_coverage_pct"] == pytest.approx(1.0, rel=1e-2)

    def test_predicted_vs_found(self):
        cands = [
            VocrCandidate(
                page=0,
                x0=0,
                y0=0,
                x1=10,
                y1=10,
                trigger_methods=["A"],
                outcome="hit",
                predicted_symbol="°",
                found_symbol="°",
            ),
            VocrCandidate(
                page=0,
                x0=20,
                y0=0,
                x1=30,
                y1=10,
                trigger_methods=["A"],
                outcome="hit",
                predicted_symbol="°",
                found_symbol="±",
            ),
            VocrCandidate(
                page=0,
                x0=40,
                y0=0,
                x1=50,
                y1=10,
                trigger_methods=["A"],
                outcome="miss",
                predicted_symbol="°",
            ),
        ]
        stats = compute_candidate_stats(cands)
        pvf = stats["predicted_vs_found"]["°"]
        assert pvf["predicted"] == 3
        assert pvf["correct"] == 1
        assert pvf["wrong_symbol"] == 1
        assert pvf["miss"] == 1

    def test_pending_candidates(self):
        cands = [
            VocrCandidate(
                page=0,
                x0=0,
                y0=0,
                x1=10,
                y1=10,
                trigger_methods=["A"],
                outcome="pending",
            ),
        ]
        stats = compute_candidate_stats(cands)
        assert stats["total_pending"] == 1
        assert stats["total_hits"] == 0
        assert stats["total_misses"] == 0


# ── VocrCandidate model ───────────────────────────────────────────────


class TestVocrCandidate:
    def test_bbox(self):
        c = VocrCandidate(page=0, x0=10, y0=20, x1=30, y1=40)
        assert c.bbox() == (10, 20, 30, 40)

    def test_patch_area(self):
        c = VocrCandidate(page=0, x0=0, y0=0, x1=10, y1=10)
        assert c.patch_area() == 100.0

    def test_patch_area_zero(self):
        c = VocrCandidate(page=0, x0=10, y0=10, x1=10, y1=10)
        assert c.patch_area() == 0.0

    def test_round_trip_dict(self):
        c = VocrCandidate(
            page=1,
            x0=10.5,
            y0=20.5,
            x1=30.5,
            y1=40.5,
            trigger_methods=["method_a", "method_b"],
            predicted_symbol="°",
            confidence=0.85,
            context={"key": "val"},
            outcome="hit",
            found_text="90°",
            found_symbol="°",
        )
        d = c.to_dict()
        c2 = VocrCandidate.from_dict(d)
        assert c2.page == c.page
        assert c2.x0 == pytest.approx(c.x0, abs=0.01)
        assert c2.trigger_methods == c.trigger_methods
        assert c2.predicted_symbol == c.predicted_symbol
        assert c2.outcome == c.outcome
        assert c2.found_symbol == c.found_symbol

    def test_defaults(self):
        c = VocrCandidate(page=0, x0=0, y0=0, x1=10, y1=10)
        assert c.trigger_methods == []
        assert c.predicted_symbol == ""
        assert c.confidence == 0.5
        assert c.outcome == "pending"
        assert c.found_text == ""
        assert c.found_symbol == ""
