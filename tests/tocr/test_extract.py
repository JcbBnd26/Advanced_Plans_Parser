"""Tests for plancheck.tocr.extract — centralised TOCR extraction API.

Covers:
- _word_to_glyph_minimal (coordinate clipping, degenerate skip)
- _word_to_glyph_full (all filters / normalisation)
- _dedup_identical_text_iou
- _build_extract_words_kwargs (both modes)
- _empty_diagnostics (schema completeness)
- TocrPageResult + to_legacy_tuple
- extract_tocr_from_page (mock-based, both modes)
- extract_tocr_page (error-handling path)
"""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from conftest import make_box

from plancheck.config import GroupingConfig
from plancheck.models import GlyphBox
from plancheck.tocr.extract import (
    TocrPageResult,
    _build_extract_words_kwargs,
    _dedup_identical_text_iou,
    _empty_diagnostics,
    _word_to_glyph_full,
    _word_to_glyph_minimal,
    extract_tocr_from_page,
    extract_tocr_page,
)

# ── Helpers ────────────────────────────────────────────────────────────


def _word(
    x0: float = 10,
    y0: float = 20,
    x1: float = 50,
    y1: float = 35,
    text: str = "HELLO",
    fontname: str = "Arial",
    size: float = 12.0,
    upright: bool = True,
) -> dict:
    """Build a dict matching pdfplumber's extract_words output."""
    return {
        "x0": x0,
        "x1": x1,
        "top": y0,
        "bottom": y1,
        "text": text,
        "fontname": fontname,
        "size": size,
        "upright": upright,
    }


def _cfg(**overrides) -> GroupingConfig:
    """Return a GroupingConfig with optional field overrides."""
    return GroupingConfig(**overrides)


# ── _word_to_glyph_minimal ────────────────────────────────────────────


class TestWordToGlyphMinimal:
    """Minimal mode: only clip coordinates + skip degenerate boxes."""

    def test_normal_word(self):
        w = _word(x0=10, y0=20, x1=50, y1=35, text="ABC")
        g = _word_to_glyph_minimal(w, page_num=0, page_w=612, page_h=792, cfg=_cfg())
        assert g is not None
        assert g.text == "ABC"
        assert g.x0 == 10
        assert g.y0 == 20
        assert g.x1 == 50
        assert g.y1 == 35
        assert g.page == 0
        assert g.origin == "text"

    def test_clips_to_page_bounds(self):
        w = _word(x0=-5, y0=-3, x1=700, y1=900)
        g = _word_to_glyph_minimal(w, 0, page_w=612, page_h=792, cfg=_cfg())
        assert g is not None
        assert g.x0 == 0.0
        assert g.y0 == 0.0
        assert g.x1 == 612
        assert g.y1 == 792

    def test_degenerate_zero_width(self):
        w = _word(x0=50, x1=50)
        g = _word_to_glyph_minimal(w, 0, 612, 792, _cfg())
        assert g is None

    def test_degenerate_zero_height(self):
        w = _word(y0=100, y1=100)
        g = _word_to_glyph_minimal(w, 0, 612, 792, _cfg())
        assert g is None

    def test_degenerate_inverted(self):
        w = _word(x0=50, x1=10, y0=35, y1=20)
        g = _word_to_glyph_minimal(w, 0, 612, 792, _cfg())
        assert g is None

    def test_preserves_font_info(self):
        w = _word(fontname="TimesNewRoman", size=14.5)
        g = _word_to_glyph_minimal(w, 0, 612, 792, _cfg())
        assert g.fontname == "TimesNewRoman"
        assert g.font_size == 14.5

    def test_no_text_modification(self):
        """Minimal mode must NOT alter text — even control chars."""
        w = _word(text="AB\x01CD")
        g = _word_to_glyph_minimal(w, 0, 612, 792, _cfg())
        assert g.text == "AB\x01CD"

    def test_whitespace_only_text_kept(self):
        """Minimal mode keeps whitespace-only tokens."""
        w = _word(text="  ")
        g = _word_to_glyph_minimal(w, 0, 612, 792, _cfg())
        assert g is not None
        assert g.text == "  "


# ── _word_to_glyph_full ───────────────────────────────────────────────


class TestWordToGlyphFull:
    """Full mode: all config-driven filters + normalisation."""

    def _diag(self, cfg=None) -> dict:
        return _empty_diagnostics(cfg or _cfg())

    def _counters(self):
        from collections import Counter

        return Counter(), Counter()

    def test_normal_word(self):
        cfg = _cfg()
        diag = self._diag(cfg)
        fn, fs = self._counters()
        w = _word(text="HELLO", fontname="Arial", size=12.0)
        g = _word_to_glyph_full(w, 0, 612, 792, cfg, diag, fn, fs)
        assert g is not None
        assert g.text == "HELLO"
        assert fn["Arial"] == 1
        assert fs["12.0"] == 1

    def test_clips_to_page_when_enabled(self):
        cfg = _cfg(tocr_clip_to_page=True)
        diag = self._diag(cfg)
        fn, fs = self._counters()
        w = _word(x0=-5, x1=700)
        g = _word_to_glyph_full(w, 0, 612, 792, cfg, diag, fn, fs)
        assert g.x0 == 0.0
        assert g.x1 == 612

    def test_no_clip_when_disabled(self):
        cfg = _cfg(tocr_clip_to_page=False)
        diag = self._diag(cfg)
        fn, fs = self._counters()
        w = _word(x0=-5, x1=700)
        g = _word_to_glyph_full(w, 0, 612, 792, cfg, diag, fn, fs)
        assert g.x0 == -5
        assert g.x1 == 700

    def test_degenerate_skipped(self):
        cfg = _cfg()
        diag = self._diag(cfg)
        fn, fs = self._counters()
        w = _word(x0=50, x1=50)
        g = _word_to_glyph_full(w, 0, 612, 792, cfg, diag, fn, fs)
        assert g is None
        assert diag["tokens_degenerate_skipped"] == 1

    def test_margin_filter(self):
        cfg = _cfg(tocr_margin_pts=20.0)
        diag = self._diag(cfg)
        fn, fs = self._counters()
        # Centre of this box is at (15, 27.5) — within 20pt margin
        w = _word(x0=10, y0=20, x1=20, y1=35)
        g = _word_to_glyph_full(w, 0, 612, 792, cfg, diag, fn, fs)
        assert g is None
        assert diag["tokens_margin_filtered"] == 1

    def test_margin_filter_right_edge(self):
        cfg = _cfg(tocr_margin_pts=20.0)
        diag = self._diag(cfg)
        fn, fs = self._counters()
        # centre x = 602.5 → within margin of right edge (612 - 20 = 592)
        w = _word(x0=595, y0=400, x1=610, y1=415)
        g = _word_to_glyph_full(w, 0, 612, 792, cfg, diag, fn, fs)
        assert g is None
        assert diag["tokens_margin_filtered"] == 1

    def test_min_font_size_filter(self):
        cfg = _cfg(tocr_min_font_size=6.0)
        diag = self._diag(cfg)
        fn, fs = self._counters()
        w = _word(size=4.0)
        g = _word_to_glyph_full(w, 0, 612, 792, cfg, diag, fn, fs)
        assert g is None
        assert diag["tokens_font_size_filtered"] == 1

    def test_max_font_size_filter(self):
        cfg = _cfg(tocr_max_font_size=50.0)
        diag = self._diag(cfg)
        fn, fs = self._counters()
        w = _word(size=72.0)
        g = _word_to_glyph_full(w, 0, 612, 792, cfg, diag, fn, fs)
        assert g is None
        assert diag["tokens_font_size_filtered"] == 1

    def test_rotated_dropped_when_configured(self):
        cfg = _cfg(tocr_keep_rotated=False)
        diag = self._diag(cfg)
        fn, fs = self._counters()
        w = _word(upright=False)
        g = _word_to_glyph_full(w, 0, 612, 792, cfg, diag, fn, fs)
        assert g is None
        assert diag["tokens_rotated_dropped"] == 1

    def test_rotated_kept_when_configured(self):
        cfg = _cfg(tocr_keep_rotated=True)
        diag = self._diag(cfg)
        fn, fs = self._counters()
        w = _word(upright=False)
        g = _word_to_glyph_full(w, 0, 612, 792, cfg, diag, fn, fs)
        assert g is not None

    def test_control_chars_cleaned(self):
        cfg = _cfg(tocr_filter_control_chars=True)
        diag = self._diag(cfg)
        fn, fs = self._counters()
        w = _word(text="HE\x01LLO")
        g = _word_to_glyph_full(w, 0, 612, 792, cfg, diag, fn, fs)
        assert g is not None
        assert g.text == "HELLO"
        assert diag["tokens_control_char_cleaned"] == 1

    def test_empty_after_control_char_clean(self):
        cfg = _cfg(tocr_filter_control_chars=True)
        diag = self._diag(cfg)
        fn, fs = self._counters()
        w = _word(text="\x01\x02")
        g = _word_to_glyph_full(w, 0, 612, 792, cfg, diag, fn, fs)
        assert g is None
        assert diag["tokens_empty_after_clean"] == 1

    def test_unicode_normalisation(self):
        cfg = _cfg(tocr_normalize_unicode=True)
        diag = self._diag(cfg)
        fn, fs = self._counters()
        # ﬁ (U+FB01) → "fi" under NFKC
        w = _word(text="ﬁnd")
        g = _word_to_glyph_full(w, 0, 612, 792, cfg, diag, fn, fs)
        assert g.text == "find"
        assert diag["tokens_unicode_normalized"] == 1

    def test_collapse_whitespace(self):
        cfg = _cfg(tocr_collapse_whitespace=True)
        diag = self._diag(cfg)
        fn, fs = self._counters()
        w = _word(text="A  B   C")
        g = _word_to_glyph_full(w, 0, 612, 792, cfg, diag, fn, fs)
        assert g.text == "A B C"
        assert diag["tokens_whitespace_collapsed"] == 1

    def test_case_fold(self):
        cfg = _cfg(tocr_case_fold=True)
        diag = self._diag(cfg)
        fn, fs = self._counters()
        w = _word(text="Hello World")
        g = _word_to_glyph_full(w, 0, 612, 792, cfg, diag, fn, fs)
        assert g.text == "hello world"
        assert diag["tokens_case_folded"] == 1

    def test_whitespace_only_stripped(self):
        cfg = _cfg(tocr_strip_whitespace_tokens=True)
        diag = self._diag(cfg)
        fn, fs = self._counters()
        w = _word(text="   ")
        g = _word_to_glyph_full(w, 0, 612, 792, cfg, diag, fn, fs)
        assert g is None
        assert diag["tokens_whitespace_filtered"] == 1

    def test_min_word_length_filter(self):
        cfg = _cfg(tocr_min_word_length=3)
        diag = self._diag(cfg)
        fn, fs = self._counters()
        w = _word(text="AB")
        g = _word_to_glyph_full(w, 0, 612, 792, cfg, diag, fn, fs)
        assert g is None
        assert diag["tokens_short_filtered"] == 1

    def test_rotation_tracking(self):
        cfg = _cfg(tocr_keep_rotated=True)
        diag = self._diag(cfg)
        fn, fs = self._counters()
        w = _word(upright=False)
        _word_to_glyph_full(w, 0, 612, 792, cfg, diag, fn, fs)
        assert diag["has_rotated_text"] is True
        assert diag["non_upright_count"] == 1
        assert diag["upright_count"] == 0

    def test_mojibake_detection(self):
        cfg = _cfg()
        diag = self._diag(cfg)
        fn, fs = self._counters()
        w = _word(text="Hello\ufffdWorld")
        _word_to_glyph_full(w, 0, 612, 792, cfg, diag, fn, fs)
        assert diag["char_encoding_issues"] == 1


# ── _dedup_identical_text_iou ──────────────────────────────────────────


class TestDedupIdenticalTextIou:
    """Text-aware IoU dedup (full mode only)."""

    def test_no_dedup_different_text(self):
        diag = _empty_diagnostics(_cfg())
        boxes = [
            make_box(0, 0, 10, 10, "AAA"),
            make_box(0, 0, 10, 10, "BBB"),
        ]
        result = _dedup_identical_text_iou(boxes, threshold=0.8, diag=diag)
        assert len(result) == 2
        assert diag["tokens_duplicate_removed"] == 0

    def test_dedup_same_text_high_overlap(self):
        diag = _empty_diagnostics(_cfg())
        boxes = [
            make_box(0, 0, 10, 10, "AAA"),
            make_box(1, 1, 11, 11, "AAA"),  # near-identical position
        ]
        result = _dedup_identical_text_iou(boxes, threshold=0.5, diag=diag)
        assert len(result) == 1
        assert diag["tokens_duplicate_removed"] == 1

    def test_no_dedup_same_text_far_apart(self):
        diag = _empty_diagnostics(_cfg())
        boxes = [
            make_box(0, 0, 10, 10, "AAA"),
            make_box(100, 100, 110, 110, "AAA"),
        ]
        result = _dedup_identical_text_iou(boxes, threshold=0.5, diag=diag)
        assert len(result) == 2

    def test_threshold_zero_disables(self):
        diag = _empty_diagnostics(_cfg())
        boxes = [
            make_box(0, 0, 10, 10, "AAA"),
            make_box(0, 0, 10, 10, "AAA"),
        ]
        result = _dedup_identical_text_iou(boxes, threshold=0, diag=diag)
        assert len(result) == 2

    def test_single_then_empty(self):
        diag = _empty_diagnostics(_cfg())
        assert _dedup_identical_text_iou([make_box(0, 0, 1, 1, "X")], 0.8, diag) == [
            make_box(0, 0, 1, 1, "X")
        ]
        assert _dedup_identical_text_iou([], 0.8, diag) == []

    def test_keeps_first_of_duplicates(self):
        diag = _empty_diagnostics(_cfg())
        b1 = make_box(0, 0, 10, 10, "DUP")
        b2 = make_box(0, 0, 10, 10, "DUP")
        result = _dedup_identical_text_iou([b1, b2], 0.8, diag)
        assert result[0] is b1


# ── _build_extract_words_kwargs ────────────────────────────────────────


class TestBuildExtractWordsKwargs:
    """extract_words argument builder."""

    def test_full_mode_defaults(self):
        cfg = _cfg()
        kw = _build_extract_words_kwargs(cfg, "full")
        assert kw["x_tolerance"] == cfg.tocr_x_tolerance
        assert kw["y_tolerance"] == cfg.tocr_y_tolerance
        assert kw["extra_attrs"] == ["fontname", "size", "upright"]

    def test_full_mode_text_flow(self):
        cfg = _cfg(tocr_use_text_flow=True)
        kw = _build_extract_words_kwargs(cfg, "full")
        assert kw["use_text_flow"] is True

    def test_full_mode_keep_blank(self):
        cfg = _cfg(tocr_keep_blank_chars=True)
        kw = _build_extract_words_kwargs(cfg, "full")
        assert kw["keep_blank_chars"] is True

    def test_minimal_mode_defaults(self):
        cfg = _cfg()
        kw = _build_extract_words_kwargs(cfg, "minimal")
        assert kw["extra_attrs"] == ["fontname", "size"]
        assert "use_text_flow" not in kw
        assert "keep_blank_chars" not in kw

    def test_extra_attrs_disabled(self):
        cfg = _cfg(tocr_extra_attrs=False)
        kw_full = _build_extract_words_kwargs(cfg, "full")
        kw_min = _build_extract_words_kwargs(cfg, "minimal")
        assert "extra_attrs" not in kw_full
        assert "extra_attrs" not in kw_min


# ── _empty_diagnostics ────────────────────────────────────────────────


class TestEmptyDiagnostics:
    """Diagnostics dict schema completeness."""

    REQUIRED_KEYS = {
        "extraction_params",
        "tokens_total",
        "tokens_raw",
        "tokens_degenerate_skipped",
        "tokens_control_char_cleaned",
        "tokens_empty_after_clean",
        "tokens_duplicate_removed",
        "tokens_font_size_filtered",
        "tokens_rotated_dropped",
        "tokens_margin_filtered",
        "tokens_short_filtered",
        "tokens_whitespace_filtered",
        "tokens_unicode_normalized",
        "tokens_case_folded",
        "tokens_whitespace_collapsed",
        "font_names",
        "font_sizes",
        "has_rotated_text",
        "upright_count",
        "non_upright_count",
        "char_encoding_issues",
        "mojibake_fraction",
        "below_min_density",
        "page_area_sqin",
        "token_density_per_sqin",
        "error",
    }

    def test_all_keys_present(self):
        diag = _empty_diagnostics(_cfg())
        assert self.REQUIRED_KEYS.issubset(set(diag.keys()))

    def test_counters_zero(self):
        diag = _empty_diagnostics(_cfg())
        for key in [
            "tokens_total",
            "tokens_raw",
            "tokens_degenerate_skipped",
            "tokens_control_char_cleaned",
            "tokens_empty_after_clean",
            "tokens_duplicate_removed",
            "tokens_font_size_filtered",
            "tokens_rotated_dropped",
            "tokens_margin_filtered",
            "tokens_short_filtered",
            "tokens_whitespace_filtered",
            "tokens_unicode_normalized",
            "tokens_case_folded",
            "tokens_whitespace_collapsed",
            "upright_count",
            "non_upright_count",
            "char_encoding_issues",
        ]:
            assert diag[key] == 0, f"{key} should be 0"

    def test_error_is_none(self):
        assert _empty_diagnostics(_cfg())["error"] is None

    def test_extraction_params_captures_config(self):
        cfg = _cfg(tocr_x_tolerance=5.0, tocr_dedup_iou=0.9)
        diag = _empty_diagnostics(cfg)
        assert diag["extraction_params"]["x_tolerance"] == 5.0
        assert diag["extraction_params"]["dedup_iou"] == 0.9


# ── TocrPageResult ─────────────────────────────────────────────────────


class TestTocrPageResult:
    """Result container and legacy adapter."""

    def test_to_legacy_tuple(self):
        boxes = [make_box(0, 0, 10, 10, "A"), make_box(20, 20, 30, 30, "B")]
        diag = {"tokens_total": 2}
        r = TocrPageResult(
            tokens=boxes, page_width=612, page_height=792, diagnostics=diag
        )
        tup = r.to_legacy_tuple()
        assert tup == (boxes, 612, 792, diag)
        assert tup[0] is boxes  # same reference

    def test_empty_result(self):
        r = TocrPageResult(tokens=[], page_width=0, page_height=0)
        assert r.to_legacy_tuple() == ([], 0, 0, {})


# ── extract_tocr_from_page (mock) ─────────────────────────────────────


def _make_mock_page(
    words: list[dict],
    width: float = 612,
    height: float = 792,
) -> MagicMock:
    """Create a mock pdfplumber Page returning *words*."""
    page = MagicMock()
    page.width = width
    page.height = height
    page.extract_words = MagicMock(return_value=words)
    return page


class TestExtractTocrFromPageMinimal:
    """Integration: minimal mode via mock page."""

    def test_basic_extraction(self):
        words = [
            _word(10, 20, 50, 35, "HELLO"),
            _word(60, 20, 100, 35, "WORLD"),
        ]
        page = _make_mock_page(words)
        result = extract_tocr_from_page(page, 0, mode="minimal")
        assert len(result.tokens) == 2
        assert result.tokens[0].text == "HELLO"
        assert result.tokens[1].text == "WORLD"
        assert result.page_width == 612
        assert result.page_height == 792
        assert result.diagnostics["tokens_raw"] == 2
        assert result.diagnostics["tokens_total"] == 2

    def test_degenerate_filtered(self):
        words = [
            _word(10, 20, 50, 35, "OK"),
            _word(50, 20, 50, 35, "BAD"),  # zero width
        ]
        page = _make_mock_page(words)
        result = extract_tocr_from_page(page, 0, mode="minimal")
        assert len(result.tokens) == 1
        assert result.tokens[0].text == "OK"

    def test_no_text_modification(self):
        """Minimal mode must not clean control chars / normalise."""
        words = [_word(10, 20, 50, 35, "AB\x01CD")]
        page = _make_mock_page(words)
        result = extract_tocr_from_page(page, 0, mode="minimal")
        assert result.tokens[0].text == "AB\x01CD"

    def test_default_cfg_created(self):
        """Omitting cfg= should create a default GroupingConfig."""
        page = _make_mock_page([_word(10, 20, 50, 35, "X")])
        result = extract_tocr_from_page(page, 0, cfg=None, mode="minimal")
        assert len(result.tokens) == 1


class TestExtractTocrFromPageFull:
    """Integration: full mode via mock page."""

    def test_basic_extraction(self):
        words = [
            _word(10, 20, 50, 35, "HELLO"),
            _word(60, 20, 100, 35, "WORLD"),
        ]
        page = _make_mock_page(words)
        result = extract_tocr_from_page(page, 0, mode="full")
        assert len(result.tokens) == 2
        assert result.diagnostics["tokens_raw"] == 2
        assert result.diagnostics["tokens_total"] == 2
        assert result.diagnostics["font_names"]["Arial"] == 2

    def test_control_char_filtered(self):
        words = [_word(10, 20, 50, 35, "HE\x01LLO")]
        page = _make_mock_page(words)
        result = extract_tocr_from_page(page, 0, mode="full")
        assert result.tokens[0].text == "HELLO"
        assert result.diagnostics["tokens_control_char_cleaned"] == 1

    def test_dedup_applied(self):
        words = [
            _word(10, 20, 50, 35, "DUP"),
            _word(10, 20, 50, 35, "DUP"),
        ]
        page = _make_mock_page(words)
        result = extract_tocr_from_page(page, 0, mode="full")
        assert len(result.tokens) == 1
        assert result.diagnostics["tokens_duplicate_removed"] == 1

    def test_font_size_filter(self):
        words = [
            _word(10, 20, 50, 35, "SMALL", size=2.0),
            _word(60, 20, 100, 35, "NORMAL", size=12.0),
        ]
        page = _make_mock_page(words)
        cfg = _cfg(tocr_min_font_size=6.0)
        result = extract_tocr_from_page(page, 0, cfg, mode="full")
        assert len(result.tokens) == 1
        assert result.tokens[0].text == "NORMAL"
        assert result.diagnostics["tokens_font_size_filtered"] == 1

    def test_margin_filter(self):
        words = [
            _word(5, 400, 15, 415, "EDGE"),  # centre x = 10, within 20pt margin
            _word(100, 400, 200, 415, "MIDDLE"),
        ]
        page = _make_mock_page(words)
        cfg = _cfg(tocr_margin_pts=20.0)
        result = extract_tocr_from_page(page, 0, cfg, mode="full")
        assert len(result.tokens) == 1
        assert result.tokens[0].text == "MIDDLE"

    def test_whitespace_filter(self):
        words = [
            _word(10, 20, 50, 35, "   "),
            _word(60, 20, 100, 35, "OK"),
        ]
        page = _make_mock_page(words)
        result = extract_tocr_from_page(page, 0, mode="full")
        assert len(result.tokens) == 1
        assert result.tokens[0].text == "OK"

    def test_density_diagnostic(self):
        """Page area and token density are computed."""
        # Use enough tokens so density rounds above 0.0 at 1-decimal precision
        words = [_word(10 + i * 50, 20, 50 + i * 50, 35, f"W{i}") for i in range(100)]
        page = _make_mock_page(words, width=612, height=792)
        result = extract_tocr_from_page(page, 0, mode="full")
        # page area = (612/72) * (792/72) = 8.5 * 11 = 93.5 sq-in
        assert result.diagnostics["page_area_sqin"] == pytest.approx(93.5, abs=0.1)
        # 100 tokens / 93.5 ≈ 1.1
        assert result.diagnostics["token_density_per_sqin"] > 0

    def test_empty_page(self):
        page = _make_mock_page([])
        result = extract_tocr_from_page(page, 0, mode="full")
        assert len(result.tokens) == 0
        assert result.diagnostics["tokens_total"] == 0
        assert result.diagnostics["tokens_raw"] == 0


# ── extract_tocr_page ─────────────────────────────────────────────────


class TestExtractTocrPage:
    """Top-level extraction from file path (error handling path)."""

    def test_invalid_path_returns_error_result(self, tmp_path):
        """Non-existent PDF → empty result with error diagnostics."""
        result = extract_tocr_page(
            tmp_path / "nonexistent.pdf",
            0,
            mode="full",
        )
        assert len(result.tokens) == 0
        assert result.diagnostics["error"] is not None

    def test_invalid_path_error_diagnostics_schema(self, tmp_path):
        """Error result still has full diagnostics schema."""
        result = extract_tocr_page(tmp_path / "missing.pdf", 0, mode="full")
        required = TestEmptyDiagnostics.REQUIRED_KEYS
        assert required.issubset(set(result.diagnostics.keys()))


# ── Mode parity sanity checks ─────────────────────────────────────────


class TestModeParity:
    """Verify minimal vs full mode behave differently where expected."""

    def _run_both_modes(self, words, cfg=None):
        page = _make_mock_page(words)
        cfg = cfg or _cfg()
        r_min = extract_tocr_from_page(page, 0, cfg, mode="minimal")
        r_full = extract_tocr_from_page(page, 0, cfg, mode="full")
        return r_min, r_full

    def test_control_chars_only_cleaned_in_full(self):
        words = [_word(10, 20, 50, 35, "AB\x01CD")]
        r_min, r_full = self._run_both_modes(words)
        assert r_min.tokens[0].text == "AB\x01CD"
        assert r_full.tokens[0].text == "ABCD"

    def test_whitespace_only_filtered_in_full(self):
        words = [_word(10, 20, 50, 35, "   ")]
        r_min, r_full = self._run_both_modes(words)
        assert len(r_min.tokens) == 1  # kept
        assert len(r_full.tokens) == 0  # filtered

    def test_dedup_only_in_full(self):
        words = [
            _word(10, 20, 50, 35, "SAME"),
            _word(10, 20, 50, 35, "SAME"),
        ]
        r_min, r_full = self._run_both_modes(words)
        assert len(r_min.tokens) == 2  # no dedup
        assert len(r_full.tokens) == 1  # deduped

    def test_font_size_only_filtered_in_full(self):
        words = [_word(10, 20, 50, 35, "TINY", size=1.0)]
        cfg = _cfg(tocr_min_font_size=6.0)
        r_min, r_full = self._run_both_modes(words, cfg)
        assert len(r_min.tokens) == 1  # kept
        assert len(r_full.tokens) == 0  # filtered

    def test_margin_only_filtered_in_full(self):
        words = [_word(2, 400, 8, 415, "EDGE")]
        cfg = _cfg(tocr_margin_pts=20.0)
        r_min, r_full = self._run_both_modes(words, cfg)
        assert len(r_min.tokens) == 1  # kept (minimal only clips coords)
        assert len(r_full.tokens) == 0  # filtered by margin

    def test_case_fold_only_in_full(self):
        words = [_word(10, 20, 50, 35, "Hello")]
        cfg = _cfg(tocr_case_fold=True)
        r_min, r_full = self._run_both_modes(words, cfg)
        assert r_min.tokens[0].text == "Hello"
        assert r_full.tokens[0].text == "hello"

    def test_unicode_normalise_only_in_full(self):
        words = [_word(10, 20, 50, 35, "ﬁnd")]
        cfg = _cfg(tocr_normalize_unicode=True)
        r_min, r_full = self._run_both_modes(words, cfg)
        assert r_min.tokens[0].text == "ﬁnd"
        assert r_full.tokens[0].text == "find"

    def test_both_modes_share_page_dimensions(self):
        words = [_word(10, 20, 50, 35, "X")]
        r_min, r_full = self._run_both_modes(words)
        assert r_min.page_width == r_full.page_width == 612
        assert r_min.page_height == r_full.page_height == 792

    def test_both_modes_have_diagnostics(self):
        words = [_word(10, 20, 50, 35, "X")]
        r_min, r_full = self._run_both_modes(words)
        required = TestEmptyDiagnostics.REQUIRED_KEYS
        assert required.issubset(set(r_min.diagnostics.keys()))
        assert required.issubset(set(r_full.diagnostics.keys()))
