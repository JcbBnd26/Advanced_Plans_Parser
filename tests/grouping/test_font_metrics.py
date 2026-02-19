"""Tests for plancheck.font_metrics — dataclass logic and pure-math helpers."""

import json
import string

from plancheck.grouping.font_metrics import (
    EXPECTED_WIDTH_RATIOS,
    FontMetricsAnalyzer,
    FontMetricsAnomaly,
    PageMetricsReport,
    VisualMetricsReport,
    WordVisualAnomaly,
    save_metrics_report,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _make_char(text: str, font_size: float, width: float) -> dict:
    """Return a synthetic pdfplumber-style char dict."""
    return {"text": text, "size": font_size, "width": width}


def _make_anomaly(
    inflation: float = 1.5,
    confidence: float = 0.9,
    fontname: str = "TestFont",
    **kwargs,
) -> FontMetricsAnomaly:
    return FontMetricsAnomaly(
        fontname=fontname,
        sample_count=kwargs.get("sample_count", 20),
        avg_width_ratio=kwargs.get("avg_width_ratio", 0.55),
        inflation_factor=inflation,
        confidence=confidence,
        sample_chars=kwargs.get("sample_chars", list("abcde")),
        detection_method=kwargs.get("detection_method", "heuristic"),
    )


# ── FontMetricsAnomaly ──────────────────────────────────────────────


class TestFontMetricsAnomaly:
    def test_is_anomalous_true_when_inflated_and_confident(self):
        a = _make_anomaly(inflation=1.5, confidence=0.9)
        assert a.is_anomalous() is True

    def test_is_anomalous_false_when_inflation_below_threshold(self):
        a = _make_anomaly(inflation=1.1, confidence=0.9)
        assert a.is_anomalous() is False

    def test_is_anomalous_false_when_confidence_below_min(self):
        a = _make_anomaly(inflation=1.5, confidence=0.5)
        assert a.is_anomalous() is False

    def test_to_dict_keys_and_types(self):
        chars = list("abcdefghijklmno")  # 15 chars
        a = _make_anomaly(sample_chars=chars)
        d = a.to_dict()

        expected_keys = {
            "fontname",
            "sample_count",
            "avg_width_ratio",
            "inflation_factor",
            "confidence",
            "is_anomalous",
            "detection_method",
            "sample_chars",
        }
        assert set(d.keys()) == expected_keys
        assert isinstance(d["avg_width_ratio"], float)
        assert isinstance(d["inflation_factor"], float)
        assert isinstance(d["confidence"], float)
        # sample_chars truncated to first 10
        assert len(d["sample_chars"]) == 10


# ── PageMetricsReport ───────────────────────────────────────────────


class TestPageMetricsReport:
    def test_has_anomalies_true(self):
        report = PageMetricsReport(
            page_num=1,
            font_anomalies={"FontA": _make_anomaly(inflation=2.0, confidence=0.9)},
        )
        assert report.has_anomalies() is True

    def test_has_anomalies_false(self):
        report = PageMetricsReport(
            page_num=1,
            font_anomalies={"FontA": _make_anomaly(inflation=1.0, confidence=0.9)},
        )
        assert report.has_anomalies() is False

    def test_get_anomalous_fonts_returns_only_anomalous(self):
        report = PageMetricsReport(
            page_num=1,
            font_anomalies={
                "BadFont": _make_anomaly(
                    inflation=2.0, confidence=0.9, fontname="BadFont"
                ),
                "GoodFont": _make_anomaly(
                    inflation=1.0, confidence=0.9, fontname="GoodFont"
                ),
            },
        )
        assert report.get_anomalous_fonts() == ["BadFont"]

    def test_get_correction_factor_anomalous(self):
        report = PageMetricsReport(
            page_num=1,
            font_anomalies={"FontA": _make_anomaly(inflation=2.0, confidence=0.9)},
        )
        assert report.get_correction_factor("FontA") == 0.5

    def test_get_correction_factor_unknown_font(self):
        report = PageMetricsReport(page_num=1)
        assert report.get_correction_factor("NoSuchFont") == 1.0

    def test_to_dict_structure(self):
        report = PageMetricsReport(
            page_num=3,
            font_anomalies={"FontA": _make_anomaly()},
            total_chars_analyzed=100,
            anomalous_char_count=20,
        )
        d = report.to_dict()
        expected_keys = {
            "page_num",
            "total_chars_analyzed",
            "anomalous_char_count",
            "has_anomalies",
            "anomalous_fonts",
            "font_details",
        }
        assert set(d.keys()) == expected_keys
        assert d["page_num"] == 3
        assert "FontA" in d["font_details"]
        assert isinstance(d["font_details"]["FontA"], dict)


# ── WordVisualAnomaly ───────────────────────────────────────────────


def _make_word_anomaly(inflation: float = 1.5) -> WordVisualAnomaly:
    return WordVisualAnomaly(
        text="TEST",
        fontname="VisFont",
        reported_bbox=(10.0, 20.0, 50.0, 30.0),
        visual_bbox=(10.0, 20.0, 40.0, 30.0),
        reported_width=40.0,
        visual_width=30.0,
        inflation_factor=inflation,
        overhang_percent=25.0,
    )


class TestWordVisualAnomaly:
    def test_is_anomalous_true(self):
        w = _make_word_anomaly(inflation=1.5)
        assert w.is_anomalous() is True

    def test_is_anomalous_false(self):
        w = _make_word_anomaly(inflation=1.2)
        assert w.is_anomalous() is False

    def test_to_dict_keys_and_bbox_types(self):
        w = _make_word_anomaly()
        d = w.to_dict()

        expected_keys = {
            "text",
            "fontname",
            "reported_bbox",
            "visual_bbox",
            "reported_width",
            "visual_width",
            "inflation_factor",
            "overhang_percent",
            "is_anomalous",
        }
        assert set(d.keys()) == expected_keys
        assert isinstance(d["reported_bbox"], list)
        assert all(isinstance(v, float) for v in d["reported_bbox"])
        assert isinstance(d["visual_bbox"], list)
        assert all(isinstance(v, float) for v in d["visual_bbox"])
        # Rounding checks
        assert d["reported_width"] == 40.0
        assert d["visual_width"] == 30.0
        assert d["inflation_factor"] == 1.5
        assert d["overhang_percent"] == 25.0


# ── VisualMetricsReport ─────────────────────────────────────────────


class TestVisualMetricsReport:
    def test_has_anomalies_true(self):
        report = VisualMetricsReport(
            page_num=1,
            resolution=300,
            word_anomalies=[_make_word_anomaly(inflation=1.5)],
        )
        assert report.has_anomalies() is True

    def test_has_anomalies_false_empty(self):
        report = VisualMetricsReport(page_num=1, resolution=300)
        assert report.has_anomalies() is False

    def test_get_anomalous_fonts_deduplicated(self):
        w1 = _make_word_anomaly(inflation=1.5)
        w2 = _make_word_anomaly(inflation=1.8)
        # Both share fontname="VisFont"
        report = VisualMetricsReport(
            page_num=1,
            resolution=300,
            word_anomalies=[w1, w2],
        )
        names = report.get_anomalous_fonts()
        assert names == ["VisFont"]

    def test_get_correction_factor(self):
        report = VisualMetricsReport(
            page_num=1,
            resolution=300,
            font_inflation_factors={"FontX": 2.0, "FontY": 1.1},
        )
        assert report.get_correction_factor("FontX") == 0.5
        # FontY is below the hard-coded 1.3 threshold
        assert report.get_correction_factor("FontY") == 1.0
        # Unknown font
        assert report.get_correction_factor("FontZ") == 1.0


# ── FontMetricsAnalyzer._analyze_font ───────────────────────────────


class TestAnalyzeFont:
    """Tests for the pure-math _analyze_font method."""

    def _build_normal_chars(self, count: int = 10) -> list:
        """Build char dicts where width == expected_ratio * font_size (inflation ≈ 1.0)."""
        chars = []
        font_size = 12.0
        for ch in list(EXPECTED_WIDTH_RATIOS.keys())[:count]:
            expected = EXPECTED_WIDTH_RATIOS[ch]
            chars.append(_make_char(ch, font_size, expected * font_size))
        return chars

    def _build_inflated_chars(self, multiplier: float = 2.0, count: int = 10) -> list:
        """Build char dicts with widths inflated by *multiplier*."""
        chars = []
        font_size = 12.0
        for ch in list(EXPECTED_WIDTH_RATIOS.keys())[:count]:
            expected = EXPECTED_WIDTH_RATIOS[ch]
            chars.append(_make_char(ch, font_size, expected * font_size * multiplier))
        return chars

    def test_normal_font_not_anomalous(self):
        analyzer = FontMetricsAnalyzer()
        result = analyzer._analyze_font("NormalFont", self._build_normal_chars())

        assert result is not None
        assert abs(result.inflation_factor - 1.0) < 0.05
        assert result.is_anomalous() is False

    def test_inflated_font_flagged(self):
        analyzer = FontMetricsAnalyzer()
        result = analyzer._analyze_font("BigFont", self._build_inflated_chars(2.0))

        assert result is not None
        assert result.inflation_factor > 1.3
        assert result.is_anomalous() is True

    def test_too_few_samples_returns_none(self):
        analyzer = FontMetricsAnalyzer(min_samples=5)
        # Only 3 qualifying chars
        chars = self._build_normal_chars(count=3)
        result = analyzer._analyze_font("SmallFont", chars)

        assert result is None

    def test_high_variance_low_confidence(self):
        analyzer = FontMetricsAnalyzer()
        font_size = 12.0
        chars = []
        keys = list(EXPECTED_WIDTH_RATIOS.keys())
        for i, ch in enumerate(keys[:10]):
            expected = EXPECTED_WIDTH_RATIOS[ch]
            # Alternate between 0.5× and 3.0× inflation
            mult = 0.5 if i % 2 == 0 else 3.0
            chars.append(_make_char(ch, font_size, expected * font_size * mult))

        result = analyzer._analyze_font("NoisyFont", chars)

        assert result is not None
        assert result.confidence < 0.7
        assert result.is_anomalous() is False


# ── FontMetricsAnalyzer.correct_box_width ────────────────────────────


class TestCorrectBoxWidth:
    def _make_report_with_anomalous_font(self) -> PageMetricsReport:
        return PageMetricsReport(
            page_num=1,
            font_anomalies={
                "BadFont": _make_anomaly(
                    inflation=2.0, confidence=0.9, fontname="BadFont"
                ),
            },
        )

    def test_anomalous_font_shrinks_x1(self):
        analyzer = FontMetricsAnalyzer()
        report = self._make_report_with_anomalous_font()
        x0, x1 = analyzer.correct_box_width(100.0, 200.0, "BadFont", report)

        assert x0 == 100.0
        assert x1 == 150.0  # 100 + (200-100)*0.5

    def test_normal_font_unchanged(self):
        analyzer = FontMetricsAnalyzer()
        report = self._make_report_with_anomalous_font()
        x0, x1 = analyzer.correct_box_width(100.0, 200.0, "CleanFont", report)

        assert x0 == 100.0
        assert x1 == 200.0


# ── save_metrics_report ──────────────────────────────────────────────


def test_save_metrics_report(tmp_path):
    report = PageMetricsReport(
        page_num=1,
        font_anomalies={"FontA": _make_anomaly()},
        total_chars_analyzed=50,
        anomalous_char_count=10,
    )
    out = tmp_path / "metrics.json"
    save_metrics_report(report, out)

    loaded = json.loads(out.read_text())
    assert loaded == report.to_dict()


# ── EXPECTED_WIDTH_RATIOS ────────────────────────────────────────────


def test_expected_width_ratios_coverage_and_range():
    expected_chars = set(
        string.ascii_uppercase + string.ascii_lowercase + string.digits
    )
    assert set(EXPECTED_WIDTH_RATIOS.keys()) == expected_chars

    for ch, ratio in EXPECTED_WIDTH_RATIOS.items():
        assert isinstance(ratio, float), f"{ch} ratio is not float"
        assert 0.0 < ratio < 1.0, f"{ch} ratio {ratio} out of range"
