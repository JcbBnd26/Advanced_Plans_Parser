"""Tests for plancheck.export.font_map — PDF font → ReportLab resolution."""

import pytest

from plancheck.export.font_map import resolve_font, strip_subset_prefix


class TestStripSubsetPrefix:
    def test_standard_prefix(self):
        assert strip_subset_prefix("BCDFEE+ArialMT") == "ArialMT"

    def test_no_prefix(self):
        assert strip_subset_prefix("ArialMT") == "ArialMT"

    def test_lowercase_not_stripped(self):
        # Only uppercase 6-letter prefix is a valid subset tag
        assert strip_subset_prefix("abcdef+ArialMT") == "abcdef+ArialMT"

    def test_short_prefix_not_stripped(self):
        assert strip_subset_prefix("ABC+ArialMT") == "ABC+ArialMT"

    def test_empty_string(self):
        assert strip_subset_prefix("") == ""

    def test_only_prefix(self):
        # Edge case: prefix with nothing after it
        assert strip_subset_prefix("ABCDEF+") == ""

    def test_multiple_plus(self):
        assert strip_subset_prefix("ABCDEF+Times+Roman") == "Times+Roman"


class TestResolveFontFamilies:
    """Test _FAMILY_MAP substring matching."""

    @pytest.mark.parametrize(
        "fontname,expected",
        [
            ("CourierNew", "Courier"),
            ("Courier-Bold", "Courier-Bold"),
            ("LucidaConsole-mono", "Courier"),
            ("Consolas", "Courier"),
        ],
    )
    def test_courier_family(self, fontname, expected):
        assert resolve_font(fontname) == expected

    @pytest.mark.parametrize(
        "fontname,expected",
        [
            ("ArialMT", "Helvetica"),
            ("Helvetica", "Helvetica"),
            ("Calibri", "Helvetica"),
            ("Verdana", "Helvetica"),
            ("Tahoma", "Helvetica"),
            ("TrebuchetMS", "Helvetica"),
        ],
    )
    def test_helvetica_family(self, fontname, expected):
        assert resolve_font(fontname) == expected

    @pytest.mark.parametrize(
        "fontname,expected",
        [
            ("TimesNewRomanPSMT", "Times-Roman"),
            ("Georgia", "Times-Roman"),
            ("Cambria", "Times-Roman"),
            ("Garamond", "Times-Roman"),
            ("Palatino", "Times-Roman"),
        ],
    )
    def test_times_family(self, fontname, expected):
        assert resolve_font(fontname) == expected


class TestResolveFontCAD:
    """AutoCAD/SHX stroke fonts → Courier."""

    @pytest.mark.parametrize(
        "fontname",
        ["simplex", "romans", "txt", "monotxt", "gothice", "scripts"],
    )
    def test_cad_fonts(self, fontname):
        assert resolve_font(fontname) == "Courier"


class TestResolveFontModifiers:
    def test_bold(self):
        assert resolve_font("Arial-Bold") == "Helvetica-Bold"

    def test_italic(self):
        assert resolve_font("Arial-Italic") == "Helvetica-Oblique"

    def test_bold_italic(self):
        assert resolve_font("Arial-BoldItalic") == "Helvetica-BoldOblique"

    def test_oblique(self):
        assert resolve_font("Helvetica-Oblique") == "Helvetica-Oblique"

    def test_times_bold(self):
        assert resolve_font("TimesNewRoman-Bold") == "Times-Bold"

    def test_times_italic(self):
        assert resolve_font("TimesNewRoman-Italic") == "Times-Italic"

    def test_courier_bold(self):
        assert resolve_font("Courier-Bold") == "Courier-Bold"

    def test_courier_bold_oblique(self):
        assert resolve_font("Courier-BoldOblique") == "Courier-BoldOblique"


class TestResolveFontFallback:
    def test_unknown_font(self):
        assert resolve_font("TotallyUnknownFont") == "Helvetica"

    def test_empty_string(self):
        assert resolve_font("") == "Helvetica"


class TestResolveFontSubsetPrefix:
    def test_subset_prefix_stripped(self):
        assert resolve_font("BCDFEE+ArialMT") == "Helvetica"

    def test_subset_prefix_with_bold(self):
        assert resolve_font("ABCDEF+Arial-Bold") == "Helvetica-Bold"
