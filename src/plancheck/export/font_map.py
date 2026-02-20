"""Font-name mapping for sheet recreation.

Maps PDF-internal font names (e.g. ``BCDFEE+ArialMT``, ``RomanT``) to
ReportLab built-in font names (``Helvetica``, ``Courier``, etc.) so the
recreation export can render text without bundling .ttf files.

Public API
----------
* ``strip_subset_prefix(fontname)`` – remove 6-letter ``+`` prefix
* ``resolve_font(fontname)`` – full mapping to a ReportLab font name
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Subset-prefix stripping
# ---------------------------------------------------------------------------

_SUBSET_RE = re.compile(r"^[A-Z]{6}\+")


def strip_subset_prefix(fontname: str) -> str:
    """Remove a 6-uppercase-letter subset prefix (e.g. ``BCDFEE+``)."""
    return _SUBSET_RE.sub("", fontname)


# ---------------------------------------------------------------------------
# PDF fontname → ReportLab base font mapping
# ---------------------------------------------------------------------------

# Checked in order; first substring match wins.
# Each entry: (substring_to_match_lowercased, rl_base_family)
_FAMILY_MAP: list[tuple[str, str]] = [
    ("courier", "Courier"),
    ("mono", "Courier"),
    ("consolas", "Courier"),
    ("arial", "Helvetica"),
    ("helvetica", "Helvetica"),
    ("calibri", "Helvetica"),
    ("verdana", "Helvetica"),
    ("tahoma", "Helvetica"),
    ("trebuchet", "Helvetica"),
    ("times", "Times-Roman"),
    ("georgia", "Times-Roman"),
    ("cambria", "Times-Roman"),
    ("garamond", "Times-Roman"),
    ("palatino", "Times-Roman"),
]

# AutoCAD / SHX stroke fonts → monospaced fallback
_CAD_FONTS = frozenset(
    {
        "romant",
        "simplex",
        "simplext",
        "complex",
        "complext",
        "txt",
        "txts",
        "monotxt",
        "romans",
        "scripts",
        "scriptc",
        "italic",
        "italict",
        "gothice",
        "gothicg",
        "gothici",
        "greeks",
        "greekc",
        "symath",
        "symap",
        "symeteo",
        "symusic",
    }
)

_DEFAULT_FONT = "Helvetica"

# ReportLab built-in style variants
_BOLD_ITALIC: dict[str, str] = {
    "Helvetica": "Helvetica-BoldOblique",
    "Courier": "Courier-BoldOblique",
    "Times-Roman": "Times-BoldItalic",
}
_BOLD: dict[str, str] = {
    "Helvetica": "Helvetica-Bold",
    "Courier": "Courier-Bold",
    "Times-Roman": "Times-Bold",
}
_ITALIC: dict[str, str] = {
    "Helvetica": "Helvetica-Oblique",
    "Courier": "Courier-Oblique",
    "Times-Roman": "Times-Italic",
}


def resolve_font(fontname: str) -> str:
    """Map a PDF ``fontname`` to a ReportLab built-in font name.

    Resolution order:
    1. Strip optional subset prefix (``BCDFEE+``).
    2. Check for known CAD/SHX fonts → ``Courier``.
    3. Substring-match against ``_FAMILY_MAP``.
    4. Fall back to ``Helvetica``.
    5. Apply bold / italic modifiers if present in the original name.
    """
    if not fontname:
        return _DEFAULT_FONT

    clean = strip_subset_prefix(fontname)
    lower = clean.lower()

    # --- CAD / SHX fonts -------------------------------------------------
    # Match the first alphanumeric token (before any dash/space)
    base_token = re.split(r"[\-\s]", lower, maxsplit=1)[0]
    if base_token in _CAD_FONTS:
        return "Courier"

    # --- Family matching --------------------------------------------------
    family = _DEFAULT_FONT
    for substr, rl_family in _FAMILY_MAP:
        if substr in lower:
            family = rl_family
            break

    # --- Style modifiers --------------------------------------------------
    is_bold = "bold" in lower
    is_italic = "italic" in lower or "oblique" in lower

    if is_bold and is_italic:
        return _BOLD_ITALIC.get(family, family)
    if is_bold:
        return _BOLD.get(family, family)
    if is_italic:
        return _ITALIC.get(family, family)

    return family
