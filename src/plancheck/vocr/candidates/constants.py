"""Constants for VOCR candidate detection.

Symbol/keyword mappings and target symbol sets used by detection methods.
"""

from __future__ import annotations

from typing import Dict, Set

# ── Symbol expectations by keyword ─────────────────────────────────────

_ANGLE_KEYWORDS: Set[str] = {"ANGLE", "BEND", "DEG", "SLOPE", "TYP", "TYPICAL"}
_TOLERANCE_KEYWORDS: Set[str] = {"ELEV", "EL", "TOL", "TOLERANCE"}
_DIAMETER_KEYWORDS: Set[str] = {"DIA", "DIAM", "DIAMETER"}
_CENTERLINE_KEYWORDS: Set[str] = {"CENTER", "CL", "C/L", "CENTERLINE"}

_KEYWORD_SYMBOL_MAP: Dict[str, str] = {}
for _kw in _ANGLE_KEYWORDS:
    _KEYWORD_SYMBOL_MAP[_kw] = "°"
for _kw in _TOLERANCE_KEYWORDS:
    _KEYWORD_SYMBOL_MAP[_kw] = "±"
for _kw in _DIAMETER_KEYWORDS:
    _KEYWORD_SYMBOL_MAP[_kw] = "Ø"

# Template adjacency: digit + gap + keyword → expected symbol
_TEMPLATE_KEYWORDS: Dict[str, str] = {
    **{k: "°" for k in ("TYP", "TYP.", "TYP)")},
    **{k: "Ø" for k in ("DIA", "DIAM", "DIAM.", "DIAMETER")},
    **{k: "±" for k in ("TOL", "TOLERANCE")},
}

# Co-occurrence pairs: keyword → expected partner symbol
_COOCCURRENCE: Dict[str, str] = {
    "OC": "@",
    "O.C.": "@",
    "O.C": "@",
    "BAR": "#",
    "REBAR": "#",
}

# Expanded symbol set we care about
_TARGET_SYMBOLS: Set[str] = set("%/°±Ø×'\"#@⌀")
