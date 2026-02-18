"""Backward compatibility — imports split between plancheck.vocr, plancheck.reconcile, and plancheck.export."""

from .export.reconcile_overlay import (
    draw_reconcile_debug,  # noqa: F401
    draw_symbol_overlay,
)
from .reconcile.reconcile import (
    _RE_AFTER_DIGIT,  # noqa: F401
    _RE_SLASH_NUMERIC,
    MatchRecord,
    ReconcileResult,
    SymbolCandidate,
    _accept_candidates,
    _build_match_index,
    _center,
    _estimate_char_width,
    _extra_symbols,
    _find_best_match,
    _find_line_neighbours,
    _generate_symbol_candidates,
    _has_allowed_symbol,
    _has_digit_neighbour_left,
    _has_numeric_symbol_context,
    _inject_symbols,
    _is_digit_group,
    _overlap_ratio,
    _overlaps_existing,
    reconcile_ocr,
)
from .vocr.extract import (
    _PADDLE_MAX_SIDE_DEFAULT,  # noqa: F401
    _TILE_OVERLAP_FRAC_DEFAULT,
    _dedup_tiles,
    _extract_ocr_tokens,
    _iou,
    _ocr_one_tile,
    extract_vocr_tokens,
)
