"""Dual-source OCR reconciliation: spatial matching and symbol injection.

Spatially aligns OCR tokens against existing PDF-text tokens, and injects
only the missing special-character tokens (%, /, °, ±) that the PDF text
layer is missing.

Public API
----------
reconcile_ocr          – run the 4-stage reconciliation pipeline
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from ..config import GroupingConfig
from ..models import GlyphBox
from ..vocr.extract import extract_ocr_tokens
from .confidence_scoring import (
    _build_match_index,
    _build_spatial_grid,
    _find_best_match,
    _grid_candidates,
    _overlap_ratio,
    _overlaps_existing,
)
from .helpers import (
    MatchRecord,
    SymbolCandidate,
    _estimate_char_width,
    _extra_symbols,
    _find_line_neighbours,
    _has_digit_neighbour_left,
    _is_digit_group,
    center,
    has_allowed_symbol,
    has_numeric_symbol_context,
)
from .symbol_injection import (
    _accept_candidates,
    _collect_candidate_results,
    _generate_symbol_candidates,
    _inject_symbols,
    _try_inject_case_a,
    _try_inject_case_b,
    _try_inject_case_c,
)

log = logging.getLogger(__name__)

# ── Backward-compatible private aliases ───────────────────────────────
_center = center
_has_allowed_symbol = has_allowed_symbol
_has_numeric_symbol_context = has_numeric_symbol_context

# ── Data structures ────────────────────────────────────────────────────


@dataclass
class ReconcileResult:
    """Aggregated output of the reconciliation pipeline."""

    added_tokens: List[GlyphBox] = field(default_factory=list)
    all_ocr_tokens: List[GlyphBox] = field(default_factory=list)
    matches: List[MatchRecord] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "added_tokens": [t.to_dict() for t in self.added_tokens],
            "all_ocr_tokens": [t.to_dict() for t in self.all_ocr_tokens],
            "matches": [m.to_dict() for m in self.matches],
            "stats": dict(self.stats),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReconcileResult":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        return cls(
            added_tokens=[GlyphBox.from_dict(t) for t in d.get("added_tokens", [])],
            all_ocr_tokens=[GlyphBox.from_dict(t) for t in d.get("all_ocr_tokens", [])],
            matches=[MatchRecord.from_dict(m) for m in d.get("matches", [])],
            stats=d.get("stats", {}),
        )


# ── Reconcile public entry point ──────────────────────────────────────


def reconcile_ocr(
    page_image,
    tokens: List[GlyphBox],
    page_num: int,
    page_width: float,
    page_height: float,
    cfg: GroupingConfig,
    *,
    ocr_tokens: Optional[List[GlyphBox]] = None,
    ocr_confs: Optional[List[float]] = None,
) -> ReconcileResult:
    """Run full-page OCR reconciliation against existing PDF tokens.

    Parameters
    ----------
    page_image : PIL.Image.Image
        Full-page render at ``cfg.ocr_reconcile_resolution`` DPI.
        Used for VOCR extraction when *ocr_tokens* is not supplied.
    tokens : list[GlyphBox]
        Existing PDF-extracted tokens (origin="text").
    page_num : int
        Zero-based page index (for GlyphBox.page).
    page_width, page_height : float
        Page dimensions in PDF points.
    cfg : GroupingConfig
        Configuration (OCR reconcile settings).
    ocr_tokens : list[GlyphBox], optional
        Pre-extracted VOCR tokens (from :func:`extract_vocr_tokens`).
        When provided, the internal VOCR extraction is skipped.
    ocr_confs : list[float], optional
        Parallel confidence scores for *ocr_tokens*.

    Returns
    -------
    ReconcileResult
        Contains ``added_tokens`` to extend the main token list.
    """
    result = ReconcileResult()

    # Stage 1 — full-page OCR (skipped when caller supplies tokens)
    if ocr_tokens is not None:
        if ocr_confs is None:
            ocr_confs = [1.0] * len(ocr_tokens)
    else:
        ocr_tokens, ocr_confs = extract_ocr_tokens(
            page_image,
            page_num,
            page_width,
            page_height,
            cfg,
        )
    result.all_ocr_tokens = ocr_tokens

    if not ocr_tokens:
        result.stats = {
            "ocr_total": 0,
            "with_symbol": 0,
            "matched": 0,
            "unmatched": 0,
            "accepted": 0,
            "symbols": "",
        }
        log.info("OCR reconcile: 0 OCR tokens detected")
        return result

    # Stage 2 — spatial alignment
    pdf_tokens = [t for t in tokens if t.origin == "text"]
    matches = _build_match_index(ocr_tokens, ocr_confs, pdf_tokens, cfg)
    result.matches = matches

    # Stage 3 — symbol injection
    added, debug_log, n_filtered_non_numeric = _inject_symbols(
        matches, pdf_tokens, cfg, page_width
    )
    result.added_tokens = added

    # Stats
    allowed = cfg.ocr_reconcile_allowed_symbols
    with_symbol = sum(
        1 for m in matches if _has_allowed_symbol(m.ocr_box.text, allowed)
    )
    matched = sum(1 for m in matches if m.match_type in ("iou", "center"))
    unmatched = sum(1 for m in matches if m.match_type == "unmatched")
    symbol_summary = ", ".join(f"{t.text}" for t in added) if added else "(none)"

    # Count candidates from debug_log
    n_candidates = sum(len(e.get("candidates", [])) for e in debug_log)
    n_accepted_c = sum(
        1
        for e in debug_log
        for c in e.get("candidates", [])
        if c.get("status") == "accepted"
    )
    n_rejected_c = n_candidates - n_accepted_c

    result.stats = {
        "ocr_total": len(ocr_tokens),
        "with_symbol": with_symbol,
        "matched": matched,
        "unmatched": unmatched,
        "accepted": len(added),
        "symbols": symbol_summary,
        "candidates_generated": n_candidates,
        "candidates_accepted": n_accepted_c,
        "candidates_rejected": n_rejected_c,
        "filtered_non_numeric": n_filtered_non_numeric,
        "injection_log": debug_log,
    }

    log.info(
        "OCR reconcile: %d OCR tokens -> %d with symbol -> %d accepted (%s) "
        "[candidates: %d gen, %d ok, %d rej, %d filtered]",
        len(ocr_tokens),
        with_symbol,
        len(added),
        symbol_summary,
        n_candidates,
        n_accepted_c,
        n_rejected_c,
        n_filtered_non_numeric,
    )

    return result
