"""Public API for VOCR candidate detection and statistics."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .helpers import _group_by_baseline
from .merge import _merge_overlapping_candidates
from .methods import (
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
)

if TYPE_CHECKING:
    from plancheck.config import GroupingConfig
    from plancheck.models import GlyphBox, VocrCandidate

__all__ = ["detect_vocr_candidates", "compute_candidate_stats"]

log = logging.getLogger(__name__)


def detect_vocr_candidates(
    tokens: List[GlyphBox],
    page_chars: List[dict],
    page_lines: List[dict],
    page_curves: List[dict],
    page_rects: List[dict],
    page_width: float,
    page_height: float,
    page_num: int,
    cfg: GroupingConfig,
    method_stats: Optional[Dict[str, Any]] = None,
    producer_stats: Optional[Dict[str, Any]] = None,
    producer_id: str = "",
) -> List[VocrCandidate]:
    """Run all detection methods and return a merged, capped candidate list.

    Parameters
    ----------
    tokens : list[GlyphBox]
        TOCR-extracted tokens (origin="text").
    page_chars : list[dict]
        Character-level dicts from ``page.chars`` (pdfplumber).
    page_lines, page_curves, page_rects : list[dict]
        Vector graphic elements from pdfplumber.
    page_width, page_height : float
        Page dimensions in PDF points.
    page_num : int
        Zero-based page index.
    cfg : GroupingConfig
        Pipeline configuration.
    method_stats : dict | None
        Persistent per-method hit/miss stats loaded via
        :func:`~plancheck.vocr.method_stats.load_method_stats`.
        When provided, each candidate's confidence is overridden with
        data-driven adaptive confidence.
    producer_stats : dict | None
        Per-producer method stats (Level 3).  When provided together
        with a non-empty *producer_id*, producer-specific hit rates
        override the global adaptive confidence.
    producer_id : str
        PDF ``/Producer`` metadata string.

    Returns
    -------
    list[VocrCandidate]
        Merged candidates sorted by confidence descending, capped at
        ``cfg.vocr_cand_max_candidates``.
    """
    m = cfg.vocr_cand_patch_margin
    pw, ph = page_width, page_height
    pn = page_num

    # Cache baseline-grouped lines once for all detection methods (efficiency fix)
    baseline_lines = _group_by_baseline(tokens)

    all_candidates: List[VocrCandidate] = []

    # --- Tier 1: exact-location signals ---
    all_candidates.extend(_detect_char_encoding_failures(page_chars, pn, pw, ph, m))
    all_candidates.extend(_detect_placeholder_tokens(tokens, pn, pw, ph, m))

    # --- Tier 2: spatial gap signals ---
    all_candidates.extend(
        _detect_intraline_gaps(
            tokens, baseline_lines, pn, pw, ph, m, cfg.vocr_cand_gap_multiplier
        )
    )
    all_candidates.extend(
        _detect_dense_cluster_holes(
            tokens, pn, pw, ph, m, cfg.vocr_cand_density_grid_size
        )
    )
    all_candidates.extend(
        _detect_baseline_style_gaps(
            tokens, baseline_lines, pn, pw, ph, m, cfg.vocr_cand_gap_multiplier
        )
    )

    # --- Tier 3: template / token-context signals ---
    all_candidates.extend(
        _detect_template_adjacency(tokens, baseline_lines, pn, pw, ph, m)
    )
    all_candidates.extend(
        _detect_regex_digit_patterns(tokens, baseline_lines, pn, pw, ph, m)
    )
    all_candidates.extend(
        _detect_impossible_sequences(tokens, baseline_lines, pn, pw, ph, m)
    )
    all_candidates.extend(_detect_vocab_triggers(tokens, baseline_lines, pn, pw, ph, m))
    all_candidates.extend(
        _detect_keyword_cooccurrence(tokens, baseline_lines, pn, pw, ph, m)
    )

    # --- Tier 4: cross-reference / consensus ---
    all_candidates.extend(
        _detect_cross_ref_phrases(tokens, baseline_lines, pn, pw, ph, m)
    )
    all_candidates.extend(
        _detect_near_duplicate_lines(tokens, baseline_lines, pn, pw, ph, m)
    )

    # --- Tier 5: structural / statistical ---
    all_candidates.extend(
        _detect_font_subset_correlation(
            tokens, baseline_lines, page_chars, pn, pw, ph, m
        )
    )
    all_candidates.extend(
        _detect_token_width_anomaly(
            tokens, pn, pw, ph, m, cfg.vocr_cand_char_width_ratio
        )
    )
    all_candidates.extend(
        _detect_vector_circles(
            page_curves,
            tokens,
            pn,
            pw,
            ph,
            m,
            cfg.vocr_cand_vector_circle_max_diameter,
        )
    )
    all_candidates.extend(
        _detect_semantic_no_units(tokens, baseline_lines, pn, pw, ph, m)
    )
    all_candidates.extend(
        _detect_dimension_geometry(page_lines, tokens, baseline_lines, pn, pw, ph, m)
    )

    # Apply adaptive confidence from accumulated stats (Level 1)
    if method_stats is not None:
        from plancheck.vocr.method_stats import get_adaptive_confidence

        for cand in all_candidates:
            if cand.trigger_methods:
                primary = cand.trigger_methods[0]
                cand.confidence = get_adaptive_confidence(
                    primary, method_stats, cand.confidence
                )

    # Apply per-producer override (Level 3) — refines Level 1 when data exists
    if producer_stats is not None and producer_id:
        from plancheck.vocr.producer_stats import get_producer_confidence

        for cand in all_candidates:
            if cand.trigger_methods:
                primary = cand.trigger_methods[0]
                cand.confidence = get_producer_confidence(
                    primary, producer_id, producer_stats, cand.confidence
                )

    # Filter by minimum confidence
    all_candidates = [
        c for c in all_candidates if c.confidence >= cfg.vocr_cand_min_confidence
    ]

    # Merge overlapping candidates
    all_candidates = _merge_overlapping_candidates(all_candidates)

    # Sort by confidence descending and cap
    all_candidates.sort(key=lambda c: c.confidence, reverse=True)
    if len(all_candidates) > cfg.vocr_cand_max_candidates:
        all_candidates = all_candidates[: cfg.vocr_cand_max_candidates]

    log.info(
        "Page %d: %d VOCR candidates from %d detection methods",
        page_num,
        len(all_candidates),
        len({m for c in all_candidates for m in c.trigger_methods}),
    )
    return all_candidates


def compute_candidate_stats(
    candidates: List[VocrCandidate],
    page_width: float = 0.0,
    page_height: float = 0.0,
) -> Dict[str, Any]:
    """Aggregate hit/miss statistics per trigger method.

    Call this **after** targeted VOCR has updated each candidate's
    ``outcome`` field (``"hit"`` or ``"miss"``).

    Returns a dict suitable for JSON serialisation and cross-run analysis.
    """
    total = len(candidates)
    hits = sum(1 for c in candidates if c.outcome == "hit")
    misses = sum(1 for c in candidates if c.outcome == "miss")
    pending = sum(1 for c in candidates if c.outcome == "pending")

    # Per-method breakdown
    by_method: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        for method in c.trigger_methods:
            entry = by_method.setdefault(method, {"flagged": 0, "hits": 0, "misses": 0})
            entry["flagged"] += 1
            if c.outcome == "hit":
                entry["hits"] += 1
            elif c.outcome == "miss":
                entry["misses"] += 1
    for entry in by_method.values():
        f = entry["flagged"]
        entry["hit_rate"] = round(entry["hits"] / f, 4) if f > 0 else 0.0

    # Area stats
    total_area = sum(c.patch_area() for c in candidates)
    page_area = page_width * page_height if page_width > 0 and page_height > 0 else 0.0

    # Predicted vs found
    predicted_vs_found: Dict[str, Dict[str, int]] = {}
    for c in candidates:
        if not c.predicted_symbol:
            continue
        entry = predicted_vs_found.setdefault(
            c.predicted_symbol,
            {"predicted": 0, "correct": 0, "wrong_symbol": 0, "miss": 0},
        )
        entry["predicted"] += 1
        if c.outcome == "hit":
            if c.found_symbol == c.predicted_symbol:
                entry["correct"] += 1
            else:
                entry["wrong_symbol"] += 1
        elif c.outcome == "miss":
            entry["miss"] += 1

    return {
        "total_candidates": total,
        "total_hits": hits,
        "total_misses": misses,
        "total_pending": pending,
        "hit_rate": round(hits / total, 4) if total > 0 else 0.0,
        "by_method": by_method,
        "area_stats": {
            "total_patch_area_pts2": round(total_area, 1),
            "mean_patch_area": round(total_area / total, 1) if total > 0 else 0.0,
            "page_coverage_pct": (
                round(100.0 * total_area / page_area, 2) if page_area > 0 else 0.0
            ),
        },
        "predicted_vs_found": predicted_vs_found,
    }
