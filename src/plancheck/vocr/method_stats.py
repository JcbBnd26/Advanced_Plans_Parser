"""Persistent method-level hit/miss statistics for VOCR candidate detection.

Level 1 of the adaptive-confidence pipeline.  Accumulates per-method
flagged / hit / miss counts across runs and computes a rolling hit-rate
that replaces hard-coded confidence values.

File format (JSON)::

    {
        "version": 1,
        "total_runs": 42,
        "methods": {
            "char_encoding_failures": {"flagged": 310, "hits": 280, "misses": 30},
            ...
        },
        "symbols": {
            "%": {"predicted": 100, "correct": 85, "wrong_symbol": 5, "miss": 10},
            ...
        },
        "updated_at": "2025-03-15T12:34:56"
    }
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

_STATS_VERSION = 1

# ── default skeleton ────────────────────────────────────────────────────

_EMPTY_STATS: Dict[str, Any] = {
    "version": _STATS_VERSION,
    "total_runs": 0,
    "methods": {},
    "symbols": {},
    "updated_at": "",
}


# ── public helpers ──────────────────────────────────────────────────────


def load_method_stats(path: str | Path) -> Dict[str, Any]:
    """Load accumulated method stats from *path*, or return empty skeleton.

    Returns a *mutable* dict — caller may pass it straight into
    :func:`get_adaptive_confidence` lookups.
    """
    path = Path(path)
    if not path.exists():
        log.debug("No method-stats file at %s – starting fresh", path)
        return _deep_copy_empty()

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or data.get("version") != _STATS_VERSION:
            log.warning("method-stats version mismatch at %s – starting fresh", path)
            return _deep_copy_empty()
        # Ensure required top-level keys
        for key in ("total_runs", "methods", "symbols"):
            if key not in data:
                data[key] = _EMPTY_STATS[key]
        return data
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not load method stats from %s: %s", path, exc)
        return _deep_copy_empty()


def update_method_stats(
    path: str | Path,
    candidate_stats: Dict[str, Any],
    *,
    stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge *candidate_stats* from one run into the persistent file.

    Parameters
    ----------
    path : str | Path
        Destination JSON file.  Created if missing.
    candidate_stats : dict
        Output of :func:`~plancheck.vocr.candidates.compute_candidate_stats`.
    stats : dict, optional
        Pre-loaded stats dict (avoids a redundant ``load_method_stats``
        call when the caller already has it in memory).

    Returns
    -------
    dict
        The updated (and saved) stats dict.
    """
    path = Path(path)
    if stats is None:
        stats = load_method_stats(path)

    stats["total_runs"] = stats.get("total_runs", 0) + 1

    # Merge per-method counts
    by_method = candidate_stats.get("by_method", {})
    methods = stats.setdefault("methods", {})
    for method_name, run_entry in by_method.items():
        acc = methods.setdefault(method_name, {"flagged": 0, "hits": 0, "misses": 0})
        acc["flagged"] += run_entry.get("flagged", 0)
        acc["hits"] += run_entry.get("hits", 0)
        acc["misses"] += run_entry.get("misses", 0)

    # Merge per-symbol counts
    pvf = candidate_stats.get("predicted_vs_found", {})
    symbols = stats.setdefault("symbols", {})
    for sym, run_entry in pvf.items():
        acc = symbols.setdefault(
            sym, {"predicted": 0, "correct": 0, "wrong_symbol": 0, "miss": 0}
        )
        acc["predicted"] += run_entry.get("predicted", 0)
        acc["correct"] += run_entry.get("correct", 0)
        acc["wrong_symbol"] += run_entry.get("wrong_symbol", 0)
        acc["miss"] += run_entry.get("miss", 0)

    stats["updated_at"] = datetime.now(timezone.utc).isoformat()

    # Persist
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    log.info("Updated method stats → %s  (total_runs=%d)", path, stats["total_runs"])
    return stats


def get_adaptive_confidence(
    method: str,
    stats: Optional[Dict[str, Any]],
    base_confidence: float,
    *,
    min_runs: int = 5,
    floor: float = 0.1,
    ceiling: float = 0.95,
) -> float:
    """Return data-driven confidence for *method*, or *base_confidence* if
    insufficient data.

    Logic
    -----
    * If *stats* is ``None`` or the method has fewer than *min_runs*
      total flagged samples, fall back to *base_confidence*.
    * Otherwise, confidence = hit_rate clamped to ``[floor, ceiling]``.
    * A small Bayesian smoothing (Laplace +1/+2) avoids extreme values
      from tiny samples.

    Parameters
    ----------
    method : str
        Detection method name (e.g. ``"char_encoding_failures"``).
    stats : dict | None
        Loaded stats dict from :func:`load_method_stats`.
    base_confidence : float
        Hard-coded fallback confidence from the detection method.
    min_runs : int
        Minimum ``flagged`` count before adaptive confidence kicks in.
    floor, ceiling : float
        Clamp bounds for the returned confidence.
    """
    if stats is None:
        return base_confidence

    entry = stats.get("methods", {}).get(method)
    if entry is None:
        return base_confidence

    flagged = entry.get("flagged", 0)
    if flagged < min_runs:
        return base_confidence

    hits = entry.get("hits", 0)
    # Laplace smoothing: (hits + 1) / (flagged + 2)
    hit_rate = (hits + 1) / (flagged + 2)
    return max(floor, min(ceiling, round(hit_rate, 4)))


# ── internals ───────────────────────────────────────────────────────────


def _deep_copy_empty() -> Dict[str, Any]:
    """Return a fresh copy of the empty skeleton."""
    return {
        "version": _STATS_VERSION,
        "total_runs": 0,
        "methods": {},
        "symbols": {},
        "updated_at": "",
    }
