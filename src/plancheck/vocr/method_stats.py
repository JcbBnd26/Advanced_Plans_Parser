"""Persistent method-level hit/miss statistics for VOCR candidate detection.

Level 1 of the adaptive-confidence pipeline.  Accumulates per-method
flagged / hit / miss counts across runs and computes a rolling hit-rate
that replaces hard-coded confidence values.

File format (JSON)::

    {
        "version": 2,
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

from .adaptive_stats_base import AdaptiveStatsBase

log = logging.getLogger(__name__)


# ── MethodStats class using common base ─────────────────────────────────


class MethodStats(AdaptiveStatsBase):
    """Method-level adaptive statistics (Level 1)."""

    VERSION = 2
    ROOT_KEY = "methods"

    @classmethod
    def _empty_skeleton(cls) -> Dict[str, Any]:
        return {
            "version": cls.VERSION,
            "total_runs": 0,
            "methods": {},
            "symbols": {},
            "updated_at": "",
        }

    @classmethod
    def load(cls, path: str | Path) -> Dict[str, Any]:
        """Load stats, handling version migration from v1 to v2."""
        path = Path(path)
        if not path.exists():
            log.debug("No method-stats file at %s – starting fresh", path)
            return cls._empty_skeleton()

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return cls._empty_skeleton()
            # Accept version 1 or 2
            version = data.get("version", 1)
            if version not in (1, 2):
                log.warning(
                    "method-stats version mismatch at %s – starting fresh", path
                )
                return cls._empty_skeleton()
            # Migrate v1 → v2 on load
            if version == 1:
                data["version"] = cls.VERSION
            # Ensure required top-level keys
            for key in ("total_runs", "methods", "symbols"):
                if key not in data:
                    data[key] = cls._empty_skeleton()[key]
            return data
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not load method stats from %s: %s", path, exc)
            return cls._empty_skeleton()

    @classmethod
    def update(
        cls,
        path: str | Path,
        candidate_stats: Dict[str, Any],
        *,
        stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Merge candidate_stats from one run into persistent file."""
        path = Path(path)
        if stats is None:
            stats = cls.load(path)

        stats["total_runs"] = stats.get("total_runs", 0) + 1

        # Merge per-method counts using base class helper
        by_method = candidate_stats.get("by_method", {})
        methods = stats.setdefault("methods", {})
        for method_name, run_entry in by_method.items():
            acc = methods.setdefault(
                method_name, {"flagged": 0, "hits": 0, "misses": 0}
            )
            cls.merge_counters(acc, run_entry)

        # Merge per-symbol counts (method_stats-specific)
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

        # Persist using base class
        cls.save(path, stats)
        log.info(
            "Updated method stats → %s  (total_runs=%d)", path, stats["total_runs"]
        )
        return stats


# ── Backward-compatible public API ──────────────────────────────────────


def load_method_stats(path: str | Path) -> Dict[str, Any]:
    """Load accumulated method stats from *path*, or return empty skeleton.

    Returns a *mutable* dict — caller may pass it straight into
    :func:`get_adaptive_confidence` lookups.
    """
    return MethodStats.load(path)


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
    return MethodStats.update(path, candidate_stats, stats=stats)


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
    return MethodStats.get_confidence(
        entry, base_confidence, min_runs=min_runs, floor=floor, ceiling=ceiling
    )
