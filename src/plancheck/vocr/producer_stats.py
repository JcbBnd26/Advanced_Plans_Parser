"""Per-producer method hit-rate statistics (Level 3 adaptation).

Different PDF producers (AutoCAD, Bluebeam, Revit, etc.) exhibit
different symbol-encoding patterns.  This module maintains producer-
specific method statistics that refine the global stats from Level 1.

File format (JSON)::

    {
        "version": 1,
        "producers": {
            "autocad": {
                "total_runs": 12,
                "methods": {
                    "char_encoding_failures": {"flagged": 80, "hits": 72, "misses": 8},
                    ...
                }
            },
            ...
        },
        "updated_at": "2025-03-15T12:34:56"
    }

When a producer-specific entry exists and has sufficient data, its
hit-rate overrides the global adaptive confidence from Level 1.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from .adaptive_stats_base import AdaptiveStatsBase

log = logging.getLogger(__name__)


def _normalise_producer(producer: str) -> str:
    """Normalise a PDF producer string into a canonical key.

    Strips version numbers, lowercases, and extracts the primary
    software name so that ``"AutoCAD 2023"`` and ``"AutoCAD 2024"``
    map to the same key.
    """
    if not producer:
        return ""
    # Strip version / build info after the name
    clean = re.sub(r"\s*[\d.]+.*$", "", producer.strip())
    return clean.lower()


# ── ProducerStats class using common base ───────────────────────────────


class ProducerStats(AdaptiveStatsBase):
    """Producer-level adaptive statistics (Level 3)."""

    VERSION = 1
    ROOT_KEY = "producers"

    @classmethod
    def _empty_skeleton(cls) -> Dict[str, Any]:
        return {
            "version": cls.VERSION,
            "producers": {},
            "updated_at": "",
        }

    @classmethod
    def _normalize_key(cls, key: str) -> str:
        """Override to normalize producer strings."""
        return _normalise_producer(key)

    @classmethod
    def update(
        cls,
        path: str | Path,
        producer: str,
        candidate_stats: Dict[str, Any],
        *,
        stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Merge one run's candidate stats under the given producer."""
        path = Path(path)
        key = cls._normalize_key(producer)
        if not key:
            log.debug("Empty producer — skipping producer stats update")
            return stats or cls.load(path)

        if stats is None:
            stats = cls.load(path)

        producers = stats.setdefault("producers", {})
        entry = producers.setdefault(key, {"total_runs": 0, "methods": {}})
        entry["total_runs"] = entry.get("total_runs", 0) + 1

        by_method = candidate_stats.get("by_method", {})
        methods = entry.setdefault("methods", {})
        for method_name, run_entry in by_method.items():
            acc = methods.setdefault(
                method_name, {"flagged": 0, "hits": 0, "misses": 0}
            )
            cls.merge_counters(acc, run_entry)

        cls.save(path, stats)
        return stats


# ── Backward-compatible public API ──────────────────────────────────────


def load_producer_stats(path: str | Path) -> Dict[str, Any]:
    """Load per-producer stats from *path*, or return empty skeleton."""
    return ProducerStats.load(path)


def update_producer_stats(
    path: str | Path,
    producer: str,
    candidate_stats: Dict[str, Any],
    *,
    stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge one run's candidate stats under the given *producer*.

    Parameters
    ----------
    path : str | Path
        Destination JSON file.
    producer : str
        Raw ``/Producer`` metadata string (normalised internally).
    candidate_stats : dict
        Output of :func:`~plancheck.vocr.candidates.compute_candidate_stats`.
    stats : dict, optional
        Pre-loaded stats dict to avoid re-reading from disk.

    Returns
    -------
    dict
        Updated stats.
    """
    return ProducerStats.update(path, producer, candidate_stats, stats=stats)


def get_producer_confidence(
    method: str,
    producer: str,
    stats: Optional[Dict[str, Any]],
    fallback: float,
    *,
    min_runs: int = 3,
    floor: float = 0.1,
    ceiling: float = 0.95,
) -> float:
    """Return producer-specific confidence; *fallback* if insufficient data.

    Parameters
    ----------
    method : str
        Detection method name.
    producer : str
        Raw producer string (normalised internally).
    stats : dict | None
        Loaded producer stats.
    fallback : float
        Confidence to return when insufficient producer-specific data.
        Typically the Level 1 adaptive confidence.
    min_runs : int
        Minimum flagged samples for this producer-method pair.
    floor, ceiling : float
        Clamp bounds.
    """
    if stats is None or not producer:
        return fallback

    key = ProducerStats._normalize_key(producer)
    prod_entry = stats.get("producers", {}).get(key)
    if prod_entry is None:
        return fallback

    m_entry = prod_entry.get("methods", {}).get(method)
    return ProducerStats.get_confidence(
        m_entry, fallback, min_runs=min_runs, floor=floor, ceiling=ceiling
    )
