"""Per-producer method hit-rate statistics (Level 3 adaptation).

Different PDF producers (AutoCAD, Bluebeam, Revit, etc.) exhibit
different symbol-encoding patterns.  This module maintains producer-
specific method statistics that refine the global stats from Level 1.

File format (JSON)::

    {
        "version": 1,
        "producers": {
            "AutoCAD": {
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

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

_STATS_VERSION = 1


def _normalise_producer(producer: str) -> str:
    """Normalise a PDF producer string into a canonical key.

    Strips version numbers, lowercases, and extracts the primary
    software name so that ``"AutoCAD 2023"`` and ``"AutoCAD 2024"``
    map to the same key.
    """
    if not producer:
        return ""
    import re

    # Strip version / build info after the name
    clean = re.sub(r"\s*[\d.]+.*$", "", producer.strip())
    return clean.lower()


# ── Public helpers ──────────────────────────────────────────────────────


def load_producer_stats(path: str | Path) -> Dict[str, Any]:
    """Load per-producer stats from *path*, or return empty skeleton."""
    path = Path(path)
    if not path.exists():
        return _empty_stats()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != _STATS_VERSION:
            return _empty_stats()
        data.setdefault("producers", {})
        return data
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not load producer stats from %s: %s", path, exc)
        return _empty_stats()


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
    path = Path(path)
    key = _normalise_producer(producer)
    if not key:
        log.debug("Empty producer — skipping producer stats update")
        return stats or load_producer_stats(path)

    if stats is None:
        stats = load_producer_stats(path)

    producers = stats.setdefault("producers", {})
    entry = producers.setdefault(key, {"total_runs": 0, "methods": {}})
    entry["total_runs"] = entry.get("total_runs", 0) + 1

    by_method = candidate_stats.get("by_method", {})
    methods = entry.setdefault("methods", {})
    for method_name, run_entry in by_method.items():
        acc = methods.setdefault(method_name, {"flagged": 0, "hits": 0, "misses": 0})
        acc["flagged"] += run_entry.get("flagged", 0)
        acc["hits"] += run_entry.get("hits", 0)
        acc["misses"] += run_entry.get("misses", 0)

    stats["updated_at"] = datetime.now(timezone.utc).isoformat()

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    return stats


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

    key = _normalise_producer(producer)
    prod_entry = stats.get("producers", {}).get(key)
    if prod_entry is None:
        return fallback

    m_entry = prod_entry.get("methods", {}).get(method)
    if m_entry is None:
        return fallback

    flagged = m_entry.get("flagged", 0)
    if flagged < min_runs:
        return fallback

    hits = m_entry.get("hits", 0)
    hit_rate = (hits + 1) / (flagged + 2)  # Laplace smoothing
    return max(floor, min(ceiling, round(hit_rate, 4)))


# ── Internals ───────────────────────────────────────────────────────────


def _empty_stats() -> Dict[str, Any]:
    return {
        "version": _STATS_VERSION,
        "producers": {},
        "updated_at": "",
    }
