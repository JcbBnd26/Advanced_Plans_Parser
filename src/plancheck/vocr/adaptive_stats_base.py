"""Base class for adaptive hit-rate statistics (methods and producers).

Both method_stats.py and producer_stats.py share the same core pattern:
  * JSON persistence with version checking
  * Counter accumulation: {flagged, hits, misses}
  * Laplace-smoothed confidence: (hits + 1) / (flagged + 2)

This base class consolidates that logic.  Subclasses override:
  - VERSION: int
  - ROOT_KEY: str  (e.g. "methods" or "producers")
  - _normalize_key(): optional key normalization
  - _empty_skeleton(): schema-specific skeleton dict
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


class AdaptiveStatsBase(ABC):
    """Abstract base for adaptive statistics storage."""

    VERSION: int = 1  # Override in subclass
    ROOT_KEY: str = "entries"  # Override: "methods" or "producers"

    # ── Abstract methods ────────────────────────────────────────────────

    @classmethod
    @abstractmethod
    def _empty_skeleton(cls) -> Dict[str, Any]:
        """Return a fresh empty stats structure."""
        ...

    # ── Optional hooks ──────────────────────────────────────────────────

    @classmethod
    def _normalize_key(cls, key: str) -> str:
        """Normalize a lookup key.  Identity by default."""
        return key

    # ── Core methods ────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str | Path) -> Dict[str, Any]:
        """Load stats from *path*, or return empty skeleton if missing/invalid."""
        path = Path(path)
        if not path.exists():
            log.debug("No stats file at %s – starting fresh", path)
            return cls._empty_skeleton()

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("version") != cls.VERSION:
                log.warning("Stats version mismatch at %s – starting fresh", path)
                return cls._empty_skeleton()
            # Ensure root key exists
            data.setdefault(cls.ROOT_KEY, {})
            return data
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not load stats from %s: %s", path, exc)
            return cls._empty_skeleton()

    @classmethod
    def save(cls, path: str | Path, stats: Dict[str, Any]) -> None:
        """Persist *stats* to *path*, creating parent dirs as needed."""
        path = Path(path)
        stats["updated_at"] = datetime.now(timezone.utc).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)

    @classmethod
    def merge_counters(
        cls,
        target: Dict[str, int],
        source: Dict[str, int],
    ) -> None:
        """Accumulate source counters into target (in-place)."""
        target["flagged"] = target.get("flagged", 0) + source.get("flagged", 0)
        target["hits"] = target.get("hits", 0) + source.get("hits", 0)
        target["misses"] = target.get("misses", 0) + source.get("misses", 0)

    @classmethod
    def get_confidence(
        cls,
        entry: Optional[Dict[str, int]],
        fallback: float,
        *,
        min_runs: int = 5,
        floor: float = 0.1,
        ceiling: float = 0.95,
    ) -> float:
        """Compute hit-rate confidence from a counters dict.

        Uses Laplace smoothing: (hits + 1) / (flagged + 2).

        Parameters
        ----------
        entry : dict | None
            Counter dict with {flagged, hits, misses} keys.
        fallback : float
            Returned if entry is None or insufficient data.
        min_runs : int
            Minimum flagged samples before adaptive confidence applies.
        floor, ceiling : float
            Clamp bounds for the returned confidence.

        Returns
        -------
        float
            Clamped hit-rate or fallback.
        """
        if entry is None:
            return fallback

        flagged = entry.get("flagged", 0)
        if flagged < min_runs:
            return fallback

        hits = entry.get("hits", 0)
        hit_rate = (hits + 1) / (flagged + 2)
        return max(floor, min(ceiling, round(hit_rate, 4)))
