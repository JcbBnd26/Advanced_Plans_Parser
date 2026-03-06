"""Feature cache mixin for CorrectionStore.

Provides caching of computed feature vectors to avoid
recomputation during classifier training.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from .store_utils import _utcnow_iso

if TYPE_CHECKING:
    pass


class FeatureCacheMixin:
    """Mixin providing feature caching operations."""

    # These attributes are provided by CorrectionStore
    _conn: object
    _write_lock: object

    def cache_features(
        self,
        detection_id: str,
        vector: list[float],
        feature_version: int,
    ) -> None:
        """Store a computed feature vector in the cache.

        Parameters
        ----------
        detection_id : str
            Detection this vector belongs to.
        vector : list[float]
            The dense feature vector (encoded by :func:`encode_features`).
        feature_version : int
            Schema version (from :data:`FEATURE_VERSION`).
        """
        cache_key = f"{detection_id}:v{feature_version}"
        with self._write_lock():
            self._conn.execute(
                "INSERT OR REPLACE INTO feature_cache "
                "(cache_key, detection_id, feature_version, vector_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    cache_key,
                    detection_id,
                    feature_version,
                    json.dumps(vector),
                    _utcnow_iso(),
                ),
            )
            self._conn.commit()

    def get_cached_features(
        self,
        detection_id: str,
        feature_version: int,
    ) -> list[float] | None:
        """Retrieve a cached feature vector, or *None* on miss.

        Parameters
        ----------
        detection_id : str
            Detection to look up.
        feature_version : int
            Required schema version — stale versions are not returned.
        """
        cache_key = f"{detection_id}:v{feature_version}"
        row = self._conn.execute(
            "SELECT vector_json FROM feature_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["vector_json"])

    def invalidate_cache(self, *, feature_version: int | None = None) -> int:
        """Delete cached feature vectors.

        Parameters
        ----------
        feature_version : int, optional
            When given, only delete entries with a *different* version.
            When *None*, delete **all** cached entries.

        Returns
        -------
        int
            Number of rows deleted.
        """
        with self._write_lock():
            if feature_version is None:
                cur = self._conn.execute("DELETE FROM feature_cache")
            else:
                cur = self._conn.execute(
                    "DELETE FROM feature_cache WHERE feature_version != ?",
                    (feature_version,),
                )
            self._conn.commit()
            return cur.rowcount

    def cache_stats(self) -> dict[str, int]:
        """Return summary stats about the feature cache.

        Returns
        -------
        dict
            ``total_entries``, ``distinct_detections``,
            ``distinct_versions``.
        """
        row = self._conn.execute("SELECT COUNT(*) AS n FROM feature_cache").fetchone()
        total = row["n"] if row else 0
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT detection_id) AS n FROM feature_cache"
        ).fetchone()
        det_count = row["n"] if row else 0
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT feature_version) AS n FROM feature_cache"
        ).fetchone()
        ver_count = row["n"] if row else 0
        return {
            "total_entries": total,
            "distinct_detections": det_count,
            "distinct_versions": ver_count,
        }


__all__ = ["FeatureCacheMixin"]
