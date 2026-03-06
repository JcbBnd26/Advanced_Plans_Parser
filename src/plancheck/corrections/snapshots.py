"""Database snapshot mixin for CorrectionStore.

Provides backup and restore functionality for the corrections database.
"""

from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class SnapshotMixin:
    """Mixin providing database snapshot operations."""

    # These attributes are provided by CorrectionStore
    _conn: sqlite3.Connection
    _db_path: Path
    _write_lock: object

    def snapshot(self, tag: str = "") -> Path:
        """Create a timestamped copy of the database.

        Parameters
        ----------
        tag : str
            Optional human-readable suffix for the snapshot filename.

        Returns
        -------
        Path
            Path to the snapshot file.
        """
        snap_dir = self._db_path.parent / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        suffix = f"_{tag}" if tag else ""
        dest = snap_dir / f"corrections_{ts}{suffix}.db"

        with self._write_lock():
            # Flush WAL to disk before copying
            self._conn.execute("PRAGMA wal_checkpoint(FULL)")
            shutil.copy2(str(self._db_path), str(dest))
        return dest

    def list_snapshots(self) -> list[dict[str, Any]]:
        """List all database snapshots.

        Returns
        -------
        list[dict]
            Each dict has ``path``, ``timestamp``, ``tag``, ``size_kb``.
        """
        snap_dir = self._db_path.parent / "snapshots"
        if not snap_dir.is_dir():
            return []

        results: list[dict[str, Any]] = []
        for p in sorted(snap_dir.glob("corrections_*.db")):
            name = p.stem  # corrections_YYYYMMDD_HHMMSS[_tag]
            parts = name.split("_", 3)  # ['corrections', date, time, ?tag]
            ts = f"{parts[1]}_{parts[2]}" if len(parts) >= 3 else ""
            tag = parts[3] if len(parts) >= 4 else ""
            results.append(
                {
                    "path": p,
                    "timestamp": ts,
                    "tag": tag,
                    "size_kb": round(p.stat().st_size / 1024, 1),
                }
            )
        return results

    def restore_snapshot(self, snapshot_path: Path) -> None:
        """Replace the current database with a snapshot.

        Parameters
        ----------
        snapshot_path : Path
            Path to the snapshot ``.db`` file.

        Raises
        ------
        FileNotFoundError
            If *snapshot_path* does not exist.
        """
        snapshot_path = Path(snapshot_path)
        if not snapshot_path.is_file():
            raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

        with self._write_lock():
            # Close current connection
            self._conn.close()

            # Replace
            shutil.copy2(str(snapshot_path), str(self._db_path))

            # Reconnect
            self._conn = sqlite3.connect(str(self._db_path), timeout=10.0)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=10000")
            self._conn.execute("PRAGMA foreign_keys=ON")


__all__ = ["SnapshotMixin"]
