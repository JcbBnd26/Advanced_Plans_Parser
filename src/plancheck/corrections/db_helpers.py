"""Database-tab helper mixin for CorrectionStore.

Provides read-only convenience queries used by the GUI's Database tab,
plus the training-data reset workflow.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import sqlite3
    from pathlib import Path

log = logging.getLogger(__name__)


class DbHelpersMixin:
    """Mixin providing database overview and summary queries.

    Requires the host class to have:
    - ``_conn: sqlite3.Connection``
    - ``_db_path: Path``
    - ``get_run_ids_for_doc(doc_id: str) -> list[str]``
    """

    _conn: "sqlite3.Connection"
    _db_path: "Path"

    def get_all_documents(self) -> list[dict[str, Any]]:
        """Return every registered document."""
        rows = self._conn.execute(
            "SELECT * FROM documents ORDER BY ingested_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_run_ids_for_doc(self, doc_id: str) -> list[str]:
        """Return distinct pipeline run_ids for *doc_id*, newest first."""
        rows = self._conn.execute(
            "SELECT DISTINCT run_id FROM detections "
            "WHERE doc_id = ? AND run_id NOT LIKE 'manual%' "
            "ORDER BY created_at DESC",
            (doc_id,),
        ).fetchall()
        return [r["run_id"] for r in rows]

    def get_db_overview(self) -> dict[str, Any]:
        """Aggregate overview stats for the whole database."""
        row = self._conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
        docs = row["n"] if row else 0
        row = self._conn.execute("SELECT COUNT(*) AS n FROM detections").fetchone()
        dets = row["n"] if row else 0
        row = self._conn.execute("SELECT COUNT(*) AS n FROM corrections").fetchone()
        corrs = row["n"] if row else 0
        row = self._conn.execute("SELECT COUNT(*) AS n FROM box_groups").fetchone()
        groups = row["n"] if row else 0
        row = self._conn.execute("SELECT COUNT(*) AS n FROM training_runs").fetchone()
        trains = row["n"] if row else 0
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM training_examples"
        ).fetchone()
        examples = row["n"] if row else 0
        row = self._conn.execute(
            "SELECT MAX(created_at) AS ts FROM detections"
        ).fetchone()
        last_det = row["ts"] if row else None
        row = self._conn.execute(
            "SELECT MAX(corrected_at) AS ts FROM corrections"
        ).fetchone()
        last_corr = row["ts"] if row else None
        db_size = self._db_path.stat().st_size if self._db_path.exists() else 0
        # Dismissed detections (table may not exist in older DBs)
        try:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM dismissed_detections"
            ).fetchone()
            dismissed = row["n"] if row else 0
        except Exception:  # noqa: BLE001 — table may not exist yet
            dismissed = 0
        return {
            "db_path": str(self._db_path.resolve()),
            "db_size_bytes": db_size,
            "total_documents": docs,
            "total_detections": dets,
            "total_corrections": corrs,
            "total_groups": groups,
            "total_training_runs": trains,
            "total_training_examples": examples,
            "total_dismissed": dismissed,
            "last_detection_at": last_det,
            "last_correction_at": last_corr,
        }

    def get_detection_type_breakdown(self) -> dict[str, int]:
        """Detection counts grouped by element_type across all documents."""
        rows = self._conn.execute(
            "SELECT element_type, COUNT(*) AS n FROM detections GROUP BY element_type"
        ).fetchall()
        return {r["element_type"]: r["n"] for r in rows}

    def get_doc_summary(self, doc_id: str) -> dict[str, Any]:
        """Summary stats for a single document."""
        doc = self._conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if not doc:
            return {}
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM detections WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        det_count = row["n"] if row else 0
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM corrections WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        corr_count = row["n"] if row else 0
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM box_groups WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        group_count = row["n"] if row else 0
        runs = self.get_run_ids_for_doc(doc_id)
        row = self._conn.execute(
            "SELECT MAX(ts) AS ts FROM ("
            "  SELECT MAX(created_at) AS ts FROM detections WHERE doc_id = ? "
            "  UNION ALL "
            "  SELECT MAX(corrected_at) FROM corrections WHERE doc_id = ?"
            ")",
            (doc_id, doc_id),
        ).fetchone()
        last_activity = row["ts"] if row else None
        return {
            **dict(doc),
            "detection_count": det_count,
            "correction_count": corr_count,
            "group_count": group_count,
            "run_ids": runs,
            "last_activity": last_activity,
        }

    def get_detection_counts_by_page(
        self, doc_id: str, run_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Per-page element_type breakdown for a document.

        Returns a list of ``{page, element_type, count}`` dicts.
        """
        if run_id:
            rows = self._conn.execute(
                "SELECT page, element_type, COUNT(*) AS count "
                "FROM detections WHERE doc_id = ? AND run_id = ? "
                "GROUP BY page, element_type ORDER BY page, element_type",
                (doc_id, run_id),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT page, element_type, COUNT(*) AS count "
                "FROM detections WHERE doc_id = ? "
                "GROUP BY page, element_type ORDER BY page, element_type",
                (doc_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_correction_type_breakdown(
        self, doc_id: str | None = None
    ) -> dict[str, int]:
        """Correction counts grouped by correction_type."""
        if doc_id:
            rows = self._conn.execute(
                "SELECT correction_type, COUNT(*) AS n FROM corrections "
                "WHERE doc_id = ? GROUP BY correction_type",
                (doc_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT correction_type, COUNT(*) AS n FROM corrections "
                "GROUP BY correction_type"
            ).fetchall()
        return {r["correction_type"]: r["n"] for r in rows}

    def get_recent_corrections(
        self, doc_id: str | None = None, limit: int = 25
    ) -> list[dict[str, Any]]:
        """Most recent corrections, optionally scoped to a document."""
        if doc_id:
            rows = self._conn.execute(
                "SELECT * FROM corrections WHERE doc_id = ? "
                "ORDER BY corrected_at DESC LIMIT ?",
                (doc_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM corrections ORDER BY corrected_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Training data reset ────────────────────────────────────────

    def _safe_table_count(self, table: str) -> int:
        """Return ``COUNT(*)`` for *table*, returning 0 if the table doesn't exist."""
        try:
            row = self._conn.execute(
                f"SELECT COUNT(*) AS n FROM [{table}]"  # noqa: S608
            ).fetchone()
            return row["n"] if row else 0
        except Exception:  # noqa: BLE001 — table may not exist in older schemas
            return 0

    def get_training_data_summary(self) -> dict[str, int | bool]:
        """Return counts for every table affected by a training-data reset.

        Also checks whether associated model/data files exist on disk.
        Used by the confirmation dialog so the user sees exactly what
        will be cleared.
        """
        from pathlib import Path

        data_dir = self._db_path.parent

        summary: dict[str, int | bool] = {}

        # Tables that would be cleared
        for table in (
            "corrections",
            "training_examples",
            "training_runs",
            "detections",
            "feature_cache",
            "candidate_outcomes",
            "dismissed_detections",
            "box_groups",
            "box_group_members",
        ):
            summary[table] = self._safe_table_count(table)

        # File existence checks
        summary["model_file_exists"] = (data_dir / "element_classifier.pkl").exists()
        summary["drift_stats_exists"] = (data_dir / "drift_stats.json").exists()
        summary["jsonl_exists"] = (data_dir / "training_data.jsonl").exists()
        summary["stage2_jsonl_exists"] = (
            data_dir / "training_data_stage2.jsonl"
        ).exists()
        summary["subtype_model_exists"] = (data_dir / "subtype_classifier.pkl").exists()

        return summary

    def reset_training_data(self) -> dict[str, int]:
        """Clear all ML training signal from the database.

        Creates an automatic ``pre-reset`` snapshot first (the user can
        restore from it via the existing Restore Snapshot button).

        **Preserved:** ``documents``, ``processing_runs``, ``page_images``
        tables and all schema structures.

        **Cleared:** ``corrections``, ``training_examples``, ``training_runs``,
        ``detections``, ``feature_cache``, ``candidate_outcomes``,
        ``dismissed_detections``, ``box_groups``, ``box_group_members``,
        and the ``session_active`` metadata flag.

        Returns a dict mapping each cleared table name to the number of
        rows that were deleted.

        .. note::
           This method only handles SQLite data.  Associated files on
           disk (model ``.pkl``, JSONL, drift stats) must be deleted
           by the caller (typically the GUI handler).
        """
        # Safety snapshot — snapshot() acquires its own write lock
        snapshot_path = self.snapshot(tag="pre-reset")
        log.info("Pre-reset snapshot saved: %s", snapshot_path.name)

        deleted: dict[str, int] = {}

        # FK-safe deletion order: children before parents
        delete_order = [
            "box_group_members",
            "box_groups",
            "feature_cache",
            "dismissed_detections",
            "training_examples",
            "corrections",
            "candidate_outcomes",
            "detections",
            "training_runs",
        ]

        with self._write_lock():
            for table in delete_order:
                try:
                    cur = self._conn.execute(f"DELETE FROM [{table}]")  # noqa: S608
                    deleted[table] = cur.rowcount
                except Exception:  # noqa: BLE001 — table may not exist
                    log.debug("Table %s not found during reset", table)
                    deleted[table] = 0

            # Clear stale session-active flag (preserve other metadata)
            try:
                self._conn.execute("DELETE FROM metadata WHERE key = 'session_active'")
            except Exception:  # noqa: BLE001 — metadata table may not exist
                pass

            self._conn.commit()

        # VACUUM must run outside the write-lock transaction
        try:
            self._conn.execute("VACUUM")
        except Exception:  # noqa: BLE001 — non-critical space reclamation
            log.warning("VACUUM failed after reset (non-critical)", exc_info=True)

        log.info(
            "Training data reset complete: %s",
            ", ".join(f"{t}={n}" for t, n in deleted.items() if n),
        )
        return deleted
