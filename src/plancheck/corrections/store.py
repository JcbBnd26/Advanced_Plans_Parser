"""SQLite-backed store for detections, corrections, and training examples.

Persists pipeline detections and human corrections to
the corrections database (by default ``data/corrections.db``).
Provides helpers to build training sets for downstream classifiers.

This module uses mixin composition — see:
- :mod:`.feature_cache` — feature vector caching
- :mod:`.candidate_outcomes` — VOCR candidate outcome tracking
- :mod:`.snapshots` — database backup/restore
- :mod:`.box_groups` — detection grouping
- :mod:`.training_data` — training set generation
- :mod:`.training_runs` — training experiment metadata
- :mod:`.db_helpers` — database overview and summary queries
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from plancheck.config import DEFAULT_CORRECTIONS_DB

from .box_groups import BoxGroupMixin
from .candidate_outcomes import CandidateOutcomesMixin
from .db_helpers import DbHelpersMixin
from .feature_cache import FeatureCacheMixin
from .snapshots import SnapshotMixin
from .store_utils import _gen_id, _utcnow_iso
from .training_data import TrainingDataMixin
from .training_runs import TrainingRunsMixin

# ── Schema DDL ─────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    pdf_path    TEXT DEFAULT '',
    page_count  INTEGER NOT NULL,
    ingested_at TEXT NOT NULL,
    project_tag TEXT DEFAULT '',
    notes       TEXT DEFAULT '',
    page_width  REAL DEFAULT 0.0,
    page_height REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS detections (
    detection_id    TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL,
    page            INTEGER NOT NULL,
    run_id          TEXT NOT NULL,
    element_type    TEXT NOT NULL,
    confidence      REAL,
    bbox_x0         REAL NOT NULL,
    bbox_y0         REAL NOT NULL,
    bbox_x1         REAL NOT NULL,
    bbox_y1         REAL NOT NULL,
    text_content    TEXT DEFAULT '',
    features_json   TEXT NOT NULL,
    polygon_json    TEXT,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);

CREATE TABLE IF NOT EXISTS corrections (
    correction_id           TEXT PRIMARY KEY,
    detection_id            TEXT,
    doc_id                  TEXT NOT NULL,
    page                    INTEGER NOT NULL,
    correction_type         TEXT NOT NULL,
    original_element_type   TEXT,
    corrected_element_type  TEXT NOT NULL,
    orig_bbox_x0            REAL,
    orig_bbox_y0            REAL,
    orig_bbox_x1            REAL,
    orig_bbox_y1            REAL,
    corr_bbox_x0            REAL NOT NULL,
    corr_bbox_y0            REAL NOT NULL,
    corr_bbox_x1            REAL NOT NULL,
    corr_bbox_y1            REAL NOT NULL,
    corrected_text          TEXT,
    corrected_features_json TEXT,
    annotator               TEXT DEFAULT 'user',
    corrected_at            TEXT NOT NULL,
    session_id              TEXT DEFAULT '',
    notes                   TEXT DEFAULT '',
    FOREIGN KEY (detection_id) REFERENCES detections(detection_id),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);

CREATE TABLE IF NOT EXISTS training_examples (
    example_id                  TEXT PRIMARY KEY,
    source                      TEXT NOT NULL,
    correction_id               TEXT,
    detection_id                TEXT NOT NULL,
    label                       TEXT NOT NULL,
    features_json               TEXT NOT NULL,
    split                       TEXT DEFAULT 'train',
    created_at                  TEXT NOT NULL,
    model_version_introduced    TEXT,
    FOREIGN KEY (correction_id) REFERENCES corrections(correction_id),
    FOREIGN KEY (detection_id)  REFERENCES detections(detection_id)
);

CREATE TABLE IF NOT EXISTS box_groups (
    group_id            TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL,
    page                INTEGER NOT NULL,
    group_label         TEXT NOT NULL,
    root_detection_id   TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id),
    FOREIGN KEY (root_detection_id) REFERENCES detections(detection_id)
);

CREATE TABLE IF NOT EXISTS box_group_members (
    group_id        TEXT NOT NULL,
    detection_id    TEXT NOT NULL,
    sort_order      INTEGER DEFAULT 0,
    PRIMARY KEY (group_id, detection_id),
    FOREIGN KEY (group_id) REFERENCES box_groups(group_id),
    FOREIGN KEY (detection_id) REFERENCES detections(detection_id)
);

CREATE TABLE IF NOT EXISTS training_runs (
    run_id          TEXT PRIMARY KEY,
    trained_at      TEXT NOT NULL,
    n_train         INTEGER NOT NULL,
    n_val           INTEGER NOT NULL,
    accuracy        REAL NOT NULL,
    f1_macro        REAL DEFAULT 0.0,
    f1_weighted     REAL DEFAULT 0.0,
    labels_json     TEXT NOT NULL,
    per_class_json  TEXT NOT NULL,
    model_path      TEXT DEFAULT '',
    notes           TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS feature_cache (
    cache_key       TEXT PRIMARY KEY,
    detection_id    TEXT NOT NULL,
    feature_version INTEGER NOT NULL DEFAULT 0,
    vector_json     TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS candidate_outcomes (
    outcome_id          TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL DEFAULT '',
    page                INTEGER NOT NULL,
    run_id              TEXT NOT NULL DEFAULT '',
    trigger_methods     TEXT NOT NULL,
    predicted_symbol    TEXT DEFAULT '',
    found_symbol        TEXT DEFAULT '',
    outcome             TEXT NOT NULL,
    confidence          REAL NOT NULL,
    bbox_x0             REAL NOT NULL,
    bbox_y0             REAL NOT NULL,
    bbox_x1             REAL NOT NULL,
    bbox_y1             REAL NOT NULL,
    page_width          REAL DEFAULT 0.0,
    page_height         REAL DEFAULT 0.0,
    features_json       TEXT DEFAULT '{}',
    created_at          TEXT NOT NULL
);

-- Indices for frequently-queried patterns (efficiency fix)
CREATE INDEX IF NOT EXISTS idx_detections_doc_page ON detections(doc_id, page);
CREATE INDEX IF NOT EXISTS idx_detections_run ON detections(run_id);
CREATE INDEX IF NOT EXISTS idx_corrections_doc_page ON corrections(doc_id, page);
CREATE INDEX IF NOT EXISTS idx_corrections_detection ON corrections(detection_id);
CREATE INDEX IF NOT EXISTS idx_candidate_outcomes_doc_page ON candidate_outcomes(doc_id, page);
"""


class CorrectionStore(
    FeatureCacheMixin,
    CandidateOutcomesMixin,
    SnapshotMixin,
    BoxGroupMixin,
    TrainingDataMixin,
    TrainingRunsMixin,
    DbHelpersMixin,
):
    """SQLite database for annotation persistence.

    Composed of mixins providing:
    - Feature caching (FeatureCacheMixin)
    - Candidate outcome tracking (CandidateOutcomesMixin)
    - Database snapshots (SnapshotMixin)
    - Box grouping (BoxGroupMixin)
    - Training data generation (TrainingDataMixin)
    - Training run metadata (TrainingRunsMixin)
    - Database overview queries (DbHelpersMixin)

    Parameters
    ----------
    db_path : Path
        Location of the database file.  Parent directories are created
        automatically on first use.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        db_path = Path(db_path) if db_path else DEFAULT_CORRECTIONS_DB
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._lock_path = db_path.with_suffix(db_path.suffix + ".lock")
        self._conn = sqlite3.connect(str(db_path), timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        from .db_lock import acquire_lock

        with acquire_lock(self._lock_path, timeout_sec=30.0):
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=10000")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_DDL)
            self._conn.commit()
            self._migrate_locked()

    def _write_lock(self):
        from .db_lock import acquire_lock

        return acquire_lock(self._lock_path, timeout_sec=30.0)

    # ── helpers ────────────────────────────────────────────────────────

    def _migrate(self) -> None:
        """Apply lightweight schema migrations for columns added after v1."""
        # Migrations are write operations and must be serialized across processes.
        with self._write_lock():
            self._migrate_locked()

    def _migrate_locked(self) -> None:
        """Implementation of migrations; caller must hold the write lock."""
        # polygon_json on detections
        det_cols = {
            r["name"]
            for r in self._conn.execute("PRAGMA table_info(detections)").fetchall()
        }
        if "polygon_json" not in det_cols:
            self._conn.execute("ALTER TABLE detections ADD COLUMN polygon_json TEXT")
            self._conn.commit()

        # pdf_path on documents
        doc_cols = {
            r["name"]
            for r in self._conn.execute("PRAGMA table_info(documents)").fetchall()
        }
        if "pdf_path" not in doc_cols:
            self._conn.execute(
                "ALTER TABLE documents ADD COLUMN pdf_path TEXT DEFAULT ''"
            )
            self._conn.commit()

        # holdout_preds_json on training_runs (Phase 1.2)
        tr_cols = {
            r["name"]
            for r in self._conn.execute("PRAGMA table_info(training_runs)").fetchall()
        }
        if "holdout_preds_json" not in tr_cols:
            self._conn.execute(
                "ALTER TABLE training_runs "
                "ADD COLUMN holdout_preds_json TEXT DEFAULT ''"
            )
            self._conn.commit()

        # hyperparams_json, feature_set_json, training_curves_json,
        # feature_version on training_runs (Phase 4.4)
        if "hyperparams_json" not in tr_cols:
            self._conn.execute(
                "ALTER TABLE training_runs "
                "ADD COLUMN hyperparams_json TEXT DEFAULT ''"
            )
            self._conn.commit()
        if "feature_set_json" not in tr_cols:
            self._conn.execute(
                "ALTER TABLE training_runs "
                "ADD COLUMN feature_set_json TEXT DEFAULT ''"
            )
            self._conn.commit()
        if "training_curves_json" not in tr_cols:
            self._conn.execute(
                "ALTER TABLE training_runs "
                "ADD COLUMN training_curves_json TEXT DEFAULT ''"
            )
            self._conn.commit()
        if "feature_version" not in tr_cols:
            self._conn.execute(
                "ALTER TABLE training_runs "
                "ADD COLUMN feature_version INTEGER DEFAULT 0"
            )
            self._conn.commit()

        # Create indices for frequently-queried patterns (idempotent)
        self._conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_detections_doc_page ON detections(doc_id, page);
            CREATE INDEX IF NOT EXISTS idx_detections_run ON detections(run_id);
            CREATE INDEX IF NOT EXISTS idx_corrections_doc_page ON corrections(doc_id, page);
            CREATE INDEX IF NOT EXISTS idx_corrections_detection ON corrections(detection_id);
            CREATE INDEX IF NOT EXISTS idx_candidate_outcomes_doc_page ON candidate_outcomes(doc_id, page);
            """
        )
        self._conn.commit()

    def refresh(self) -> None:
        """Ensure the connection sees the latest committed data from other connections.

        Call this before reading if another process or connection may have
        modified the database (e.g., after a pipeline run completes in a
        background thread).
        """
        # Commit ends any implicit read transaction, ensuring subsequent
        # queries see the latest committed data from all connections.
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    # ── documents ──────────────────────────────────────────────────────

    def register_document(self, pdf_path: Path) -> str:
        """Register a PDF and return its ``sha256:…`` *doc_id*.

        Reads the file once for hashing and uses ``pdfplumber`` to
        determine page count. Duplicate registrations are silently
        ignored (INSERT OR IGNORE).
        """
        pdf_path = Path(pdf_path)
        pdf_bytes = pdf_path.read_bytes()
        doc_id = "sha256:" + hashlib.sha256(pdf_bytes).hexdigest()

        # Check if already registered — avoid re-opening with pdfplumber
        row = self._conn.execute(
            "SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if row:
            return doc_id

        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            # Store first-page dimensions (width × height in points)
            pw, ph = 0.0, 0.0
            if pdf.pages:
                first_page = pdf.pages[0]
                try:
                    pw = float(first_page.width)
                    ph = float(first_page.height)
                except (AttributeError, TypeError):
                    pass

        with self._write_lock():
            self._conn.execute(
                "INSERT OR IGNORE INTO documents "
                "(doc_id, filename, pdf_path, page_count, ingested_at, "
                " page_width, page_height) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    doc_id,
                    pdf_path.name,
                    str(pdf_path.resolve()),
                    page_count,
                    _utcnow_iso(),
                    pw,
                    ph,
                ),
            )
            self._conn.commit()
        return doc_id

    # ── detections ─────────────────────────────────────────────────────

    def save_detection(
        self,
        doc_id: str,
        page: int,
        run_id: str,
        element_type: str,
        bbox: tuple[float, float, float, float],
        text_content: str,
        features: dict,
        confidence: float | None = None,
        polygon: list[tuple[float, float]] | None = None,
    ) -> str:
        """Persist one detection and return its ``det_…`` ID."""
        detection_id = _gen_id("det_")
        poly_json = json.dumps(polygon) if polygon else None
        with self._write_lock():
            self._conn.execute(
                "INSERT INTO detections "
                "(detection_id, doc_id, page, run_id, element_type, confidence, "
                " bbox_x0, bbox_y0, bbox_x1, bbox_y1, text_content, "
                " features_json, polygon_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    detection_id,
                    doc_id,
                    page,
                    run_id,
                    element_type,
                    confidence,
                    *bbox,
                    text_content,
                    json.dumps(features),
                    poly_json,
                    _utcnow_iso(),
                ),
            )
            self._conn.commit()
        return detection_id

    def purge_old_detections_for_doc(self, doc_id: str, keep_run_id: str) -> int:
        """Delete old pipeline detections for *doc_id*, keeping only *keep_run_id*.

        Manual annotations (``run_id`` starting with ``'manual'``) are
        always preserved.  Orphaned ``box_groups`` / ``box_group_members``
        whose detections no longer exist are cleaned up as well.

        Corrections and training_examples that reference purged detections
        have their detection_id set to NULL (preserving user annotations).

        Returns the number of deleted detection rows.
        """
        with self._write_lock():
            # Build list of detection_ids that will be deleted
            to_delete = [
                row[0]
                for row in self._conn.execute(
                    "SELECT detection_id FROM detections "
                    "WHERE doc_id = ? AND run_id != ? AND run_id NOT LIKE 'manual%'",
                    (doc_id, keep_run_id),
                ).fetchall()
            ]

            if not to_delete:
                return 0

            placeholders = ",".join("?" * len(to_delete))

            # Nullify FK references in corrections (preserve user annotations)
            self._conn.execute(
                f"UPDATE corrections SET detection_id = NULL "
                f"WHERE detection_id IN ({placeholders})",
                to_delete,
            )

            # Delete training_examples referencing these detections
            # (training examples without a detection are not useful)
            self._conn.execute(
                f"DELETE FROM training_examples "
                f"WHERE detection_id IN ({placeholders})",
                to_delete,
            )

            # Delete box_group_members referencing these detections
            self._conn.execute(
                f"DELETE FROM box_group_members "
                f"WHERE detection_id IN ({placeholders})",
                to_delete,
            )

            # Delete box_groups whose root_detection_id will be deleted
            self._conn.execute(
                f"DELETE FROM box_groups "
                f"WHERE root_detection_id IN ({placeholders})",
                to_delete,
            )

            # Remove empty groups (no remaining members)
            self._conn.execute(
                "DELETE FROM box_groups WHERE group_id NOT IN "
                "(SELECT DISTINCT group_id FROM box_group_members)"
            )

            # Now delete the detections
            cur = self._conn.execute(
                f"DELETE FROM detections WHERE detection_id IN ({placeholders})",
                to_delete,
            )
            n_deleted = cur.rowcount

            self._conn.commit()
        return n_deleted

    def purge_all_stale_detections(self) -> int:
        """For every document, keep only the latest pipeline run's detections.

        Manual annotations are always preserved.  Returns the total
        number of deleted detection rows.
        """
        with self._write_lock():
            # Find the latest non-manual run_id per doc_id.
            # Use a correlated subquery so we get the run_id that
            # actually corresponds to MAX(created_at), not an
            # arbitrary row from the GROUP BY.
            latest_rows = self._conn.execute(
                "SELECT DISTINCT d.doc_id, d.run_id "
                "FROM detections d "
                "WHERE d.run_id NOT LIKE 'manual%' "
                "  AND d.created_at = ("
                "      SELECT MAX(d2.created_at) FROM detections d2 "
                "      WHERE d2.doc_id = d.doc_id "
                "        AND d2.run_id NOT LIKE 'manual%'"
                "  )"
            ).fetchall()
            keep = {r["doc_id"]: r["run_id"] for r in latest_rows}

            total = 0
            for did, rid in keep.items():
                cur = self._conn.execute(
                    "DELETE FROM detections "
                    "WHERE doc_id = ? AND run_id != ? AND run_id NOT LIKE 'manual%'",
                    (did, rid),
                )
                total += cur.rowcount

            # Cascade-clean orphaned group members / groups
            self._conn.execute(
                "DELETE FROM box_group_members "
                "WHERE detection_id NOT IN (SELECT detection_id FROM detections)"
            )
            self._conn.execute(
                "DELETE FROM box_groups WHERE group_id NOT IN "
                "(SELECT DISTINCT group_id FROM box_group_members)"
            )
            self._conn.commit()
        return total

    def update_detection_polygon(
        self,
        detection_id: str,
        polygon: list[tuple[float, float]] | None,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> None:
        """Update the polygon (and optionally bbox) of an existing detection."""
        poly_json = json.dumps(polygon) if polygon else None
        with self._write_lock():
            if bbox is not None:
                self._conn.execute(
                    "UPDATE detections SET polygon_json = ?, "
                    "bbox_x0 = ?, bbox_y0 = ?, bbox_x1 = ?, bbox_y1 = ? "
                    "WHERE detection_id = ?",
                    (poly_json, *bbox, detection_id),
                )
            else:
                self._conn.execute(
                    "UPDATE detections SET polygon_json = ? WHERE detection_id = ?",
                    (poly_json, detection_id),
                )
            self._conn.commit()

    def update_detection_confidence(
        self,
        detection_id: str,
        confidence: float,
    ) -> None:
        """Update the confidence score of an existing detection.

        Parameters
        ----------
        detection_id : str
            The ID of the detection to update.
        confidence : float
            The new confidence score (0.0-1.0).
        """
        with self._write_lock():
            self._conn.execute(
                "UPDATE detections SET confidence = ? WHERE detection_id = ?",
                (confidence, detection_id),
            )
            self._conn.commit()

    def update_detection_label(
        self,
        detection_id: str,
        element_type: str,
    ) -> None:
        """Update the element type (label) of an existing detection.

        Parameters
        ----------
        detection_id : str
            The ID of the detection to update.
        element_type : str
            The new element type label.
        """
        with self._write_lock():
            self._conn.execute(
                "UPDATE detections SET element_type = ? WHERE detection_id = ?",
                (element_type, detection_id),
            )
            self._conn.commit()

    def update_detection_label_and_bbox(
        self,
        detection_id: str,
        element_type: str,
        bbox: tuple[float, float, float, float],
    ) -> None:
        """Update both label and bounding box of an existing detection.

        Parameters
        ----------
        detection_id : str
            The ID of the detection to update.
        element_type : str
            The new element type label.
        bbox : tuple[float, float, float, float]
            The new bounding box (x0, y0, x1, y1).
        """
        with self._write_lock():
            self._conn.execute(
                "UPDATE detections SET element_type = ?, "
                "bbox_x0 = ?, bbox_y0 = ?, bbox_x1 = ?, bbox_y1 = ? "
                "WHERE detection_id = ?",
                (element_type, *bbox, detection_id),
            )
            self._conn.commit()

    def update_detection_text_and_features(
        self,
        detection_id: str,
        text_content: str,
        features: dict | None = None,
    ) -> None:
        """Update the text content and optionally features of a detection.

        Parameters
        ----------
        detection_id : str
            The ID of the detection to update.
        text_content : str
            The new text content extracted from the PDF region.
        features : dict | None
            If provided, replaces the stored features_json.
        """
        with self._write_lock():
            if features is not None:
                self._conn.execute(
                    "UPDATE detections SET text_content = ?, features_json = ? "
                    "WHERE detection_id = ?",
                    (text_content, json.dumps(features), detection_id),
                )
            else:
                self._conn.execute(
                    "UPDATE detections SET text_content = ? "
                    "WHERE detection_id = ?",
                    (text_content, detection_id),
                )
            self._conn.commit()

    # ── corrections ────────────────────────────────────────────────────

    def save_correction(
        self,
        doc_id: str,
        page: int,
        correction_type: str,
        corrected_label: str,
        corrected_bbox: tuple[float, float, float, float],
        detection_id: str | None = None,
        original_label: str | None = None,
        original_bbox: tuple[float, float, float, float] | None = None,
        corrected_text: str | None = None,
        corrected_features: dict | None = None,
        annotator: str = "user",
        session_id: str = "",
        notes: str = "",
    ) -> str:
        """Persist a human correction and return its ``cor_…`` ID."""
        correction_id = _gen_id("cor_")
        orig = original_bbox or (None, None, None, None)
        with self._write_lock():
            self._conn.execute(
                "INSERT INTO corrections "
                "(correction_id, detection_id, doc_id, page, correction_type, "
                " original_element_type, corrected_element_type, "
                " orig_bbox_x0, orig_bbox_y0, orig_bbox_x1, orig_bbox_y1, "
                " corr_bbox_x0, corr_bbox_y0, corr_bbox_x1, corr_bbox_y1, "
                " corrected_text, corrected_features_json, "
                " annotator, corrected_at, session_id, notes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    correction_id,
                    detection_id,
                    doc_id,
                    page,
                    correction_type,
                    original_label,
                    corrected_label,
                    *orig,
                    *corrected_bbox,
                    corrected_text,
                    json.dumps(corrected_features) if corrected_features else None,
                    annotator,
                    _utcnow_iso(),
                    session_id,
                    notes,
                ),
            )
            self._conn.commit()
        return correction_id

    def accept_detection(self, detection_id: str, doc_id: str, page: int) -> str:
        """Mark a detection as correct (no change needed).

        Looks up the detection's element_type and bbox, then writes a
        correction with ``correction_type='accept'`` so it enters the
        training set as a positive example.
        """
        row = self._conn.execute(
            "SELECT element_type, bbox_x0, bbox_y0, bbox_x1, bbox_y1 "
            "FROM detections WHERE detection_id = ?",
            (detection_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Detection {detection_id!r} not found")
        original_label = row["element_type"]
        original_bbox = (
            row["bbox_x0"],
            row["bbox_y0"],
            row["bbox_x1"],
            row["bbox_y1"],
        )
        return self.save_correction(
            doc_id=doc_id,
            page=page,
            correction_type="accept",
            corrected_label=original_label,
            corrected_bbox=original_bbox,
            detection_id=detection_id,
            original_label=original_label,
            original_bbox=original_bbox,
        )

    # ── queries ────────────────────────────────────────────────────────

    def get_detections_for_page(self, doc_id: str, page: int) -> list[dict[str, Any]]:
        """Return all detections for *doc_id* + *page* as a list of dicts."""
        rows = self._conn.execute(
            "SELECT * FROM detections WHERE doc_id = ? AND page = ? "
            "ORDER BY created_at",
            (doc_id, page),
        ).fetchall()
        return self._rows_to_detection_dicts(rows)

    def get_latest_detections_for_page(
        self, doc_id: str, page: int
    ) -> list[dict[str, Any]]:
        """Return detections for this page from the *globally* latest run.

        The latest ``run_id`` is determined across **all** pages for this
        document (not per-page), so pages that were not part of the most
        recent pipeline run return an empty list instead of stale data.

        Manual annotations are only included for pages that ALSO have
        pipeline detections from the latest run. This prevents showing
        stale manual annotations on pages that weren't processed.
        """
        # Find the globally latest pipeline run_id for this document
        row = self._conn.execute(
            "SELECT run_id FROM detections "
            "WHERE doc_id = ? AND run_id NOT LIKE 'manual%' "
            "ORDER BY created_at DESC LIMIT 1",
            (doc_id,),
        ).fetchone()
        if not row:
            # No pipeline detections at all — return empty
            # (manual-only pages are not shown until a pipeline run)
            return []

        latest_run_id = row["run_id"]

        # Check if this specific page was processed in the latest run
        page_in_latest_run = self._conn.execute(
            "SELECT 1 FROM detections "
            "WHERE doc_id = ? AND page = ? AND run_id = ? LIMIT 1",
            (doc_id, page, latest_run_id),
        ).fetchone()

        if not page_in_latest_run:
            # This page was NOT processed in the latest run — return empty
            return []

        # This page WAS processed — include pipeline detections + manual annotations
        rows = self._conn.execute(
            "SELECT * FROM detections "
            "WHERE doc_id = ? AND page = ? "
            "  AND (run_id = ? OR run_id LIKE 'manual%') "
            "ORDER BY created_at",
            (doc_id, page, latest_run_id),
        ).fetchall()
        return self._rows_to_detection_dicts(rows)

    @staticmethod
    def _rows_to_detection_dicts(rows) -> list[dict[str, Any]]:
        """Convert SQLite rows to detection dicts."""
        results: list[dict[str, Any]] = []
        for r in rows:
            poly_raw = r["polygon_json"]
            polygon = json.loads(poly_raw) if poly_raw else None
            if polygon:
                polygon = [tuple(p) for p in polygon]
            results.append(
                {
                    "detection_id": r["detection_id"],
                    "element_type": r["element_type"],
                    "confidence": r["confidence"],
                    "bbox": (
                        r["bbox_x0"],
                        r["bbox_y0"],
                        r["bbox_x1"],
                        r["bbox_y1"],
                    ),
                    "text_content": r["text_content"],
                    "features": json.loads(r["features_json"]),
                    "polygon": polygon,
                    "created_at": r["created_at"],
                }
            )
        return results

    def has_detections_for_doc(self, doc_id: str) -> bool:
        """Return *True* if any pipeline detection exists for *doc_id*."""
        row = self._conn.execute(
            "SELECT 1 FROM detections "
            "WHERE doc_id = ? AND run_id NOT LIKE 'manual%' LIMIT 1",
            (doc_id,),
        ).fetchone()
        return row is not None

    def get_corrections_for_page(self, doc_id: str, page: int) -> list[dict[str, Any]]:
        """Return all corrections for *doc_id* + *page*, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM corrections WHERE doc_id = ? AND page = ? "
            "ORDER BY corrected_at DESC",
            (doc_id, page),
        ).fetchall()
        results: list[dict[str, Any]] = []
        for r in rows:
            d: dict[str, Any] = dict(r)
            # Reconstruct bbox tuples
            d["original_bbox"] = (
                d.pop("orig_bbox_x0"),
                d.pop("orig_bbox_y0"),
                d.pop("orig_bbox_x1"),
                d.pop("orig_bbox_y1"),
            )
            d["corrected_bbox"] = (
                d.pop("corr_bbox_x0"),
                d.pop("corr_bbox_y0"),
                d.pop("corr_bbox_x1"),
                d.pop("corr_bbox_y1"),
            )
            if d.get("corrected_features_json"):
                d["corrected_features"] = json.loads(d["corrected_features_json"])
            else:
                d["corrected_features"] = None
            results.append(d)
        return results

    # ── feedback helpers ─────────────────────────────────────────────────
    # Box group methods are provided by BoxGroupMixin

    def get_prior_corrections_by_bbox(
        self,
        doc_id: str,
        page: int,
        *,
        iou_threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Return prior corrections matched by bounding-box overlap.

        For each detection on *doc_id* + *page* that has at least one
        non-delete correction, returns the **most recent** correction
        with the corrected label, bbox, and type.  This enables
        carrying forward user corrections across pipeline re-runs
        (where detection IDs change) by matching spatially.

        Parameters
        ----------
        doc_id : str
            Document hash.
        page : int
            Zero-based page number.
        iou_threshold : float
            Not used for DB-side lookup (we rely on the FK link between
            detections and corrections), but kept for future spatial
            matching when the detection_id link is unavailable.

        Returns
        -------
        list[dict]
            Each dict: ``detection_id``, ``correction_type``,
            ``corrected_label``, ``corrected_bbox``, ``original_label``,
            ``original_bbox``.
        """
        rows = self._conn.execute(
            "SELECT c.detection_id, c.correction_type, "
            "       c.corrected_element_type AS corrected_label, "
            "       c.corr_bbox_x0, c.corr_bbox_y0, "
            "       c.corr_bbox_x1, c.corr_bbox_y1, "
            "       c.original_element_type AS original_label, "
            "       c.orig_bbox_x0, c.orig_bbox_y0, "
            "       c.orig_bbox_x1, c.orig_bbox_y1 "
            "FROM corrections c "
            "JOIN detections d ON c.detection_id = d.detection_id "
            "WHERE c.doc_id = ? AND c.page = ? "
            "  AND c.correction_id = ("
            "      SELECT c2.correction_id FROM corrections c2 "
            "      WHERE c2.detection_id = c.detection_id "
            "      ORDER BY c2.corrected_at DESC, c2.ROWID DESC LIMIT 1"
            "  )",
            (doc_id, page),
        ).fetchall()

        results: list[dict[str, Any]] = []
        for r in rows:
            results.append(
                {
                    "detection_id": r["detection_id"],
                    "correction_type": r["correction_type"],
                    "corrected_label": r["corrected_label"],
                    "corrected_bbox": (
                        r["corr_bbox_x0"],
                        r["corr_bbox_y0"],
                        r["corr_bbox_x1"],
                        r["corr_bbox_y1"],
                    ),
                    "original_label": r["original_label"],
                    "original_bbox": (
                        r["orig_bbox_x0"],
                        r["orig_bbox_y0"],
                        r["orig_bbox_x1"],
                        r["orig_bbox_y1"],
                    ),
                }
            )
        return results

    # ── Database-tab helpers ───────────────────────────────────────────
    # Database overview and summary methods are provided by DbHelpersMixin
    # (see module docstring for full list of mixins)
