"""SQLite-backed store for detections, corrections, and training examples.

Persists pipeline detections and human corrections to
``data/corrections.db`` (by default).  Provides helpers to build
training sets for downstream classifiers.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def _utcnow_iso() -> str:
    """Return current UTC time as ISO-8601 string with 'Z' suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _gen_id(prefix: str) -> str:
    """Generate a short prefixed ID — e.g. ``det_a1b2c3d4``."""
    return f"{prefix}{uuid4().hex[:8]}"


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
"""


class CorrectionStore:
    """Thin wrapper around an SQLite database for annotation persistence.

    Parameters
    ----------
    db_path : Path
        Location of the database file.  Parent directories are created
        automatically on first use.
    """

    def __init__(self, db_path: Path = Path("data/corrections.db")) -> None:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_DDL)
        self._conn.commit()
        self._migrate()

    # ── helpers ────────────────────────────────────────────────────────

    def _migrate(self) -> None:
        """Apply lightweight schema migrations for columns added after v1."""
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

    def update_detection_polygon(
        self,
        detection_id: str,
        polygon: list[tuple[float, float]] | None,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> None:
        """Update the polygon (and optionally bbox) of an existing detection."""
        poly_json = json.dumps(polygon) if polygon else None
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
        results: list[dict[str, Any]] = []
        for r in rows:
            poly_raw = r["polygon_json"]
            polygon = json.loads(poly_raw) if poly_raw else None
            # Convert inner lists back to tuples for consistency
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

    # ── box groups ─────────────────────────────────────────────────────

    def create_group(
        self,
        doc_id: str,
        page: int,
        group_label: str,
        root_detection_id: str,
    ) -> str:
        """Create a box group with *root_detection_id* as the parent.

        The root detection is also added as the first member (sort_order 0).
        Returns the ``grp_…`` group ID.
        """
        group_id = _gen_id("grp_")
        self._conn.execute(
            "INSERT INTO box_groups "
            "(group_id, doc_id, page, group_label, root_detection_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (group_id, doc_id, page, group_label, root_detection_id, _utcnow_iso()),
        )
        # Root is always member 0
        self._conn.execute(
            "INSERT OR IGNORE INTO box_group_members "
            "(group_id, detection_id, sort_order) VALUES (?, ?, 0)",
            (group_id, root_detection_id),
        )
        self._conn.commit()
        return group_id

    def add_to_group(
        self, group_id: str, detection_id: str, sort_order: int = 0
    ) -> None:
        """Add a detection as a child member of a group."""
        self._conn.execute(
            "INSERT OR IGNORE INTO box_group_members "
            "(group_id, detection_id, sort_order) VALUES (?, ?, ?)",
            (group_id, detection_id, sort_order),
        )
        self._conn.commit()

    def remove_from_group(self, group_id: str, detection_id: str) -> None:
        """Remove a detection from a group.

        If the removed detection is the group root, the entire group
        is deleted (all members removed).
        """
        # Check if this is the root
        row = self._conn.execute(
            "SELECT root_detection_id FROM box_groups WHERE group_id = ?",
            (group_id,),
        ).fetchone()
        if row and row["root_detection_id"] == detection_id:
            self.delete_group(group_id)
            return
        self._conn.execute(
            "DELETE FROM box_group_members " "WHERE group_id = ? AND detection_id = ?",
            (group_id, detection_id),
        )
        self._conn.commit()

    def delete_group(self, group_id: str) -> None:
        """Delete a group and all its member associations."""
        self._conn.execute(
            "DELETE FROM box_group_members WHERE group_id = ?", (group_id,)
        )
        self._conn.execute("DELETE FROM box_groups WHERE group_id = ?", (group_id,))
        self._conn.commit()

    def get_groups_for_page(self, doc_id: str, page: int) -> list[dict[str, Any]]:
        """Return all groups on a page with their members."""
        groups = self._conn.execute(
            "SELECT * FROM box_groups WHERE doc_id = ? AND page = ? "
            "ORDER BY created_at",
            (doc_id, page),
        ).fetchall()
        results: list[dict[str, Any]] = []
        for g in groups:
            members = self._conn.execute(
                "SELECT detection_id, sort_order FROM box_group_members "
                "WHERE group_id = ? ORDER BY sort_order",
                (g["group_id"],),
            ).fetchall()
            results.append(
                {
                    "group_id": g["group_id"],
                    "group_label": g["group_label"],
                    "root_detection_id": g["root_detection_id"],
                    "members": [
                        {
                            "detection_id": m["detection_id"],
                            "sort_order": m["sort_order"],
                        }
                        for m in members
                    ],
                }
            )
        return results

    def get_group_for_detection(self, detection_id: str) -> dict[str, Any] | None:
        """Return the group a detection belongs to, or *None*."""
        row = self._conn.execute(
            "SELECT g.group_id, g.group_label, g.root_detection_id "
            "FROM box_group_members m "
            "JOIN box_groups g ON m.group_id = g.group_id "
            "WHERE m.detection_id = ?",
            (detection_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "group_id": row["group_id"],
            "group_label": row["group_label"],
            "root_detection_id": row["root_detection_id"],
        }

    # ── feedback helpers ─────────────────────────────────────────────────

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

    # ── training set ───────────────────────────────────────────────────

    @staticmethod
    def _deterministic_sort_key(detection_id: str) -> str:
        """Return a deterministic sort key for a detection_id.

        Uses ``hashlib.md5`` to produce a stable hex digest that is
        consistent across ``PYTHONHASHSEED`` values, Python versions,
        and machines.
        """
        return hashlib.md5(detection_id.encode()).hexdigest()

    def build_training_set(self) -> int:
        """Rebuild the ``training_examples`` table from corrections.

        For each non-delete correction, creates a training example
        whose label is the *corrected* element type.  The split is
        **truly stratified**: within each label group, examples are
        sorted by a deterministic MD5 hash of ``detection_id`` then
        the first 70 % are assigned to ``train``, the next 20 % to
        ``val``, and the remaining 10 % to ``test``.

        This guarantees every class with ≥2 examples has
        representation in at least two splits.

        Returns the number of examples inserted.
        """
        from collections import defaultdict

        self._conn.execute("DELETE FROM training_examples")

        rows = self._conn.execute(
            "SELECT c.correction_id, c.detection_id, c.corrected_element_type, "
            "       c.corrected_features_json, d.features_json "
            "FROM corrections c "
            "JOIN detections d ON c.detection_id = d.detection_id "
            "WHERE c.correction_type != 'delete' "
            "  AND c.detection_id IS NOT NULL"
        ).fetchall()

        # ── Stratified split: group by label, force distribution ────
        by_label: dict[str, list] = defaultdict(list)
        for r in rows:
            by_label[r["corrected_element_type"]].append(r)

        now = _utcnow_iso()
        count = 0
        for _label, group in sorted(by_label.items()):
            # Sort deterministically within this class using MD5
            group.sort(key=lambda r: self._deterministic_sort_key(r["detection_id"]))
            n = len(group)
            train_end = max(1, int(n * 0.7))  # at least 1 in train
            val_end = max(train_end + 1, int(n * 0.9)) if n >= 2 else train_end
            for i, r in enumerate(group):
                features_json = r["corrected_features_json"] or r["features_json"]
                if i < train_end:
                    split = "train"
                elif i < val_end:
                    split = "val"
                else:
                    split = "test"
                self._conn.execute(
                    "INSERT INTO training_examples "
                    "(example_id, source, correction_id, detection_id, "
                    " label, features_json, split, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        _gen_id("ex_"),
                        "correction",
                        r["correction_id"],
                        r["detection_id"],
                        r["corrected_element_type"],
                        features_json,
                        split,
                        now,
                    ),
                )
                count += 1

        self._conn.commit()
        return count

    def export_training_jsonl(self, output_path: Path) -> int:
        """Write training examples as JSON-Lines to *output_path*.

        Each line: ``{"example_id": …, "label": …, "features": {…}, "split": …}``

        Returns the number of lines written.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows = self._conn.execute(
            "SELECT example_id, label, features_json, split " "FROM training_examples"
        ).fetchall()

        count = 0
        with open(output_path, "w", encoding="utf-8") as fh:
            for r in rows:
                obj = {
                    "example_id": r["example_id"],
                    "label": r["label"],
                    "features": json.loads(r["features_json"]),
                    "split": r["split"],
                }
                fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count += 1
        return count

    # ── snapshots ──────────────────────────────────────────────────────

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

        # Close current connection
        self._conn.close()

        # Replace
        shutil.copy2(str(snapshot_path), str(self._db_path))

        # Reconnect
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    # ── training runs ──────────────────────────────────────────────────

    def save_training_run(
        self,
        metrics: dict,
        model_path: str = "",
        notes: str = "",
        holdout_predictions: list[dict] | None = None,
        hyperparams: dict | None = None,
        feature_set: dict | None = None,
        training_curves: dict | None = None,
        feature_version: int = 0,
    ) -> str:
        """Persist a training-run record and return its ``run_…`` ID.

        Parameters
        ----------
        metrics : dict
            Output of :func:`~plancheck.corrections.metrics.compute_metrics`,
            augmented with ``n_train`` and ``n_val``.
        model_path : str
            Path to the saved model file (informational).
        notes : str
            Free-text annotation for this run.
        holdout_predictions : list[dict] | None
            Optional list of ``{"label_true", "label_pred", "confidence"}``
            dicts from the validation set evaluation.
        hyperparams : dict | None
            Hyperparameters used for training (Phase 4.4).
        feature_set : dict | None
            Feature schema description (Phase 4.4).
        training_curves : dict | None
            Epoch-level training/validation loss curves (Phase 4.4).
        feature_version : int
            Feature schema version (Phase 4.3).
        """
        run_id = _gen_id("run_")
        hp_json = json.dumps(holdout_predictions) if holdout_predictions else ""
        hyper_json = json.dumps(hyperparams) if hyperparams else ""
        fs_json = json.dumps(feature_set) if feature_set else ""
        tc_json = json.dumps(training_curves) if training_curves else ""
        self._conn.execute(
            "INSERT INTO training_runs "
            "(run_id, trained_at, n_train, n_val, accuracy, f1_macro, f1_weighted, "
            " labels_json, per_class_json, model_path, notes, holdout_preds_json, "
            " hyperparams_json, feature_set_json, training_curves_json, "
            " feature_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                _utcnow_iso(),
                metrics.get("n_train", 0),
                metrics.get("n_val", 0),
                metrics.get("accuracy", 0.0),
                metrics.get("f1_macro", 0.0),
                metrics.get("f1_weighted", 0.0),
                json.dumps(metrics.get("labels", [])),
                json.dumps(metrics.get("per_class", {})),
                model_path,
                notes,
                hp_json,
                hyper_json,
                fs_json,
                tc_json,
                feature_version,
            ),
        )
        self._conn.commit()
        return run_id

    def get_training_history(self) -> list[dict[str, Any]]:
        """Return all training runs ordered newest-first.

        Returns
        -------
        list[dict]
            Each dict has: ``run_id``, ``trained_at``, ``n_train``,
            ``n_val``, ``accuracy``, ``f1_macro``, ``f1_weighted``,
            ``labels``, ``per_class``, ``model_path``, ``notes``,
            ``holdout_predictions``, ``hyperparams``, ``feature_set``,
            ``training_curves``, ``feature_version``.
        """
        rows = self._conn.execute(
            "SELECT * FROM training_runs ORDER BY trained_at DESC, rowid DESC"
        ).fetchall()
        results: list[dict[str, Any]] = []
        for r in rows:
            keys = r.keys()
            hp_raw = r["holdout_preds_json"] if "holdout_preds_json" in keys else ""
            hyper_raw = r["hyperparams_json"] if "hyperparams_json" in keys else ""
            fs_raw = r["feature_set_json"] if "feature_set_json" in keys else ""
            tc_raw = r["training_curves_json"] if "training_curves_json" in keys else ""
            fv = r["feature_version"] if "feature_version" in keys else 0
            results.append(
                {
                    "run_id": r["run_id"],
                    "trained_at": r["trained_at"],
                    "n_train": r["n_train"],
                    "n_val": r["n_val"],
                    "accuracy": r["accuracy"],
                    "f1_macro": r["f1_macro"],
                    "f1_weighted": r["f1_weighted"],
                    "labels": json.loads(r["labels_json"]),
                    "per_class": json.loads(r["per_class_json"]),
                    "model_path": r["model_path"],
                    "notes": r["notes"],
                    "holdout_predictions": json.loads(hp_raw) if hp_raw else [],
                    "hyperparams": json.loads(hyper_raw) if hyper_raw else {},
                    "feature_set": json.loads(fs_raw) if fs_raw else {},
                    "training_curves": json.loads(tc_raw) if tc_raw else {},
                    "feature_version": fv,
                }
            )
        return results

    def compare_runs(self, run_id_a: str, run_id_b: str) -> dict[str, Any]:
        """Compare two training runs and return metric deltas.

        Parameters
        ----------
        run_id_a, run_id_b : str
            Run IDs to compare.  Convention: *a* is the baseline (older),
            *b* is the candidate (newer).

        Returns
        -------
        dict
            ``f1_weighted_delta``, ``accuracy_delta``,
            ``improved_classes``, ``regressed_classes``,
            ``per_class_deltas``.

        Raises
        ------
        ValueError
            If either *run_id* is not found.
        """
        history = {r["run_id"]: r for r in self.get_training_history()}
        if run_id_a not in history:
            raise ValueError(f"Run not found: {run_id_a}")
        if run_id_b not in history:
            raise ValueError(f"Run not found: {run_id_b}")

        a = history[run_id_a]
        b = history[run_id_b]

        f1_delta = b["f1_weighted"] - a["f1_weighted"]
        acc_delta = b["accuracy"] - a["accuracy"]

        # Per-class F1 deltas
        all_classes = sorted(set(a["per_class"]) | set(b["per_class"]))
        per_class_deltas: dict[str, float] = {}
        improved: list[str] = []
        regressed: list[str] = []
        for cls in all_classes:
            f1_a = a["per_class"].get(cls, {}).get("f1", 0.0)
            f1_b = b["per_class"].get(cls, {}).get("f1", 0.0)
            delta = f1_b - f1_a
            per_class_deltas[cls] = round(delta, 6)
            if delta > 0.005:
                improved.append(cls)
            elif delta < -0.005:
                regressed.append(cls)

        return {
            "f1_weighted_delta": round(f1_delta, 6),
            "accuracy_delta": round(acc_delta, 6),
            "improved_classes": improved,
            "regressed_classes": regressed,
            "per_class_deltas": per_class_deltas,
        }

    # ── retrain helpers (Phase 4.2) ────────────────────────────────

    def last_train_date(self) -> str | None:
        """Return ISO-8601 timestamp of the most recent training run.

        Returns *None* when no runs have been recorded.
        """
        row = self._conn.execute(
            "SELECT trained_at FROM training_runs ORDER BY trained_at DESC LIMIT 1"
        ).fetchone()
        return row["trained_at"] if row else None

    def count_corrections_since(self, since_iso: str | None = None) -> int:
        """Count corrections added after *since_iso*.

        Parameters
        ----------
        since_iso : str | None
            ISO-8601 timestamp.  When *None*, returns the total count.
        """
        if since_iso is None:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM corrections").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM corrections WHERE corrected_at > ?",
                (since_iso,),
            ).fetchone()
        return row["n"] if row else 0

    def count_corrections_since_last_train(self) -> int:
        """Return number of new corrections since the last training run."""
        return self.count_corrections_since(self.last_train_date())

    def should_retrain(self, threshold: int = 50) -> bool:
        """Return *True* when enough new corrections have accumulated.

        Parameters
        ----------
        threshold : int
            Minimum number of new corrections needed (default 50).
        """
        return self.count_corrections_since_last_train() >= threshold

    # ── feature cache (Phase 4.3) ─────────────────────────────────

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
        total = self._conn.execute(
            "SELECT COUNT(*) AS n FROM feature_cache"
        ).fetchone()["n"]
        det_count = self._conn.execute(
            "SELECT COUNT(DISTINCT detection_id) AS n FROM feature_cache"
        ).fetchone()["n"]
        ver_count = self._conn.execute(
            "SELECT COUNT(DISTINCT feature_version) AS n FROM feature_cache"
        ).fetchone()["n"]
        return {
            "total_entries": total,
            "distinct_detections": det_count,
            "distinct_versions": ver_count,
        }

    # ── candidate outcomes (Level 2 learning) ──────────────────────

    def save_candidate_outcome(
        self,
        page: int,
        trigger_methods: list[str],
        outcome: str,
        confidence: float,
        bbox: tuple[float, float, float, float],
        *,
        doc_id: str = "",
        run_id: str = "",
        predicted_symbol: str = "",
        found_symbol: str = "",
        page_width: float = 0.0,
        page_height: float = 0.0,
        features: dict | None = None,
    ) -> str:
        """Persist one VOCR candidate outcome and return its ``co_…`` ID.

        Parameters
        ----------
        page : int
            Zero-based page index.
        trigger_methods : list[str]
            Detection methods that flagged this candidate.
        outcome : str
            ``"hit"`` or ``"miss"``.
        confidence : float
            Candidate confidence at decision time.
        bbox : tuple
            ``(x0, y0, x1, y1)`` in PDF points.
        doc_id, run_id : str
            Optional document / run identifiers.
        predicted_symbol, found_symbol : str
            The symbol the candidate predicted and what VOCR actually found.
        page_width, page_height : float
            Page dimensions (for normalisation later).
        features : dict | None
            Pre-computed feature vector for the classifier.
        """
        outcome_id = _gen_id("co_")
        self._conn.execute(
            "INSERT INTO candidate_outcomes "
            "(outcome_id, doc_id, page, run_id, trigger_methods, "
            " predicted_symbol, found_symbol, outcome, confidence, "
            " bbox_x0, bbox_y0, bbox_x1, bbox_y1, "
            " page_width, page_height, features_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                outcome_id,
                doc_id,
                page,
                run_id,
                ",".join(trigger_methods),
                predicted_symbol,
                found_symbol,
                outcome,
                confidence,
                *bbox,
                page_width,
                page_height,
                json.dumps(features or {}),
                _utcnow_iso(),
            ),
        )
        self._conn.commit()
        return outcome_id

    def save_candidate_outcomes_batch(
        self,
        candidates: list,
        *,
        doc_id: str = "",
        run_id: str = "",
        page_width: float = 0.0,
        page_height: float = 0.0,
    ) -> int:
        """Persist a batch of :class:`VocrCandidate` objects.

        Parameters
        ----------
        candidates : list[VocrCandidate]
            Candidates with ``outcome`` set to ``"hit"`` or ``"miss"``.
        doc_id, run_id : str
            Optional context identifiers.
        page_width, page_height : float
            Page dimensions.

        Returns
        -------
        int
            Number of rows inserted.
        """
        now = _utcnow_iso()
        count = 0
        for c in candidates:
            if c.outcome not in ("hit", "miss"):
                continue
            outcome_id = _gen_id("co_")
            self._conn.execute(
                "INSERT INTO candidate_outcomes "
                "(outcome_id, doc_id, page, run_id, trigger_methods, "
                " predicted_symbol, found_symbol, outcome, confidence, "
                " bbox_x0, bbox_y0, bbox_x1, bbox_y1, "
                " page_width, page_height, features_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    outcome_id,
                    doc_id,
                    c.page,
                    run_id,
                    ",".join(c.trigger_methods),
                    c.predicted_symbol or "",
                    c.found_symbol or "",
                    c.outcome,
                    c.confidence,
                    c.x0,
                    c.y0,
                    c.x1,
                    c.y1,
                    page_width,
                    page_height,
                    json.dumps({}),
                    now,
                ),
            )
            count += 1
        if count:
            self._conn.commit()
        return count

    def get_candidate_outcomes(
        self,
        *,
        min_rows: int = 0,
        outcome: str | None = None,
    ) -> list[dict]:
        """Retrieve candidate outcome rows for classifier training.

        Parameters
        ----------
        min_rows : int
            If the table has fewer than this many rows, return an empty
            list (avoids training on tiny datasets).
        outcome : str | None
            Filter to ``"hit"`` or ``"miss"`` only.  ``None`` = all.

        Returns
        -------
        list[dict]
            Each dict has keys matching the ``candidate_outcomes`` schema.
        """
        total = self._conn.execute(
            "SELECT COUNT(*) AS n FROM candidate_outcomes"
        ).fetchone()["n"]
        if total < min_rows:
            return []

        query = "SELECT * FROM candidate_outcomes"
        params: list = []
        if outcome:
            query += " WHERE outcome = ?"
            params.append(outcome)
        query += " ORDER BY created_at"

        return [dict(r) for r in self._conn.execute(query, params).fetchall()]

    def count_candidate_outcomes(self) -> dict[str, int]:
        """Return ``{total, hits, misses}`` counts."""
        rows = self._conn.execute(
            "SELECT outcome, COUNT(*) AS n FROM candidate_outcomes GROUP BY outcome"
        ).fetchall()
        counts = {r["outcome"]: r["n"] for r in rows}
        return {
            "total": sum(counts.values()),
            "hits": counts.get("hit", 0),
            "misses": counts.get("miss", 0),
        }
