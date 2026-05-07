"""Group snapshot and correction storage mixin for CorrectionStore.

Provides persistence for the two new grouper training tables:

- ``group_snapshots``  — IsGroup fingerprint records (the Example Store).
  One row per human-confirmed group, carrying full spatial geometry.

- ``group_corrections`` — Four-signal correction log (Yes / IsGroup / Edit /
  NotGroup).  Records what the machine had and what the human changed.

This module follows the same mixin pattern as :mod:`.box_groups` and
:mod:`.training_data` — pure DB operations only; all geometry computation
lives in :mod:`plancheck.grouping.snapshot_geometry`.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional

from .store_utils import _gen_id, _utcnow_iso

# Valid grouper correction signals — kept as a frozenset for O(1) lookup.
_VALID_SIGNALS: frozenset[str] = frozenset({"yes", "is_group", "edit", "not_group"})


class GroupSnapshotMixin:
    """Mixin providing group snapshot and correction persistence.

    Requires the host class to expose ``_conn`` (sqlite3 connection) and
    ``_write_lock()`` (context manager), both provided by ``CorrectionStore``.
    """

    # Provided by CorrectionStore
    _conn: Any
    _write_lock: Any

    # ── Example Store ─────────────────────────────────────────────────

    def save_group_snapshot(
        self,
        session_id: str,
        pdf_filename: str,
        page_number: int,
        boxes_dicts: List[dict],
        group_geometry: dict,
        normalized_geom: dict,
        page_context: dict,
        project_id: Optional[str] = None,
        is_verified: bool = True,
    ) -> str:
        """Persist an IsGroup fingerprint to ``group_snapshots``.

        Parameters
        ----------
        session_id:
            UUID of the active Grouper session.
        pdf_filename:
            Source PDF filename (basename only, not the full path).
        page_number:
            0-based page index.
        boxes_dicts:
            List of ``GlyphBox.to_dict()`` records belonging to this group.
        group_geometry:
            Output of ``compute_group_geometry()``.
        normalized_geom:
            Output of ``compute_normalized_geometry()``.
        page_context:
            Output of ``compute_page_context()``.
        project_id:
            Optional project identifier string.
        is_verified:
            ``True`` for human-captured records (default).
            ``False`` for machine-suggested records awaiting confirmation.

        Returns
        -------
        str
            The generated ``example_id`` with ``ex_`` prefix.
        """
        example_id = _gen_id("ex_")
        with self._write_lock():
            self._conn.execute(
                """
                INSERT INTO group_snapshots
                  (example_id, session_id, project_id, pdf_filename, page_number,
                   timestamp, boxes_json, group_geometry, normalized_geom,
                   page_context, label, is_verified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', ?)
                """,
                (
                    example_id,
                    session_id,
                    project_id or "",
                    pdf_filename,
                    page_number,
                    _utcnow_iso(),
                    json.dumps(boxes_dicts),
                    json.dumps(group_geometry),
                    json.dumps(normalized_geom),
                    json.dumps(page_context),
                    int(is_verified),
                ),
            )
            self._conn.commit()
        return example_id

    def get_snapshots_for_page(
        self,
        pdf_filename: str,
        page_number: int,
    ) -> List[dict]:
        """Return all group snapshots for a given PDF page, oldest first."""
        rows = self._conn.execute(
            """
            SELECT * FROM group_snapshots
            WHERE pdf_filename = ? AND page_number = ?
            ORDER BY timestamp ASC
            """,
            (pdf_filename, page_number),
        ).fetchall()
        return [_snapshot_row_to_dict(row) for row in rows]

    def get_snapshots_for_session(self, session_id: str) -> List[dict]:
        """Return all snapshots captured in a given session, oldest first."""
        rows = self._conn.execute(
            """
            SELECT * FROM group_snapshots
            WHERE session_id = ?
            ORDER BY timestamp ASC
            """,
            (session_id,),
        ).fetchall()
        return [_snapshot_row_to_dict(row) for row in rows]

    # ── Correction Store ──────────────────────────────────────────────

    def save_group_correction(
        self,
        session_id: str,
        doc_id: str,
        page_number: int,
        signal: str,
        machine_grouping: Optional[dict] = None,
        corrected_grouping: Optional[dict] = None,
        delta: Optional[dict] = None,
        example_id: Optional[str] = None,
    ) -> str:
        """Record a four-signal grouper correction to ``group_corrections``.

        Parameters
        ----------
        session_id:
            UUID of the active Grouper session.
        doc_id:
            Document ID (``sha256:…``) from the ``documents`` table.
        page_number:
            0-based page index.
        signal:
            One of ``"yes"``, ``"is_group"``, ``"edit"``, ``"not_group"``.
        machine_grouping:
            Dict representation of what the machine produced.  May be
            ``None`` in Learn Session where no machine output existed.
        corrected_grouping:
            Dict representation of the human-corrected result.  Omit for
            ``"yes"`` and ``"not_group"`` signals.
        delta:
            ``{op, added, removed}`` dict for ``"edit"`` signals only.
        example_id:
            Links to a ``group_snapshots`` record for ``"is_group"`` signals.

        Returns
        -------
        str
            The generated ``correction_id`` with ``gc_`` prefix.

        Raises
        ------
        ValueError
            If *signal* is not one of the four valid values.
        """
        if signal not in _VALID_SIGNALS:
            raise ValueError(
                f"Invalid grouper signal {signal!r}. "
                f"Must be one of: {sorted(_VALID_SIGNALS)}"
            )
        correction_id = _gen_id("gc_")
        with self._write_lock():
            self._conn.execute(
                """
                INSERT INTO group_corrections
                  (correction_id, session_id, doc_id, page_number, timestamp,
                   signal, machine_grouping, corrected_grouping, delta, example_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    correction_id,
                    session_id,
                    doc_id,
                    page_number,
                    _utcnow_iso(),
                    signal,
                    (
                        json.dumps(machine_grouping)
                        if machine_grouping is not None
                        else None
                    ),
                    (
                        json.dumps(corrected_grouping)
                        if corrected_grouping is not None
                        else None
                    ),
                    json.dumps(delta) if delta is not None else None,
                    example_id,
                ),
            )
            self._conn.commit()
        return correction_id

    def get_corrections_for_session(self, session_id: str) -> List[dict]:
        """Return all group corrections for a session, ordered by timestamp."""
        rows = self._conn.execute(
            """
            SELECT * FROM group_corrections
            WHERE session_id = ?
            ORDER BY timestamp ASC
            """,
            (session_id,),
        ).fetchall()
        return [_correction_row_to_dict(row) for row in rows]


# ── Row deserializers ─────────────────────────────────────────────────────────


def _snapshot_row_to_dict(row: Any) -> dict:
    """Convert a SQLite row from ``group_snapshots`` to a plain dict."""
    return {
        "example_id": row["example_id"],
        "session_id": row["session_id"],
        "project_id": row["project_id"],
        "pdf_filename": row["pdf_filename"],
        "page_number": row["page_number"],
        "timestamp": row["timestamp"],
        "boxes": json.loads(row["boxes_json"]),
        "group_geometry": json.loads(row["group_geometry"]),
        "normalized_geom": json.loads(row["normalized_geom"]),
        "page_context": json.loads(row["page_context"]),
        "label": row["label"],
        "is_verified": bool(row["is_verified"]),
    }


def _correction_row_to_dict(row: Any) -> dict:
    """Convert a SQLite row from ``group_corrections`` to a plain dict."""
    return {
        "correction_id": row["correction_id"],
        "session_id": row["session_id"],
        "doc_id": row["doc_id"],
        "page_number": row["page_number"],
        "timestamp": row["timestamp"],
        "signal": row["signal"],
        "machine_grouping": (
            json.loads(row["machine_grouping"])
            if row["machine_grouping"] is not None
            else None
        ),
        "corrected_grouping": (
            json.loads(row["corrected_grouping"])
            if row["corrected_grouping"] is not None
            else None
        ),
        "delta": json.loads(row["delta"]) if row["delta"] is not None else None,
        "example_id": row["example_id"],
    }
