"""Box groups mixin for CorrectionStore.

Provides detection grouping operations for related detections
that should be treated as a single unit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .store_utils import _gen_id, _utcnow_iso

if TYPE_CHECKING:
    pass


class BoxGroupMixin:
    """Mixin providing box group operations."""

    # These attributes are provided by CorrectionStore
    _conn: object
    _write_lock: object

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
        with self._write_lock():
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
        with self._write_lock():
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
        with self._write_lock():
            self._conn.execute(
                "DELETE FROM box_group_members "
                "WHERE group_id = ? AND detection_id = ?",
                (group_id, detection_id),
            )
            self._conn.commit()

    def delete_group(self, group_id: str) -> None:
        """Delete a group and all its member associations."""
        with self._write_lock():
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


__all__ = ["BoxGroupMixin"]
