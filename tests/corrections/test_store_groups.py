"""Tests for box_groups / box_group_members persistence in CorrectionStore."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from plancheck.corrections.store import CorrectionStore


@pytest.fixture()
def store(tmp_path: Path) -> CorrectionStore:
    """Create a CorrectionStore with two detections pre-inserted."""
    db = tmp_path / "test.db"
    s = CorrectionStore(db_path=db)
    # Insert a minimal document
    s._conn.execute(
        "INSERT INTO documents (doc_id, filename, page_count, ingested_at) "
        "VALUES ('doc1', 'test.pdf', 1, '2026-01-01T00:00:00')"
    )
    # Insert detections
    for det_id in ("det_a", "det_b", "det_c", "det_d"):
        s._conn.execute(
            "INSERT INTO detections "
            "(detection_id, doc_id, page, run_id, element_type, "
            " bbox_x0, bbox_y0, bbox_x1, bbox_y1, text_content, "
            " features_json, created_at) "
            "VALUES (?, 'doc1', 0, 'run1', 'header', "
            " 0, 0, 100, 50, '', '{}', '2026-01-01T00:00:00')",
            (det_id,),
        )
    s._conn.commit()
    return s


class TestCreateGroup:
    def test_returns_grp_id(self, store: CorrectionStore) -> None:
        gid = store.create_group("doc1", 0, "Test Group", "det_a")
        assert gid.startswith("grp_")

    def test_root_is_member(self, store: CorrectionStore) -> None:
        gid = store.create_group("doc1", 0, "Test Group", "det_a")
        members = store._conn.execute(
            "SELECT detection_id FROM box_group_members WHERE group_id = ?",
            (gid,),
        ).fetchall()
        ids = [m["detection_id"] for m in members]
        assert "det_a" in ids

    def test_group_row_stored(self, store: CorrectionStore) -> None:
        gid = store.create_group("doc1", 0, "My Label", "det_a")
        row = store._conn.execute(
            "SELECT * FROM box_groups WHERE group_id = ?", (gid,)
        ).fetchone()
        assert row is not None
        assert row["group_label"] == "My Label"
        assert row["root_detection_id"] == "det_a"
        assert row["page"] == 0


class TestAddToGroup:
    def test_add_child(self, store: CorrectionStore) -> None:
        gid = store.create_group("doc1", 0, "G", "det_a")
        store.add_to_group(gid, "det_b", sort_order=1)
        members = store._conn.execute(
            "SELECT detection_id, sort_order FROM box_group_members "
            "WHERE group_id = ? ORDER BY sort_order",
            (gid,),
        ).fetchall()
        assert len(members) == 2
        assert members[1]["detection_id"] == "det_b"
        assert members[1]["sort_order"] == 1

    def test_duplicate_ignored(self, store: CorrectionStore) -> None:
        gid = store.create_group("doc1", 0, "G", "det_a")
        store.add_to_group(gid, "det_b")
        store.add_to_group(gid, "det_b")  # INSERT OR IGNORE
        members = store._conn.execute(
            "SELECT detection_id FROM box_group_members WHERE group_id = ?",
            (gid,),
        ).fetchall()
        assert len(members) == 2  # root + one child, not duplicated


class TestRemoveFromGroup:
    def test_remove_child(self, store: CorrectionStore) -> None:
        gid = store.create_group("doc1", 0, "G", "det_a")
        store.add_to_group(gid, "det_b")
        store.remove_from_group(gid, "det_b")
        members = store._conn.execute(
            "SELECT detection_id FROM box_group_members WHERE group_id = ?",
            (gid,),
        ).fetchall()
        ids = [m["detection_id"] for m in members]
        assert "det_b" not in ids
        # Group still exists
        row = store._conn.execute(
            "SELECT 1 FROM box_groups WHERE group_id = ?", (gid,)
        ).fetchone()
        assert row is not None

    def test_remove_root_deletes_group(self, store: CorrectionStore) -> None:
        gid = store.create_group("doc1", 0, "G", "det_a")
        store.add_to_group(gid, "det_b")
        store.add_to_group(gid, "det_c")
        store.remove_from_group(gid, "det_a")  # root
        # Group gone
        row = store._conn.execute(
            "SELECT 1 FROM box_groups WHERE group_id = ?", (gid,)
        ).fetchone()
        assert row is None
        # Members gone
        members = store._conn.execute(
            "SELECT 1 FROM box_group_members WHERE group_id = ?", (gid,)
        ).fetchall()
        assert len(members) == 0


class TestDeleteGroup:
    def test_explicit_delete(self, store: CorrectionStore) -> None:
        gid = store.create_group("doc1", 0, "G", "det_a")
        store.add_to_group(gid, "det_b")
        store.delete_group(gid)
        assert (
            store._conn.execute(
                "SELECT 1 FROM box_groups WHERE group_id = ?", (gid,)
            ).fetchone()
            is None
        )
        assert (
            len(
                store._conn.execute(
                    "SELECT 1 FROM box_group_members WHERE group_id = ?",
                    (gid,),
                ).fetchall()
            )
            == 0
        )


class TestGetGroupsForPage:
    def test_returns_groups_with_members(self, store: CorrectionStore) -> None:
        gid = store.create_group("doc1", 0, "Notes", "det_a")
        store.add_to_group(gid, "det_b", sort_order=1)
        store.add_to_group(gid, "det_c", sort_order=2)
        groups = store.get_groups_for_page("doc1", 0)
        assert len(groups) == 1
        g = groups[0]
        assert g["group_id"] == gid
        assert g["group_label"] == "Notes"
        assert g["root_detection_id"] == "det_a"
        assert len(g["members"]) == 3
        member_ids = [m["detection_id"] for m in g["members"]]
        assert "det_a" in member_ids
        assert "det_b" in member_ids
        assert "det_c" in member_ids

    def test_empty_page(self, store: CorrectionStore) -> None:
        assert store.get_groups_for_page("doc1", 5) == []

    def test_sort_order_preserved(self, store: CorrectionStore) -> None:
        gid = store.create_group("doc1", 0, "G", "det_a")
        store.add_to_group(gid, "det_c", sort_order=2)
        store.add_to_group(gid, "det_b", sort_order=1)
        groups = store.get_groups_for_page("doc1", 0)
        orders = [m["sort_order"] for m in groups[0]["members"]]
        assert orders == sorted(orders)


class TestGetGroupForDetection:
    def test_root_detection(self, store: CorrectionStore) -> None:
        gid = store.create_group("doc1", 0, "G", "det_a")
        result = store.get_group_for_detection("det_a")
        assert result is not None
        assert result["group_id"] == gid
        assert result["root_detection_id"] == "det_a"

    def test_child_detection(self, store: CorrectionStore) -> None:
        gid = store.create_group("doc1", 0, "G", "det_a")
        store.add_to_group(gid, "det_b")
        result = store.get_group_for_detection("det_b")
        assert result is not None
        assert result["group_id"] == gid

    def test_ungrouped_returns_none(self, store: CorrectionStore) -> None:
        assert store.get_group_for_detection("det_d") is None

    def test_after_removal_returns_none(self, store: CorrectionStore) -> None:
        gid = store.create_group("doc1", 0, "G", "det_a")
        store.add_to_group(gid, "det_b")
        store.remove_from_group(gid, "det_b")
        assert store.get_group_for_detection("det_b") is None


# ── Edge-case / stress tests ──────────────────────────────────────────


class TestMultipleGroupsOnPage:
    """Multiple independent groups on the same page."""

    def test_two_groups_returned(self, store: CorrectionStore) -> None:
        gid1 = store.create_group("doc1", 0, "Alpha", "det_a")
        gid2 = store.create_group("doc1", 0, "Beta", "det_c")
        store.add_to_group(gid1, "det_b")
        store.add_to_group(gid2, "det_d")
        groups = store.get_groups_for_page("doc1", 0)
        assert len(groups) == 2
        labels = {g["group_label"] for g in groups}
        assert labels == {"Alpha", "Beta"}

    def test_deleting_one_group_keeps_other(self, store: CorrectionStore) -> None:
        gid1 = store.create_group("doc1", 0, "Keep", "det_a")
        gid2 = store.create_group("doc1", 0, "Drop", "det_c")
        store.add_to_group(gid1, "det_b")
        store.delete_group(gid2)
        groups = store.get_groups_for_page("doc1", 0)
        assert len(groups) == 1
        assert groups[0]["group_label"] == "Keep"
        assert len(groups[0]["members"]) == 2


class TestGroupReuse:
    """After a group is deleted, detections can be re-grouped."""

    def test_regroup_after_root_removal(self, store: CorrectionStore) -> None:
        gid1 = store.create_group("doc1", 0, "Old", "det_a")
        store.add_to_group(gid1, "det_b")
        store.remove_from_group(gid1, "det_a")  # root → deletes group
        # Both det_a and det_b should be free
        assert store.get_group_for_detection("det_a") is None
        assert store.get_group_for_detection("det_b") is None
        # Can create a new group with the former child as root
        gid2 = store.create_group("doc1", 0, "New", "det_b")
        store.add_to_group(gid2, "det_a")
        assert store.get_group_for_detection("det_a")["group_id"] == gid2
        assert store.get_group_for_detection("det_b")["group_id"] == gid2

    def test_regroup_after_explicit_delete(self, store: CorrectionStore) -> None:
        gid1 = store.create_group("doc1", 0, "Old", "det_a")
        store.delete_group(gid1)
        gid2 = store.create_group("doc1", 0, "Fresh", "det_a")
        assert gid2 != gid1  # new UUID
        result = store.get_group_for_detection("det_a")
        assert result is not None
        assert result["group_label"] == "Fresh"


class TestGroupDataIntegrity:
    """Verify foreign-key and unique constraints behave correctly."""

    def test_duplicate_root_in_members_ignored(self, store: CorrectionStore) -> None:
        """Root is auto-added at sort_order 0; re-adding should be a no-op."""
        gid = store.create_group("doc1", 0, "G", "det_a")
        store.add_to_group(gid, "det_a", sort_order=5)  # INSERT OR IGNORE
        members = store._conn.execute(
            "SELECT * FROM box_group_members WHERE group_id = ?", (gid,)
        ).fetchall()
        assert len(members) == 1
        assert members[0]["sort_order"] == 0  # original, not 5

    def test_remove_nonexistent_member_no_error(self, store: CorrectionStore) -> None:
        """Removing a detection that's not in the group is a no-op."""
        gid = store.create_group("doc1", 0, "G", "det_a")
        store.remove_from_group(gid, "det_d")  # not a member — should not raise
        members = store._conn.execute(
            "SELECT * FROM box_group_members WHERE group_id = ?", (gid,)
        ).fetchall()
        assert len(members) == 1  # root still present

    def test_group_label_preserved_through_member_ops(
        self, store: CorrectionStore
    ) -> None:
        """Adding/removing children should never alter the group metadata."""
        gid = store.create_group("doc1", 0, "Immutable Label", "det_a")
        store.add_to_group(gid, "det_b")
        store.add_to_group(gid, "det_c")
        store.remove_from_group(gid, "det_b")
        row = store._conn.execute(
            "SELECT group_label, root_detection_id FROM box_groups "
            "WHERE group_id = ?",
            (gid,),
        ).fetchone()
        assert row["group_label"] == "Immutable Label"
        assert row["root_detection_id"] == "det_a"

    def test_sort_order_survives_removal(self, store: CorrectionStore) -> None:
        """Sort order of remaining members isn't affected by removal."""
        gid = store.create_group("doc1", 0, "G", "det_a")
        store.add_to_group(gid, "det_b", sort_order=1)
        store.add_to_group(gid, "det_c", sort_order=2)
        store.add_to_group(gid, "det_d", sort_order=3)
        store.remove_from_group(gid, "det_c")  # remove middle
        members = store._conn.execute(
            "SELECT detection_id, sort_order FROM box_group_members "
            "WHERE group_id = ? ORDER BY sort_order",
            (gid,),
        ).fetchall()
        ids = [m["detection_id"] for m in members]
        assert ids == ["det_a", "det_b", "det_d"]
        assert [m["sort_order"] for m in members] == [0, 1, 3]

    def test_created_at_timestamp(self, store: CorrectionStore) -> None:
        """Groups should have a non-empty created_at timestamp."""
        gid = store.create_group("doc1", 0, "Ts", "det_a")
        row = store._conn.execute(
            "SELECT created_at FROM box_groups WHERE group_id = ?", (gid,)
        ).fetchone()
        assert row["created_at"] is not None
        assert len(row["created_at"]) > 10  # ISO format string
