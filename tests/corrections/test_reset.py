"""Tests for training-data reset methods in DbHelpersMixin."""

from __future__ import annotations

from pathlib import Path

import pytest

from plancheck.corrections.store import CorrectionStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_register(store: CorrectionStore, doc_id: str = "sha256:abc123") -> str:
    """Insert a minimal document row (avoids needing a real PDF)."""
    store._conn.execute(
        "INSERT OR IGNORE INTO documents "
        "(doc_id, filename, page_count, ingested_at) VALUES (?, ?, ?, ?)",
        (doc_id, "test.pdf", 3, "2026-01-01T00:00:00Z"),
    )
    store._conn.commit()
    return doc_id


def _populate_training_data(
    store: CorrectionStore,
    sample_features: dict,
) -> dict[str, int]:
    """Insert rows into all 9 clearable tables and return expected counts."""
    doc_id = _mock_register(store)

    # detections × 3
    det_ids = []
    for i in range(3):
        did = store.save_detection(
            doc_id=doc_id,
            page=0,
            run_id="run_reset",
            element_type="notes_column",
            bbox=(i * 10, 0, i * 10 + 50, 20),
            text_content=f"note {i}",
            features=sample_features,
        )
        det_ids.append(did)

    # corrections × 2  (relabel + accept)
    store.save_correction(
        doc_id=doc_id,
        page=0,
        correction_type="relabel",
        corrected_label="header",
        corrected_bbox=(0, 0, 50, 20),
        detection_id=det_ids[0],
        original_label="notes_column",
        original_bbox=(0, 0, 50, 20),
    )
    store.accept_detection(det_ids[1], doc_id, page=0)

    # training_examples via build_training_set
    store.build_training_set()

    # training_runs × 1
    store._conn.execute(
        "INSERT INTO training_runs "
        "(run_id, trained_at, n_train, n_val, accuracy, labels_json, per_class_json) "
        "VALUES (?, datetime('now'), 10, 2, 0.9, '[]', '{}')",
        ("trun_1",),
    )

    # candidate_outcomes × 1
    store._conn.execute(
        "INSERT INTO candidate_outcomes "
        "(outcome_id, doc_id, page, run_id, trigger_methods, outcome, "
        "confidence, bbox_x0, bbox_y0, bbox_x1, bbox_y1, created_at) "
        "VALUES ('co_1', ?, 0, 'run_reset', 'test', 'accepted', "
        "0.9, 0, 0, 50, 20, datetime('now'))",
        (doc_id,),
    )

    # dismissed_detections × 1 (table may have been created by schema)
    try:
        store._conn.execute(
            "INSERT INTO dismissed_detections "
            "(dismiss_id, detection_id, doc_id, page, dismissed_at) "
            "VALUES ('dis_1', ?, ?, 0, datetime('now'))",
            (det_ids[2], doc_id),
        )
    except Exception:  # noqa: BLE001
        pass

    # box_groups + box_group_members
    store._conn.execute(
        "INSERT INTO box_groups "
        "(group_id, doc_id, page, group_label, root_detection_id, created_at) "
        "VALUES ('grp_1', ?, 0, 'notes_column', ?, datetime('now'))",
        (doc_id, det_ids[0]),
    )
    store._conn.execute(
        "INSERT INTO box_group_members "
        "(group_id, detection_id, sort_order) VALUES ('grp_1', ?, 0)",
        (det_ids[0],),
    )

    # feature_cache × 1
    try:
        store._conn.execute(
            "INSERT INTO feature_cache "
            "(cache_key, detection_id, feature_version, vector_json, created_at) "
            "VALUES (?, ?, 1, '[]', datetime('now'))",
            (f"fc_{det_ids[0]}", det_ids[0]),
        )
    except Exception:  # noqa: BLE001
        pass

    store._conn.commit()

    return {
        "detections": 3,
        "corrections": 2,
        "training_examples": 2,
        "training_runs": 1,
        "candidate_outcomes": 1,
        "box_groups": 1,
        "box_group_members": 1,
    }


# ---------------------------------------------------------------------------
# Tests: get_training_data_summary
# ---------------------------------------------------------------------------


class TestGetTrainingDataSummary:
    """Tests for CorrectionStore.get_training_data_summary()."""

    @pytest.mark.unit
    def test_empty_db_returns_all_zeros(self, tmp_store: CorrectionStore) -> None:
        """On a fresh DB every table count should be 0."""
        summary = tmp_store.get_training_data_summary()

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
            assert summary[table] == 0, f"{table} should be 0 on empty DB"

    @pytest.mark.unit
    def test_file_booleans_all_false_on_empty(self, tmp_store: CorrectionStore) -> None:
        """File-existence booleans should be False when no files exist."""
        summary = tmp_store.get_training_data_summary()
        assert summary["model_file_exists"] is False
        assert summary["drift_stats_exists"] is False
        assert summary["jsonl_exists"] is False
        assert summary["stage2_jsonl_exists"] is False
        assert summary["subtype_model_exists"] is False

    @pytest.mark.unit
    def test_file_booleans_true_when_files_exist(
        self, tmp_store: CorrectionStore, tmp_path: Path
    ) -> None:
        """File-existence flags should reflect actual disk state."""
        data_dir = tmp_path  # tmp_store's db lives in tmp_path
        (data_dir / "element_classifier.pkl").write_bytes(b"fake")
        (data_dir / "drift_stats.json").write_text("{}")
        (data_dir / "training_data.jsonl").write_text("")
        (data_dir / "training_data_stage2.jsonl").write_text("")
        (data_dir / "subtype_classifier.pkl").write_bytes(b"fake")

        summary = tmp_store.get_training_data_summary()
        assert summary["model_file_exists"] is True
        assert summary["drift_stats_exists"] is True
        assert summary["jsonl_exists"] is True
        assert summary["stage2_jsonl_exists"] is True
        assert summary["subtype_model_exists"] is True

    @pytest.mark.unit
    def test_counts_match_populated_data(
        self, tmp_store: CorrectionStore, sample_features: dict
    ) -> None:
        """After populating tables, summary counts should be non-zero."""
        expected = _populate_training_data(tmp_store, sample_features)
        summary = tmp_store.get_training_data_summary()

        assert summary["detections"] == expected["detections"]
        assert summary["corrections"] == expected["corrections"]
        assert summary["training_examples"] == expected["training_examples"]
        assert summary["training_runs"] == expected["training_runs"]
        assert summary["candidate_outcomes"] == expected["candidate_outcomes"]
        assert summary["box_groups"] == expected["box_groups"]
        assert summary["box_group_members"] == expected["box_group_members"]


# ---------------------------------------------------------------------------
# Tests: reset_training_data
# ---------------------------------------------------------------------------


class TestResetTrainingData:
    """Tests for CorrectionStore.reset_training_data()."""

    @pytest.mark.unit
    def test_reset_on_empty_db_no_crash(self, tmp_store: CorrectionStore) -> None:
        """Resetting a fresh empty DB should work without errors."""
        deleted = tmp_store.reset_training_data()

        # Every table should report 0 deleted rows
        for count in deleted.values():
            assert count == 0

    @pytest.mark.unit
    def test_reset_clears_all_trainable_tables(
        self, tmp_store: CorrectionStore, sample_features: dict
    ) -> None:
        """After reset, all 9 clearable tables should be empty."""
        _populate_training_data(tmp_store, sample_features)

        # Sanity: at least some data exists before reset
        pre = tmp_store.get_training_data_summary()
        assert pre["detections"] > 0
        assert pre["corrections"] > 0

        deleted = tmp_store.reset_training_data()

        # Verify deleted counts are positive for populated tables
        assert deleted["detections"] > 0
        assert deleted["corrections"] > 0

        # Verify all tables now empty
        post = tmp_store.get_training_data_summary()
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
            assert post[table] == 0, f"{table} should be 0 after reset"

    @pytest.mark.unit
    def test_reset_preserves_documents(
        self, tmp_store: CorrectionStore, sample_features: dict
    ) -> None:
        """Documents table must survive the reset unchanged."""
        _populate_training_data(tmp_store, sample_features)

        docs_before = tmp_store._conn.execute(
            "SELECT COUNT(*) AS n FROM documents"
        ).fetchone()["n"]
        assert docs_before > 0

        tmp_store.reset_training_data()

        docs_after = tmp_store._conn.execute(
            "SELECT COUNT(*) AS n FROM documents"
        ).fetchone()["n"]
        assert docs_after == docs_before

    @pytest.mark.unit
    def test_reset_creates_snapshot(
        self, tmp_store: CorrectionStore, tmp_path: Path
    ) -> None:
        """A pre-reset snapshot file should be created in the snapshots dir."""
        snap_dir = tmp_path / "snapshots"

        # No snapshots before
        assert not snap_dir.exists() or len(list(snap_dir.glob("*.db"))) == 0

        tmp_store.reset_training_data()

        # Snapshot should exist with 'pre-reset' in the name
        snaps = list(snap_dir.glob("corrections_*_pre-reset.db"))
        assert len(snaps) == 1, f"Expected 1 pre-reset snapshot, found {snaps}"
        assert snaps[0].stat().st_size > 0

    @pytest.mark.unit
    def test_reset_returns_deleted_counts(
        self, tmp_store: CorrectionStore, sample_features: dict
    ) -> None:
        """Return value should be a dict mapping table names to row counts."""
        expected = _populate_training_data(tmp_store, sample_features)
        deleted = tmp_store.reset_training_data()

        assert isinstance(deleted, dict)
        assert deleted["detections"] == expected["detections"]
        assert deleted["corrections"] == expected["corrections"]
        assert deleted["training_runs"] == expected["training_runs"]

    @pytest.mark.unit
    def test_reset_clears_session_active_metadata(
        self, tmp_store: CorrectionStore
    ) -> None:
        """The session_active metadata flag should be removed by reset."""
        # Manually set the metadata flag
        tmp_store._conn.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)"
        )
        tmp_store._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('session_active', '1')"
        )
        tmp_store._conn.commit()

        tmp_store.reset_training_data()

        row = tmp_store._conn.execute(
            "SELECT value FROM metadata WHERE key = 'session_active'"
        ).fetchone()
        assert row is None, "session_active should be cleared after reset"

    @pytest.mark.unit
    def test_double_reset_is_safe(
        self, tmp_store: CorrectionStore, sample_features: dict
    ) -> None:
        """Calling reset twice in a row should not crash."""
        _populate_training_data(tmp_store, sample_features)
        tmp_store.reset_training_data()

        # Second reset on already-empty tables
        deleted = tmp_store.reset_training_data()
        for count in deleted.values():
            assert count == 0
