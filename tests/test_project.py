"""Tests for plancheck.config.project — project folder CRUD, config building, recent projects, export/import."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from plancheck.config.project import (
    _RECENT_PROJECTS_PATH,
    add_recent_project,
    build_project_config,
    clear_recent_projects,
    create_project,
    export_project,
    get_master_label_defs,
    get_recent_projects,
    import_project,
    load_project,
    slugify,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_labels() -> list[dict]:
    """A small subset of label definitions for testing."""
    return [
        {
            "label": "notes_column",
            "display_name": "General Notes Column",
            "color": "#1ea01e",
            "description": "Test label",
        },
        {
            "label": "header",
            "display_name": "Section Header",
            "color": "#dc1e1e",
            "description": "Test label 2",
        },
    ]


@pytest.fixture()
def project_dir(tmp_path: Path, sample_labels: list[dict]) -> Path:
    """Create a temporary project and return its directory."""
    pdir = tmp_path / "test_project"
    create_project(pdir, "Test Project", sample_labels)
    return pdir


# ---------------------------------------------------------------------------
# slugify
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_basic(self):
        assert slugify("ODOT Highway Plans") == "ODOT_Highway_Plans"

    def test_special_chars(self):
        assert slugify("My Project (v2)!") == "My_Project_v2"

    def test_multiple_underscores(self):
        assert slugify("a   b   c") == "a_b_c"


# ---------------------------------------------------------------------------
# create_project
# ---------------------------------------------------------------------------


class TestCreateProject:
    def test_creates_folder_structure(self, tmp_path: Path, sample_labels: list[dict]):
        pdir = tmp_path / "new_proj"
        result = create_project(pdir, "New Project", sample_labels)

        assert result == pdir
        assert (pdir / "project.json").exists()
        assert (pdir / "label_registry.json").exists()
        assert (pdir / "corrections.db").exists()
        assert (pdir / "snapshots").is_dir()

    def test_project_json_contents(self, tmp_path: Path, sample_labels: list[dict]):
        pdir = tmp_path / "meta_test"
        create_project(
            pdir,
            "Meta Test",
            sample_labels,
            description="A test project",
            config_overrides={"ml_retrain_threshold": 25},
        )

        data = json.loads((pdir / "project.json").read_text(encoding="utf-8"))
        assert data["name"] == "Meta Test"
        assert data["version"] == "1.0"
        assert data["description"] == "A test project"
        assert data["config_overrides"]["ml_retrain_threshold"] == 25
        assert "created_at" in data

    def test_label_registry_contents(self, tmp_path: Path, sample_labels: list[dict]):
        pdir = tmp_path / "reg_test"
        create_project(pdir, "Reg Test", sample_labels)

        data = json.loads((pdir / "label_registry.json").read_text(encoding="utf-8"))
        assert data["version"] == "1.0"
        assert len(data["label_registry"]) == 2
        assert data["label_registry"][0]["label"] == "notes_column"

    def test_raises_on_existing(self, project_dir: Path, sample_labels: list[dict]):
        with pytest.raises(FileExistsError, match="already exists"):
            create_project(project_dir, "Duplicate", sample_labels)

    def test_no_overrides(self, tmp_path: Path, sample_labels: list[dict]):
        pdir = tmp_path / "no_overrides"
        create_project(pdir, "No Overrides", sample_labels)
        data = json.loads((pdir / "project.json").read_text(encoding="utf-8"))
        assert "config_overrides" not in data


# ---------------------------------------------------------------------------
# load_project
# ---------------------------------------------------------------------------


class TestLoadProject:
    def test_load_valid(self, project_dir: Path):
        meta = load_project(project_dir)
        assert meta["name"] == "Test Project"
        assert meta["version"] == "1.0"

    def test_raises_on_missing_dir(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="no project.json"):
            load_project(tmp_path / "nonexistent")

    def test_raises_on_missing_name(self, tmp_path: Path):
        pdir = tmp_path / "bad_proj"
        pdir.mkdir()
        (pdir / "project.json").write_text(
            json.dumps({"version": "1.0"}), encoding="utf-8"
        )
        from plancheck.config.exceptions import ConfigLoadError

        with pytest.raises(ConfigLoadError, match="name"):
            load_project(pdir)

    def test_raises_on_bad_json(self, tmp_path: Path):
        pdir = tmp_path / "bad_json"
        pdir.mkdir()
        (pdir / "project.json").write_text("not valid json {{{", encoding="utf-8")
        from plancheck.config.exceptions import ConfigLoadError

        with pytest.raises(ConfigLoadError):
            load_project(pdir)


# ---------------------------------------------------------------------------
# build_project_config
# ---------------------------------------------------------------------------


class TestBuildProjectConfig:
    def test_paths_redirect(self, project_dir: Path):
        cfg = build_project_config(project_dir)

        assert cfg.ml_model_path == str(project_dir / "element_classifier.pkl")
        assert cfg.ml_stage2_model_path == str(
            project_dir / "title_subtype_classifier.pkl"
        )
        assert cfg.ml_gnn_model_path == str(project_dir / "document_gnn.pt")
        assert cfg.ml_drift_stats_path == str(project_dir / "drift_stats.json")

    def test_overrides_applied(self, tmp_path: Path, sample_labels: list[dict]):
        pdir = tmp_path / "override_proj"
        create_project(
            pdir,
            "Override Test",
            sample_labels,
            config_overrides={"ml_retrain_threshold": 15},
        )
        cfg = build_project_config(pdir)
        assert cfg.ml_retrain_threshold == 15

    def test_runtime_overrides(self, project_dir: Path):
        cfg = build_project_config(
            project_dir, overrides={"ml_relabel_confidence": 0.6}
        )
        assert cfg.ml_relabel_confidence == 0.6

    def test_defaults_preserved(self, project_dir: Path):
        cfg = build_project_config(project_dir)
        # Non-overridden fields should keep defaults
        assert cfg.iou_prune == 0.5
        assert cfg.enable_tocr is True


# ---------------------------------------------------------------------------
# get_master_label_defs
# ---------------------------------------------------------------------------


class TestGetMasterLabelDefs:
    def test_returns_list(self):
        defs = get_master_label_defs()
        assert isinstance(defs, list)
        assert len(defs) > 0

    def test_label_structure(self):
        defs = get_master_label_defs()
        first = defs[0]
        assert "label" in first
        assert "display_name" in first
        assert "color" in first


# ---------------------------------------------------------------------------
# Recent projects
# ---------------------------------------------------------------------------


class TestRecentProjects:
    def setup_method(self):
        """Clear recent projects before each test."""
        clear_recent_projects()

    def teardown_method(self):
        """Clean up after each test."""
        clear_recent_projects()

    def test_add_and_get(self, project_dir: Path):
        add_recent_project(project_dir, "Test Project")
        recents = get_recent_projects()
        assert len(recents) == 1
        assert recents[0]["name"] == "Test Project"

    def test_deduplication(self, project_dir: Path):
        add_recent_project(project_dir, "Test")
        add_recent_project(project_dir, "Test")
        recents = get_recent_projects()
        assert len(recents) == 1

    def test_ordering(self, tmp_path: Path, sample_labels: list[dict]):
        p1 = tmp_path / "proj1"
        p2 = tmp_path / "proj2"
        create_project(p1, "First", sample_labels)
        create_project(p2, "Second", sample_labels)

        add_recent_project(p1, "First")
        add_recent_project(p2, "Second")

        recents = get_recent_projects()
        assert recents[0]["name"] == "Second"  # Most recent first
        assert recents[1]["name"] == "First"

    def test_clear(self, project_dir: Path):
        add_recent_project(project_dir, "Test")
        clear_recent_projects()
        assert get_recent_projects() == []

    def test_filters_deleted(self, tmp_path: Path, sample_labels: list[dict]):
        pdir = tmp_path / "temp_proj"
        create_project(pdir, "Temp", sample_labels)
        add_recent_project(pdir, "Temp")

        # Delete project folder
        import shutil

        shutil.rmtree(pdir)

        recents = get_recent_projects()
        assert len(recents) == 0


# ---------------------------------------------------------------------------
# Export / Import
# ---------------------------------------------------------------------------


class TestExportImport:
    def test_export_creates_zip(self, project_dir: Path, tmp_path: Path):
        out = tmp_path / "exported.plancheck"
        result = export_project(project_dir, out)
        assert result == out
        assert zipfile.is_zipfile(out)

    def test_export_contains_files(self, project_dir: Path, tmp_path: Path):
        out = tmp_path / "check.zip"
        export_project(project_dir, out)

        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
            assert "project.json" in names
            assert "label_registry.json" in names
            assert "_export_manifest.json" in names

    def test_export_excludes_lock_files(self, project_dir: Path, tmp_path: Path):
        # Create a .lock file
        (project_dir / "corrections.db.lock").write_text("lock", encoding="utf-8")
        out = tmp_path / "nolocks.zip"
        export_project(project_dir, out)

        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
            assert not any(".lock" in n for n in names)

    def test_import_round_trip(self, project_dir: Path, tmp_path: Path):
        zip_out = tmp_path / "roundtrip.zip"
        export_project(project_dir, zip_out)

        dest = tmp_path / "imported"
        dest.mkdir()
        imported_dir = import_project(zip_out, dest)

        assert (imported_dir / "project.json").exists()
        meta = load_project(imported_dir)
        assert meta["name"] == "Test Project"

    def test_import_invalid_zip(self, tmp_path: Path):
        bad_file = tmp_path / "not_a_zip.plancheck"
        bad_file.write_text("not a zip", encoding="utf-8")
        from plancheck.config.exceptions import ConfigLoadError

        with pytest.raises(ConfigLoadError, match="Not a valid zip"):
            import_project(bad_file, tmp_path / "dest")

    def test_import_missing_project_json(self, tmp_path: Path):
        # Create a zip without project.json
        zip_path = tmp_path / "empty.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("readme.txt", "hello")
        from plancheck.config.exceptions import ConfigLoadError

        with pytest.raises(ConfigLoadError, match="does not contain project.json"):
            import_project(zip_path, tmp_path / "dest")

    def test_export_raises_for_invalid_project(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            export_project(tmp_path / "nonexistent", tmp_path / "out.zip")
