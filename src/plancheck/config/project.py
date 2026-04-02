"""Project folder management — create, load, and configure per-project workspaces.

A "project" is a directory containing a ``project.json`` metadata file,
a ``label_registry.json`` with the project's chosen classes, and
runtime-created artefacts (database, trained models, drift stats).

This module provides pure-logic functions with no GUI dependencies.

Typical usage
-------------
>>> from plancheck.config.project import create_project, load_project, build_project_config
>>> project_dir = create_project(Path("projects/ODOT"), "ODOT Highway Plans", label_defs)
>>> meta = load_project(project_dir)
>>> cfg = build_project_config(project_dir)
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

from .constants import DATA_DIR, DEFAULT_LABEL_REGISTRY
from .exceptions import ConfigLoadError

if TYPE_CHECKING:
    from .pipeline import PipelineConfig

log = logging.getLogger(__name__)

# Current schema version for project.json
_PROJECT_SCHEMA_VERSION = "1.0"

# Recent projects storage
_RECENT_PROJECTS_PATH = DATA_DIR / "recent_projects.json"
_RECENT_PROJECTS_MAX = 10

# Files/patterns excluded from project export
_EXPORT_EXCLUDE_PATTERNS = {"__pycache__", ".lock", ".pyc"}


# ══════════════════════════════════════════════════════════════════════
# Project CRUD
# ══════════════════════════════════════════════════════════════════════


def create_project(
    project_dir: Path,
    name: str,
    label_defs: List[dict],
    *,
    description: str = "",
    config_overrides: Optional[dict] = None,
) -> Path:
    """Create a new project folder with its initial files.

    Parameters
    ----------
    project_dir
        Target directory for the project.  Created if it does not exist.
    name
        Human-readable project name stored in ``project.json``.
    label_defs
        List of label definition dicts (same schema as ``label_registry.json``
        entries).  Typically a subset of :func:`get_master_label_defs`.
    description
        Optional project description.
    config_overrides
        Optional dict of ``PipelineConfig`` field overrides.

    Returns
    -------
    Path
        The *project_dir* that was created.

    Raises
    ------
    FileExistsError
        If ``project.json`` already exists in *project_dir*.
    """
    project_dir = Path(project_dir)
    project_json = project_dir / "project.json"

    if project_json.exists():
        raise FileExistsError(f"project.json already exists in {project_dir}")

    # Create directory structure
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "snapshots").mkdir(exist_ok=True)

    # Write project.json
    meta: dict[str, Any] = {
        "version": _PROJECT_SCHEMA_VERSION,
        "name": name,
        "description": description,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if config_overrides:
        meta["config_overrides"] = config_overrides

    project_json.write_text(
        json.dumps(meta, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Created project.json in %s", project_dir)

    # Write label_registry.json
    registry = {"version": "1.0", "label_registry": label_defs}
    (project_dir / "label_registry.json").write_text(
        json.dumps(registry, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Wrote label_registry.json with %d labels", len(label_defs))

    # Initialise empty corrections.db (schema creation happens in constructor)
    try:
        from plancheck.corrections.store import CorrectionStore

        store = CorrectionStore(project_dir / "corrections.db")
        store.close()
        log.info("Initialised corrections.db")
    except Exception:  # noqa: BLE001 — best-effort; DB created on first use anyway
        log.warning("Could not pre-create corrections.db", exc_info=True)

    return project_dir


def load_project(project_dir: Path) -> dict:
    """Read and validate ``project.json`` from *project_dir*.

    Parameters
    ----------
    project_dir
        Directory expected to contain ``project.json``.

    Returns
    -------
    dict
        Parsed project metadata.

    Raises
    ------
    FileNotFoundError
        If ``project.json`` does not exist.
    ConfigLoadError
        If the file cannot be parsed or is missing required fields.
    """
    project_dir = Path(project_dir)
    project_json = project_dir / "project.json"

    if not project_json.exists():
        raise FileNotFoundError(
            f"Not a valid project folder: {project_dir} " f"(no project.json found)"
        )

    try:
        raw = project_json.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError) as exc:
        raise ConfigLoadError(
            f"Failed to load project.json from {project_dir}: {exc}"
        ) from exc

    # Validate required fields
    for field in ("version", "name"):
        if field not in data:
            raise ConfigLoadError(f"project.json missing required field: {field!r}")

    return data


def build_project_config(
    project_dir: Path,
    overrides: Optional[dict] = None,
) -> "PipelineConfig":
    """Build a ``PipelineConfig`` with all paths pointing at *project_dir*.

    Starts from a fresh default config, redirects ML/data paths to the
    project folder, applies any ``config_overrides`` from ``project.json``,
    then applies additional runtime *overrides*.

    Parameters
    ----------
    project_dir
        An existing project directory (must contain ``project.json``).
    overrides
        Additional runtime overrides applied on top of the project config.

    Returns
    -------
    PipelineConfig
        Fully configured instance with project-scoped paths.
    """
    from .pipeline import PipelineConfig

    project_dir = Path(project_dir)
    meta = load_project(project_dir)

    cfg = PipelineConfig()

    # Redirect ML / data paths to project folder
    cfg.ml_model_path = str(project_dir / "element_classifier.pkl")
    cfg.ml_stage2_model_path = str(project_dir / "title_subtype_classifier.pkl")
    cfg.ml_gnn_model_path = str(project_dir / "document_gnn.pt")
    cfg.ml_drift_stats_path = str(project_dir / "drift_stats.json")

    # Apply config_overrides from project.json
    project_overrides = meta.get("config_overrides", {})
    if project_overrides:
        _apply_overrides(cfg, project_overrides, source="project.json")

    # Apply additional runtime overrides
    if overrides:
        _apply_overrides(cfg, overrides, source="runtime")

    # Re-validate after applying overrides
    cfg.validate()

    return cfg


def _apply_overrides(
    cfg: "PipelineConfig",
    overrides: dict,
    source: str = "unknown",
) -> None:
    """Apply a dict of field overrides to a PipelineConfig, skipping unknowns."""
    import dataclasses

    valid_fields = {f.name for f in dataclasses.fields(cfg)}
    for key, value in overrides.items():
        if key in valid_fields:
            setattr(cfg, key, value)
        else:
            log.warning("Ignoring unknown config override %r from %s", key, source)


# ══════════════════════════════════════════════════════════════════════
# Master label definitions
# ══════════════════════════════════════════════════════════════════════


def get_master_label_defs() -> List[dict]:
    """Return the full set of available label definitions.

    Reads from the app-level ``data/label_registry.json``. Returns an
    empty list if the file does not exist or cannot be parsed.
    """
    try:
        raw = DEFAULT_LABEL_REGISTRY.read_text(encoding="utf-8")
        data = json.loads(raw)
        return data.get("label_registry", [])
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        log.warning(
            "Could not load master label registry from %s: %s",
            DEFAULT_LABEL_REGISTRY,
            exc,
        )
        return []


# ══════════════════════════════════════════════════════════════════════
# Recent projects
# ══════════════════════════════════════════════════════════════════════


def add_recent_project(project_dir: Path, name: str) -> None:
    """Add or promote a project to the top of the recent projects list."""
    project_dir = Path(project_dir).resolve()
    recents = _load_recents()

    # Remove existing entry for this path (if any) to avoid duplicates
    recents = [r for r in recents if Path(r["path"]).resolve() != project_dir]

    # Prepend
    recents.insert(
        0,
        {
            "path": str(project_dir),
            "name": name,
            "last_opened": datetime.now(timezone.utc).isoformat(),
        },
    )

    # Trim to max
    recents = recents[:_RECENT_PROJECTS_MAX]
    _save_recents(recents)


def get_recent_projects() -> List[dict]:
    """Return recent projects whose directories still exist."""
    recents = _load_recents()
    valid = [r for r in recents if Path(r["path"]).is_dir()]
    # Persist the filtered list to clean up stale entries
    if len(valid) != len(recents):
        _save_recents(valid)
    return valid


def clear_recent_projects() -> None:
    """Remove all entries from the recent projects list."""
    _save_recents([])


def _load_recents() -> List[dict]:
    """Load the recent projects JSON file."""
    try:
        raw = _RECENT_PROJECTS_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        return data.get("recent", [])
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def _save_recents(entries: List[dict]) -> None:
    """Persist the recent projects list to disk."""
    try:
        _RECENT_PROJECTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _RECENT_PROJECTS_PATH.write_text(
            json.dumps({"recent": entries}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:  # noqa: BLE001
        log.warning("Could not save recent projects list", exc_info=True)


# ══════════════════════════════════════════════════════════════════════
# Project export / import
# ══════════════════════════════════════════════════════════════════════


def export_project(project_dir: Path, output_path: Path) -> Path:
    """Export a project folder to a zip archive.

    Parameters
    ----------
    project_dir
        Source project directory (must contain ``project.json``).
    output_path
        Destination path for the zip file.

    Returns
    -------
    Path
        The *output_path* that was written.

    Raises
    ------
    FileNotFoundError
        If *project_dir* is not a valid project folder.
    """
    project_dir = Path(project_dir)
    output_path = Path(output_path)

    # Validate it's a real project
    load_project(project_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Write version manifest
        manifest = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "project_name": load_project(project_dir).get("name", ""),
            "schema_version": _PROJECT_SCHEMA_VERSION,
        }
        zf.writestr("_export_manifest.json", json.dumps(manifest, indent=2))

        # Walk the project directory
        for file_path in sorted(project_dir.rglob("*")):
            if file_path.is_dir():
                continue
            # Skip excluded patterns
            if any(pat in file_path.name for pat in _EXPORT_EXCLUDE_PATTERNS):
                continue
            arcname = file_path.relative_to(project_dir).as_posix()
            zf.write(file_path, arcname)

    log.info("Exported project to %s", output_path)
    return output_path


def import_project(zip_path: Path, target_parent_dir: Path) -> Path:
    """Import a project from a zip archive.

    Parameters
    ----------
    zip_path
        Path to the zip file to import.
    target_parent_dir
        Parent directory where the project folder will be extracted.

    Returns
    -------
    Path
        The extracted project directory.

    Raises
    ------
    ConfigLoadError
        If the archive is invalid or does not contain ``project.json``.
    """
    zip_path = Path(zip_path)
    target_parent_dir = Path(target_parent_dir)

    if not zipfile.is_zipfile(zip_path):
        raise ConfigLoadError(f"Not a valid zip file: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

        # Determine project folder name from project.json location
        project_json_candidates = [
            n for n in names if n == "project.json" or n.endswith("/project.json")
        ]
        if not project_json_candidates:
            raise ConfigLoadError(f"Archive does not contain project.json: {zip_path}")

        # Security: check for path traversal
        for name in names:
            resolved = (target_parent_dir / name).resolve()
            if not str(resolved).startswith(str(target_parent_dir.resolve())):
                raise ConfigLoadError(f"Archive contains unsafe path: {name}")

        # Determine project folder name from the archive
        pj = project_json_candidates[0]
        if pj == "project.json":
            # project.json is at root — use the zip filename as folder name
            folder_name = zip_path.stem
        else:
            # project.json is nested — use the top-level folder
            folder_name = pj.split("/")[0]

        project_dir = target_parent_dir / folder_name
        project_dir.mkdir(parents=True, exist_ok=True)

        # Extract files
        for member in names:
            if member.endswith("/"):
                continue
            # Skip manifest
            if member == "_export_manifest.json":
                continue

            if pj == "project.json":
                # Root-level layout — extract directly into project_dir
                dest = project_dir / member
            else:
                # Nested layout — strip the top-level folder
                rel = "/".join(member.split("/")[1:])
                if not rel:
                    continue
                dest = project_dir / rel

            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(dest, "wb") as tgt:
                tgt.write(src.read())

    # Validate the result
    load_project(project_dir)
    log.info("Imported project from %s to %s", zip_path, project_dir)
    return project_dir


# ══════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════


def slugify(name: str) -> str:
    """Convert a project name to a filesystem-safe slug.

    >>> slugify("ODOT Highway Plans")
    'ODOT_Highway_Plans'
    """
    # Replace spaces and non-alphanum (except underscores/hyphens) with _
    slug = re.sub(r"[^\w\-]", "_", name)
    # Collapse multiple underscores
    slug = re.sub(r"_+", "_", slug)
    return slug.strip("_")
