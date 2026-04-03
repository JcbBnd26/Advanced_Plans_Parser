"""Config package exposing PipelineConfig and supporting types.

Backward-compatible public API:
>>> from plancheck.config import PipelineConfig, ConfigValidationError

Project-management helpers (``create_project``, ``load_project``, etc.)
are lazy-imported on first access to avoid pulling in ``zipfile``,
``json``, ``shutil`` and their transitive dependencies at startup.
"""

from .constants import (
    DATA_DIR,
    DEFAULT_CORRECTIONS_DB,
    DEFAULT_DRIFT_STATS,
    DEFAULT_GNN_MODEL,
    DEFAULT_LABEL_REGISTRY,
    DEFAULT_ML_MODEL,
)
from .exceptions import ConfigLoadError, ConfigValidationError, OCRBackendTimeoutError
from .pipeline import PipelineConfig, migrate_config
from .subconfigs import (
    AnalysisConfig,
    ExportConfig,
    GroupingStageConfig,
    MLConfig,
    ReconcileConfig,
    TOCRConfig,
    VOCRConfig,
)

# Backward-compatibility alias
GroupingConfig = PipelineConfig

# ── Lazy imports for project management helpers ──────────────────────
# These pull in zipfile → bz2/lzma/zlib and shutil at import time,
# adding ~200-300 ms that is unnecessary until the user actually
# creates, loads, or exports a project.
_PROJECT_ATTRS: dict[str, str] = {
    "build_project_config": "project",
    "create_project": "project",
    "export_project": "project",
    "get_master_label_defs": "project",
    "get_recent_projects": "project",
    "import_project": "project",
    "load_project": "project",
    "slugify": "project",
}


def __getattr__(name: str):
    if name in _PROJECT_ATTRS:
        import importlib

        mod = importlib.import_module(f".{_PROJECT_ATTRS[name]}", __package__)
        value = getattr(mod, name)
        # Cache in module namespace so __getattr__ is only called once
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Main config class
    "PipelineConfig",
    # Backward-compat alias
    "GroupingConfig",
    # Exceptions
    "ConfigValidationError",
    "ConfigLoadError",
    "OCRBackendTimeoutError",
    # Migration helper
    "migrate_config",
    # Sub-config views
    "TOCRConfig",
    "VOCRConfig",
    "ReconcileConfig",
    "GroupingStageConfig",
    "AnalysisConfig",
    "ExportConfig",
    "MLConfig",
    # Constants
    "DATA_DIR",
    "DEFAULT_ML_MODEL",
    "DEFAULT_GNN_MODEL",
    "DEFAULT_DRIFT_STATS",
    "DEFAULT_CORRECTIONS_DB",
    "DEFAULT_LABEL_REGISTRY",
    # Project management (lazy)
    "create_project",
    "load_project",
    "build_project_config",
    "get_master_label_defs",
    "get_recent_projects",
    "export_project",
    "import_project",
    "slugify",
]
