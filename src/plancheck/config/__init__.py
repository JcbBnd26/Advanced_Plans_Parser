"""Config package exposing PipelineConfig and supporting types.

Backward-compatible public API:
>>> from plancheck.config import PipelineConfig, ConfigValidationError
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
from .project import (
    build_project_config,
    create_project,
    export_project,
    get_master_label_defs,
    get_recent_projects,
    import_project,
    load_project,
    slugify,
)
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
    # Project management
    "create_project",
    "load_project",
    "build_project_config",
    "get_master_label_defs",
    "get_recent_projects",
    "export_project",
    "import_project",
    "slugify",
]
