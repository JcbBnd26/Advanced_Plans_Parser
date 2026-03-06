"""Config package exposing PipelineConfig and supporting types.

Backward-compatible public API:
>>> from plancheck.config import PipelineConfig, ConfigValidationError
"""

from .constants import (
    DATA_DIR,
    DEFAULT_CORRECTIONS_DB,
    DEFAULT_DRIFT_STATS,
    DEFAULT_GNN_MODEL,
    DEFAULT_ML_MODEL,
)
from .exceptions import ConfigLoadError, ConfigValidationError
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

__all__ = [
    # Main config class
    "PipelineConfig",
    # Backward-compat alias
    "GroupingConfig",
    # Exceptions
    "ConfigValidationError",
    "ConfigLoadError",
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
]
