"""Global constants and default paths for plancheck configuration."""

from __future__ import annotations

import os
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
# Global data directory - can be overridden via PLANCHECK_DATA_DIR env var
# ══════════════════════════════════════════════════════════════════════
DATA_DIR = Path(os.environ.get("PLANCHECK_DATA_DIR", "data"))

# Default paths derived from DATA_DIR
DEFAULT_CORRECTIONS_DB = DATA_DIR / "corrections.db"
DEFAULT_ML_MODEL = DATA_DIR / "element_classifier.pkl"
DEFAULT_GNN_MODEL = DATA_DIR / "document_gnn.pt"
DEFAULT_DRIFT_STATS = DATA_DIR / "drift_stats.json"
DEFAULT_LABEL_REGISTRY = DATA_DIR / "label_registry.json"
