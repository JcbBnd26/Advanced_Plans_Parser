"""Diagnostics section widgets for the Diagnostics tab.

Sections are grouped by domain and re-exported here for convenient
import by ``tab_diagnostics.DiagnosticsTab``.
"""

from __future__ import annotations

from .font_benchmark import BenchmarkSection, FontDiagnosticsSection
from .ml_tools import (
    MLCalibrationSection,
    MLRuntimeSummarySection,
    ModelComparisonSection,
    TrainingProgressSection,
)
from .external_tools import (
    CrossPageGNNSection,
    LayoutModelSection,
    LLMSemanticChecksSection,
    TextEmbeddingsSection,
)

__all__ = [
    "FontDiagnosticsSection",
    "BenchmarkSection",
    "MLCalibrationSection",
    "MLRuntimeSummarySection",
    "TrainingProgressSection",
    "ModelComparisonSection",
    "LayoutModelSection",
    "TextEmbeddingsSection",
    "LLMSemanticChecksSection",
    "CrossPageGNNSection",
]
