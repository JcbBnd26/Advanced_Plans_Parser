"""GUI mixin modules extracted from AnnotationStoreMixin.

This package contains focused mixins that compose into the AnnotationTab:

- label_registry: Label type management and persistence
- filter_controls: Filter UI for element types and confidence
- model_training: ML model training, metrics, and active learning
"""

from __future__ import annotations

from .filter_controls import FilterControlsMixin
from .label_registry import LabelRegistryMixin
from .model_training import ModelTrainingMixin

__all__ = [
    "LabelRegistryMixin",
    "FilterControlsMixin",
    "ModelTrainingMixin",
]
