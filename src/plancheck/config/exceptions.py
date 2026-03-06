"""Configuration-related exceptions."""

from __future__ import annotations


class ConfigValidationError(ValueError):
    """Raised when a GroupingConfig field has an invalid value."""


class ConfigLoadError(ValueError):
    """Raised when a config file cannot be loaded or parsed."""
