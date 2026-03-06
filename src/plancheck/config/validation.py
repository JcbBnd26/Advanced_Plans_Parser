"""Validation helper functions for configuration values."""

from __future__ import annotations

from .exceptions import ConfigValidationError


def _check_range(
    name: str, value: float, lo: float, hi: float, *, inclusive: bool = True
) -> None:
    """Raise ConfigValidationError if *value* is outside [lo, hi]."""
    if inclusive:
        if not (lo <= value <= hi):
            raise ConfigValidationError(f"{name}={value} out of range [{lo}, {hi}]")
    else:
        if not (lo < value < hi):
            raise ConfigValidationError(f"{name}={value} out of range ({lo}, {hi})")


def _check_positive(name: str, value: float) -> None:
    """Raise ConfigValidationError if *value* is not positive."""
    if value <= 0:
        raise ConfigValidationError(f"{name}={value} must be > 0")


def _check_non_negative(name: str, value: float) -> None:
    """Raise ConfigValidationError if *value* is negative."""
    if value < 0:
        raise ConfigValidationError(f"{name}={value} must be >= 0")


def _check_odd(name: str, value: int, floor: int = 3) -> None:
    """Raise ConfigValidationError if *value* is even or below *floor*."""
    if value < floor:
        raise ConfigValidationError(f"{name}={value} must be >= {floor}")
    if value % 2 == 0:
        raise ConfigValidationError(f"{name}={value} must be odd")
