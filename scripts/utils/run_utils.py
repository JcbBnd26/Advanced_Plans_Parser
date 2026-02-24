"""Shared utility functions for scripts.

Centralises helpers that were previously duplicated across overlay scripts,
runners, and the GUI overlay viewer.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence


def latest_overlays_dir(runs_root: Path | str = "runs") -> Path:
    """Return the ``overlays/`` subdirectory of the most recent run folder.

    Falls back to the current directory if no run folders exist.
    """
    runs_dir = Path(runs_root)
    if runs_dir.is_dir():
        run_dirs = sorted(runs_dir.iterdir(), reverse=True)
        if run_dirs:
            d = run_dirs[0] / "overlays"
            d.mkdir(parents=True, exist_ok=True)
            return d
    return Path(".")


def scale(x: float, y: float, s: float) -> tuple[int, int]:
    """Scale a coordinate pair by factor *s* and return integer pixel coords."""
    return int(x * s), int(y * s)


_DEFAULT_SUBDIRS = ("inputs", "artifacts", "overlays", "exports", "logs")


def make_run_dir(
    runs_root: Path | str = "runs",
    label: str | None = None,
    subdirs: Sequence[str] = _DEFAULT_SUBDIRS,
) -> Path:
    """Create a timestamped run directory with standard sub-folders.

    Parameters
    ----------
    runs_root : base directory that holds all run folders (default ``runs/``).
    label : optional suffix appended to the timestamp, e.g. ``"IFC_page2"``.
    subdirs : sub-directories to create inside the run folder.
              Defaults to ``("inputs", "artifacts", "overlays", "exports", "logs")``.

    Returns
    -------
    Path to the newly created run directory.
    """
    runs_root = Path(runs_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{stamp}_{label}" if label else f"run_{stamp}"
    run_dir = runs_root / run_name
    for sub in subdirs:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir
