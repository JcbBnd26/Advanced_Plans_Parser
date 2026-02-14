"""Pipeline stage infrastructure: gating, timing, and stage-result recording.

Provides a canonical pipeline contract for the 5-stage OCR flow:

    ingest → tocr → vocrpp → vocr → reconcile

Every stage produces a :class:`StageResult` that is serialised into the
run manifest. Gating logic is centralised in :func:`gate` so that
GUI / CLI / benchmark runners all behave identically.
"""

from __future__ import annotations

import hashlib
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from .config import GroupingConfig

# ── Skip reasons (exhaustive enumeration) ──────────────────────────────


class SkipReason(str, Enum):
    """Why a pipeline stage was skipped."""

    disabled_by_config = "disabled_by_config"
    missing_dependency = "missing_dependency"
    missing_inputs = "missing_inputs"
    no_pages = "no_pages"
    no_images = "no_images"
    no_tokens = "no_tokens"
    upstream_failed = "upstream_failed"
    cache_hit = "cache_hit"
    not_applicable = "not_applicable"


# ── Stage result ───────────────────────────────────────────────────────


@dataclass
class StageResult:
    """Outcome record for a single pipeline stage."""

    stage: str
    enabled: bool = False
    ran: bool = False
    status: str = "skipped"  # "success" | "skipped" | "failed"
    skip_reason: Optional[str] = None
    duration_ms: int = 0
    counts: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "stage": self.stage,
            "enabled": self.enabled,
            "ran": self.ran,
            "status": self.status,
        }
        if self.skip_reason is not None:
            d["skip_reason"] = self.skip_reason
        d["duration_ms"] = self.duration_ms
        if self.counts:
            d["counts"] = self.counts
        if self.inputs:
            d["inputs"] = self.inputs
        if self.outputs:
            d["outputs"] = self.outputs
        if self.error is not None:
            d["error"] = self.error
        return d


# ── Dependency probes (cached at module level) ─────────────────────────


def _has_paddleocr() -> bool:
    try:
        import os

        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        import paddleocr  # noqa: F401

        return True
    except ImportError:
        return False


def _has_cv2() -> bool:
    try:
        import cv2  # noqa: F401

        return True
    except ImportError:
        return False


# ── Canonical gating function ──────────────────────────────────────────

# Ordered stage names — the canonical pipeline sequence.
STAGE_ORDER: List[str] = ["ingest", "tocr", "vocrpp", "vocr", "reconcile"]


def gate(
    stage: str,
    cfg: GroupingConfig,
    inputs: Dict[str, Any] | None = None,
) -> tuple[bool, Optional[str]]:
    """Decide whether *stage* should run.

    Parameters
    ----------
    stage : str
        One of :data:`STAGE_ORDER`.
    cfg : GroupingConfig
        Effective configuration for the run.
    inputs : dict, optional
        Lightweight metadata about upstream outputs (e.g.
        ``{"tocr_tokens": 1500, "vocrpp_image": True}``).

    Returns
    -------
    (should_run, skip_reason)
        *should_run* is ``True`` when the stage should execute.
        When ``False``, *skip_reason* explains why.
    """
    if inputs is None:
        inputs = {}

    if stage == "ingest":
        # Always runs.
        return True, None

    if stage == "tocr":
        if not cfg.enable_tocr:
            return False, SkipReason.disabled_by_config.value
        if not inputs.get("has_pdf", True):
            return False, SkipReason.missing_inputs.value
        return True, None

    if stage == "vocrpp":
        if not cfg.enable_ocr_preprocess:
            return False, SkipReason.disabled_by_config.value
        if not cfg.enable_vocr:
            return False, SkipReason.disabled_by_config.value
        if not _has_cv2():
            return False, SkipReason.missing_dependency.value
        return True, None

    if stage == "vocr":
        if not cfg.enable_vocr:
            return False, SkipReason.disabled_by_config.value
        if not _has_paddleocr():
            return False, SkipReason.missing_dependency.value
        return True, None

    if stage == "reconcile":
        if not cfg.enable_ocr_reconcile:
            return False, SkipReason.disabled_by_config.value
        if not cfg.enable_vocr:
            # Reconcile requires VOCR tokens to merge.
            return False, SkipReason.disabled_by_config.value
        if not _has_paddleocr():
            return False, SkipReason.missing_dependency.value
        if inputs.get("vocr_failed") and not inputs.get("tocr_tokens"):
            return False, SkipReason.upstream_failed.value
        return True, None

    # Unknown stage — defensive.
    return False, SkipReason.not_applicable.value


# ── Stage context manager ──────────────────────────────────────────────


@contextmanager
def run_stage(
    stage: str,
    cfg: GroupingConfig,
    inputs: Dict[str, Any] | None = None,
) -> Generator[StageResult, None, None]:
    """Context manager that wraps a pipeline stage with gating + timing.

    Usage::

        with run_stage("tocr", cfg) as sr:
            if sr.ran:
                # … do the work …
                sr.counts["tokens_total"] = 1234
                sr.status = "success"

    The yielded :class:`StageResult` has ``ran=True`` only when
    :func:`gate` approves the stage. The caller should check ``sr.ran``
    before doing expensive work. Timing is handled automatically.
    """
    should_run, skip_reason = gate(stage, cfg, inputs)

    sr = StageResult(stage=stage)
    sr.enabled = should_run or (skip_reason == SkipReason.missing_dependency.value)
    # A stage is "enabled" in config even if a dependency is missing.
    # It only truly *runs* when gate approves.
    if stage == "vocrpp":
        sr.enabled = cfg.enable_ocr_preprocess and cfg.enable_ocr_reconcile
    elif stage == "vocr":
        sr.enabled = cfg.enable_ocr_reconcile
    elif stage == "reconcile":
        sr.enabled = cfg.enable_ocr_reconcile
    else:
        sr.enabled = True  # ingest / tocr always enabled

    if inputs:
        sr.inputs = inputs

    if not should_run:
        sr.ran = False
        sr.status = "skipped"
        sr.skip_reason = skip_reason
        yield sr
        return

    sr.ran = True
    t0 = time.perf_counter()
    try:
        yield sr
        # If the caller didn't explicitly set status, mark success if no error.
        if sr.status not in ("success", "failed"):
            sr.status = "success"
    except Exception as exc:
        sr.status = "failed"
        sr.error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "stack": traceback.format_exc(),
        }
        # Re-raise so the outer handler can decide fallback policy.
        raise
    finally:
        elapsed = time.perf_counter() - t0
        sr.duration_ms = int(elapsed * 1000)


# ── Input fingerprint (determinism aid) ────────────────────────────────


def input_fingerprint(
    pdf_path: Path,
    pages: List[int],
    cfg: GroupingConfig,
) -> str:
    """Compute a reproducibility fingerprint for a pipeline run.

    Uses file size + mtime (fast) rather than full content hash.
    """
    stat = pdf_path.stat()
    parts = [
        str(pdf_path.resolve()),
        str(stat.st_size),
        str(stat.st_mtime_ns),
        ",".join(str(p) for p in sorted(pages)),
        str(sorted(vars(cfg).items())),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
