"""Standalone TOCR + grouping runner for the Grouper tab.

Provides a lightweight alternative to the full pipeline — only the two
stages the Grouper tab actually needs:

1. **TOCR** — extract ``GlyphBox`` tokens from one PDF page
2. **Grouping** — run ``build_clusters_v2`` to produce machine group
   suggestions for the "Show Machine Groups" toggle

Results are cached per PDF/page to
``~/.plancheck/grouper_cache/<pdf_hash>/page_<n>.json`` so repeated
navigation to the same page is instant.

This module is part of the ``plancheck.grouping`` package because it
orchestrates grouping-stage logic.  It imports from ``plancheck.tocr``
and ``plancheck.ingest`` but never from the GUI layer.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pdfplumber

from plancheck.config import GroupingConfig
from plancheck.grouping.clustering import build_clusters_v2
from plancheck.grouping.labeling import mark_headers, mark_notes
from plancheck.models.blocks import BlockCluster
from plancheck.models.tokens import GlyphBox
from plancheck.tocr.extract import build_extract_words_kwargs, extract_tocr_from_words

log = logging.getLogger(__name__)

# Cache root: ~/.plancheck/grouper_cache/
_CACHE_ROOT = Path.home() / ".plancheck" / "grouper_cache"


@dataclass
class GrouperPageData:
    """All data needed by the Grouper tab for one page.

    Attributes
    ----------
    boxes:
        TOCR-extracted ``GlyphBox`` tokens.  One token per word.
    machine_groups:
        ``BlockCluster`` instances produced by ``build_clusters_v2``.
        Used by the "Show Machine Groups" toggle.
    page_w, page_h:
        Page dimensions in PDF points.
    """

    boxes: List[GlyphBox] = field(default_factory=list)
    machine_groups: List[BlockCluster] = field(default_factory=list)
    page_w: float = 0.0
    page_h: float = 0.0


# ── Public API ────────────────────────────────────────────────────────────────


def run_grouper_pipeline(
    pdf_path: Path,
    page_num: int,
    cfg: Optional[GroupingConfig] = None,
) -> GrouperPageData:
    """Extract GlyphBoxes and machine group suggestions for one PDF page.

    Calls TOCR and grouping stages directly — no full pipeline overhead.
    Results are cached; a second call on the same page returns from cache.

    Parameters
    ----------
    pdf_path:
        Path to the source PDF.
    page_num:
        Zero-based page index.
    cfg:
        Grouping configuration.  ``GroupingConfig()`` defaults are used
        if omitted.

    Returns
    -------
    GrouperPageData
    """
    if cfg is None:
        cfg = GroupingConfig()

    # Cache hit path
    cached = load_cached_page(pdf_path, page_num)
    if cached is not None:
        log.debug("grouper_pipeline: cache hit — %s page %d", pdf_path.name, page_num)
        return cached

    log.info(
        "grouper_pipeline: running TOCR+grouping for %s page %d",
        pdf_path.name,
        page_num,
    )

    extract_kw = build_extract_words_kwargs(cfg, mode="full")

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        words: list[dict] = page.extract_words(**extract_kw)
        page_w = float(page.width)
        page_h = float(page.height)

    tocr_result = extract_tocr_from_words(
        words, page_num, page_w, page_h, cfg, mode="full"
    )
    boxes = tocr_result.tokens

    machine_groups = build_clusters_v2(boxes, cfg)
    mark_headers(machine_groups)
    mark_notes(machine_groups)

    data = GrouperPageData(
        boxes=boxes,
        machine_groups=machine_groups,
        page_w=page_w,
        page_h=page_h,
    )

    save_page_cache(pdf_path, page_num, data)
    return data


def get_grouper_cache_path(pdf_path: Path, page_num: int) -> Path:
    """Return the cache file path for a given PDF/page combination.

    Cache lives under ``~/.plancheck/grouper_cache/<hash12>/page_<n>.json``.
    The hash is derived from the resolved PDF path string so that the
    same logical file always maps to the same cache directory even when
    accessed via different relative paths.
    """
    path_hash = hashlib.sha256(str(pdf_path.resolve()).encode()).hexdigest()[:12]
    return _CACHE_ROOT / path_hash / f"page_{page_num}.json"


def load_cached_page(pdf_path: Path, page_num: int) -> Optional[GrouperPageData]:
    """Load a cached ``GrouperPageData`` from disk.

    Returns ``None`` if no cache file exists or the file is malformed.
    """
    cache_file = get_grouper_cache_path(pdf_path, page_num)
    if not cache_file.exists():
        return None
    try:
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
        boxes = [GlyphBox.from_dict(b) for b in raw["boxes"]]
        machine_groups = [
            BlockCluster.from_dict(g, boxes) for g in raw["machine_groups"]
        ]
        return GrouperPageData(
            boxes=boxes,
            machine_groups=machine_groups,
            page_w=raw["page_w"],
            page_h=raw["page_h"],
        )
    except Exception:  # noqa: BLE001 — corrupt cache; treat as miss
        log.warning("grouper_pipeline: cache file corrupt, ignoring — %s", cache_file)
        return None


def save_page_cache(pdf_path: Path, page_num: int, data: GrouperPageData) -> None:
    """Persist a ``GrouperPageData`` to the on-disk cache.

    Silently skips on any I/O error so a cache write failure never breaks
    the grouper session.
    """
    cache_file = get_grouper_cache_path(pdf_path, page_num)
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "boxes": [b.to_dict() for b in data.boxes],
            "machine_groups": [g.to_dict() for g in data.machine_groups],
            "page_w": data.page_w,
            "page_h": data.page_h,
        }
        cache_file.write_text(json.dumps(payload, indent=None), encoding="utf-8")
    except Exception:  # noqa: BLE001 — best-effort cache write
        log.warning(
            "grouper_pipeline: failed to write cache — %s", cache_file, exc_info=True
        )


def invalidate_page_cache(pdf_path: Path, page_num: int) -> None:
    """Delete the cache entry for one page, forcing a fresh TOCR run."""
    cache_file = get_grouper_cache_path(pdf_path, page_num)
    if cache_file.exists():
        cache_file.unlink()
        log.debug("grouper_pipeline: invalidated cache — %s", cache_file)
