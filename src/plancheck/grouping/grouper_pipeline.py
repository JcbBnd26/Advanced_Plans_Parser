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
import statistics
from dataclasses import dataclass, field, replace as _dc_replace
from pathlib import Path
from typing import Dict, List, Optional

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
    force_refresh: bool = False,
) -> GrouperPageData:
    """Extract GlyphBoxes and machine group suggestions for one PDF page.

    Calls TOCR and grouping stages directly — no full pipeline overhead.
    Results are cached; a second call on the same page returns from cache
    unless *force_refresh* is ``True``.

    Parameters
    ----------
    pdf_path:
        Path to the source PDF.
    page_num:
        Zero-based page index.
    cfg:
        Grouping configuration.  ``GroupingConfig()`` defaults are used
        if omitted.
    force_refresh:
        When ``True`` the on-disk cache is ignored and the page is
        re-processed with *cfg*.  The new result is written back to cache.

    Returns
    -------
    GrouperPageData
    """
    if cfg is None:
        cfg = GroupingConfig()

    # Cache hit path (skipped when force_refresh is set)
    if not force_refresh:
        cached = load_cached_page(pdf_path, page_num)
        if cached is not None:
            log.debug(
                "grouper_pipeline: cache hit — %s page %d", pdf_path.name, page_num
            )
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


# ── Pattern propagation ───────────────────────────────────────────────────────


def propagate_page_patterns(
    source_groups: list,
    source_page_data: "GrouperPageData",
    target_page_data: "GrouperPageData",
    position_tol: float = 0.02,
) -> List[dict]:
    """Match confirmed groups from one page onto the tokens of another page.

    Uses normalised bounding boxes (0–1 of page dimensions) so that
    groups that occupy the same *relative* region on the page are
    matched regardless of absolute coordinate differences.

    Parameters
    ----------
    source_groups:
        List of :class:`~scripts.gui.grouper_state.ConfirmedGroup` objects
        from the previously confirmed page.
    source_page_data:
        :class:`GrouperPageData` for the source page (provides ``page_w``
        / ``page_h`` for normalisation).
    target_page_data:
        :class:`GrouperPageData` for the new page being loaded.
    position_tol:
        Extra margin added to each normalised bbox side when searching
        for matching tokens on the target page.  Default is 2 % of the
        page dimension.

    Returns
    -------
    list of dicts
        Each dict has ``indices`` (list of int into
        ``target_page_data.boxes``) and ``bbox`` (tuple of 4 floats in
        PDF points).  Returns an empty list when nothing matches.
    """
    if not source_groups or not target_page_data.boxes:
        return []

    sw, sh = source_page_data.page_w, source_page_data.page_h
    tw, th = target_page_data.page_w, target_page_data.page_h

    if sw <= 0 or sh <= 0 or tw <= 0 or th <= 0:
        return []

    results: List[dict] = []

    for grp in source_groups:
        bx0, by0, bx1, by1 = grp.bbox

        # Normalise source bbox
        nx0 = bx0 / sw - position_tol
        ny0 = by0 / sh - position_tol
        nx1 = bx1 / sw + position_tol
        ny1 = by1 / sh + position_tol

        # Find target tokens whose centre falls within the normalised region
        hits: List[int] = []
        for i, box in enumerate(target_page_data.boxes):
            cx = ((box.x0 + box.x1) / 2.0) / tw
            cy = ((box.y0 + box.y1) / 2.0) / th
            if nx0 <= cx <= nx1 and ny0 <= cy <= ny1:
                hits.append(i)

        if not hits:
            continue

        hit_boxes = [target_page_data.boxes[i] for i in hits]
        bbox = (
            min(b.x0 for b in hit_boxes),
            min(b.y0 for b in hit_boxes),
            max(b.x1 for b in hit_boxes),
            max(b.y1 for b in hit_boxes),
        )
        results.append({"indices": hits, "bbox": bbox})

    log.info(
        "grouper_pipeline: propagated %d / %d patterns to new page",
        len(results),
        len(source_groups),
    )
    return results


# ── Online learning: adapt config from confirmed groups ───────────────────────


def learn_cfg_from_confirmed_groups(
    saved_pages: "Dict[int, list]",
    pdf_path: Path,
    base_cfg: GroupingConfig,
) -> GroupingConfig:
    """Derive a document-adapted ``GroupingConfig`` from confirmed groups.

    Examines every confirmed group that has been saved across all pages
    and measures:

    * **max internal gap** — the largest vertical gap between consecutive
      pseudo-lines *within* any single confirmed group.  The new
      ``block_gap_mult`` is set so that this gap would *not* have caused
      a split.
    * **token height** — estimated from the median GlyphBox height across
      all confirmed tokens, used to convert the raw gap to a multiplier.

    The returned config is a shallow copy of *base_cfg* with only
    ``block_gap_mult`` updated.  It falls back to *base_cfg* if there is
    not enough data to make a reliable estimate.

    Parameters
    ----------
    saved_pages:
        ``GrouperSessionState.saved_pages`` — maps page index to a list
        of :class:`~scripts.gui.grouper_state.ConfirmedGroup` objects.
    pdf_path:
        Path to the source PDF (needed to load page caches).
    base_cfg:
        Config to copy and update.
    """
    if not saved_pages:
        return base_cfg

    token_heights: List[float] = []
    max_internal_gap: float = 0.0

    for page_num, groups in saved_pages.items():
        page_data = load_cached_page(pdf_path, page_num)
        if page_data is None:
            continue
        boxes = page_data.boxes

        for grp in groups:
            idxs = [i for i in grp.indices if i < len(boxes)]
            if len(idxs) < 2:
                continue

            grp_boxes = [boxes[i] for i in idxs]

            # Collect token heights
            for b in grp_boxes:
                h = b.y1 - b.y0
                if h > 0:
                    token_heights.append(h)

            # Cluster tokens into pseudo-lines by y-centre proximity
            sorted_boxes = sorted(grp_boxes, key=lambda b: (b.y0 + b.y1) / 2.0)
            if not sorted_boxes:
                continue

            median_h = statistics.median(b.y1 - b.y0 for b in sorted_boxes)
            line_tol = median_h * 0.6

            # Build pseudo-lines: group tokens whose y-centres are within tol
            pseudo_lines: List[List[GlyphBox]] = [[sorted_boxes[0]]]
            for b in sorted_boxes[1:]:
                ref_y = sum((pb.y0 + pb.y1) / 2.0 for pb in pseudo_lines[-1]) / len(
                    pseudo_lines[-1]
                )
                this_y = (b.y0 + b.y1) / 2.0
                if abs(this_y - ref_y) <= line_tol:
                    pseudo_lines[-1].append(b)
                else:
                    pseudo_lines.append([b])

            if len(pseudo_lines) < 2:
                continue

            # Measure vertical gaps between consecutive pseudo-lines
            for i in range(1, len(pseudo_lines)):
                prev_bottom = max(b.y1 for b in pseudo_lines[i - 1])
                cur_top = min(b.y0 for b in pseudo_lines[i])
                gap = cur_top - prev_bottom
                if gap > max_internal_gap:
                    max_internal_gap = gap

    if max_internal_gap <= 0 or not token_heights:
        return base_cfg

    median_token_h = statistics.median(token_heights)
    if median_token_h <= 0:
        return base_cfg

    # Set block_gap_mult so that max_internal_gap is just within threshold.
    # Add a 20 % margin so borderline gaps don't cause splits on the next page.
    new_mult = (max_internal_gap / median_token_h) * 1.2

    # Clamp to a sensible range — never go below the default or above 15×
    new_mult = max(base_cfg.block_gap_mult, min(15.0, new_mult))

    if abs(new_mult - base_cfg.block_gap_mult) < 0.05:
        return base_cfg  # no meaningful change

    log.info(
        "grouper_pipeline: learned block_gap_mult %.2f → %.2f "
        "(max_internal_gap=%.1f pt, median_token_h=%.1f pt, %d groups from %d pages)",
        base_cfg.block_gap_mult,
        new_mult,
        max_internal_gap,
        median_token_h,
        sum(len(gs) for gs in saved_pages.values()),
        len(saved_pages),
    )
    return _dc_replace(base_cfg, block_gap_mult=new_mult)
