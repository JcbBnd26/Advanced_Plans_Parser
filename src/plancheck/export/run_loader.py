"""Load a saved pipeline run from disk into a DocumentResult.

This module reconstructs ``PageResult`` and ``DocumentResult`` objects
from the JSON artifacts written by ``run_pdf_batch._materialise_page()``.

Usage::

    from plancheck.export.run_loader import load_run

    doc = load_run("runs/run_20260219_180300_IFC_page2")
    for page in doc.pages:
        print(page.page, len(page.tokens), len(page.blocks))

The loader reads the ``manifest.json`` for run metadata, then locates
per-page extraction and region JSON files in ``artifacts/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import GroupingConfig
from ..models import (
    AbbreviationEntry,
    AbbreviationRegion,
    LegendEntry,
    LegendRegion,
    MiscTitleRegion,
    RevisionEntry,
    RevisionRegion,
    StandardDetailEntry,
    StandardDetailRegion,
)
from ..pipeline import DocumentResult, PageResult, StageResult

log = logging.getLogger(__name__)


def _read_json(path: Path) -> Any:
    """Read and parse a JSON file, returning None on failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        log.debug("Could not read %s: %s", path, exc)
        return None


def _find_artifact(artifacts_dir: Path, suffix: str) -> Optional[Path]:
    """Find an artifact file by suffix pattern (e.g. '_extraction.json')."""
    for p in artifacts_dir.iterdir():
        if p.name.endswith(suffix):
            return p
    return None


def _load_page_from_artifacts(
    artifacts_dir: Path,
    page_num: int,
    page_info: Optional[Dict[str, Any]] = None,
) -> Optional[PageResult]:
    """Reconstruct a PageResult from artifact JSON files.

    Parameters
    ----------
    artifacts_dir : Path
        The ``artifacts/`` directory inside a run folder.
    page_num : int
        The page number (used to locate artifact files).
    page_info : dict, optional
        Per-page metadata from ``manifest.json`` (stages, counts, etc.).
    """
    from ..analysis.structural_boxes import SemanticRegion, StructuralBox
    from ..analysis.zoning import PageZone

    # Find extraction file (contains tokens, blocks, notes_columns)
    extraction_suffix = f"_page_{page_num}_extraction.json"
    extraction_file = _find_artifact(artifacts_dir, extraction_suffix)
    if not extraction_file:
        log.warning(
            "No extraction file found for page %d in %s", page_num, artifacts_dir
        )
        return None

    extraction_data = _read_json(extraction_file)
    if not extraction_data:
        return None

    # Deserialize core artefacts
    from ..export.page_data import deserialize_page

    tokens, blocks, notes_columns, page_width, page_height = deserialize_page(
        extraction_data
    )

    # Load region artifacts
    def _load_region_file(suffix: str) -> list:
        fpath = _find_artifact(artifacts_dir, f"_page_{page_num}_{suffix}.json")
        if not fpath:
            return []
        data = _read_json(fpath)
        return data if isinstance(data, list) else []

    # Abbreviation regions
    abbrev_data = _load_region_file("abbreviations")
    abbreviation_regions = []
    for rd in abbrev_data:
        entries = [
            AbbreviationEntry(
                page=page_num,
                code=e.get("code", ""),
                meaning=e.get("meaning", ""),
                code_bbox=tuple(e["code_bbox"]) if e.get("code_bbox") else None,
                meaning_bbox=(
                    tuple(e["meaning_bbox"]) if e.get("meaning_bbox") else None
                ),
            )
            for e in rd.get("entries", [])
        ]
        abbreviation_regions.append(
            AbbreviationRegion(
                page=page_num,
                entries=entries,
                is_boxed=rd.get("is_boxed", False),
                box_bbox=tuple(rd["box_bbox"]) if rd.get("box_bbox") else None,
                confidence=rd.get("confidence", 0.0),
            )
        )

    # Legend regions
    legend_data = _load_region_file("legends")
    legend_regions = []
    for rd in legend_data:
        entries = [
            LegendEntry(
                page=page_num,
                symbol_bbox=tuple(e["symbol_bbox"]) if e.get("symbol_bbox") else None,
                description=e.get("description", ""),
                description_bbox=(
                    tuple(e["description_bbox"]) if e.get("description_bbox") else None
                ),
            )
            for e in rd.get("entries", [])
        ]
        legend_regions.append(
            LegendRegion(
                page=page_num,
                entries=entries,
                is_boxed=rd.get("is_boxed", False),
                box_bbox=tuple(rd["box_bbox"]) if rd.get("box_bbox") else None,
                confidence=rd.get("confidence", 0.0),
            )
        )

    # Revision regions
    revision_data = _load_region_file("revisions")
    revision_regions = []
    for rd in revision_data:
        entries = [
            RevisionEntry(
                page=page_num,
                number=e.get("number", ""),
                description=e.get("description", ""),
                date=e.get("date", ""),
                row_bbox=tuple(e["row_bbox"]) if e.get("row_bbox") else None,
            )
            for e in rd.get("entries", [])
        ]
        revision_regions.append(
            RevisionRegion(
                page=page_num,
                entries=entries,
                is_boxed=rd.get("is_boxed", False),
                box_bbox=tuple(rd["box_bbox"]) if rd.get("box_bbox") else None,
                confidence=rd.get("confidence", 0.0),
            )
        )

    # Standard detail regions
    std_data = _load_region_file("standard_details")
    standard_detail_regions = []
    for rd in std_data:
        entries = [
            StandardDetailEntry(
                page=page_num,
                sheet_number=e.get("sheet_number", ""),
                description=e.get("description", ""),
                sheet_bbox=tuple(e["sheet_bbox"]) if e.get("sheet_bbox") else None,
                description_bbox=(
                    tuple(e["description_bbox"]) if e.get("description_bbox") else None
                ),
            )
            for e in rd.get("entries", [])
        ]
        standard_detail_regions.append(
            StandardDetailRegion(
                page=page_num,
                subheader=rd.get("subheader"),
                subheader_bbox=(
                    tuple(rd["subheader_bbox"]) if rd.get("subheader_bbox") else None
                ),
                entries=entries,
                is_boxed=rd.get("is_boxed", False),
                box_bbox=tuple(rd["box_bbox"]) if rd.get("box_bbox") else None,
                confidence=rd.get("confidence", 0.0),
            )
        )

    # Misc title regions
    misc_data = _load_region_file("misc_titles")
    misc_title_regions = [
        MiscTitleRegion(
            page=page_num,
            text=rd.get("text", ""),
            is_boxed=rd.get("is_boxed", False),
            box_bbox=tuple(rd["box_bbox"]) if rd.get("box_bbox") else None,
            confidence=rd.get("confidence", 0.0),
        )
        for rd in misc_data
    ]

    # Structural boxes
    struct_data = _load_region_file("structural_boxes")
    structural_boxes = [StructuralBox.from_dict(sd) for sd in struct_data]

    # Semantic regions
    sem_data = _load_region_file("semantic_regions")
    semantic_regions = [SemanticRegion.from_dict(sd, blocks) for sd in sem_data]

    # Zones
    zones_suffix = f"_page_{page_num}_zones.json"
    zones_file = _find_artifact(artifacts_dir, zones_suffix)
    page_zones: list = []
    if zones_file:
        zones_data = _read_json(zones_file)
        if zones_data and isinstance(zones_data, dict):
            for zd in zones_data.get("zones", []):
                page_zones.append(PageZone.from_dict(zd))
        elif zones_data and isinstance(zones_data, list):
            for zd in zones_data:
                page_zones.append(PageZone.from_dict(zd))

    # Reconstruct stages from manifest page_info if available
    stages: Dict[str, StageResult] = {}
    if page_info and "stages" in page_info:
        for name, sr_data in page_info["stages"].items():
            stages[name] = StageResult.from_dict(sr_data)

    # Build PageResult
    pr = PageResult(
        page=page_num,
        page_width=page_width,
        page_height=page_height,
        stages=stages,
        tokens=tokens,
        blocks=blocks,
        notes_columns=notes_columns,
        structural_boxes=structural_boxes,
        semantic_regions=semantic_regions,
        abbreviation_regions=abbreviation_regions,
        legend_regions=legend_regions,
        revision_regions=revision_regions,
        standard_detail_regions=standard_detail_regions,
        misc_title_regions=misc_title_regions,
        page_zones=page_zones,
    )
    return pr


def load_run(run_dir: str | Path) -> DocumentResult:
    """Load a saved pipeline run and reconstruct a DocumentResult.

    Parameters
    ----------
    run_dir : str or Path
        Path to the run directory (e.g. ``runs/run_20260219_180300_IFC_page2``).

    Returns
    -------
    DocumentResult
        Reconstructed document with all available page data.

    Raises
    ------
    FileNotFoundError
        If the run directory or manifest does not exist.
    """
    run_path = Path(run_dir)
    if not run_path.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_path}")

    manifest_path = run_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = _read_json(manifest_path)
    if manifest is None:
        raise ValueError(f"Could not parse manifest: {manifest_path}")

    artifacts_dir = run_path / "artifacts"
    if not artifacts_dir.is_dir():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")

    # Reconstruct config from manifest
    cfg = None
    config_data = manifest.get("config_snapshot") or manifest.get("settings")
    if config_data:
        try:
            cfg = GroupingConfig.from_dict(config_data)
        except Exception:
            log.debug("Could not reconstruct config from manifest", exc_info=True)

    # Determine pages from manifest or by scanning artifacts
    page_infos: List[Dict[str, Any]] = manifest.get("page_results", [])

    pages: List[PageResult] = []

    if page_infos:
        # Manifest has per-page metadata
        for pi in page_infos:
            page_num = pi.get("page", 0)
            pr = _load_page_from_artifacts(artifacts_dir, page_num, page_info=pi)
            if pr:
                pages.append(pr)
    else:
        # Scan artifacts directory for extraction files
        extraction_files = sorted(artifacts_dir.glob("*_extraction.json"))
        for ef in extraction_files:
            # Extract page number from filename pattern: ..._page_N_extraction.json
            parts = ef.stem.split("_page_")
            if len(parts) >= 2:
                try:
                    page_num = int(parts[-1].replace("_extraction", ""))
                except ValueError:
                    continue
                pr = _load_page_from_artifacts(artifacts_dir, page_num)
                if pr:
                    pages.append(pr)

    pdf_path = None
    if manifest.get("source_pdf"):
        pdf_path = Path(manifest["source_pdf"])

    return DocumentResult(
        pdf_path=pdf_path,
        pages=pages,
        config=cfg,
    )
