from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
import argparse
import json
import shutil
from datetime import datetime

from plancheck import (
    BlockCluster,
    GlyphBox,
    GroupingConfig,
    StageResult,
    ingest_pdf,
    zone_summary,
)
from plancheck.analysis.structural_boxes import BoxType
from plancheck.export import export_page_results
from plancheck.export.overlay import draw_columns_overlay, draw_lines_overlay
from plancheck.export.page_data import serialize_page
from plancheck.export.report import generate_html_report, generate_json_report
from plancheck.pipeline import (
    DocumentResult,
    PageResult,
    _run_document_checks,
    input_fingerprint,
    run_pipeline,
)


def make_run_dir(base: Path, name: str) -> Path:
    run_dir = base / name
    for sub in ["artifacts", "overlays", "exports", "logs"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def cleanup_old_runs(run_root: Path, keep: int = 50) -> None:
    """Delete old run folders, keeping only the most recent `keep` runs."""
    run_dirs = sorted(
        [d for d in run_root.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old_dir in run_dirs[keep:]:
        shutil.rmtree(old_dir, ignore_errors=True)
        print(f"Cleaned up old run: {old_dir.name}")


def save_boxes_json(boxes: list[GlyphBox], out_path: Path) -> None:
    serializable = [
        {
            "page": b.page,
            "x0": b.x0,
            "y0": b.y0,
            "x1": b.x1,
            "y1": b.y1,
            "text": b.text,
            "origin": b.origin,
        }
        for b in boxes
    ]
    out_path.write_text(json.dumps(serializable, indent=2))


def summarize(blocks: list[BlockCluster]) -> str:
    summary = [f"Blocks: {len(blocks)}"]
    table_count = sum(1 for b in blocks if b.is_table)
    summary.append(f"Marked tables: {table_count}")
    for i, blk in enumerate(blocks, start=1):
        x0, y0, x1, y1 = blk.bbox()
        num_items = len(blk.lines) if blk.lines else len(blk.rows)
        item_label = "lines" if blk.lines else "rows"
        summary.append(
            f"Block {i}: {item_label}={num_items} table={blk.is_table} bbox=({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f})"
        )
    return "\n".join(summary)


def process_page(
    pdf: Path,
    page_num: int,
    run_dir: Path,
    resolution: int,
    color_overrides: dict | None = None,
    cfg: GroupingConfig | None = None,
) -> dict:
    """Process a single page and return page results for manifest.

    Delegates the full 9-stage pipeline to :func:`plancheck.pipeline.run_pipeline`
    then materialises all file artefacts (JSON, overlays, CSVs) to *run_dir*.
    """
    import time as _time

    if cfg is None:
        cfg = GroupingConfig()

    pdf_stem = pdf.stem.replace(" ", "_")
    t0 = _time.perf_counter()

    # ── Run the canonical library pipeline ────────────────────────────
    pr: PageResult = run_pipeline(pdf, page_num, cfg=cfg, resolution=resolution)

    # ── Materialise artefacts to disk ─────────────────────────────────
    _materialise_page(pr, pdf, run_dir, pdf_stem, resolution, cfg)

    elapsed = _time.perf_counter() - t0
    print(f"  page {page_num}: done ({elapsed:.1f}s)", flush=True)
    print(summarize(pr.blocks), flush=True)

    return _build_page_manifest(pr, pdf_stem, run_dir, cfg)


# ── Artefact materialisation ──────────────────────────────────────────


def _materialise_page(
    pr: PageResult,
    pdf: Path,
    run_dir: Path,
    pdf_stem: str,
    resolution: int,
    cfg: GroupingConfig,
) -> None:
    """Write all JSON artefacts, overlays, and exports for one page."""
    page_num = pr.page
    art = run_dir / "artifacts"

    # Boxes JSON
    boxes_path = art / f"{pdf_stem}_page_{page_num}_boxes.json"
    save_boxes_json(pr.tokens, boxes_path)

    # Blocks JSON
    blocks_path = art / f"{pdf_stem}_page_{page_num}_blocks.json"
    blocks_serialized = [_serialize_block(blk) for blk in pr.blocks]
    blocks_path.write_text(json.dumps(blocks_serialized, indent=2))

    # Columns JSON
    columns_path = art / f"{pdf_stem}_page_{page_num}_columns.json"
    columns_serialized = [_serialize_column(col) for col in pr.notes_columns]
    columns_path.write_text(json.dumps(columns_serialized, indent=2))

    # Extraction JSON (consumed by sheet recreation)
    extraction_path = art / f"{pdf_stem}_page_{page_num}_extraction.json"
    extraction_data = serialize_page(
        page=page_num,
        page_width=pr.page_width,
        page_height=pr.page_height,
        tokens=pr.tokens,
        blocks=pr.blocks,
        notes_columns=pr.notes_columns,
    )
    extraction_path.write_text(json.dumps(extraction_data, indent=2))

    # Structural boxes JSON
    struct_path = art / f"{pdf_stem}_page_{page_num}_structural_boxes.json"
    struct_serialized = [
        {
            "box_type": sb.box_type.value,
            "bbox": list(sb.bbox()),
            "confidence": round(sb.confidence, 3),
            "is_synthetic": sb.is_synthetic,
            "contained_blocks": len(sb.contained_block_indices),
            "contained_text_preview": (
                sb.contained_text[:120] if sb.contained_text else ""
            ),
        }
        for sb in pr.structural_boxes
    ]
    struct_path.write_text(json.dumps(struct_serialized, indent=2))

    # Semantic regions JSON
    regions_path = art / f"{pdf_stem}_page_{page_num}_semantic_regions.json"
    regions_serialized = [
        {
            "label": sr.label,
            "bbox": list(sr.bbox()),
            "confidence": round(sr.confidence, 3),
            "has_enclosing_box": sr.enclosing_box is not None,
            "anchor_text": (
                " ".join(
                    b.text for r in sr.anchor_block.rows for b in r.boxes if b.text
                )[:80]
                if sr.anchor_block
                else ""
            ),
            "child_blocks": len(sr.child_blocks),
        }
        for sr in pr.semantic_regions
    ]
    regions_path.write_text(json.dumps(regions_serialized, indent=2))

    # Abbreviation regions JSON
    abbrev_path = art / f"{pdf_stem}_page_{page_num}_abbreviations.json"
    abbrev_serialized = [
        {
            "header_text": ab.header_text(),
            "is_boxed": ab.is_boxed,
            "box_bbox": list(ab.box_bbox) if ab.box_bbox else None,
            "bbox": list(ab.bbox()),
            "confidence": ab.confidence,
            "entries_count": len(ab.entries),
            "entries": [
                {
                    "code": e.code,
                    "meaning": e.meaning,
                    "code_bbox": list(e.code_bbox) if e.code_bbox else None,
                    "meaning_bbox": list(e.meaning_bbox) if e.meaning_bbox else None,
                }
                for e in ab.entries
            ],
        }
        for ab in pr.abbreviation_regions
    ]
    abbrev_path.write_text(json.dumps(abbrev_serialized, indent=2))

    # Legend regions JSON
    legends_path = art / f"{pdf_stem}_page_{page_num}_legends.json"
    legends_serialized = [
        {
            "header_text": leg.header_text(),
            "is_boxed": leg.is_boxed,
            "box_bbox": list(leg.box_bbox) if leg.box_bbox else None,
            "bbox": list(leg.bbox()),
            "confidence": leg.confidence,
            "entries_count": len(leg.entries),
            "entries": [
                {
                    "symbol_bbox": list(e.symbol_bbox) if e.symbol_bbox else None,
                    "description": e.description,
                    "description_bbox": (
                        list(e.description_bbox) if e.description_bbox else None
                    ),
                }
                for e in leg.entries
            ],
        }
        for leg in pr.legend_regions
    ]
    legends_path.write_text(json.dumps(legends_serialized, indent=2))

    # Revision regions JSON
    revisions_path = art / f"{pdf_stem}_page_{page_num}_revisions.json"
    revisions_serialized = [
        {
            "header_text": rev.header_text(),
            "is_boxed": rev.is_boxed,
            "box_bbox": list(rev.box_bbox) if rev.box_bbox else None,
            "bbox": list(rev.bbox()),
            "confidence": rev.confidence,
            "entries_count": len(rev.entries),
            "entries": [
                {
                    "number": e.number,
                    "description": e.description,
                    "date": e.date,
                    "row_bbox": list(e.row_bbox) if e.row_bbox else None,
                }
                for e in rev.entries
            ],
        }
        for rev in pr.revision_regions
    ]
    revisions_path.write_text(json.dumps(revisions_serialized, indent=2))

    # Misc title regions JSON
    misc_titles_path = art / f"{pdf_stem}_page_{page_num}_misc_titles.json"
    misc_titles_serialized = [
        {
            "text": mt.text,
            "is_boxed": mt.is_boxed,
            "box_bbox": list(mt.box_bbox) if mt.box_bbox else None,
            "bbox": list(mt.bbox()),
            "confidence": mt.confidence,
        }
        for mt in pr.misc_title_regions
    ]
    misc_titles_path.write_text(json.dumps(misc_titles_serialized, indent=2))

    # Standard detail regions JSON
    std_details_path = art / f"{pdf_stem}_page_{page_num}_standard_details.json"
    std_details_serialized = [
        {
            "header_text": sd.header_text(),
            "subheader": sd.subheader,
            "subheader_bbox": list(sd.subheader_bbox) if sd.subheader_bbox else None,
            "is_boxed": sd.is_boxed,
            "box_bbox": list(sd.box_bbox) if sd.box_bbox else None,
            "bbox": list(sd.bbox()),
            "confidence": sd.confidence,
            "entries_count": len(sd.entries),
            "entries": [
                {
                    "sheet_number": e.sheet_number,
                    "description": e.description,
                    "sheet_bbox": list(e.sheet_bbox) if e.sheet_bbox else None,
                    "description_bbox": (
                        list(e.description_bbox) if e.description_bbox else None
                    ),
                }
                for e in sd.entries
            ],
        }
        for sd in pr.standard_detail_regions
    ]
    std_details_path.write_text(json.dumps(std_details_serialized, indent=2))

    # Zones JSON
    zones_path = art / f"{pdf_stem}_page_{page_num}_zones.json"
    zones_path.write_text(json.dumps(zone_summary(pr.page_zones), indent=2))

    # Semantic findings JSON
    if pr.semantic_findings:
        findings_path = art / f"{pdf_stem}_page_{page_num}_findings.json"
        findings_path.write_text(
            json.dumps([f.to_dict() for f in pr.semantic_findings], indent=2)
        )
        print(
            f"  page {page_num}: {len(pr.semantic_findings)} semantic finding(s)",
            flush=True,
        )

    # ── Overlays ──────────────────────────────────────────────────────
    scale = resolution / 72.0
    all_lines = [ln for blk in pr.blocks for ln in (blk.lines or [])]
    overlay_path = run_dir / "overlays" / f"{pdf_stem}_page_{page_num}_overlay.png"
    draw_lines_overlay(
        page_width=pr.page_width,
        page_height=pr.page_height,
        lines=all_lines,
        tokens=pr.tokens,
        out_path=overlay_path,
        scale=scale,
        background=pr.background_image,
        cfg=cfg,
    )

    col_overlay_path = run_dir / "overlays" / f"{pdf_stem}_page_{page_num}_columns.png"
    draw_columns_overlay(
        page_width=pr.page_width,
        page_height=pr.page_height,
        blocks=pr.blocks,
        tokens=pr.tokens,
        out_path=col_overlay_path,
        scale=scale,
        background=pr.background_image,
        cfg=cfg,
    )

    # OCR reconcile overlays
    if pr.reconcile_result is not None:
        from plancheck import draw_reconcile_debug, draw_symbol_overlay

        if pr.reconcile_result.added_tokens or cfg.ocr_reconcile_debug:
            recon_path = (
                run_dir / "overlays" / f"{pdf_stem}_page_{page_num}_ocr_reconcile.png"
            )
            draw_reconcile_debug(
                result=pr.reconcile_result,
                page_width=pr.page_width,
                page_height=pr.page_height,
                out_path=recon_path,
                scale=scale,
                background=pr.background_image,
            )
        if cfg.ocr_reconcile_debug:
            sym_path = run_dir / "overlays" / f"{pdf_stem}_page_{page_num}_symbols.png"
            draw_symbol_overlay(
                result=pr.reconcile_result,
                page_width=pr.page_width,
                page_height=pr.page_height,
                out_path=sym_path,
                scale=scale,
                background=pr.background_image,
            )


# ── Block / column serialisation helpers ─────────────────────────────


def _serialize_block(blk: BlockCluster) -> dict:
    x0, y0, x1, y1 = blk.bbox()
    result = {
        "page": blk.page,
        "bbox": [x0, y0, x1, y1],
        "rows": [
            {"bbox": list(row.bbox()), "texts": [b.text for b in row.boxes]}
            for row in blk.rows
        ],
        "label": blk.label,
        "is_table": bool(blk.is_table),
        "is_notes": bool(blk.is_notes),
    }
    if blk.lines and blk._tokens:
        result["lines"] = [
            {
                "line_id": line.line_id,
                "baseline_y": line.baseline_y,
                "bbox": list(line.bbox(blk._tokens)),
                "text": line.text(blk._tokens),
                "spans": [
                    {
                        "col_id": span.col_id,
                        "bbox": list(span.bbox(blk._tokens)),
                        "text": span.text(blk._tokens),
                    }
                    for span in line.spans
                ],
            }
            for line in blk.lines
        ]
    return result


def _serialize_column(col) -> dict:
    return {
        "header_text": col.header_text(),
        "base_header_text": col.base_header_text(),
        "is_continuation": col.is_continuation(),
        "column_group_id": col.column_group_id,
        "continues_from": col.continues_from,
        "notes_count": len(col.notes_blocks),
        "bbox": list(col.bbox()),
    }


# ── Manifest builder ─────────────────────────────────────────────────


def _compute_page_quality(pr: PageResult) -> float:
    """Compute a 0–1 page quality score from a PageResult."""
    tocr_counts = pr.stages.get("tocr", StageResult(stage="tocr")).counts or {}
    token_density = tocr_counts.get("token_density_per_sqin", 0.0)
    encoding_issues = tocr_counts.get("char_encoding_issues", 0)
    tokens_total = tocr_counts.get("tokens_total", 0)

    density_score = min(1.0, token_density / 2.0)
    region_count = (
        len(pr.notes_columns)
        + len(pr.abbreviation_regions)
        + len(pr.legend_regions)
        + len(pr.revision_regions)
        + len(pr.standard_detail_regions)
        + len(pr.misc_title_regions)
    )
    region_score = min(1.0, region_count / 3.0)
    error_frac = encoding_issues / tokens_total if tokens_total > 0 else 0.0
    error_penalty = min(1.0, error_frac * 10.0)
    return round(
        max(
            0.0,
            (density_score * 0.4 + region_score * 0.4) * (1.0 - error_penalty) + 0.2,
        ),
        3,
    )


def _build_page_manifest(
    pr: PageResult, pdf_stem: str, run_dir: Path, cfg: GroupingConfig
) -> dict:
    """Build a manifest dict for one page from a PageResult."""
    page_num = pr.page
    quality = _compute_page_quality(pr)

    # Stage health flags
    stage_health: dict = {}
    vocr_sr = pr.stages.get("vocr", StageResult(stage="vocr"))
    if vocr_sr.ran:
        vocr_tokens = (vocr_sr.counts or {}).get("tokens_total", 0)
        stage_health["vocr_degraded"] = (
            vocr_tokens == 0 and vocr_sr.duration_ms > 30_000
        )
    recon_sr = pr.stages.get("reconcile", StageResult(stage="reconcile"))
    if recon_sr.ran:
        recon_counts = recon_sr.counts or {}
        stage_health["reconcile_no_candidates"] = recon_counts.get("ocr_total", 0) == 0

    # Region confidences
    region_confidences: dict = {}
    if pr.abbreviation_regions:
        region_confidences["abbreviation"] = [
            round(r.confidence, 2) for r in pr.abbreviation_regions
        ]
    if pr.legend_regions:
        region_confidences["legend"] = [
            round(r.confidence, 2) for r in pr.legend_regions
        ]
    if pr.revision_regions:
        region_confidences["revision"] = [
            round(r.confidence, 2) for r in pr.revision_regions
        ]
    if pr.standard_detail_regions:
        region_confidences["standard_detail"] = [
            round(r.confidence, 2) for r in pr.standard_detail_regions
        ]
    if pr.misc_title_regions:
        region_confidences["misc_title"] = [
            round(r.confidence, 2) for r in pr.misc_title_regions
        ]

    # Build reconcile-related counts
    rr = pr.reconcile_result
    recon_counts_dict = {}
    if rr:
        recon_counts_dict = {
            "ocr_reconcile_accepted": len(rr.added_tokens),
            "ocr_reconcile_total": rr.stats.get("ocr_total", 0),
            "ocr_reconcile_candidates": rr.stats.get("candidates_generated", 0),
            "ocr_reconcile_candidates_accepted": rr.stats.get("candidates_accepted", 0),
            "ocr_reconcile_candidates_rejected": rr.stats.get("candidates_rejected", 0),
            "ocr_reconcile_filtered_non_numeric": rr.stats.get(
                "filtered_non_numeric", 0
            ),
        }

    art = run_dir / "artifacts"
    result = {
        "page": page_num,
        "page_width": pr.page_width,
        "page_height": pr.page_height,
        "skew_degrees": pr.skew_degrees,
        "page_quality": quality,
        "stage_health": stage_health,
        "region_confidences": region_confidences,
        "semantic_findings": [f.to_dict() for f in pr.semantic_findings],
        "semantic_findings_count": len(pr.semantic_findings),
        "stages": {name: sr.to_dict() for name, sr in pr.stages.items()},
        "counts": {
            "boxes": len(pr.tokens),
            "rows": sum(len(blk.rows) for blk in pr.blocks),
            "lines": sum(len(blk.lines or []) for blk in pr.blocks),
            "blocks": len(pr.blocks),
            "tables": sum(1 for b in pr.blocks if b.is_table),
            "notes_columns": len(pr.notes_columns),
            "graphics": len(pr.graphics),
            "abbreviation_regions": len(pr.abbreviation_regions),
            "abbreviation_entries": sum(
                len(ab.entries) for ab in pr.abbreviation_regions
            ),
            "legend_regions": len(pr.legend_regions),
            "legend_entries": sum(len(leg.entries) for leg in pr.legend_regions),
            "revision_regions": len(pr.revision_regions),
            "revision_entries": sum(len(r.entries) for r in pr.revision_regions),
            "misc_title_regions": len(pr.misc_title_regions),
            "standard_detail_regions": len(pr.standard_detail_regions),
            "standard_detail_entries": sum(
                len(sd.entries) for sd in pr.standard_detail_regions
            ),
            "structural_boxes": len(pr.structural_boxes),
            "structural_boxes_classified": sum(
                1 for sb in pr.structural_boxes if sb.box_type != BoxType.unknown
            ),
            "semantic_regions": len(pr.semantic_regions),
            "title_blocks": len(pr.title_blocks),
            **recon_counts_dict,
        },
        "artifacts": {
            "boxes_json": str(art / f"{pdf_stem}_page_{page_num}_boxes.json"),
            "overlay_png": str(
                run_dir / "overlays" / f"{pdf_stem}_page_{page_num}_overlay.png"
            ),
            "legends_json": str(art / f"{pdf_stem}_page_{page_num}_legends.json"),
            "abbreviations_json": str(
                art / f"{pdf_stem}_page_{page_num}_abbreviations.json"
            ),
            "revisions_json": str(art / f"{pdf_stem}_page_{page_num}_revisions.json"),
            "standard_details_json": str(
                art / f"{pdf_stem}_page_{page_num}_standard_details.json"
            ),
            "zones_json": str(art / f"{pdf_stem}_page_{page_num}_zones.json"),
        },
        "zones": zone_summary(pr.page_zones),
    }

    # CSV export
    try:
        page_exports = export_page_results(result, run_dir, pdf_stem)
        result["exports"] = page_exports
    except Exception as exc:  # pragma: no cover
        print(f"  page {page_num}: export warning: {exc}", flush=True)

    # Stash the PageResult for cross-page checks (stripped before serialisation)
    result["_page_result"] = pr

    return result


def run_pdf(
    pdf: Path,
    start: int,
    end: int | None,
    resolution: int,
    run_root: Path,
    run_prefix: str,
    color_overrides: dict | None = None,
    cfg: GroupingConfig | None = None,
) -> Path:
    """Process pages of a single PDF and create one run folder with all results."""
    if cfg is None:
        cfg = GroupingConfig()

    pdf_meta = ingest_pdf(pdf)
    total_pages = pdf_meta.num_pages
    end_page = end if end is not None else total_pages

    # Create single run folder for this PDF
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{stamp}_{run_prefix}"
    run_dir = make_run_dir(run_root, run_name)

    print(f"Processing {pdf.name} -> {run_dir}", flush=True)

    page_results = []
    for page_num in range(start, end_page):
        try:
            result = process_page(
                pdf,
                page_num,
                run_dir,
                resolution,
                color_overrides,
                cfg=cfg,
            )
            page_results.append(result)
        except Exception as exc:  # pragma: no cover
            print(f"  page {page_num}: ERROR {exc}", flush=True)
            page_results.append({"page": page_num, "error": str(exc)})

    # ── Cross-page checks ─────────────────────────────────────────────
    page_result_objects = [
        pr_obj
        for pr_obj in page_results
        if isinstance(pr_obj, dict) and "_page_result" in pr_obj
    ]
    cross_page_findings = []
    if len(page_result_objects) > 1:
        pr_list = [pr_obj["_page_result"] for pr_obj in page_result_objects]
        cross_page_findings = _run_document_checks(pr_list)

    # Write single manifest for entire run
    pages_list = list(range(start, end_page))
    # Strip internal _page_result refs before serialisation
    clean_results = [
        (
            {k: v for k, v in pr.items() if k != "_page_result"}
            if isinstance(pr, dict)
            else pr
        )
        for pr in page_results
    ]
    manifest = {
        "run_id": run_dir.name,
        "created_at": datetime.now().isoformat(),
        "source_pdf": str(pdf.resolve()),
        "pdf_name": pdf.name,
        "input_fingerprint": input_fingerprint(pdf, pages_list, cfg),
        "render_resolution_dpi": resolution,
        "overlay_scale": resolution / 72.0,
        "config_snapshot": vars(cfg),
        # Keep legacy key for backward compat
        "settings": vars(cfg),
        "pages_processed": pages_list,
        "pages": clean_results,
        "cross_page_findings": [f.to_dict() for f in cross_page_findings],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # ── Generate HTML + JSON reports ──────────────────────────────────
    try:
        generate_html_report(manifest, run_dir / "report.html")
        generate_json_report(manifest, run_dir / "report.json")
    except Exception as exc:  # pragma: no cover
        print(f"  report generation warning: {exc}", flush=True)

    print(f"Run complete: {run_dir}", flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch run pages of a PDF")
    parser.add_argument("pdf", type=Path, help="Path to PDF")
    parser.add_argument("--start", type=int, default=0, help="Start page (inclusive)")
    parser.add_argument(
        "--end", type=int, default=None, help="End page (exclusive); default = all"
    )
    parser.add_argument(
        "--resolution", type=int, default=200, help="Overlay render DPI"
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default=None,
        help="Prefix for run folder name (default: derived from PDF name)",
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("runs"), help="Root directory for runs"
    )
    parser.add_argument(
        "--keep-runs",
        type=int,
        default=50,
        help="Number of runs to keep (oldest deleted)",
    )
    parser.add_argument(
        "--ocr-full-reconcile",
        action="store_true",
        default=False,
        help="Enable full-page OCR reconciliation (inject missing %% / ° ± symbols)",
    )
    parser.add_argument(
        "--ocr-debug",
        action="store_true",
        default=False,
        help="Force OCR reconcile debug overlay even when no tokens are injected",
    )
    parser.add_argument(
        "--ocr-resolution",
        type=int,
        default=300,
        help="DPI for OCR page render (default 300; use 120 to avoid Paddle resize)",
    )
    parser.add_argument(
        "--ocr-preprocess",
        action="store_true",
        default=False,
        help="Preprocess OCR image (grayscale + CLAHE contrast) before PaddleOCR",
    )
    args = parser.parse_args()

    cfg = GroupingConfig(
        enable_ocr_reconcile=args.ocr_full_reconcile,
        ocr_reconcile_debug=args.ocr_debug,
        ocr_reconcile_resolution=args.ocr_resolution,
        enable_ocr_preprocess=args.ocr_preprocess,
    )

    # Derive run prefix from PDF name if not provided
    run_prefix = args.run_prefix or args.pdf.stem.replace(" ", "_")[:20]

    run_pdf(
        pdf=args.pdf,
        start=args.start,
        end=args.end,
        resolution=args.resolution,
        run_root=args.run_root,
        run_prefix=run_prefix,
        cfg=cfg,
    )

    # Cleanup old runs
    cleanup_old_runs(args.run_root, args.keep_runs)


if __name__ == "__main__":
    main()
