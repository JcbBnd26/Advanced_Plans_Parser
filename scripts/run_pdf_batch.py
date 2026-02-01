import re


def detect_headers_from_words(boxes: list[GlyphBox]) -> list[GlyphBox]:
    """Tag header candidates directly from word boxes (before grouping)."""
    # Improved: group words into lines by y0 and horizontal proximity
    from collections import defaultdict

    header_re = re.compile(r"^[A-Z0-9\s\-\(\)\'\.]+: *$", re.ASCII)
    # Step 1: group by y0 (row)
    y_tol = 2.0  # tolerance for y alignment in points
    rows = defaultdict(list)
    for b in boxes:
        found = False
        for y in rows:
            if abs(b.y0 - y) < y_tol:
                rows[y].append(b)
                found = True
                break
        if not found:
            rows[b.y0].append(b)
    header_boxes = []
    for row in rows.values():
        # Step 2: sort by x0 and group horizontally close words into lines
        row = sorted(row, key=lambda b: b.x0)
        line = []
        lines = []
        x_gap_tol = 20.0  # max gap between words in a line (points)
        for b in row:
            if not line:
                line.append(b)
            else:
                prev = line[-1]
                if b.x0 - prev.x1 < x_gap_tol:
                    line.append(b)
                else:
                    lines.append(line)
                    line = [b]
        if line:
            lines.append(line)
        # Step 3: apply header regex to each line
        for line_words in lines:
            line_text = " ".join(b.text for b in line_words if b.text).strip()
            line_text_norm = re.sub(r"\s+", " ", line_text).upper()
            if header_re.match(line_text_norm):
                for b in line_words:
                    b.origin = "header_candidate"
                header_boxes.extend(line_words)
    return header_boxes


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pdfplumber

from plancheck import (
    BlockCluster,
    GlyphBox,
    GroupingConfig,
    draw_overlay,
    estimate_skew_degrees,
    group_blocks,
    group_rows,
    mark_notes,
    mark_tables,
    nms_prune,
    rotate_boxes,
)
from plancheck.grouping import group_notes_columns, link_continued_columns, mark_headers
from plancheck.legends import (
    detect_abbreviation_regions,
    detect_legend_regions,
    detect_misc_title_regions,
    detect_revision_regions,
    extract_graphics,
    filter_graphics_outside_regions,
)
from plancheck.models import (
    AbbreviationEntry,
    AbbreviationRegion,
    GraphicElement,
    LegendEntry,
    LegendRegion,
    MiscTitleRegion,
    RevisionEntry,
    RevisionRegion,
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


def page_boxes(pdf_path: Path, page_num: int) -> tuple[list[GlyphBox], float, float]:
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        page_w, page_h = float(page.width), float(page.height)
        words = page.extract_words()
        boxes: list[GlyphBox] = []
        for w in words:
            # Clip coordinates to page bounds (PDF content can extend past page edge)
            x0 = max(0.0, min(page_w, float(w.get("x0", 0.0))))
            x1 = max(0.0, min(page_w, float(w.get("x1", 0.0))))
            y0 = max(0.0, min(page_h, float(w.get("top", 0.0))))
            y1 = max(0.0, min(page_h, float(w.get("bottom", 0.0))))
            text = w.get("text", "")
            # Skip degenerate boxes (fully clipped)
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append(
                GlyphBox(
                    page=page_num, x0=x0, y0=y0, x1=x1, y1=y1, text=text, origin="text"
                )
            )
    return boxes, page_w, page_h


def render_page_image(pdf_path: Path, page_num: int, resolution: int = 200):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        img_page = page.to_image(resolution=resolution)
        return img_page.original.copy()


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
    lines = [f"Blocks: {len(blocks)}"]
    table_count = sum(1 for b in blocks if b.is_table)
    lines.append(f"Marked tables: {table_count}")
    for i, blk in enumerate(blocks, start=1):
        x0, y0, x1, y1 = blk.bbox()
        lines.append(
            f"Block {i}: rows={len(blk.rows)} table={blk.is_table} bbox=({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f})"
        )
    return "\n".join(lines)


def process_page(pdf: Path, page_num: int, run_dir: Path, resolution: int) -> dict:
    """Process a single page and return page results for manifest."""
    pdf_stem = pdf.stem.replace(" ", "_")

    boxes, page_w, page_h = page_boxes(pdf, page_num)
    bg_img = render_page_image(pdf, page_num, resolution=resolution)

    cfg = GroupingConfig()
    boxes = nms_prune(boxes, cfg.iou_prune)
    if cfg.enable_skew:
        skew = estimate_skew_degrees(boxes, cfg.max_skew_degrees)
        boxes = rotate_boxes(boxes, -skew, page_w, page_h)
    else:
        skew = 0.0

    rows = group_rows(boxes, cfg)
    blocks = group_blocks(rows, cfg)
    mark_tables(blocks, cfg)
    # Block-level header detection and debug output
    debug_path = str(run_dir / "artifacts" / "debug_headers.txt")
    mark_headers(blocks, debug_path=debug_path)
    # Notes detection, will skip header blocks
    mark_notes(blocks, debug_path=debug_path)
    # Group headers with their notes blocks into columns
    notes_columns = group_notes_columns(blocks, debug_path=debug_path)
    # Link continued columns (e.g., "NOTES" and "NOTES (CONT'D)")
    link_continued_columns(notes_columns, blocks=blocks, debug_path=debug_path)

    boxes_path = run_dir / "artifacts" / f"{pdf_stem}_page_{page_num}_boxes.json"
    save_boxes_json(boxes, boxes_path)

    # Save block-level clusters with tags
    blocks_path = run_dir / "artifacts" / f"{pdf_stem}_page_{page_num}_blocks.json"

    def serialize_block(blk):
        x0, y0, x1, y1 = blk.bbox()
        return {
            "page": blk.page,
            "bbox": [x0, y0, x1, y1],
            "rows": [
                {
                    "bbox": list(row.bbox()),
                    "texts": [b.text for b in row.boxes],
                }
                for row in blk.rows
            ],
            "label": blk.label,
            "is_table": blk.is_table,
            "is_notes": blk.is_notes,
        }

    blocks_serialized = [serialize_block(blk) for blk in blocks]
    blocks_path.write_text(json.dumps(blocks_serialized, indent=2))

    # Save notes columns with continuation info
    columns_path = run_dir / "artifacts" / f"{pdf_stem}_page_{page_num}_columns.json"

    def serialize_column(col):
        return {
            "header_text": col.header_text(),
            "base_header_text": col.base_header_text(),
            "is_continuation": col.is_continuation(),
            "column_group_id": col.column_group_id,
            "continues_from": col.continues_from,
            "notes_count": len(col.notes_blocks),
            "bbox": list(col.bbox()),
        }

    columns_serialized = [serialize_column(col) for col in notes_columns]
    columns_path.write_text(json.dumps(columns_serialized, indent=2))

    # Extract graphics and detect legends
    graphics = extract_graphics(str(pdf), page_num)

    # Detect abbreviation regions FIRST (pure text, no graphics)
    abbreviation_regions = detect_abbreviation_regions(
        blocks=blocks,
        graphics=graphics,
        page_width=page_w,
        page_height=page_h,
        debug_path=debug_path,
    )

    # Get exclusion zones from abbreviation regions
    exclusion_zones = [abbrev.bbox() for abbrev in abbreviation_regions]

    # Detect misc title regions (e.g., 'OKLAHOMA DEPARTMENT OF TRANSPORTATION')
    misc_title_regions = detect_misc_title_regions(
        blocks=blocks,
        graphics=graphics,
        page_width=page_w,
        page_height=page_h,
        debug_path=debug_path,
        exclusion_zones=exclusion_zones,
    )

    # Add misc title regions to exclusion zones
    for mt in misc_title_regions:
        exclusion_zones.append(mt.bbox())

    # Detect revision regions BEFORE legends (title block element)
    revision_regions = detect_revision_regions(
        blocks=blocks,
        graphics=graphics,
        page_width=page_w,
        page_height=page_h,
        debug_path=debug_path,
        exclusion_zones=exclusion_zones,
    )

    # Add revision regions to exclusion zones for legend detection
    for rev in revision_regions:
        exclusion_zones.append(rev.bbox())

    # Filter graphics to exclude those in abbreviation/revision regions
    filtered_graphics = filter_graphics_outside_regions(graphics, exclusion_zones)

    # Now detect legend regions with filtered graphics AND exclusion zones for text
    legend_regions = detect_legend_regions(
        blocks=blocks,
        graphics=filtered_graphics,
        page_width=page_w,
        page_height=page_h,
        debug_path=debug_path,
        exclusion_zones=exclusion_zones,
    )
    # Save abbreviation regions
    abbrev_path = (
        run_dir / "artifacts" / f"{pdf_stem}_page_{page_num}_abbreviations.json"
    )

    def serialize_abbreviation(abbrev):
        return {
            "header_text": abbrev.header_text(),
            "is_boxed": abbrev.is_boxed,
            "box_bbox": list(abbrev.box_bbox) if abbrev.box_bbox else None,
            "bbox": list(abbrev.bbox()),
            "entries_count": len(abbrev.entries),
            "entries": [
                {
                    "code": e.code,
                    "meaning": e.meaning,
                    "code_bbox": list(e.code_bbox) if e.code_bbox else None,
                    "meaning_bbox": list(e.meaning_bbox) if e.meaning_bbox else None,
                }
                for e in abbrev.entries
            ],
        }

    abbrev_serialized = [serialize_abbreviation(ab) for ab in abbreviation_regions]
    abbrev_path.write_text(json.dumps(abbrev_serialized, indent=2))

    # Save legend regions
    legends_path = run_dir / "artifacts" / f"{pdf_stem}_page_{page_num}_legends.json"

    def serialize_legend(legend):
        return {
            "header_text": legend.header_text(),
            "is_boxed": legend.is_boxed,
            "box_bbox": list(legend.box_bbox) if legend.box_bbox else None,
            "bbox": list(legend.bbox()),
            "entries_count": len(legend.entries),
            "entries": [
                {
                    "symbol_bbox": list(e.symbol_bbox) if e.symbol_bbox else None,
                    "description": e.description,
                    "description_bbox": (
                        list(e.description_bbox) if e.description_bbox else None
                    ),
                }
                for e in legend.entries
            ],
        }

    legends_serialized = [serialize_legend(leg) for leg in legend_regions]
    legends_path.write_text(json.dumps(legends_serialized, indent=2))

    # Save revision regions
    revisions_path = (
        run_dir / "artifacts" / f"{pdf_stem}_page_{page_num}_revisions.json"
    )

    def serialize_revision(rev):
        return {
            "header_text": rev.header_text(),
            "is_boxed": rev.is_boxed,
            "box_bbox": list(rev.box_bbox) if rev.box_bbox else None,
            "bbox": list(rev.bbox()),
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

    revisions_serialized = [serialize_revision(r) for r in revision_regions]
    revisions_path.write_text(json.dumps(revisions_serialized, indent=2))

    # Save misc title regions
    misc_titles_path = (
        run_dir / "artifacts" / f"{pdf_stem}_page_{page_num}_misc_titles.json"
    )

    def serialize_misc_title(mt):
        return {
            "text": mt.text,
            "is_boxed": mt.is_boxed,
            "box_bbox": list(mt.box_bbox) if mt.box_bbox else None,
            "bbox": list(mt.bbox()),
        }

    misc_titles_serialized = [serialize_misc_title(mt) for mt in misc_title_regions]
    misc_titles_path.write_text(json.dumps(misc_titles_serialized, indent=2))

    overlay_path = run_dir / "overlays" / f"{pdf_stem}_page_{page_num}_overlay.png"
    scale = resolution / 72.0
    draw_overlay(
        page_width=page_w,
        page_height=page_h,
        boxes=boxes,
        rows=rows,
        blocks=blocks,
        out_path=overlay_path,
        scale=scale,
        background=bg_img,
        notes_columns=notes_columns,
        legend_regions=legend_regions,
        abbreviation_regions=abbreviation_regions,
        revision_regions=revision_regions,
        misc_title_regions=misc_title_regions,
    )

    # Return page results for manifest (don't write manifest here)
    page_result = {
        "page": page_num,
        "page_width": page_w,
        "page_height": page_h,
        "skew_degrees": skew,
        "counts": {
            "boxes": len(boxes),
            "rows": len(rows),
            "blocks": len(blocks),
            "tables": sum(1 for b in blocks if b.is_table),
            "notes_columns": len(notes_columns),
            "graphics": len(graphics),
            "filtered_graphics": len(filtered_graphics),
            "abbreviation_regions": len(abbreviation_regions),
            "abbreviation_entries": sum(len(ab.entries) for ab in abbreviation_regions),
            "legend_regions": len(legend_regions),
            "legend_entries": sum(len(leg.entries) for leg in legend_regions),
            "revision_regions": len(revision_regions),
            "revision_entries": sum(len(r.entries) for r in revision_regions),
            "misc_title_regions": len(misc_title_regions),
        },
        "artifacts": {
            "boxes_json": str(boxes_path),
            "overlay_png": str(overlay_path),
            "legends_json": str(legends_path),
            "abbreviations_json": str(abbrev_path),
            "revisions_json": str(revisions_path),
        },
    }

    print(f"  page {page_num}: done", flush=True)
    print(summarize(blocks), flush=True)
    return page_result


def run_pdf(
    pdf: Path,
    start: int,
    end: int | None,
    resolution: int,
    run_root: Path,
    run_prefix: str,
) -> Path:
    """Process pages of a single PDF and create one run folder with all results."""
    with pdfplumber.open(pdf) as pdf_doc:
        total_pages = len(pdf_doc.pages)
    end_page = end if end is not None else total_pages

    # Create single run folder for this PDF
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{stamp}_{run_prefix}"
    run_dir = make_run_dir(run_root, run_name)

    print(f"Processing {pdf.name} -> {run_dir}", flush=True)

    page_results = []
    for page_num in range(start, end_page):
        try:
            result = process_page(pdf, page_num, run_dir, resolution)
            page_results.append(result)
        except Exception as exc:  # pragma: no cover
            print(f"  page {page_num}: ERROR {exc}", flush=True)
            page_results.append({"page": page_num, "error": str(exc)})

    # Write single manifest for entire run
    manifest = {
        "run_id": run_dir.name,
        "created_at": datetime.now().isoformat(),
        "source_pdf": str(pdf.resolve()),
        "pdf_name": pdf.name,
        "render_resolution_dpi": resolution,
        "overlay_scale": resolution / 72.0,
        "settings": vars(GroupingConfig()),
        "pages_processed": list(range(start, end_page)),
        "pages": page_results,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
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
    args = parser.parse_args()

    # Derive run prefix from PDF name if not provided
    run_prefix = args.run_prefix or args.pdf.stem.replace(" ", "_")[:20]

    run_pdf(
        pdf=args.pdf,
        start=args.start,
        end=args.end,
        resolution=args.resolution,
        run_root=args.run_root,
        run_prefix=run_prefix,
    )

    # Cleanup old runs
    cleanup_old_runs(args.run_root, args.keep_runs)


if __name__ == "__main__":
    main()
