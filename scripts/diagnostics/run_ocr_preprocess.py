from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pdfplumber

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from plancheck.vocrpp.preprocess import (
    OcrPreprocessConfig,  # noqa: E402
    preprocess_image_for_ocr,
)


def render_page_image(pdf_path: Path, page_num: int, resolution: int = 200):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        img_page = page.to_image(resolution=resolution)
        return img_page.original.copy(), float(page.width), float(page.height)


def _make_run_dir(run_root: Path, run_prefix: str) -> Path:
    """Create a timestamped run folder matching the main pipeline convention."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{stamp}_ocr_preproc_{run_prefix}"
    run_dir = run_root / run_name
    for sub in ["artifacts", "overlays", "exports", "logs"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def process_pdf_pages(
    pdf: Path,
    out_dir: Path,
    start: int,
    end: int | None,
    render_dpi: int,
    cfg: OcrPreprocessConfig,
) -> Path:
    """Run OCR preprocessing and save all outputs under *out_dir*.

    When called from the standalone CLI, *out_dir* is a timestamped run
    folder under ``runs/``.  When called from ``run_from_args.py``, the
    caller passes ``run_dir / "ocr_preprocess"`` so everything stays
    inside the main pipeline's run folder.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "artifacts"
    images_dir.mkdir(parents=True, exist_ok=True)
    steps_dir = out_dir / "artifacts" / "steps"
    if cfg.save_intermediate:
        steps_dir.mkdir(parents=True, exist_ok=True)

    with pdfplumber.open(pdf) as pdf_doc:
        total_pages = len(pdf_doc.pages)

    end_page = end if end is not None else total_pages

    results = []
    for page_num in range(start, min(end_page, total_pages)):
        print(f"  ocr-preprocess page {page_num} ...", flush=True)
        page_image, page_w, page_h = render_page_image(
            pdf_path=pdf,
            page_num=page_num,
            resolution=render_dpi,
        )

        # Raw render bytes for A/B reporting
        import io

        raw_buf = io.BytesIO()
        page_image.save(raw_buf, format="PNG")
        raw_bytes = raw_buf.tell()

        page_steps_dir = (
            str(steps_dir / f"page_{page_num}") if cfg.save_intermediate else None
        )
        result = preprocess_image_for_ocr(
            image=page_image,
            cfg=cfg,
            intermediate_dir=page_steps_dir,
        )

        image_out = images_dir / f"page_{page_num:04d}_ocr_input.png"
        result.image.save(image_out)
        preprocessed_bytes = image_out.stat().st_size

        results.append(
            {
                "page": page_num,
                "page_width": page_w,
                "page_height": page_h,
                "render_dpi": render_dpi,
                "output_image": str(image_out),
                "applied_steps": result.applied_steps,
                "metrics": result.metrics,
                "raw_render_bytes": raw_bytes,
                "preprocessed_bytes": preprocessed_bytes,
            }
        )
        print(
            f"    -> {image_out.name}  "
            f"raw={raw_bytes:,}B  preprocessed={preprocessed_bytes:,}B  "
            f"steps={result.applied_steps}",
            flush=True,
        )

    manifest = {
        "source_pdf": str(pdf.resolve()),
        "created_at": datetime.now().isoformat(),
        "settings": {
            "enabled": cfg.enabled,
            "grayscale": cfg.grayscale,
            "autocontrast": cfg.autocontrast,
            "clahe": cfg.clahe,
            "clahe_clip_limit": cfg.clahe_clip_limit,
            "clahe_tile_size": cfg.clahe_tile_size,
            "median_denoise": cfg.median_denoise,
            "median_kernel_size": cfg.median_kernel_size,
            "adaptive_binarize": cfg.adaptive_binarize,
            "adaptive_block_size": cfg.adaptive_block_size,
            "adaptive_c": cfg.adaptive_c,
            "sharpen": cfg.sharpen,
            "save_intermediate": cfg.save_intermediate,
        },
        "pages": results,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  ocr-preprocess done: {manifest_path}", flush=True)
    return manifest_path


def process_pdf_to_pdf(
    pdf: Path,
    out_pdf: Path,
    start: int,
    end: int | None,
    render_dpi: int,
    cfg: OcrPreprocessConfig,
) -> Path:
    """Run OCR preprocessing and write only a single processed PDF file."""
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    source_pdf_bytes = pdf.stat().st_size

    with pdfplumber.open(pdf) as pdf_doc:
        total_pages = len(pdf_doc.pages)

    end_page = end if end is not None else total_pages
    page_images = []
    steps_seen: list[str] = []
    seen = set()
    total_raw_png_bytes = 0
    total_preprocessed_png_bytes = 0

    for page_num in range(start, min(end_page, total_pages)):
        print(f"  ocr-preprocess page {page_num} ...", flush=True)
        page_image, _, _ = render_page_image(
            pdf_path=pdf,
            page_num=page_num,
            resolution=render_dpi,
        )

        import io

        raw_buf = io.BytesIO()
        page_image.save(raw_buf, format="PNG")
        total_raw_png_bytes += raw_buf.tell()

        result = preprocess_image_for_ocr(image=page_image, cfg=cfg)

        pre_buf = io.BytesIO()
        result.image.save(pre_buf, format="PNG")
        total_preprocessed_png_bytes += pre_buf.tell()

        for step in result.applied_steps:
            if step not in seen:
                seen.add(step)
                steps_seen.append(step)
        page_images.append(result.image.convert("RGB"))

    if not page_images:
        raise ValueError("No pages selected for preprocessing")

    page_span = f"{start + 1}-{min(end_page, total_pages)}"
    changes_summary = ", ".join(steps_seen) if steps_seen else "none"

    first_page, *other_pages = page_images
    output_pdf_bytes: int | None = None
    for _ in range(3):
        output_text = (
            str(output_pdf_bytes) if output_pdf_bytes is not None else "pending"
        )
        first_page.save(
            out_pdf,
            save_all=True,
            append_images=other_pages,
            title=f"OCR Preprocessed (Optimized for Parsing) - {pdf.stem}",
            author="Advanced Plan Parser",
            subject=(
                "OCR preprocessing changes: "
                f"{changes_summary}; "
                f"pdf_size_bytes:{source_pdf_bytes}->{output_text}; "
                "comments: Optimized for Parsing"
            ),
            keywords=(
                "ocr, preprocessing, "
                f"steps:{changes_summary}, "
                f"dpi:{render_dpi}, pages:{page_span}, "
                f"pdf_size_bytes:{source_pdf_bytes}->{output_text}, "
                "comments:Optimized for Parsing, "
                f"raw_png_total_bytes:{total_raw_png_bytes}, "
                f"preprocessed_png_total_bytes:{total_preprocessed_png_bytes}"
            ),
            creator="Advanced Plan Parser",
        )
        new_size = out_pdf.stat().st_size
        if new_size == output_pdf_bytes:
            break
        output_pdf_bytes = new_size

    print(f"  ocr-preprocess done: {out_pdf}", flush=True)
    return out_pdf


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone OCR preprocessing pipeline. Produces OCR-input images "
            "that can be fed into the main OCR stage later.  All outputs go "
            "under runs/ using the same naming convention as the main pipeline."
        )
    )
    parser.add_argument("pdf", type=Path, help="Path to PDF")
    parser.add_argument("--start", type=int, default=0, help="Start page (inclusive)")
    parser.add_argument(
        "--end", type=int, default=None, help="End page (exclusive); default all"
    )
    parser.add_argument("--render-dpi", type=int, default=220)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs"),
        help="Root directory for run folders (default: runs/)",
    )

    parser.add_argument("--disable", action="store_true")
    parser.add_argument("--no-grayscale", action="store_true")
    parser.add_argument("--autocontrast", action="store_true")
    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument("--clahe-clip", type=float, default=2.0)
    parser.add_argument("--clahe-tile", type=int, default=8)
    parser.add_argument("--median-denoise", action="store_true")
    parser.add_argument("--median-kernel", type=int, default=3)
    parser.add_argument("--adaptive-binarize", action="store_true")
    parser.add_argument("--adaptive-block", type=int, default=11)
    parser.add_argument("--adaptive-c", type=float, default=2.0)
    parser.add_argument("--sharpen", action="store_true")
    parser.add_argument("--save-intermediate", action="store_true")
    parser.add_argument(
        "--pdf-only",
        action="store_true",
        help="Write only one processed PDF file (no run folders/artifacts)",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=None,
        help="Output PDF path for --pdf-only mode",
    )

    args = parser.parse_args()

    cfg = OcrPreprocessConfig(
        enabled=not args.disable,
        grayscale=not args.no_grayscale,
        autocontrast=args.autocontrast,
        clahe=not args.no_clahe,
        clahe_clip_limit=args.clahe_clip,
        clahe_tile_size=args.clahe_tile,
        median_denoise=args.median_denoise,
        median_kernel_size=args.median_kernel,
        adaptive_binarize=args.adaptive_binarize,
        adaptive_block_size=args.adaptive_block,
        adaptive_c=args.adaptive_c,
        sharpen=args.sharpen,
        save_intermediate=args.save_intermediate,
    )

    if args.pdf_only:
        out_pdf = args.output_pdf
        if out_pdf is None:
            # Create a proper run subfolder for the output PDF
            run_prefix = args.pdf.stem.replace(" ", "_")[:20]
            pdf_run_dir = _make_run_dir(args.run_root, run_prefix)
            out_pdf = pdf_run_dir / "exports" / f"{args.pdf.stem}_ocr_preprocessed.pdf"
        print(f"OCR preprocess (PDF only) -> {out_pdf}", flush=True)
        process_pdf_to_pdf(
            pdf=args.pdf,
            out_pdf=out_pdf,
            start=args.start,
            end=args.end,
            render_dpi=args.render_dpi,
            cfg=cfg,
        )
    else:
        # Derive run prefix from PDF name
        run_prefix = args.pdf.stem.replace(" ", "_")[:20]
        run_dir = _make_run_dir(args.run_root, run_prefix)
        print(f"OCR preprocess -> {run_dir}", flush=True)

        process_pdf_pages(
            pdf=args.pdf,
            out_dir=run_dir,
            start=args.start,
            end=args.end,
            render_dpi=args.render_dpi,
            cfg=cfg,
        )


if __name__ == "__main__":
    main()
