from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pdfplumber

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from plancheck.font_metrics import (
    FontMetricsAnalyzer,  # noqa: E402
    VisualMetricsAnalyzer,
)


def run_diagnostics(
    pdf: Path,
    out_dir: Path,
    start: int,
    end: int | None,
    run_heuristic: bool,
    run_visual: bool,
    visual_resolution: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    with pdfplumber.open(str(pdf)) as pdf_doc:
        total_pages = len(pdf_doc.pages)

    end_page = end if end is not None else total_pages
    pages = list(range(start, min(end_page, total_pages)))

    if not pages:
        raise ValueError("No pages selected for diagnostics")

    payload: dict[str, object] = {
        "created_at": datetime.now().isoformat(),
        "source_pdf": str(pdf.resolve()),
        "pdf_name": pdf.name,
        "pages": pages,
        "tools_enabled": {
            "FontMetricsAnalyzer": run_heuristic,
            "VisualMetricsAnalyzer": run_visual,
        },
        "heuristic_reports": [],
        "visual_reports": [],
    }

    heuristic = FontMetricsAnalyzer() if run_heuristic else None
    visual = VisualMetricsAnalyzer(resolution=visual_resolution) if run_visual else None

    for page_num in pages:
        if heuristic is not None:
            report = heuristic.analyze_page(pdf, page_num)
            payload["heuristic_reports"].append(report.to_dict())

        if visual is not None:
            report = visual.analyze_page(pdf, page_num)
            payload["visual_reports"].append(report.to_dict())

    out_path = out_dir / f"{pdf.stem}_font_metrics_diagnostics.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run standalone font diagnostics and save JSON report"
    )
    parser.add_argument("pdf", type=Path, help="Path to PDF")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs"),
        help="Output directory for diagnostics JSON",
    )
    parser.add_argument("--start", type=int, default=0, help="Start page (inclusive)")
    parser.add_argument(
        "--end", type=int, default=None, help="End page (exclusive); default all"
    )
    parser.add_argument(
        "--heuristic",
        action="store_true",
        help="Run FontMetricsAnalyzer heuristic analysis",
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Run VisualMetricsAnalyzer pixel analysis",
    )
    parser.add_argument(
        "--visual-resolution",
        type=int,
        default=200,
        help="DPI for VisualMetricsAnalyzer",
    )

    args = parser.parse_args()

    if not args.pdf.exists():
        raise FileNotFoundError(f"PDF not found: {args.pdf}")

    if not (args.heuristic or args.visual):
        raise ValueError("Enable at least one tool: --heuristic and/or --visual")

    out_path = run_diagnostics(
        pdf=args.pdf,
        out_dir=args.out_dir,
        start=max(0, args.start),
        end=args.end,
        run_heuristic=args.heuristic,
        run_visual=args.visual,
        visual_resolution=args.visual_resolution,
    )
    print(f"Diagnostics report written: {out_path}", flush=True)


if __name__ == "__main__":
    main()
