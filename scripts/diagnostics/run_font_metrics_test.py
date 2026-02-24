#!/usr/bin/env python
"""Quick font-metrics test: run heuristic + visual analysis on a PDF page.

Usage::

    python scripts/diagnostics/run_font_metrics_test.py <pdf> --page 2

Replaces the old ``if __name__ == '__main__'`` block that was embedded in
``src/plancheck/grouping/font_metrics.py`` with a proper CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from plancheck.grouping.font_metrics import (
    FontMetricsAnalyzer,  # noqa: E402
    VisualMetricsAnalyzer,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run heuristic + visual font-metrics analysis on a PDF page"
    )
    parser.add_argument("pdf", type=Path, help="Path to PDF file")
    parser.add_argument("--page", type=int, default=2, help="Zero-based page index")
    parser.add_argument(
        "--resolution", type=int, default=200, help="DPI for visual analysis"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("font_metrics_report.json"),
        help="Output JSON report path",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"PDF not found: {args.pdf}")
        sys.exit(1)

    print(f"Analyzing: {args.pdf.name}")
    print("=" * 70)

    # ── Heuristic analysis (fast) ─────────────────────────────────────
    print("\n1. HEURISTIC ANALYSIS (based on expected char width ratios)")
    print("-" * 70)

    analyzer = FontMetricsAnalyzer()
    report = analyzer.analyze_page(args.pdf, args.page)

    print(f"  Total chars analyzed: {report.total_chars_analyzed}")
    print(f"  Has anomalies: {report.has_anomalies()}")

    for fontname, anomaly in report.font_anomalies.items():
        status = "ANOMALOUS" if anomaly.is_anomalous() else "Normal"
        print(f"\n  {fontname}: {status}")
        print(
            f"    Samples: {anomaly.sample_count}, "
            f"Inflation: {anomaly.inflation_factor:.2f}x"
        )

    # ── Visual analysis (accurate but slower) ─────────────────────────
    print("\n2. VISUAL ANALYSIS (pixel-based comparison)")
    print("-" * 70)

    visual = VisualMetricsAnalyzer(resolution=args.resolution)
    visual_report = visual.analyze_page(args.pdf, args.page)

    anomalous_words = [a for a in visual_report.word_anomalies if a.is_anomalous()]
    normal_words = [a for a in visual_report.word_anomalies if not a.is_anomalous()]

    print(f"  Words analyzed: {len(visual_report.word_anomalies)}")
    print(f"  Anomalous words: {len(anomalous_words)}")
    print(f"  Normal words: {len(normal_words)}")

    if anomalous_words:
        print("\n  ANOMALOUS WORDS DETECTED:")
        for anomaly in anomalous_words:
            print(
                f"\n    '{anomaly.text}' at "
                f"({anomaly.reported_bbox[0]:.0f}, {anomaly.reported_bbox[1]:.0f})"
            )
            print(f"      Font: {anomaly.fontname}")
            print(f"      Reported width: {anomaly.reported_width:.1f}")
            print(f"      Visual width:   {anomaly.visual_width:.1f}")
            print(
                f"      Inflation:      {anomaly.inflation_factor:.2f}x "
                f"({anomaly.overhang_percent:.0f}% empty)"
            )
            print(
                f"      -> Correction: shrink x1 by "
                f"{anomaly.reported_width - anomaly.visual_width:.1f}"
            )

    # ── Save report ───────────────────────────────────────────────────
    args.out.write_text(json.dumps(visual_report.to_dict(), indent=2))
    print(f"\n  Report saved to: {args.out}")


if __name__ == "__main__":
    main()
