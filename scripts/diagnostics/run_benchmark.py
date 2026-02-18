"""A/B/C benchmark runner for the OCR pipeline.

Runs three configurations on the same pages and produces a comparison
report so you can decide whether VOCRPP is worth always-on.

Usage::

    python scripts/run_benchmark.py \\
        --pdf input/IFC*.pdf \\
        --pages 2 \\
        --conditions all \\
        [--ground-truth samples/benchmark_ground_truth.json]

Conditions
----------
A  TOCR only  (no VOCR, no preprocessing)
B  TOCR + VOCR  (no preprocessing)
C  TOCR + VOCRPP + VOCR  (full pipeline)

Output
------
* One run directory per condition (named ``run_…_benchA``, ``_benchB``, ``_benchC``)
* A ``runs/benchmark_<timestamp>.json`` comparison report
* Console summary table
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_project = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project / "src"))
sys.path.insert(0, str(_project / "scripts" / "runners"))

from run_pdf_batch import cleanup_old_runs, run_pdf  # noqa: E402

from plancheck.config import GroupingConfig

# ── Condition definitions ──────────────────────────────────────────────

_CONDITIONS = {
    "A": {
        "label": "TOCR only",
        "cfg": GroupingConfig(
            enable_vocr=False,
            enable_ocr_reconcile=False,
            enable_ocr_preprocess=False,
        ),
    },
    "B": {
        "label": "TOCR + VOCR",
        "cfg": GroupingConfig(
            enable_vocr=True,
            enable_ocr_reconcile=False,
            enable_ocr_preprocess=False,
            ocr_reconcile_debug=True,
        ),
    },
    "C": {
        "label": "TOCR + VOCR + Reconcile",
        "cfg": GroupingConfig(
            enable_vocr=True,
            enable_ocr_reconcile=True,
            enable_ocr_preprocess=False,
            ocr_reconcile_debug=True,
        ),
    },
    "D": {
        "label": "TOCR + VOCRPP + VOCR + Reconcile",
        "cfg": GroupingConfig(
            enable_vocr=True,
            enable_ocr_reconcile=True,
            enable_ocr_preprocess=True,
            ocr_reconcile_debug=True,
        ),
    },
}


# ── Report builder ─────────────────────────────────────────────────────


def _read_manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text())


def _extract_stage_summary(manifest: dict) -> dict:
    """Pull per-stage metrics out of the first page in a manifest."""
    pages = manifest.get("pages", [])
    if not pages:
        return {}
    page = pages[0]
    stages = page.get("stages", {})
    summary: dict = {}
    for name in ("ingest", "tocr", "vocrpp", "vocr", "reconcile"):
        st = stages.get(name, {})
        summary[name] = {
            "enabled": st.get("enabled", False),
            "ran": st.get("ran", False),
            "status": st.get("status", "n/a"),
            "duration_ms": st.get("duration_ms", 0),
            "counts": st.get("counts", {}),
        }
    return summary


def _build_comparison(results: dict[str, dict]) -> dict:
    """Build a structured comparison from {condition: manifest}."""
    rows: list[dict] = []
    for cond, manifest in results.items():
        ss = _extract_stage_summary(manifest)
        page = manifest.get("pages", [{}])[0]
        counts = page.get("counts", {})
        row: dict = {
            "condition": cond,
            "label": _CONDITIONS[cond]["label"],
            "total_tokens": counts.get("boxes", 0),
            "ocr_reconcile_accepted": counts.get("ocr_reconcile_accepted", 0),
            "ocr_reconcile_total": counts.get("ocr_reconcile_total", 0),
            "ocr_reconcile_candidates": counts.get("ocr_reconcile_candidates", 0),
            "ocr_rejected": counts.get("ocr_reconcile_candidates_rejected", 0),
            "ocr_filtered_non_numeric": counts.get(
                "ocr_reconcile_filtered_non_numeric", 0
            ),
        }
        # Stage durations
        for stage_name in ("ingest", "tocr", "vocrpp", "vocr", "reconcile"):
            row[f"{stage_name}_ms"] = ss.get(stage_name, {}).get("duration_ms", 0)
            row[f"{stage_name}_ran"] = ss.get(stage_name, {}).get("ran", False)
        rows.append(row)
    return {"conditions": rows}


def _print_table(comparison: dict) -> None:
    """Print a human-readable comparison table to stdout."""
    rows = comparison["conditions"]
    if not rows:
        print("No results to compare.")
        return

    cols = [
        ("Cond", "condition", 5),
        ("Label", "label", 24),
        ("Tokens", "total_tokens", 7),
        ("Accepted", "ocr_reconcile_accepted", 8),
        ("OCR Tot", "ocr_reconcile_total", 7),
        ("Rejected", "ocr_rejected", 8),
        ("Filtered", "ocr_filtered_non_numeric", 8),
        ("TOCR ms", "tocr_ms", 8),
        ("VOCRPP ms", "vocrpp_ms", 9),
        ("VOCR ms", "vocr_ms", 8),
        ("Recon ms", "reconcile_ms", 8),
    ]
    header = " | ".join(name.ljust(w) for name, _, w in cols)
    sep = "-+-".join("-" * w for _, _, w in cols)
    print()
    print(header)
    print(sep)
    for row in rows:
        line = " | ".join(str(row.get(key, "")).ljust(w) for _, key, w in cols)
        print(line)
    print()


# ── Ground-truth scoring (optional) ───────────────────────────────────


def _score_ground_truth(
    manifest: dict, ground_truth: dict
) -> dict[str, dict[str, float]]:
    """Score reconciliation against expected symbols.

    Ground truth format::

        { "page_2": { "expected_symbols": [
            { "text": "%", "near_text": "2" }, ...
        ]}}

    Returns per-symbol precision / recall / f1 (best-effort).
    """
    pages = manifest.get("pages", [])
    if not pages:
        return {}

    page = pages[0]
    page_key = f"page_{page.get('page', 0)}"
    gt_page = ground_truth.get(page_key, {})
    expected = gt_page.get("expected_symbols", [])
    if not expected:
        return {}

    # Accepted tokens from injection log
    inj_log = page.get("ocr_injection_log", [])
    accepted_symbols: list[str] = []
    for entry in inj_log:
        for cand in entry.get("candidates", []):
            if cand.get("status") == "accepted":
                accepted_symbols.append(cand.get("symbol", ""))

    # Per-symbol scoring
    symbol_types = set(e["text"] for e in expected) | set(accepted_symbols)
    scores: dict[str, dict[str, float]] = {}
    for sym in symbol_types:
        tp = min(
            sum(1 for s in accepted_symbols if s == sym),
            sum(1 for e in expected if e["text"] == sym),
        )
        fp = max(0, sum(1 for s in accepted_symbols if s == sym) - tp)
        fn = max(0, sum(1 for e in expected if e["text"] == sym) - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        scores[sym] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }
    return scores


# ── Main ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run A/B/C benchmark for OCR pipeline evaluation"
    )
    parser.add_argument("--pdf", type=Path, required=True, help="PDF to process")
    parser.add_argument(
        "--pages",
        type=int,
        nargs="+",
        default=[2],
        help="Page numbers (1-indexed) to benchmark",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["A", "B", "C"],
        help="Conditions to run (A, B, C, or 'all')",
    )
    parser.add_argument(
        "--resolution", type=int, default=200, help="Overlay render DPI"
    )
    parser.add_argument(
        "--ocr-resolution",
        type=int,
        default=180,
        help="OCR render DPI (for conditions B and C)",
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("runs"), help="Root for run outputs"
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="JSON file with expected symbols per page",
    )
    args = parser.parse_args()

    conditions = args.conditions
    if "all" in conditions:
        conditions = ["A", "B", "C"]
    conditions = [c.upper() for c in conditions]

    # Convert 1-indexed pages to 0-indexed
    pages_0 = [p - 1 for p in args.pages]
    start = min(pages_0)
    end = max(pages_0) + 1

    # Load ground truth if provided
    ground_truth = None
    if args.ground_truth and args.ground_truth.exists():
        ground_truth = json.loads(args.ground_truth.read_text())

    run_dirs: dict[str, Path] = {}
    manifests: dict[str, dict] = {}

    print(f"Benchmark: {args.pdf.name}, pages {args.pages}, conditions {conditions}")
    print("=" * 70)

    for cond in conditions:
        if cond not in _CONDITIONS:
            print(f"  Unknown condition: {cond}, skipping")
            continue

        cond_def = _CONDITIONS[cond]
        cfg = GroupingConfig(**vars(cond_def["cfg"]))
        # Apply OCR resolution override
        cfg.ocr_reconcile_resolution = args.ocr_resolution

        prefix = f"{args.pdf.stem[:15]}_bench{cond}".replace(" ", "_")
        print(f"\n--- Condition {cond}: {cond_def['label']} ---")

        rd = run_pdf(
            pdf=args.pdf,
            start=start,
            end=end,
            resolution=args.resolution,
            run_root=args.run_root,
            run_prefix=prefix,
            cfg=cfg,
        )
        run_dirs[cond] = rd
        manifests[cond] = _read_manifest(rd)

    # Build comparison
    comparison = _build_comparison(manifests)

    # Score against ground truth if available
    if ground_truth:
        comparison["ground_truth_scores"] = {}
        for cond, manifest in manifests.items():
            scores = _score_ground_truth(manifest, ground_truth)
            if scores:
                comparison["ground_truth_scores"][cond] = scores

    # Save report
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.run_root / f"benchmark_{stamp}.json"
    report_path.write_text(json.dumps(comparison, indent=2))

    # Console output
    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON")
    print("=" * 70)
    _print_table(comparison)

    if "ground_truth_scores" in comparison:
        print("Ground Truth Scores:")
        for cond, scores in comparison["ground_truth_scores"].items():
            print(f"  Condition {cond}:")
            for sym, s in scores.items():
                print(
                    f"    '{sym}': precision={s['precision']:.3f} "
                    f"recall={s['recall']:.3f} f1={s['f1']:.3f} "
                    f"(tp={s['tp']} fp={s['fp']} fn={s['fn']})"
                )
        print()

    print(f"Report saved: {report_path}")
    for cond, rd in run_dirs.items():
        print(f"  {cond}: {rd}")

    cleanup_old_runs(args.run_root, keep=50)


if __name__ == "__main__":
    main()
