#!/usr/bin/env python
"""Train an element-type classifier from accumulated corrections.

Usage
-----
    python scripts/train_model.py [--db PATH] [--format coco|voc]

Steps:
  1. Build training set from corrections database
  2. Export features to JSONL
  3. Train GradientBoostingClassifier
  4. Print metrics (accuracy, per-class, confusion matrix)
  5. Optionally export annotations in COCO or VOC format
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from plancheck.corrections.classifier import ElementClassifier
from plancheck.corrections.metrics import format_metrics_table
from plancheck.corrections.store import CorrectionStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Train element classifier")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/corrections.db"),
        help="Path to corrections database (default: data/corrections.db)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("data/element_classifier.pkl"),
        help="Output model path (default: data/element_classifier.pkl)",
    )
    parser.add_argument(
        "--format",
        choices=["coco", "voc"],
        default=None,
        help="Also export annotations in COCO or VOC format",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("data/exports"),
        help="Directory for COCO/VOC exports (default: data/exports)",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        default=False,
        help="Skip confidence calibration (faster, but raw probabilities)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Accept model even if F1 regresses vs. previous run",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        default=False,
        help="Use soft-voting ensemble (GBM + HistGBM + optional LightGBM/XGBoost)",
    )
    args = parser.parse_args()

    # 1. Open store
    if not args.db.exists():
        print(f"ERROR: Database not found: {args.db}", file=sys.stderr)
        return 1

    store = CorrectionStore(args.db)

    # 2. Snapshot before training
    try:
        snap = store.snapshot(tag="pre_training")
        print(f"Snapshot saved: {snap}")
    except Exception as exc:
        print(f"Warning: snapshot failed: {exc}")

    # 3. Build training set
    n_examples = store.build_training_set()
    print(f"Training examples built: {n_examples}")

    if n_examples < 10:
        print(
            f"ERROR: Need at least 10 training examples, got {n_examples}.",
            file=sys.stderr,
        )
        store.close()
        return 1

    # 4. Export JSONL
    jsonl_path = args.db.parent / "training_data.jsonl"
    n_lines = store.export_training_jsonl(jsonl_path)
    print(f"Exported {n_lines} examples to {jsonl_path}")

    # 5. Train
    clf = ElementClassifier(model_path=args.model)
    metrics = clf.train(
        jsonl_path,
        calibrate=not args.no_calibration,
        ensemble=args.ensemble,
    )

    print()
    print(format_metrics_table(metrics))
    print()
    print(f"Model saved: {args.model}")
    print(f"  Training examples: {metrics['n_train']}")
    print(f"  Validation examples: {metrics['n_val']}")
    if metrics.get("calibrated"):
        print("  Calibration: isotonic (confidence scores are calibrated)")
    else:
        print("  Calibration: none (raw probabilities)")
    if metrics.get("ensemble"):
        members = metrics.get("ensemble_members", [])
        print(f"  Ensemble: soft-voting ({', '.join(members)})")
    else:
        print("  Ensemble: disabled (single GBM)")

    # 6. Record training run in the database
    holdout_preds = metrics.get("holdout_predictions")
    try:
        run_id = store.save_training_run(
            metrics,
            model_path=str(args.model),
            notes="CLI train",
            holdout_predictions=holdout_preds,
        )
        print(f"  Training run recorded: {run_id}")
    except Exception as exc:
        print(f"Warning: could not record training run: {exc}")
        run_id = None

    # 7. Auto-rollback check — compare F1 against previous run
    try:
        history = store.get_training_history()
        # history is newest-first; current run is history[0] if just saved
        prior_runs = [r for r in history if r.get("run_id") != run_id]
        if prior_runs:
            prev = prior_runs[0]
            prev_f1 = prev["f1_weighted"]
            curr_f1 = metrics.get("f1_weighted", 0.0)
            delta = curr_f1 - prev_f1
            print(
                f"\n  F1 weighted: {curr_f1:.4f} (prev {prev_f1:.4f}, "
                f"delta {delta:+.4f})"
            )
            if delta < 0 and not args.force:
                print(
                    f"\n  WARNING: F1 regressed by {abs(delta):.4f}. "
                    "Rolling back to previous snapshot."
                )
                print("  Use --force to accept the model anyway.")
                try:
                    store.restore_snapshot(tag="pre_training")
                    print("  Restored pre-training snapshot.")
                except Exception as restore_exc:
                    print(f"  Warning: rollback failed: {restore_exc}")
                store.close()
                return 1
            elif delta < 0 and args.force:
                print(
                    f"\n  WARNING: F1 regressed by {abs(delta):.4f}, "
                    "but --force was set. Accepting model."
                )
    except Exception as exc:
        print(f"Warning: could not compare against previous run: {exc}")

    # 8. Optional export
    if args.format:
        from plancheck.corrections.export_formats import export_coco, export_voc

        if args.format == "coco":
            out = export_coco(store, args.export_dir)
            print(f"\nCOCO annotations exported: {out}")
        elif args.format == "voc":
            out = export_voc(store, args.export_dir)
            print(f"\nVOC annotations exported to: {out}")

    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
