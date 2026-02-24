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
    metrics = clf.train(jsonl_path)

    print()
    print(format_metrics_table(metrics))
    print()
    print(f"Model saved: {args.model}")
    print(f"  Training examples: {metrics['n_train']}")
    print(f"  Validation examples: {metrics['n_val']}")

    # 6. Record training run in the database
    try:
        run_id = store.save_training_run(
            metrics, model_path=str(args.model), notes="CLI train"
        )
        print(f"  Training run recorded: {run_id}")
    except Exception as exc:
        print(f"Warning: could not record training run: {exc}")

    # 6. Optional export
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
