#!/usr/bin/env python
"""Train the VOCR candidate hit/miss classifier (Level 2).

Usage::

    python scripts/train_candidate_classifier.py [--db data/corrections.db]
                                                  [--model data/candidate_classifier.pkl]
                                                  [--min-rows 100]

Reads outcome data from the ``candidate_outcomes`` table in the
corrections database and trains a binary HistGradientBoostingClassifier
to predict P(hit) for each candidate.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from plancheck.corrections.retrain_trigger import auto_retrain_candidate_classifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train candidate hit/miss classifier")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/corrections.db"),
        help="Path to corrections database (default: data/corrections.db)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("data/candidate_classifier.pkl"),
        help="Output model path (default: data/candidate_classifier.pkl)",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=100,
        help="Min outcome rows to train (default: 100)",
    )
    args = parser.parse_args()

    result = auto_retrain_candidate_classifier(
        db_path=args.db,
        model_path=args.model,
        min_rows=args.min_rows,
    )

    print(json.dumps(result, indent=2, default=str))

    if result.get("skipped"):
        log.info("Training skipped: %s", result.get("reason", "unknown"))
        sys.exit(0)
    elif result.get("error"):
        log.error("Training failed: %s", result["error"])
        sys.exit(1)
    else:
        log.info(
            "Training complete — accuracy=%.3f  f1=%.3f  auc=%.3f",
            result.get("accuracy", 0),
            result.get("f1", 0),
            result.get("auc_roc", 0),
        )


if __name__ == "__main__":
    main()
