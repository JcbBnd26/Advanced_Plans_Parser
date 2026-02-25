#!/usr/bin/env python
"""Train the cross-page Document GNN from labelled page results.

Usage
-----
::

    python scripts/train_gnn.py --run-dir runs/my_run --epochs 200

The script:
1. Loads page results (manifests) from one or more run directories.
2. Builds a cross-page document graph via ``build_document_graph``.
3. Uses the region-type labels already present in each node as
   ground-truth (self-supervised / label propagation scenario).
4. Trains a :class:`DocumentGNN` and saves it to ``data/document_gnn.pt``.

Requires ``torch`` and ``torch-geometric`` to be installed::

    pip install "plancheck[gnn]"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train cross-page Document GNN")
    p.add_argument(
        "--run-dir",
        type=Path,
        nargs="+",
        default=[],
        help="Run directories containing manifests to train on",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/document_gnn.pt"),
        help="Output model path (default: data/document_gnn.pt)",
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument(
        "--train-split", type=float, default=0.8, help="Fraction used for training"
    )
    p.add_argument("--embeddings", action="store_true", help="Include text embeddings")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    # Lazy-import heavy deps
    try:
        import torch
    except ImportError:
        log.error("PyTorch is required. Install with: pip install 'plancheck[gnn]'")
        sys.exit(1)

    try:
        from torch_geometric.data import Data  # noqa: F401
    except ImportError:
        log.error(
            "PyTorch Geometric is required. Install with: pip install 'plancheck[gnn]'"
        )
        sys.exit(1)

    from plancheck.analysis.document_graph import GraphNodeType, build_document_graph
    from plancheck.analysis.gnn_model import DocumentGNN, save_gnn, train_gnn

    # ── Load page results ─────────────────────────────────────────
    # For the pilot, we create a synthetic graph from run manifests
    # since we don't have serialised PageResult objects.
    # In production this would load actual PageResult pickles.
    if not args.run_dir:
        log.warning("No --run-dir specified; creating a small demo graph for testing")
        import numpy as np

        # Create a small synthetic graph
        num_nodes = 20
        num_classes = GraphNodeType.num_types()
        feat_dim = 14

        x = np.random.randn(num_nodes, feat_dim).astype(np.float32)
        # Set one-hot region type properly
        for i in range(num_nodes):
            x[i, :num_classes] = 0.0
            x[i, i % num_classes] = 1.0

        labels = torch.tensor(
            [i % num_classes for i in range(num_nodes)], dtype=torch.long
        )

        # Create edges (fully connected within groups of 5)
        src, dst = [], []
        for g in range(0, num_nodes, 5):
            for i in range(g, min(g + 5, num_nodes)):
                for j in range(i + 1, min(g + 5, num_nodes)):
                    src.extend([i, j])
                    dst.extend([j, i])

        edge_index = torch.tensor([src, dst], dtype=torch.long)

        data = Data(
            x=torch.from_numpy(x),
            edge_index=edge_index,
        )
    else:
        # Load real data from run directories
        log.info("Loading page results from %d run directories", len(args.run_dir))
        # Placeholder — real implementation would deserialize PageResult
        log.error(
            "Real page-result loading not yet implemented. Use without --run-dir for demo."
        )
        sys.exit(1)

    # ── Create train/val masks ────────────────────────────────────
    num_nodes = data.num_nodes
    perm = torch.randperm(num_nodes)
    n_train = int(num_nodes * args.train_split)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:]] = True

    # ── Build and train model ─────────────────────────────────────
    in_channels = data.x.shape[1]
    num_classes = GraphNodeType.num_types()

    model = DocumentGNN(
        in_channels=in_channels,
        hidden_channels=args.hidden_dim,
        num_classes=num_classes,
        heads=args.heads,
    )

    log.info(
        "Model: in=%d, hidden=%d, classes=%d, heads=%d, params=%d",
        in_channels,
        args.hidden_dim,
        num_classes,
        args.heads,
        sum(p.numel() for p in model.parameters()),
    )

    history = train_gnn(
        model,
        data,
        labels,
        epochs=args.epochs,
        lr=args.lr,
        train_mask=train_mask,
        val_mask=val_mask,
        verbose=args.verbose,
    )

    final_train_acc = history["train_acc"][-1] if history["train_acc"] else 0.0
    final_val_acc = history["val_acc"][-1] if history["val_acc"] else 0.0
    log.info("Final — train_acc: %.3f  val_acc: %.3f", final_train_acc, final_val_acc)

    # ── Save ──────────────────────────────────────────────────────
    save_gnn(model, args.output)
    log.info("Done. Model saved to %s", args.output)


if __name__ == "__main__":
    main()
