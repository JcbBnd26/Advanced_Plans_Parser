"""GNN-based document reasoning sub-package.

This package contains:
  - model.py: Graph Attention Network (GAT) for cross-page understanding
  - graph.py: Document graph builder for GNN input

The package gracefully handles missing torch_geometric - if not installed,
the exports are set to None and __all__ is empty.

Example usage::

    from plancheck.analysis.gnn import GNNModel, build_document_graph, is_gnn_available

    if is_gnn_available():
        graph = build_document_graph(page_results)
        model = GNNModel(input_dim=graph.x.size(1), hidden_dim=64, n_classes=9)
        # ... train or predict
"""

from __future__ import annotations

try:
    from .graph import EdgeType, GraphNode, GraphNodeType, build_document_graph
    from .model import DocumentGNN, is_gnn_available, predict_with_gnn, train_gnn

    __all__ = [
        "DocumentGNN",
        "EdgeType",
        "GraphNode",
        "GraphNodeType",
        "build_document_graph",
        "is_gnn_available",
        "predict_with_gnn",
        "train_gnn",
    ]
except ImportError:
    # torch_geometric not installed - provide graceful fallback
    DocumentGNN = None  # type: ignore[assignment,misc]
    EdgeType = None  # type: ignore[assignment,misc]
    GraphNode = None  # type: ignore[assignment,misc]
    GraphNodeType = None  # type: ignore[assignment,misc]
    build_document_graph = None  # type: ignore[assignment,misc]
    is_gnn_available = lambda: False  # type: ignore[assignment,misc]
    predict_with_gnn = None  # type: ignore[assignment,misc]
    train_gnn = None  # type: ignore[assignment,misc]
    __all__ = []
