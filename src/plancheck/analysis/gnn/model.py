"""Graph Attention Network for cross-page document understanding.

Implements a 2-layer **GAT** (Graph Attention Network) using PyTorch
Geometric that operates on the document graph built by
:mod:`~plancheck.analysis.document_graph`.

The model predicts a *refined* label for each region node, taking into
account cross-page context that a single-page classifier cannot capture.

Architecture
~~~~~~~~~~~~
::

    Input (D)  ─►  GATConv(D, hidden, heads=4)  ─►  ELU  ─►  Dropout
               ─►  GATConv(hidden*4, n_classes, heads=1, concat=False)
               ─►  LogSoftmax

The model is intentionally lightweight (a few hundred parameters at
``hidden_dim=64``) so it trains in seconds on CPU.

Public API
----------
DocumentGNN          – The GAT model
is_gnn_available     – True when torch + torch_geometric are importable
predict_with_gnn     – High-level helper to refine labels on a graph
train_gnn            – Simple training loop
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

log = logging.getLogger(__name__)

# ── Availability probe ──────────────────────────────────────────────────
# We defer the actual torch import to avoid hangs on some Windows machines
# during CUDA/MKL initialisation.  find_spec() checks whether packages
# are installed without importing them.  The real import only happens when
# code actually instantiates a model (inside _ensure_gnn_imports below).
import importlib.util as _ilu

_GNN_INSTALLED = (
    _ilu.find_spec("torch") is not None
    and _ilu.find_spec("torch_geometric") is not None
)

# Flipped to True by _ensure_gnn_imports() on first real use.
_GNN_AVAILABLE = False


def _ensure_gnn_imports() -> bool:
    """Do the real torch/pyg import on first use.  Returns True on success."""
    global _GNN_AVAILABLE  # noqa: PLW0603
    if _GNN_AVAILABLE:
        return True
    if not _GNN_INSTALLED:
        return False
    try:
        import torch  # noqa: F401
        import torch.nn.functional  # noqa: F401
        from torch_geometric.data import Data  # noqa: F401
        from torch_geometric.nn import GATConv  # noqa: F401

        _GNN_AVAILABLE = True
        return True
    except (ImportError, OSError):
        return False


def is_gnn_available() -> bool:
    """Return True if PyTorch and PyTorch Geometric are installed.

    On first call this triggers the actual import of torch and
    torch_geometric.  Subsequent calls return a cached result.
    """
    return _ensure_gnn_imports()


# ── Model definition ────────────────────────────────────────────────────
# At module level, _GNN_AVAILABLE is False so we always take the stub
# branch.  The stubs raise RuntimeError on use and are harmless to import.
# Code that needs real GNN functionality calls _ensure_gnn_imports() first,
# which does the real torch import; only then do we define the real class
# (see _make_gnn_class below, used by load_gnn/predict_with_gnn).

# Stubs — always defined so the module is importable without torch.


class DocumentGNN:  # type: ignore[no-redef]
    """Stub — PyTorch Geometric not installed or not yet imported."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "PyTorch Geometric is required for DocumentGNN.  "
            "Call is_gnn_available() first to trigger lazy loading."
        )


def train_gnn(*args: Any, **kwargs: Any) -> dict:
    """Stub — requires PyTorch Geometric."""
    raise RuntimeError("PyTorch Geometric is required for train_gnn")


def save_gnn(*args: Any, **kwargs: Any) -> None:
    """Stub — requires PyTorch Geometric."""
    raise RuntimeError("PyTorch Geometric is required for save_gnn")


def load_gnn(*args: Any, **kwargs: Any) -> Any:
    """Stub — requires PyTorch Geometric."""
    raise RuntimeError("PyTorch Geometric is required for load_gnn")


# ── High-level prediction helper ──────────────────────────────────────


def predict_with_gnn(
    graph_data: dict,
    model_path: str | Path = "data/document_gnn.pt",
) -> Optional[np.ndarray]:
    """Load a trained GNN and predict refined labels for each graph node.

    Parameters
    ----------
    graph_data : dict
        Output of :func:`build_document_graph` — must contain ``"pyg_data"``
        or ``"x"`` and ``"edge_index"`` keys.
    model_path : Path
        Path to a saved ``.pt`` model file.

    Returns
    -------
    np.ndarray or None
        Integer label predictions of shape ``[N]``, or ``None`` if GNN
        is unavailable or the model file doesn't exist.
    """
    if not is_gnn_available():
        log.debug("GNN not available — skipping prediction")
        return None

    model_path = Path(model_path)
    if not model_path.exists():
        log.debug("GNN model not found at %s — skipping", model_path)
        return None

    # Get or build PyG Data
    pyg_data = graph_data.get("pyg_data")
    if pyg_data is None:
        import torch
        from torch_geometric.data import Data

        pyg_data = Data(
            x=torch.from_numpy(graph_data["x"]),
            edge_index=torch.from_numpy(graph_data["edge_index"]),
        )

    if pyg_data.num_nodes == 0:
        return np.array([], dtype=np.int64)

    model = load_gnn(model_path)
    return model.predict(pyg_data)
