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


_RealDocumentGNN = None  # populated by _ensure_gnn_imports()


def _ensure_gnn_imports() -> bool:
    """Do the real torch/pyg import on first use.  Returns True on success."""
    global _GNN_AVAILABLE, _RealDocumentGNN  # noqa: PLW0603
    if _GNN_AVAILABLE:
        return True
    if not _GNN_INSTALLED:
        return False
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.nn import GATConv

        _GNN_AVAILABLE = True
    except (ImportError, OSError):
        return False

    class _DocumentGNN(torch.nn.Module):
        """2-layer GAT for cross-page document understanding."""

        def __init__(
            self,
            in_channels: int,
            hidden_channels: int = 64,
            num_classes: int = 9,
            heads: int = 4,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
            self.conv2 = GATConv(
                hidden_channels * heads, num_classes, heads=1, concat=False
            )
            self.dropout = dropout

        def forward(self, data: Any) -> torch.Tensor:
            x, edge_index = data.x.float(), data.edge_index
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

        def get_embeddings(self, data: Any) -> np.ndarray:
            """Return penultimate-layer embeddings as a numpy array."""
            self.eval()
            with torch.no_grad():
                x, edge_index = data.x.float(), data.edge_index
                x = self.conv1(x, edge_index)
                x = F.elu(x)
            return x.cpu().numpy()

        def predict(self, data: Any) -> np.ndarray:
            """Return integer label predictions."""
            self.eval()
            with torch.no_grad():
                out = self.forward(data)
            return out.argmax(dim=1).cpu().numpy()

    _RealDocumentGNN = _DocumentGNN
    globals()["DocumentGNN"] = _DocumentGNN
    return True


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
    """Stub — auto-delegates to real GAT class when PyG is available."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        if _ensure_gnn_imports():
            return _RealDocumentGNN(*args, **kwargs)
        raise RuntimeError(
            "PyTorch Geometric is required for DocumentGNN.  "
            "Install with: pip install torch_geometric"
        )


def train_gnn(*args: Any, **kwargs: Any) -> dict:
    """Stub — requires PyTorch Geometric."""
    raise RuntimeError("PyTorch Geometric is required for train_gnn")


def save_gnn(*args: Any, **kwargs: Any) -> None:
    """Stub — requires PyTorch Geometric."""
    raise RuntimeError("PyTorch Geometric is required for save_gnn")


def load_gnn(path: str | Path, device: str = "cpu") -> Any:
    """Load a saved DocumentGNN model.  Returns None if unavailable."""
    if not _ensure_gnn_imports():
        raise RuntimeError("PyTorch Geometric is required for load_gnn")
    import torch

    path = Path(path)
    state = torch.load(path, map_location=device, weights_only=False)
    model = _RealDocumentGNN(**state["init_args"])
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model


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
