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

_GNN_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv

    _GNN_AVAILABLE = True
except ImportError:
    pass


def is_gnn_available() -> bool:
    """Return True if PyTorch and PyTorch Geometric are installed."""
    return _GNN_AVAILABLE


# ── Model definition ────────────────────────────────────────────────────

if _GNN_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv

    class DocumentGNN(nn.Module):
        """2-layer Graph Attention Network for document region classification.

        Parameters
        ----------
        in_channels : int
            Node feature dimensionality (14 or 398 with embeddings).
        hidden_channels : int
            Hidden layer size (default 64).
        num_classes : int
            Number of output classes (default 9, matching ``GraphNodeType``).
        heads : int
            Number of attention heads in the first layer (default 4).
        dropout : float
            Dropout probability (default 0.3).
        """

        def __init__(
            self,
            in_channels: int = 14,
            hidden_channels: int = 64,
            num_classes: int = 9,
            heads: int = 4,
            dropout: float = 0.3,
        ) -> None:
            super().__init__()
            self.conv1 = GATConv(
                in_channels, hidden_channels, heads=heads, dropout=dropout
            )
            self.conv2 = GATConv(
                hidden_channels * heads,
                num_classes,
                heads=1,
                concat=False,
                dropout=dropout,
            )
            self.dropout = dropout

        def forward(self, data: Data) -> torch.Tensor:
            """Forward pass.

            Returns log-softmax class probabilities of shape ``[N, num_classes]``.
            """
            x, edge_index = data.x, data.edge_index

            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

        def predict(self, data: Data) -> np.ndarray:
            """Return class predictions as a numpy array of shape ``[N]``."""
            self.eval()
            with torch.no_grad():
                logits = self.forward(data)
                return logits.argmax(dim=1).cpu().numpy()

        def predict_proba(self, data: Data) -> np.ndarray:
            """Return class probabilities as a numpy array ``[N, C]``."""
            self.eval()
            with torch.no_grad():
                logits = self.forward(data)
                return torch.exp(logits).cpu().numpy()

        def get_embeddings(self, data: Data) -> np.ndarray:
            """Return penultimate-layer embeddings ``[N, hidden*heads]``.

            These are the outputs of conv1 + ELU *before* the
            classification head, useful as feature representations
            for downstream tasks (e.g. GNN candidate prior).
            """
            self.eval()
            with torch.no_grad():
                x, edge_index = data.x, data.edge_index
                x = self.conv1(x, edge_index)
                x = F.elu(x)
                return x.cpu().numpy()

    # ── Training loop ──────────────────────────────────────────────

    def train_gnn(
        model: DocumentGNN,
        data: Data,
        labels: torch.Tensor,
        *,
        epochs: int = 200,
        lr: float = 0.005,
        weight_decay: float = 5e-4,
        patience: int = 0,
        train_mask: Optional[torch.Tensor] = None,
        val_mask: Optional[torch.Tensor] = None,
        verbose: bool = True,
    ) -> dict[str, list[float] | int]:
        """Train the GNN on a single document graph.

        Parameters
        ----------
        model : DocumentGNN
            The model to train (will be mutated in-place).
        data : torch_geometric.data.Data
            The document graph with node features and edges.
        labels : torch.Tensor
            Integer class labels for each node, shape ``[N]``.
        epochs : int
            Number of training epochs.
        lr : float
            Learning rate.
        weight_decay : float
            L2 regularisation.
        patience : int
            Early-stopping patience (0 = disabled).  Training stops when
            validation accuracy has not improved for *patience* consecutive
            epochs.  Requires *val_mask* to be set.
        train_mask : Tensor, optional
            Boolean mask for training nodes (defaults to all nodes).
        val_mask : Tensor, optional
            Boolean mask for validation nodes.
        verbose : bool
            If True, log training progress every 20 epochs.

        Returns
        -------
        dict
            ``{"train_loss": [...], "train_acc": [...], "val_acc": [...],
            "stopped_epoch": int}``
        """
        device = next(model.parameters()).device
        data = data.to(device)
        labels = labels.to(device)

        if train_mask is None:
            train_mask = torch.ones(data.num_nodes, dtype=torch.bool, device=device)
        else:
            train_mask = train_mask.to(device)
        if val_mask is not None:
            val_mask = val_mask.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        history: dict[str, list[float] | int] = {
            "train_loss": [],
            "train_acc": [],
            "val_acc": [],
            "stopped_epoch": epochs,
        }

        # Early stopping state
        use_early_stop = patience > 0 and val_mask is not None
        best_val_acc = -1.0
        epochs_without_improvement = 0
        best_state: dict | None = None

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()

            out = model(data)
            loss = F.nll_loss(out[train_mask], labels[train_mask])
            loss.backward()
            optimizer.step()

            # Record metrics — re-run forward pass so metrics
            # reflect the updated parameters (not pre-step logits).
            model.eval()
            with torch.no_grad():
                out = model(data)
                pred = out.argmax(dim=1)
                train_correct = (pred[train_mask] == labels[train_mask]).sum().item()
                train_total = train_mask.sum().item()
                train_acc = train_correct / max(train_total, 1)

                val_acc = 0.0
                if val_mask is not None and val_mask.sum() > 0:
                    val_correct = (pred[val_mask] == labels[val_mask]).sum().item()
                    val_total = val_mask.sum().item()
                    val_acc = val_correct / max(val_total, 1)

            history["train_loss"].append(loss.item())
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            # Early stopping
            if use_early_stop:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    epochs_without_improvement = 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        history["stopped_epoch"] = epoch
                        if verbose:
                            log.info(
                                "Early stopping at epoch %d (no improvement for %d epochs)",
                                epoch,
                                patience,
                            )
                        break

            if verbose and epoch % 20 == 0:
                msg = f"Epoch {epoch:3d} | loss {loss.item():.4f} | train_acc {train_acc:.3f}"
                if val_mask is not None:
                    msg += f" | val_acc {val_acc:.3f}"
                log.info(msg)

        # Restore best weights when early stopping was active
        if use_early_stop and best_state is not None:
            model.load_state_dict(best_state)

        return history

    # ── Save / Load helpers ────────────────────────────────────────

    def save_gnn(model: DocumentGNN, path: str | Path) -> None:
        """Save model weights and config to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_state_dict": model.state_dict(),
            "config": {
                "in_channels": model.conv1.in_channels,
                "hidden_channels": model.conv1.out_channels,
                "num_classes": model.conv2.out_channels,
                "heads": model.conv1.heads,
                "dropout": model.dropout,
            },
        }
        torch.save(state, path)
        log.info("Saved GNN model to %s", path)

    def load_gnn(path: str | Path, device: str = "cpu") -> DocumentGNN:
        """Load a saved GNN model from disk."""
        path = Path(path)
        state = torch.load(path, map_location=device, weights_only=False)
        cfg = state["config"]
        model = DocumentGNN(
            in_channels=cfg["in_channels"],
            hidden_channels=cfg["hidden_channels"],
            num_classes=cfg["num_classes"],
            heads=cfg["heads"],
            dropout=cfg["dropout"],
        )
        model.load_state_dict(state["model_state_dict"])
        model.to(device)
        model.eval()
        log.info("Loaded GNN model from %s", path)
        return model

else:
    # Stubs when PyG is not installed

    class DocumentGNN:  # type: ignore[no-redef]
        """Stub — PyTorch Geometric not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PyTorch Geometric is required for DocumentGNN")

    def train_gnn(*args: Any, **kwargs: Any) -> dict:
        raise RuntimeError("PyTorch Geometric is required for train_gnn")

    def save_gnn(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("PyTorch Geometric is required for save_gnn")

    def load_gnn(*args: Any, **kwargs: Any) -> Any:
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
    if not _GNN_AVAILABLE:
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
