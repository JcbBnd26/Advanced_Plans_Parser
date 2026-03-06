"""Level 4 — GNN-based candidate priors for VOCR candidate detection.

Uses the cross-page document graph to predict which regions are likely to
contain true missing symbols, providing a posterior adjustment to VOCR
candidate confidence values.

Architecture
~~~~~~~~~~~~
After all pages are processed and the document graph is built, this
module:

1. **Annotates** graph nodes with per-region candidate statistics
   (count, mean confidence, hit rate).
2. **Extracts** penultimate-layer GNN embeddings (hidden*heads dims)
   for each region, capturing cross-page structural context.
3. **Predicts** P(true positive | region) using a small binary MLP head
   trained on top of (GNN embeddings + candidate features).
4. **Adjusts** candidate confidence on each ``VocrCandidate`` based on
   the per-region prior, blending the original confidence with the GNN
   prior via a configurable weight.

The prior head is intentionally lightweight (a few hundred parameters)
and trains in seconds on CPU.  When no trained model is available the
module is a no-op (fail-open).

Public API
----------
assign_candidates_to_nodes   – Map candidates to graph nodes by bbox
annotate_graph_with_candidates – Populate candidate stats on graph nodes
GNNCandidatePriorHead        – Small MLP on GNN embeddings
apply_gnn_prior              – Adjust candidate confidences
train_gnn_candidate_prior    – Train the prior head from outcomes
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..analysis.gnn.graph import GraphNode
from ..models.geometry import bbox_intersection_area as _bbox_overlap

log = logging.getLogger(__name__)

# ── Torch availability ──────────────────────────────────────────────────

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    pass

# Number of candidate-level features appended to GNN embeddings.
CANDIDATE_FEATURE_DIM = 3  # count, mean_confidence, hit_rate


def assign_candidates_to_nodes(
    nodes: Sequence[GraphNode],
    candidates: Sequence[Any],
    page_width: float = 1.0,
    page_height: float = 1.0,
) -> Dict[int, List[Any]]:
    """Map VOCR candidates to graph nodes by spatial overlap.

    Each candidate is assigned to the node whose normalised bbox has the
    largest intersection with the candidate's normalised bbox.  Candidates
    with zero overlap to all nodes are assigned to the nearest node on the
    same page.

    Parameters
    ----------
    nodes : sequence of GraphNode
        Graph nodes (with normalised bboxes already set).
    candidates : sequence of VocrCandidate
        VOCR candidates with absolute-coordinate bboxes.
    page_width, page_height : float
        Page dimensions for normalising candidate bboxes.

    Returns
    -------
    dict[int, list]
        Mapping from node index to list of candidates assigned to it.
    """
    node_cands: Dict[int, List[Any]] = {i: [] for i in range(len(nodes))}
    if not nodes or not candidates:
        return node_cands

    pw = max(page_width, 1.0)
    ph = max(page_height, 1.0)

    for cand in candidates:
        cand_page = getattr(cand, "page", 0)
        # Normalise candidate bbox to [0, 1]
        cand_bbox = (
            cand.x0 / pw,
            cand.y0 / ph,
            cand.x1 / pw,
            cand.y1 / ph,
        )

        best_idx = -1
        best_overlap = 0.0

        for i, node in enumerate(nodes):
            if node.page != cand_page:
                continue
            ov = _bbox_overlap(node.bbox, cand_bbox)
            if ov > best_overlap:
                best_overlap = ov
                best_idx = i

        # Fallback: nearest same-page node by centre distance
        if best_idx < 0:
            cx = (cand_bbox[0] + cand_bbox[2]) / 2.0
            cy = (cand_bbox[1] + cand_bbox[3]) / 2.0
            best_dist = float("inf")
            for i, node in enumerate(nodes):
                if node.page != cand_page:
                    continue
                nx = (node.bbox[0] + node.bbox[2]) / 2.0
                ny = (node.bbox[1] + node.bbox[3]) / 2.0
                d = (cx - nx) ** 2 + (cy - ny) ** 2
                if d < best_dist:
                    best_dist = d
                    best_idx = i

        if best_idx >= 0:
            node_cands[best_idx].append(cand)

    return node_cands


def annotate_graph_with_candidates(
    nodes: Sequence[GraphNode],
    node_cands: Dict[int, List[Any]],
) -> None:
    """Populate candidate statistics on each :class:`GraphNode`.

    Mutates nodes in-place, setting ``candidate_count``,
    ``mean_candidate_confidence``, and ``candidate_hit_rate``.
    """
    for i, node in enumerate(nodes):
        cands = node_cands.get(i, [])
        node.candidate_count = len(cands)
        if cands:
            node.mean_candidate_confidence = float(
                np.mean([getattr(c, "confidence", 0.5) for c in cands])
            )
            hits = sum(1 for c in cands if getattr(c, "outcome", "pending") == "hit")
            total_resolved = sum(
                1 for c in cands if getattr(c, "outcome", "pending") in ("hit", "miss")
            )
            node.candidate_hit_rate = (
                hits / total_resolved if total_resolved > 0 else 0.0
            )
        else:
            node.mean_candidate_confidence = 0.0
            node.candidate_hit_rate = 0.0


# ── GNN Candidate Prior Head ───────────────────────────────────────────


if _TORCH_AVAILABLE:

    class GNNCandidatePriorHead(nn.Module):
        """Small binary MLP predicting P(true positive | region).

        Input: concatenation of GNN embedding (``embed_dim``) and
        candidate features (3 dims).
        Output: scalar probability via sigmoid.

        Parameters
        ----------
        embed_dim : int
            Dimensionality of the GNN penultimate-layer embeddings
            (default 256 = hidden_channels * heads for default GNN config).
        hidden : int
            Hidden layer size (default 32).
        """

        def __init__(self, embed_dim: int = 256, hidden: int = 32) -> None:
            super().__init__()
            in_dim = embed_dim + CANDIDATE_FEATURE_DIM
            self.head = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden, 1),
            )
            self._embed_dim = embed_dim

        @property
        def embed_dim(self) -> int:
            return self._embed_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return logits of shape ``[N, 1]``."""
            return self.head(x)

        def predict_proba(
            self,
            embeddings: np.ndarray,
            candidate_features: np.ndarray,
        ) -> np.ndarray:
            """Return P(hit) for each node as a 1-D numpy array ``[N]``.

            Parameters
            ----------
            embeddings : ndarray, shape [N, embed_dim]
                GNN penultimate-layer embeddings.
            candidate_features : ndarray, shape [N, 3]
                Per-node candidate statistics from
                ``GraphNode.to_candidate_features()``.
            """
            self.eval()
            x = np.concatenate([embeddings, candidate_features], axis=1)
            with torch.no_grad():
                logits = self.forward(torch.from_numpy(x.astype(np.float32)))
                return torch.sigmoid(logits).squeeze(-1).cpu().numpy()

    # ── Save / Load ────────────────────────────────────────────────

    def save_gnn_candidate_prior(head: GNNCandidatePriorHead, path: str | Path) -> None:
        """Save the prior head to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_state_dict": head.state_dict(),
            "embed_dim": head.embed_dim,
        }
        torch.save(state, path)
        log.info("Saved GNN candidate prior head to %s", path)

    def load_gnn_candidate_prior(
        path: str | Path,
        device: str = "cpu",
    ) -> Optional[GNNCandidatePriorHead]:
        """Load a saved prior head.  Returns ``None`` if file missing."""
        path = Path(path)
        if not path.exists():
            return None
        state = torch.load(path, map_location=device, weights_only=False)
        embed_dim = state.get("embed_dim", 256)
        head = GNNCandidatePriorHead(embed_dim=embed_dim)
        head.load_state_dict(state["model_state_dict"])
        head.to(device)
        head.eval()
        log.info("Loaded GNN candidate prior head from %s", path)
        return head

    # ── Training loop ──────────────────────────────────────────────

    def train_gnn_candidate_prior(
        embeddings: np.ndarray,
        candidate_features: np.ndarray,
        labels: np.ndarray,
        *,
        embed_dim: int = 256,
        epochs: int = 100,
        lr: float = 0.001,
        val_fraction: float = 0.2,
        verbose: bool = False,
    ) -> Tuple[GNNCandidatePriorHead, Dict[str, Any]]:
        """Train a :class:`GNNCandidatePriorHead` from labelled data.

        Parameters
        ----------
        embeddings : ndarray, shape [N, embed_dim]
            GNN penultimate-layer embeddings for each node.
        candidate_features : ndarray, shape [N, 3]
            Per-node candidate statistics.
        labels : ndarray, shape [N]
            Binary labels — 1 if any candidate in the region was a hit.
        embed_dim : int
            Must match ``embeddings.shape[1]``.
        epochs, lr : training hyperparameters.
        val_fraction : float
            Hold-out fraction for validation.
        verbose : bool
            Log progress every 10 epochs.

        Returns
        -------
        (head, metrics)
            Trained model and a dict of training metrics.
        """
        n = len(labels)
        x = np.concatenate([embeddings, candidate_features], axis=1).astype(np.float32)
        y = labels.astype(np.float32)

        # Train/val split
        perm = np.random.permutation(n)
        n_val = max(1, int(n * val_fraction))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        x_train = torch.from_numpy(x[train_idx])
        y_train = torch.from_numpy(y[train_idx]).unsqueeze(1)
        x_val = torch.from_numpy(x[val_idx])
        y_val = torch.from_numpy(y[val_idx]).unsqueeze(1)

        head = GNNCandidatePriorHead(embed_dim=embed_dim)
        optimizer = torch.optim.Adam(head.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, epochs + 1):
            head.train()
            optimizer.zero_grad()
            logits = head(x_train)
            loss = loss_fn(logits, y_train)
            loss.backward()
            optimizer.step()

            # Validation
            head.eval()
            with torch.no_grad():
                val_logits = head(x_val)
                val_loss = loss_fn(val_logits, y_val).item()
                val_preds = (torch.sigmoid(val_logits) >= 0.5).float()
                val_acc = (val_preds == y_val).float().mean().item()

            history["train_loss"].append(loss.item())
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if verbose and epoch % 10 == 0:
                log.info(
                    "GNN prior epoch %3d | train_loss %.4f | val_loss %.4f | val_acc %.3f",
                    epoch,
                    loss.item(),
                    val_loss,
                    val_acc,
                )

        metrics = {
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "final_train_loss": (
                history["train_loss"][-1] if history["train_loss"] else 0.0
            ),
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else 0.0,
            "final_val_acc": history["val_acc"][-1] if history["val_acc"] else 0.0,
            "epochs": epochs,
        }
        log.info(
            "GNN candidate prior trained: val_acc=%.3f  (%d train, %d val)",
            metrics["final_val_acc"],
            metrics["n_train"],
            metrics["n_val"],
        )
        return head, metrics

else:
    # ── Stubs when PyTorch not installed ───────────────────────────

    class GNNCandidatePriorHead:  # type: ignore[no-redef]
        """Stub — PyTorch not installed."""

        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("PyTorch is required for GNNCandidatePriorHead")

        def predict_proba(self, *a: Any, **kw: Any) -> np.ndarray:
            raise RuntimeError("PyTorch is required for GNNCandidatePriorHead")

    def save_gnn_candidate_prior(*a: Any, **kw: Any) -> None:
        raise RuntimeError("PyTorch is required")

    def load_gnn_candidate_prior(*a: Any, **kw: Any) -> None:
        return None  # graceful no-op

    def train_gnn_candidate_prior(*a: Any, **kw: Any) -> tuple:
        raise RuntimeError("PyTorch is required")


# ── Confidence Adjustment ──────────────────────────────────────────────


def apply_gnn_prior(
    page_results: Sequence[Any],
    graph_data: Dict[str, Any],
    gnn_model: Any,
    prior_head: Any,
    *,
    page_width: float = 1.0,
    page_height: float = 1.0,
    blend_weight: float = 0.25,
) -> int:
    """Adjust candidate confidences across a document using GNN priors.

    This is a **document-level post-processing** step run after all pages
    have been processed individually.  It:

    1. Collects all candidates across pages.
    2. Assigns them to graph nodes.
    3. Annotates nodes with candidate stats.
    4. Extracts GNN embeddings + candidate features.
    5. Predicts per-region P(hit) via the prior head.
    6. Blends the prior into each candidate's confidence:
       ``conf = (1 - w) * conf + w * region_prior``

    Parameters
    ----------
    page_results : sequence of PageResult
        Each must have a ``vocr_candidates`` attribute.
    graph_data : dict
        Output of ``build_document_graph()`` — must contain ``"nodes"``
        and either ``"pyg_data"`` or (``"x"``, ``"edge_index"``).
    gnn_model : DocumentGNN
        A loaded GNN model with ``get_embeddings()``.
    prior_head : GNNCandidatePriorHead
        A loaded prior head with ``predict_proba()``.
    page_width, page_height : float
        Page dimensions for normalising candidate bboxes.
    blend_weight : float
        How much weight to give the GNN prior (0 = ignore, 1 = full).

    Returns
    -------
    int
        Number of candidates whose confidence was adjusted.
    """
    nodes = graph_data.get("nodes", [])
    if not nodes:
        return 0

    # Collect all candidates across pages
    all_candidates: List[Any] = []
    for pr in page_results:
        all_candidates.extend(getattr(pr, "vocr_candidates", []))
    if not all_candidates:
        return 0

    # 1. Map candidates to nodes
    node_cands = assign_candidates_to_nodes(
        nodes, all_candidates, page_width=page_width, page_height=page_height
    )

    # 2. Annotate nodes with candidate stats
    annotate_graph_with_candidates(nodes, node_cands)

    # 3. Get GNN embeddings
    pyg_data = graph_data.get("pyg_data")
    if pyg_data is None:
        if not _TORCH_AVAILABLE:
            return 0
        import torch as _torch
        from torch_geometric.data import Data as _Data

        pyg_data = _Data(
            x=_torch.from_numpy(graph_data["x"]),
            edge_index=_torch.from_numpy(graph_data["edge_index"]),
        )

    embeddings = gnn_model.get_embeddings(pyg_data)  # [N, embed_dim]

    # 4. Candidate features per node
    cand_feats = np.stack([n.to_candidate_features() for n in nodes])  # [N, 3]

    # 5. Predict per-region P(hit)
    priors = prior_head.predict_proba(embeddings, cand_feats)  # [N]

    # 6. Blend into candidate confidence
    adjusted = 0
    for i, node in enumerate(nodes):
        cands = node_cands.get(i, [])
        if not cands:
            continue
        region_prior = float(priors[i])
        for cand in cands:
            old_conf = cand.confidence
            cand.confidence = (
                1.0 - blend_weight
            ) * old_conf + blend_weight * region_prior
            cand.confidence = max(0.0, min(1.0, cand.confidence))
            if cand.confidence != old_conf:
                adjusted += 1
                # Annotate context with GNN prior info
                cand.context["gnn_region_prior"] = round(region_prior, 4)
                cand.context["gnn_prior_blend"] = round(blend_weight, 4)

    log.info(
        "GNN candidate prior: adjusted %d / %d candidates across %d nodes",
        adjusted,
        len(all_candidates),
        len(nodes),
    )
    return adjusted
