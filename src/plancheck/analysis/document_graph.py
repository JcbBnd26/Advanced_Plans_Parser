"""Cross-page document graph for GNN-based reasoning.

Builds a **homogeneous** graph whose:

* **Nodes** = semantic regions (one per region per page — notes columns,
  title blocks, legends, abbreviation tables, revision blocks, detail
  callouts, etc.).
* **Edges** = relationships:
  - *same-page proximity* — two regions are on the same page
  - *shared entity* — two regions mention the same abbreviation /
    detail number / revision ID across pages
  - *page sequence* — link page-N title block to page-(N±1) title block

Each node carries a feature vector composed of:
  - one-hot region type (9 slots matching LayoutLMv3 labels)
  - normalised bbox  (x0, y0, x1, y1)
  - page number (int)
  - optional text embedding (384-d from ``TextEmbedder``)

The graph can be passed directly to :class:`~plancheck.analysis.gnn_model.DocumentGNN`.

Public API
----------
build_document_graph  – Build a PyG Data object from a list of PageResults
GraphNodeType         – Enum of region types
"""

from __future__ import annotations

import enum
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ── Node types (aligned with LayoutLMv3 label set) ─────────────────────


class GraphNodeType(enum.IntEnum):
    """Region label IDs used as node type.

    Kept in sync with ``LAYOUT_LABELS`` in ``classifier.py``.
    """

    BORDER = 0
    DRAWING = 1
    NOTES = 2
    TITLE_BLOCK = 3
    LEGEND = 4
    ABBREVIATIONS = 5
    REVISIONS = 6
    DETAILS = 7
    UNKNOWN = 8

    @classmethod
    def from_string(cls, s: str) -> "GraphNodeType":
        """Convert a label string to enum, case-insensitive."""
        try:
            return cls[s.upper()]
        except KeyError:
            return cls.UNKNOWN

    @classmethod
    def num_types(cls) -> int:
        return len(cls)


# ── Edge types ──────────────────────────────────────────────────────────


class EdgeType(enum.IntEnum):
    """Categorical edge types for the document graph."""

    SAME_PAGE = 0
    SHARED_ENTITY = 1
    PAGE_SEQUENCE = 2


# ── Node data container ────────────────────────────────────────────────


@dataclass
class GraphNode:
    """Intermediate container for a single graph node before tensorizing."""

    node_type: GraphNodeType
    page: int
    bbox: Tuple[float, float, float, float]  # normalised 0-1
    text_embedding: Optional[np.ndarray] = None  # 384-d or None
    source_region: Any = None  # reference back to model object

    def to_feature_vector(self, include_embedding: bool = False) -> np.ndarray:
        """Return the feature vector for this node.

        Shape: (9 + 4 + 1) = 14  or  (14 + 384) = 398 when *include_embedding*.
        """
        # one-hot region type
        one_hot = np.zeros(GraphNodeType.num_types(), dtype=np.float32)
        one_hot[int(self.node_type)] = 1.0

        # normalised bbox
        bbox_arr = np.array(self.bbox, dtype=np.float32)

        # page (normalised — caller should post-normalise across doc)
        page_arr = np.array([float(self.page)], dtype=np.float32)

        parts = [one_hot, bbox_arr, page_arr]

        if include_embedding and self.text_embedding is not None:
            parts.append(self.text_embedding.astype(np.float32))
        elif include_embedding:
            # zero-pad if no embedding available
            parts.append(np.zeros(384, dtype=np.float32))

        return np.concatenate(parts)


# ── Entity extraction helpers ───────────────────────────────────────────

_DETAIL_RE = re.compile(r"\b(\d{1,2})/([A-Z]-?\d{1,4})\b")
_ABBREV_RE = re.compile(r"\b([A-Z]{2,6})\b")  # simple upper-case tokens


def _extract_entities(text: str) -> dict[str, set[str]]:
    """Pull out named entities that can link regions across pages.

    Returns a dict with keys ``"details"``, ``"abbreviations"`` each
    mapping to a set of normalised strings.
    """
    entities: dict[str, set[str]] = {"details": set(), "abbreviations": set()}

    if not text:
        return entities

    for m in _DETAIL_RE.finditer(text):
        entities["details"].add(m.group(0))

    for m in _ABBREV_RE.finditer(text):
        tok = m.group(1)
        # Filter out very common words that happen to be uppercase
        if len(tok) >= 2 and tok not in {
            "THE",
            "AND",
            "FOR",
            "NOT",
            "ALL",
            "ARE",
            "BUT",
            "THIS",
            "THAT",
            "WITH",
        }:
            entities["abbreviations"].add(tok)

    return entities


def _get_region_text(region: Any) -> str:
    """Best-effort text extraction from a model region object."""
    if hasattr(region, "full_text") and callable(region.full_text):
        return region.full_text()
    if hasattr(region, "blocks"):
        parts = []
        for block in region.blocks:
            if hasattr(block, "get_all_boxes"):
                parts.extend(b.text for b in block.get_all_boxes())
            elif hasattr(block, "text"):
                parts.append(block.text)
        return " ".join(parts)
    if hasattr(region, "text"):
        return region.text
    return ""


def _get_bbox(
    region: Any, page_width: float, page_height: float
) -> Tuple[float, float, float, float]:
    """Return normalised (0-1) bounding box from a region."""
    if hasattr(region, "bbox"):
        b = region.bbox
        if isinstance(b, (list, tuple)) and len(b) == 4:
            pw = max(page_width, 1.0)
            ph = max(page_height, 1.0)
            return (b[0] / pw, b[1] / ph, b[2] / pw, b[3] / ph)
    return (0.0, 0.0, 1.0, 1.0)


# ── Main builder ────────────────────────────────────────────────────────


def build_document_graph(
    page_results: Sequence[Any],
    *,
    include_embeddings: bool = False,
    text_embedder: Any = None,
) -> dict:
    """Build a cross-page document graph from pipeline results.

    Parameters
    ----------
    page_results : sequence of PageResult
        One per page, as returned by :func:`plancheck.pipeline.run_pipeline`.
    include_embeddings : bool
        Whether to include 384-d text embeddings in node features.
    text_embedder : TextEmbedder | None
        If *include_embeddings* is True and a text embedder is given,
        it will embed each region's text.

    Returns
    -------
    dict
        With keys ``"x"`` (np.ndarray, node features, shape [N, D]),
        ``"edge_index"`` (np.ndarray, shape [2, E]),
        ``"edge_attr"`` (np.ndarray, shape [E]),
        ``"nodes"`` (list of GraphNode),
        ``"num_nodes"``, ``"num_edges"``.

        If PyTorch Geometric is available, also includes ``"pyg_data"``
        with a ready-to-use ``torch_geometric.data.Data`` object.
    """
    nodes: list[GraphNode] = []
    node_entities: list[dict[str, set[str]]] = []
    node_page: list[int] = []

    # ── Collect nodes from all pages ──────────────────────────────
    for pr in page_results:
        page = getattr(pr, "page", 0)
        pw = getattr(pr, "page_width", 1.0)
        ph = getattr(pr, "page_height", 1.0)

        region_map: list[tuple[GraphNodeType, Any]] = []

        # Notes columns
        for nc in getattr(pr, "notes_columns", []):
            region_map.append((GraphNodeType.NOTES, nc))
        # Title blocks
        for tb in getattr(pr, "title_blocks", []):
            region_map.append((GraphNodeType.TITLE_BLOCK, tb))
        # Legends
        for lg in getattr(pr, "legend_regions", []):
            region_map.append((GraphNodeType.LEGEND, lg))
        # Abbreviation regions
        for ab in getattr(pr, "abbreviation_regions", []):
            region_map.append((GraphNodeType.ABBREVIATIONS, ab))
        # Revision regions
        for rv in getattr(pr, "revision_regions", []):
            region_map.append((GraphNodeType.REVISIONS, rv))
        # Standard detail regions
        for sd in getattr(pr, "standard_detail_regions", []):
            region_map.append((GraphNodeType.DETAILS, sd))
        # Misc title regions
        for mt in getattr(pr, "misc_title_regions", []):
            region_map.append((GraphNodeType.UNKNOWN, mt))
        # Structural boxes (drawing areas)
        for sb in getattr(pr, "structural_boxes", []):
            region_map.append((GraphNodeType.DRAWING, sb))

        for ntype, region in region_map:
            bbox = _get_bbox(region, pw, ph)
            text = _get_region_text(region)

            text_emb = None
            if include_embeddings and text_embedder is not None and text:
                try:
                    text_emb = text_embedder.embed(text)
                except Exception:
                    pass

            node = GraphNode(
                node_type=ntype,
                page=page,
                bbox=bbox,
                text_embedding=text_emb,
                source_region=region,
            )
            nodes.append(node)
            node_entities.append(_extract_entities(text))
            node_page.append(page)

    if not nodes:
        return {
            "x": np.zeros((0, 14), dtype=np.float32),
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "edge_attr": np.zeros(0, dtype=np.int64),
            "nodes": [],
            "num_nodes": 0,
            "num_edges": 0,
        }

    # ── Build feature matrix ──────────────────────────────────────
    x = np.stack(
        [n.to_feature_vector(include_embedding=include_embeddings) for n in nodes]
    )

    # Normalise page feature (column index 13) across document
    max_page = max(node_page) if node_page else 1
    if max_page > 0:
        x[:, GraphNodeType.num_types() + 4] /= float(max_page)

    # ── Build edges ───────────────────────────────────────────────
    src_list: list[int] = []
    dst_list: list[int] = []
    edge_types: list[int] = []

    n = len(nodes)

    for i in range(n):
        for j in range(i + 1, n):
            # Same-page → bidirectional
            if node_page[i] == node_page[j]:
                src_list.extend([i, j])
                dst_list.extend([j, i])
                edge_types.extend([EdgeType.SAME_PAGE, EdgeType.SAME_PAGE])

            # Shared entities (cross-page or same-page)
            ent_i = node_entities[i]
            ent_j = node_entities[j]
            shared = False
            for key in ("details", "abbreviations"):
                if ent_i[key] & ent_j[key]:
                    shared = True
                    break
            if shared:
                src_list.extend([i, j])
                dst_list.extend([j, i])
                edge_types.extend([EdgeType.SHARED_ENTITY, EdgeType.SHARED_ENTITY])

            # Page sequence (adjacent pages, both title blocks)
            if (
                nodes[i].node_type == GraphNodeType.TITLE_BLOCK
                and nodes[j].node_type == GraphNodeType.TITLE_BLOCK
                and abs(node_page[i] - node_page[j]) == 1
            ):
                src_list.extend([i, j])
                dst_list.extend([j, i])
                edge_types.extend([EdgeType.PAGE_SEQUENCE, EdgeType.PAGE_SEQUENCE])

    edge_index = (
        np.array([src_list, dst_list], dtype=np.int64)
        if src_list
        else np.zeros((2, 0), dtype=np.int64)
    )
    edge_attr = (
        np.array(edge_types, dtype=np.int64)
        if edge_types
        else np.zeros(0, dtype=np.int64)
    )

    result: dict[str, Any] = {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "nodes": nodes,
        "num_nodes": n,
        "num_edges": edge_index.shape[1],
    }

    # ── Optionally wrap as PyG Data ───────────────────────────────
    try:
        import torch
        from torch_geometric.data import Data

        result["pyg_data"] = Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.from_numpy(edge_attr),
        )
    except ImportError:
        pass

    log.info(
        "Document graph: %d nodes, %d edges (from %d pages)",
        n,
        edge_index.shape[1],
        len(page_results),
    )
    return result
