"""Tests for Phase 3.3 — Document Graph and GNN Model.

Covers:
- GraphNodeType enum
- build_document_graph() with mock PageResults
- Edge creation (same-page, shared entity, page sequence)
- Feature vector shape
- GNN model stub (graceful degradation)
- predict_with_gnn() returns None when model unavailable
- Config fields (ml_gnn_enabled, ml_gnn_model_path, ml_gnn_hidden_dim)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from plancheck.analysis.document_graph import (
    EdgeType,
    GraphNode,
    GraphNodeType,
    _extract_entities,
    build_document_graph,
)
from plancheck.analysis.gnn_model import is_gnn_available, predict_with_gnn
from plancheck.config import GroupingConfig

# ── Helpers ────────────────────────────────────────────────────────────


@dataclass
class MockRegion:
    """Minimal region for testing."""

    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 100.0, 100.0)
    text: str = ""


@dataclass
class MockPageResult:
    """Minimal PageResult for graph building."""

    page: int = 0
    page_width: float = 612.0
    page_height: float = 792.0
    notes_columns: list = field(default_factory=list)
    title_blocks: list = field(default_factory=list)
    legend_regions: list = field(default_factory=list)
    abbreviation_regions: list = field(default_factory=list)
    revision_regions: list = field(default_factory=list)
    standard_detail_regions: list = field(default_factory=list)
    misc_title_regions: list = field(default_factory=list)
    structural_boxes: list = field(default_factory=list)


# ── GraphNodeType ─────────────────────────────────────────────────────


class TestGraphNodeType:
    def test_num_types(self):
        assert GraphNodeType.num_types() == 9

    def test_from_string(self):
        assert GraphNodeType.from_string("notes") == GraphNodeType.NOTES
        assert GraphNodeType.from_string("TITLE_BLOCK") == GraphNodeType.TITLE_BLOCK

    def test_from_string_unknown(self):
        assert GraphNodeType.from_string("nonexistent") == GraphNodeType.UNKNOWN

    def test_values_contiguous(self):
        """Values should be 0..8."""
        vals = sorted(GraphNodeType)
        assert vals == list(range(9))


# ── GraphNode ─────────────────────────────────────────────────────────


class TestGraphNode:
    def test_feature_vector_shape_basic(self):
        node = GraphNode(
            node_type=GraphNodeType.NOTES,
            page=0,
            bbox=(0.1, 0.2, 0.3, 0.4),
        )
        vec = node.to_feature_vector()
        assert vec.shape == (14,)  # 9 type + 4 bbox + 1 page

    def test_feature_vector_shape_with_embedding(self):
        node = GraphNode(
            node_type=GraphNodeType.NOTES,
            page=0,
            bbox=(0.1, 0.2, 0.3, 0.4),
            text_embedding=np.zeros(384, dtype=np.float32),
        )
        vec = node.to_feature_vector(include_embedding=True)
        assert vec.shape == (398,)  # 14 + 384

    def test_feature_vector_onehot(self):
        node = GraphNode(
            node_type=GraphNodeType.LEGEND,
            page=0,
            bbox=(0.0, 0.0, 1.0, 1.0),
        )
        vec = node.to_feature_vector()
        # One-hot: only index 4 (LEGEND) should be 1.0
        assert vec[GraphNodeType.LEGEND] == 1.0
        assert sum(vec[:9]) == 1.0

    def test_feature_vector_embedding_pad_zeros(self):
        """If include_embedding but no embedding set, pads with zeros."""
        node = GraphNode(
            node_type=GraphNodeType.NOTES,
            page=0,
            bbox=(0.0, 0.0, 1.0, 1.0),
            text_embedding=None,
        )
        vec = node.to_feature_vector(include_embedding=True)
        assert vec.shape == (398,)
        assert np.allclose(vec[14:], 0.0)


# ── Entity extraction ─────────────────────────────────────────────────


class TestExtractEntities:
    def test_extract_details(self):
        ent = _extract_entities("See detail 5/A-101 and 3/S-200")
        assert "5/A-101" in ent["details"]
        assert "3/S-200" in ent["details"]

    def test_extract_abbreviations(self):
        ent = _extract_entities("HVAC system with ADA compliance")
        assert "HVAC" in ent["abbreviations"]
        assert "ADA" in ent["abbreviations"]

    def test_common_words_filtered(self):
        ent = _extract_entities("THE AND FOR NOT ALL")
        assert "THE" not in ent["abbreviations"]
        assert "AND" not in ent["abbreviations"]

    def test_empty_text(self):
        ent = _extract_entities("")
        assert ent["details"] == set()
        assert ent["abbreviations"] == set()


# ── build_document_graph ──────────────────────────────────────────────


class TestBuildDocumentGraph:
    def test_empty_input(self):
        result = build_document_graph([])
        assert result["num_nodes"] == 0
        assert result["num_edges"] == 0
        assert result["x"].shape == (0, 14)

    def test_single_page_single_region(self):
        pr = MockPageResult(
            page=0,
            notes_columns=[MockRegion(bbox=(100, 200, 300, 400), text="General notes")],
        )
        result = build_document_graph([pr])
        assert result["num_nodes"] == 1
        assert result["x"].shape == (1, 14)

    def test_same_page_edges(self):
        """Nodes on the same page should be connected."""
        pr = MockPageResult(
            page=0,
            notes_columns=[MockRegion(text="notes 1"), MockRegion(text="notes 2")],
            title_blocks=[MockRegion(text="sheet A-101")],
        )
        result = build_document_graph([pr])
        assert result["num_nodes"] == 3
        # Each pair gets 2 directed edges = 3 pairs * 2 = 6 same-page edges
        assert result["num_edges"] >= 6

    def test_shared_entity_edges(self):
        """Nodes mentioning the same abbreviation get SHARED_ENTITY edges."""
        pr1 = MockPageResult(
            page=0,
            notes_columns=[MockRegion(text="HVAC system described here")],
        )
        pr2 = MockPageResult(
            page=1,
            abbreviation_regions=[
                MockRegion(text="HVAC - Heating Ventilation Air Conditioning")
            ],
        )
        result = build_document_graph([pr1, pr2])
        assert result["num_nodes"] == 2
        # Should have shared entity edges
        edge_types = result["edge_attr"]
        assert EdgeType.SHARED_ENTITY in edge_types

    def test_page_sequence_edges(self):
        """Adjacent-page title blocks get PAGE_SEQUENCE edges."""
        pr1 = MockPageResult(page=0, title_blocks=[MockRegion(text="Sheet 1")])
        pr2 = MockPageResult(page=1, title_blocks=[MockRegion(text="Sheet 2")])
        result = build_document_graph([pr1, pr2])
        edge_types = result["edge_attr"]
        assert EdgeType.PAGE_SEQUENCE in edge_types

    def test_no_page_sequence_for_non_adjacent(self):
        """Non-adjacent pages shouldn't get PAGE_SEQUENCE edges."""
        pr1 = MockPageResult(page=0, title_blocks=[MockRegion(text="Sheet 1")])
        pr2 = MockPageResult(page=5, title_blocks=[MockRegion(text="Sheet 6")])
        result = build_document_graph([pr1, pr2])
        edge_types = result["edge_attr"]
        has_seq = EdgeType.PAGE_SEQUENCE in edge_types
        assert not has_seq

    def test_multiple_pages_multiple_regions(self):
        """Smoke test with realistic multi-page data."""
        pages = []
        for i in range(3):
            pr = MockPageResult(
                page=i,
                notes_columns=[MockRegion(text=f"General notes page {i}")],
                title_blocks=[MockRegion(text=f"Sheet A-{100 + i}")],
                legend_regions=[MockRegion(text="HVAC Equipment Legend")],
            )
            pages.append(pr)
        result = build_document_graph(pages)
        assert result["num_nodes"] == 9  # 3 pages * 3 regions
        assert result["num_edges"] > 0

    def test_feature_matrix_dtype(self):
        pr = MockPageResult(
            page=0,
            notes_columns=[MockRegion(text="test")],
        )
        result = build_document_graph([pr])
        assert result["x"].dtype == np.float32

    def test_edge_index_shape(self):
        pr = MockPageResult(
            page=0,
            notes_columns=[MockRegion(text="a"), MockRegion(text="b")],
        )
        result = build_document_graph([pr])
        assert result["edge_index"].shape[0] == 2  # [2, E]


# ── GNN config ────────────────────────────────────────────────────────


class TestGNNConfig:
    def test_defaults(self):
        cfg = GroupingConfig()
        assert cfg.ml_gnn_enabled is False
        assert cfg.ml_gnn_model_path == "data/document_gnn.pt"
        assert cfg.ml_gnn_hidden_dim == 64

    def test_hidden_dim_validation(self):
        with pytest.raises(ValueError):
            GroupingConfig(ml_gnn_hidden_dim=0)

    def test_custom_values(self):
        cfg = GroupingConfig(
            ml_gnn_enabled=True,
            ml_gnn_model_path="models/my_gnn.pt",
            ml_gnn_hidden_dim=128,
        )
        assert cfg.ml_gnn_enabled is True
        assert cfg.ml_gnn_model_path == "models/my_gnn.pt"
        assert cfg.ml_gnn_hidden_dim == 128


# ── GNN availability ─────────────────────────────────────────────────


class TestGNNAvailability:
    def test_returns_bool(self):
        assert isinstance(is_gnn_available(), bool)


# ── predict_with_gnn ──────────────────────────────────────────────────


class TestPredictWithGNN:
    def test_returns_none_when_no_model_file(self, tmp_path):
        graph = {
            "x": np.random.randn(5, 14).astype(np.float32),
            "edge_index": np.array([[0, 1], [1, 0]], dtype=np.int64),
        }
        result = predict_with_gnn(graph, model_path=tmp_path / "nonexistent.pt")
        assert result is None

    def test_empty_graph(self, tmp_path):
        graph = {
            "x": np.zeros((0, 14), dtype=np.float32),
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "pyg_data": None,
        }
        # Should not crash, even though model doesn't exist
        result = predict_with_gnn(graph, model_path=tmp_path / "nonexistent.pt")
        assert result is None


# ── EdgeType ──────────────────────────────────────────────────────────


class TestEdgeType:
    def test_values(self):
        assert EdgeType.SAME_PAGE == 0
        assert EdgeType.SHARED_ENTITY == 1
        assert EdgeType.PAGE_SEQUENCE == 2
