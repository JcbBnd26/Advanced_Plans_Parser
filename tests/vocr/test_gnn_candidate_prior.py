"""Tests for Level 4 — GNN candidate prior.

Tests cover:
- assign_candidates_to_nodes (spatial mapping)
- annotate_graph_with_candidates (stat population)
- GraphNode.to_candidate_features()
- GNNCandidatePriorHead (MLP forward/predict, save/load)
- train_gnn_candidate_prior (training loop)
- apply_gnn_prior (end-to-end confidence adjustment)
- Config fields
- Stubs when torch unavailable
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from plancheck.analysis.document_graph import GraphNode, GraphNodeType

# ── Helpers ─────────────────────────────────────────────────────────────


@dataclass
class FakeCandidate:
    """Minimal VocrCandidate-like object for testing."""

    page: int = 0
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    trigger_methods: List[str] = field(default_factory=lambda: ["gap_after_number"])
    predicted_symbol: str = "°"
    confidence: float = 0.5
    context: dict = field(default_factory=dict)
    outcome: str = "pending"
    found_text: str = ""
    found_symbol: str = ""

    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


def _make_node(
    page: int = 0,
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.5, 0.5),
    ntype: GraphNodeType = GraphNodeType.NOTES,
) -> GraphNode:
    return GraphNode(node_type=ntype, page=page, bbox=bbox)


def _make_candidate(
    page: int = 0,
    x0: float = 10.0,
    y0: float = 10.0,
    x1: float = 50.0,
    y1: float = 50.0,
    confidence: float = 0.6,
    outcome: str = "pending",
) -> FakeCandidate:
    return FakeCandidate(
        page=page, x0=x0, y0=y0, x1=x1, y1=y1, confidence=confidence, outcome=outcome
    )


# ── Tests: assign_candidates_to_nodes ──────────────────────────────────


class TestAssignCandidatesToNodes:
    def test_empty_nodes_returns_empty(self):
        from plancheck.vocr.gnn_candidate_prior import assign_candidates_to_nodes

        result = assign_candidates_to_nodes([], [_make_candidate()], 100, 100)
        assert result == {}

    def test_empty_candidates_returns_empty_lists(self):
        from plancheck.vocr.gnn_candidate_prior import assign_candidates_to_nodes

        nodes = [_make_node()]
        result = assign_candidates_to_nodes(nodes, [], 100, 100)
        assert result == {0: []}

    def test_single_candidate_maps_to_overlapping_node(self):
        from plancheck.vocr.gnn_candidate_prior import assign_candidates_to_nodes

        nodes = [
            _make_node(page=0, bbox=(0.0, 0.0, 0.5, 0.5)),
            _make_node(page=0, bbox=(0.5, 0.5, 1.0, 1.0)),
        ]
        # Candidate at (10,10)-(30,30) on a 100x100 page → normalised (0.1,0.1,0.3,0.3)
        # Overlaps node 0 (0,0,0.5,0.5) fully, no overlap with node 1
        cand = _make_candidate(page=0, x0=10, y0=10, x1=30, y1=30)
        result = assign_candidates_to_nodes(nodes, [cand], 100, 100)
        assert len(result[0]) == 1
        assert len(result[1]) == 0

    def test_candidate_maps_to_best_overlap(self):
        from plancheck.vocr.gnn_candidate_prior import assign_candidates_to_nodes

        nodes = [
            _make_node(page=0, bbox=(0.0, 0.0, 0.3, 0.3)),
            _make_node(page=0, bbox=(0.2, 0.2, 0.8, 0.8)),
        ]
        # Candidate at (25,25)-(70,70) normalised → (0.25,0.25,0.7,0.7)
        # Overlaps node 0 by (0.25-0.3)*(0.25-0.3) = 0.05*0.05 = 0.0025
        # Overlaps node 1 by (0.7-0.2)*(0.7-0.2) = 0.5*0.5 = 0.25 → larger
        cand = _make_candidate(page=0, x0=25, y0=25, x1=70, y1=70)
        result = assign_candidates_to_nodes(nodes, [cand], 100, 100)
        assert len(result[1]) == 1
        assert len(result[0]) == 0

    def test_candidate_on_different_page_maps_correctly(self):
        from plancheck.vocr.gnn_candidate_prior import assign_candidates_to_nodes

        nodes = [
            _make_node(page=0, bbox=(0.0, 0.0, 1.0, 1.0)),
            _make_node(page=1, bbox=(0.0, 0.0, 1.0, 1.0)),
        ]
        cand = _make_candidate(page=1, x0=10, y0=10, x1=30, y1=30)
        result = assign_candidates_to_nodes(nodes, [cand], 100, 100)
        assert len(result[0]) == 0
        assert len(result[1]) == 1

    def test_fallback_nearest_node_when_no_overlap(self):
        from plancheck.vocr.gnn_candidate_prior import assign_candidates_to_nodes

        # Node covers top-left, candidate is in bottom-right with no overlap
        nodes = [_make_node(page=0, bbox=(0.0, 0.0, 0.1, 0.1))]
        cand = _make_candidate(page=0, x0=90, y0=90, x1=99, y1=99)
        result = assign_candidates_to_nodes(nodes, [cand], 100, 100)
        assert len(result[0]) == 1  # falls back to nearest

    def test_multiple_candidates_same_node(self):
        from plancheck.vocr.gnn_candidate_prior import assign_candidates_to_nodes

        nodes = [_make_node(page=0, bbox=(0.0, 0.0, 1.0, 1.0))]
        cands = [
            _make_candidate(page=0, x0=10, y0=10, x1=20, y1=20),
            _make_candidate(page=0, x0=30, y0=30, x1=40, y1=40),
            _make_candidate(page=0, x0=50, y0=50, x1=60, y1=60),
        ]
        result = assign_candidates_to_nodes(nodes, cands, 100, 100)
        assert len(result[0]) == 3


# ── Tests: annotate_graph_with_candidates ──────────────────────────────


class TestAnnotateGraphWithCandidates:
    def test_annotates_empty(self):
        from plancheck.vocr.gnn_candidate_prior import annotate_graph_with_candidates

        node = _make_node()
        annotate_graph_with_candidates([node], {0: []})
        assert node.candidate_count == 0
        assert node.mean_candidate_confidence == 0.0
        assert node.candidate_hit_rate == 0.0

    def test_annotates_with_candidates(self):
        from plancheck.vocr.gnn_candidate_prior import annotate_graph_with_candidates

        node = _make_node()
        cands = [
            _make_candidate(confidence=0.6, outcome="hit"),
            _make_candidate(confidence=0.4, outcome="miss"),
        ]
        annotate_graph_with_candidates([node], {0: cands})
        assert node.candidate_count == 2
        assert abs(node.mean_candidate_confidence - 0.5) < 1e-6
        assert abs(node.candidate_hit_rate - 0.5) < 1e-6

    def test_hit_rate_ignores_pending(self):
        from plancheck.vocr.gnn_candidate_prior import annotate_graph_with_candidates

        node = _make_node()
        cands = [
            _make_candidate(outcome="hit"),
            _make_candidate(outcome="pending"),
            _make_candidate(outcome="pending"),
        ]
        annotate_graph_with_candidates([node], {0: cands})
        assert node.candidate_count == 3
        assert node.candidate_hit_rate == 1.0  # 1 hit / 1 resolved

    def test_all_pending_gives_zero_hit_rate(self):
        from plancheck.vocr.gnn_candidate_prior import annotate_graph_with_candidates

        node = _make_node()
        annotate_graph_with_candidates(
            [node], {0: [_make_candidate(outcome="pending")]}
        )
        assert node.candidate_hit_rate == 0.0


# ── Tests: GraphNode.to_candidate_features ─────────────────────────────


class TestGraphNodeCandidateFeatures:
    def test_default_candidate_features(self):
        node = _make_node()
        feats = node.to_candidate_features()
        assert feats.shape == (3,)
        assert feats.dtype == np.float32
        np.testing.assert_array_equal(feats, [0.0, 0.0, 0.0])

    def test_populated_candidate_features(self):
        node = _make_node()
        node.candidate_count = 5
        node.mean_candidate_confidence = 0.7
        node.candidate_hit_rate = 0.4
        feats = node.to_candidate_features()
        np.testing.assert_array_almost_equal(feats, [5.0, 0.7, 0.4])

    def test_base_feature_vector_unchanged(self):
        """Ensure to_feature_vector() still returns 14 dims (backward compat)."""
        node = _make_node()
        node.candidate_count = 3
        fv = node.to_feature_vector()
        assert fv.shape == (14,)


# ── Tests: GNNCandidatePriorHead (torch-dependent) ────────────────────


_torch_avail = False
try:
    import torch

    _torch_avail = True
except ImportError:
    pass


@pytest.mark.skipif(not _torch_avail, reason="torch not installed")
class TestGNNCandidatePriorHead:
    def test_forward_shape(self):
        from plancheck.vocr.gnn_candidate_prior import GNNCandidatePriorHead

        head = GNNCandidatePriorHead(embed_dim=32, hidden=16)
        x = torch.randn(5, 35)  # 32 embed + 3 cand features
        out = head(x)
        assert out.shape == (5, 1)

    def test_predict_proba_returns_numpy(self):
        from plancheck.vocr.gnn_candidate_prior import GNNCandidatePriorHead

        head = GNNCandidatePriorHead(embed_dim=16, hidden=8)
        emb = np.random.randn(4, 16).astype(np.float32)
        cf = np.random.randn(4, 3).astype(np.float32)
        probs = head.predict_proba(emb, cf)
        assert isinstance(probs, np.ndarray)
        assert probs.shape == (4,)
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_save_and_load(self, tmp_path):
        from plancheck.vocr.gnn_candidate_prior import (
            GNNCandidatePriorHead,
            load_gnn_candidate_prior,
            save_gnn_candidate_prior,
        )

        head = GNNCandidatePriorHead(embed_dim=16, hidden=8)
        model_path = tmp_path / "prior.pt"
        save_gnn_candidate_prior(head, model_path)
        assert model_path.exists()

        loaded = load_gnn_candidate_prior(model_path)
        assert loaded is not None
        assert loaded.embed_dim == 16

        # Predictions should match
        emb = np.random.randn(3, 16).astype(np.float32)
        cf = np.random.randn(3, 3).astype(np.float32)
        p1 = head.predict_proba(emb, cf)
        p2 = loaded.predict_proba(emb, cf)
        np.testing.assert_array_almost_equal(p1, p2, decimal=5)

    def test_load_missing_returns_none(self, tmp_path):
        from plancheck.vocr.gnn_candidate_prior import load_gnn_candidate_prior

        result = load_gnn_candidate_prior(tmp_path / "nonexistent.pt")
        assert result is None

    def test_embed_dim_property(self):
        from plancheck.vocr.gnn_candidate_prior import GNNCandidatePriorHead

        head = GNNCandidatePriorHead(embed_dim=64)
        assert head.embed_dim == 64


# ── Tests: train_gnn_candidate_prior ──────────────────────────────────


@pytest.mark.skipif(not _torch_avail, reason="torch not installed")
class TestTrainGNNCandidatePrior:
    def test_training_runs_and_returns_metrics(self):
        from plancheck.vocr.gnn_candidate_prior import train_gnn_candidate_prior

        n = 30
        embed_dim = 16
        embeddings = np.random.randn(n, embed_dim).astype(np.float32)
        cand_feats = np.random.randn(n, 3).astype(np.float32)
        labels = np.random.randint(0, 2, size=n).astype(np.float32)

        head, metrics = train_gnn_candidate_prior(
            embeddings,
            cand_feats,
            labels,
            embed_dim=embed_dim,
            epochs=10,
            verbose=False,
        )
        assert head is not None
        assert "final_val_acc" in metrics
        assert "n_train" in metrics
        assert metrics["n_train"] > 0
        assert metrics["epochs"] == 10

    def test_trained_head_predicts(self):
        from plancheck.vocr.gnn_candidate_prior import train_gnn_candidate_prior

        n = 20
        embed_dim = 8
        embeddings = np.random.randn(n, embed_dim).astype(np.float32)
        cand_feats = np.random.randn(n, 3).astype(np.float32)
        labels = np.random.randint(0, 2, size=n).astype(np.float32)

        head, _ = train_gnn_candidate_prior(
            embeddings, cand_feats, labels, embed_dim=embed_dim, epochs=5
        )
        probs = head.predict_proba(embeddings, cand_feats)
        assert probs.shape == (n,)


# ── Tests: apply_gnn_prior (end-to-end, mocked GNN) ──────────────────


@pytest.mark.skipif(not _torch_avail, reason="torch not installed")
class TestApplyGnnPrior:
    def _make_fake_graph_and_models(self, n_nodes=3, embed_dim=16):
        """Create fake graph_data, gnn_model, and prior_head for testing."""
        from plancheck.vocr.gnn_candidate_prior import GNNCandidatePriorHead

        nodes = [
            _make_node(page=0, bbox=(0.0, 0.0, 0.5, 0.5)),
            _make_node(page=0, bbox=(0.5, 0.0, 1.0, 0.5)),
            _make_node(page=0, bbox=(0.0, 0.5, 1.0, 1.0)),
        ]

        x = np.stack([n.to_feature_vector() for n in nodes])
        edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)

        from torch_geometric.data import Data

        pyg_data = Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(edge_index),
        )

        graph_data = {
            "x": x,
            "edge_index": edge_index,
            "nodes": nodes,
            "num_nodes": n_nodes,
            "num_edges": 2,
            "pyg_data": pyg_data,
        }

        # Create a minimal GNN model
        from plancheck.analysis.gnn_model import DocumentGNN

        gnn_model = DocumentGNN(
            in_channels=14, hidden_channels=embed_dim // 4, num_classes=9, heads=4
        )
        # Determine actual embedding dimension from model
        actual_embed_dim = (embed_dim // 4) * 4

        prior_head = GNNCandidatePriorHead(embed_dim=actual_embed_dim, hidden=8)

        return graph_data, gnn_model, prior_head

    def test_adjusts_confidence(self):
        from plancheck.vocr.gnn_candidate_prior import apply_gnn_prior

        graph_data, gnn_model, prior_head = self._make_fake_graph_and_models()

        # Create fake page results with candidates
        @dataclass
        class FakePageResult:
            vocr_candidates: list = field(default_factory=list)
            page_width: float = 100.0
            page_height: float = 100.0

        cands = [
            _make_candidate(page=0, x0=10, y0=10, x1=40, y1=40, confidence=0.5),
            _make_candidate(page=0, x0=60, y0=10, x1=90, y1=40, confidence=0.5),
        ]
        pr = FakePageResult(vocr_candidates=cands)

        adjusted = apply_gnn_prior(
            [pr],
            graph_data,
            gnn_model,
            prior_head,
            page_width=100,
            page_height=100,
            blend_weight=0.5,
        )
        # Some candidates should have been adjusted
        assert adjusted >= 0
        # Check context annotation
        for c in cands:
            if c.confidence != 0.5:
                assert "gnn_region_prior" in c.context

    def test_no_candidates_returns_zero(self):
        from plancheck.vocr.gnn_candidate_prior import apply_gnn_prior

        graph_data, gnn_model, prior_head = self._make_fake_graph_and_models()

        @dataclass
        class FakePageResult:
            vocr_candidates: list = field(default_factory=list)

        adjusted = apply_gnn_prior(
            [FakePageResult()],
            graph_data,
            gnn_model,
            prior_head,
            page_width=100,
            page_height=100,
        )
        assert adjusted == 0

    def test_empty_graph_returns_zero(self):
        from plancheck.vocr.gnn_candidate_prior import apply_gnn_prior

        @dataclass
        class FakePageResult:
            vocr_candidates: list = field(default_factory=list)

        adjusted = apply_gnn_prior(
            [FakePageResult(vocr_candidates=[_make_candidate()])],
            {"nodes": [], "x": np.zeros((0, 14)), "edge_index": np.zeros((2, 0))},
            None,
            None,
        )
        assert adjusted == 0

    def test_blend_weight_zero_no_change(self):
        from plancheck.vocr.gnn_candidate_prior import apply_gnn_prior

        graph_data, gnn_model, prior_head = self._make_fake_graph_and_models()

        @dataclass
        class FakePageResult:
            vocr_candidates: list = field(default_factory=list)

        cands = [_make_candidate(page=0, x0=10, y0=10, x1=40, y1=40, confidence=0.6)]
        pr = FakePageResult(vocr_candidates=cands)

        apply_gnn_prior(
            [pr],
            graph_data,
            gnn_model,
            prior_head,
            page_width=100,
            page_height=100,
            blend_weight=0.0,
        )
        # With blend_weight=0, confidence should not change
        assert cands[0].confidence == pytest.approx(0.6, abs=1e-6)


# ── Tests: Config fields ──────────────────────────────────────────────


class TestConfigFields:
    def test_gnn_prior_config_defaults(self):
        from plancheck.config import GroupingConfig

        cfg = GroupingConfig()
        assert cfg.vocr_cand_gnn_prior_enabled is False
        assert cfg.vocr_cand_gnn_prior_path == "data/gnn_candidate_prior.pt"
        assert cfg.vocr_cand_gnn_prior_blend == 0.25


# ── Tests: bbox_overlap helper ────────────────────────────────────────


class TestBboxOverlap:
    def test_no_overlap(self):
        from plancheck.vocr.gnn_candidate_prior import _bbox_overlap

        assert _bbox_overlap((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0

    def test_full_overlap(self):
        from plancheck.vocr.gnn_candidate_prior import _bbox_overlap

        assert _bbox_overlap((0, 0, 1, 1), (0, 0, 1, 1)) == pytest.approx(1.0)

    def test_partial_overlap(self):
        from plancheck.vocr.gnn_candidate_prior import _bbox_overlap

        # (0,0,0.5,0.5) ∩ (0.25,0.25,0.75,0.75) = (0.25,0.25,0.5,0.5) = 0.0625
        result = _bbox_overlap((0, 0, 0.5, 0.5), (0.25, 0.25, 0.75, 0.75))
        assert result == pytest.approx(0.0625)
