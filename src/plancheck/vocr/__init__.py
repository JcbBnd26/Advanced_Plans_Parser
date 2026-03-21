"""Visual OCR package exports.

The public helpers in this package are resolved lazily so importing a
specific VOCR submodule does not eagerly import the full candidate,
targeted, and graph-prior stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "compute_candidate_stats": (
        "plancheck.vocr.candidates",
        "compute_candidate_stats",
    ),
    "detect_vocr_candidates": (
        "plancheck.vocr.candidates",
        "detect_vocr_candidates",
    ),
    "extract_vocr_tokens": ("plancheck.vocr.extract", "extract_vocr_tokens"),
    "GNNCandidatePriorHead": (
        "plancheck.vocr.gnn_candidate_prior",
        "GNNCandidatePriorHead",
    ),
    "annotate_graph_with_candidates": (
        "plancheck.vocr.gnn_candidate_prior",
        "annotate_graph_with_candidates",
    ),
    "apply_gnn_prior": ("plancheck.vocr.gnn_candidate_prior", "apply_gnn_prior"),
    "assign_candidates_to_nodes": (
        "plancheck.vocr.gnn_candidate_prior",
        "assign_candidates_to_nodes",
    ),
    "load_gnn_candidate_prior": (
        "plancheck.vocr.gnn_candidate_prior",
        "load_gnn_candidate_prior",
    ),
    "save_gnn_candidate_prior": (
        "plancheck.vocr.gnn_candidate_prior",
        "save_gnn_candidate_prior",
    ),
    "train_gnn_candidate_prior": (
        "plancheck.vocr.gnn_candidate_prior",
        "train_gnn_candidate_prior",
    ),
    "get_adaptive_confidence": (
        "plancheck.vocr.method_stats",
        "get_adaptive_confidence",
    ),
    "load_method_stats": ("plancheck.vocr.method_stats", "load_method_stats"),
    "update_method_stats": (
        "plancheck.vocr.method_stats",
        "update_method_stats",
    ),
    "get_producer_confidence": (
        "plancheck.vocr.producer_stats",
        "get_producer_confidence",
    ),
    "load_producer_stats": (
        "plancheck.vocr.producer_stats",
        "load_producer_stats",
    ),
    "update_producer_stats": (
        "plancheck.vocr.producer_stats",
        "update_producer_stats",
    ),
    "extract_vocr_targeted": ("plancheck.vocr.targeted", "extract_vocr_targeted"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Resolve public VOCR exports lazily on first access."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module attributes including lazy public exports."""
    return sorted(set(globals()) | set(__all__))
