"""Merging and deduplication logic for VOCR candidates."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from .helpers import _iou_bbox

if TYPE_CHECKING:
    from plancheck.models import VocrCandidate

__all__ = ["_merge_overlapping_candidates"]


def _merge_overlapping_candidates(
    candidates: List[VocrCandidate],
    iou_threshold: float = 0.5,
) -> List[VocrCandidate]:
    """Merge candidates whose bboxes overlap above *iou_threshold*.

    When two candidates overlap, the one with higher confidence survives
    and inherits all ``trigger_methods`` from the absorbed candidate.
    """
    if len(candidates) <= 1:
        return candidates

    # Sort by confidence descending so the best candidate absorbs others
    cands = sorted(candidates, key=lambda c: c.confidence, reverse=True)
    merged: List[VocrCandidate] = []
    absorbed = [False] * len(cands)

    for i in range(len(cands)):
        if absorbed[i]:
            continue
        keeper = cands[i]
        keeper_bbox = keeper.bbox()
        for j in range(i + 1, len(cands)):
            if absorbed[j]:
                continue
            other_bbox = cands[j].bbox()
            if _iou_bbox(keeper_bbox, other_bbox) >= iou_threshold:
                # Merge trigger methods
                for m in cands[j].trigger_methods:
                    if m not in keeper.trigger_methods:
                        keeper.trigger_methods.append(m)
                # Keep the higher confidence
                keeper.confidence = max(keeper.confidence, cands[j].confidence)
                # Prefer a predicted symbol if one has it and the other doesn't
                if not keeper.predicted_symbol and cands[j].predicted_symbol:
                    keeper.predicted_symbol = cands[j].predicted_symbol
                absorbed[j] = True
        merged.append(keeper)

    return merged
