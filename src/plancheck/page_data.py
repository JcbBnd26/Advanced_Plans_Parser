"""Serialization helpers for the per-page pipeline output.

``serialize_page`` converts the in-memory output of ``build_clusters_v2`` +
``group_notes_columns`` into a JSON-friendly dict that can be written to
``page_N_extraction.json``.

``deserialize_page`` reconstructs lightweight ``BlockCluster`` and
``NotesColumn`` objects from that dict.  The reconstructed blocks have their
``_tokens`` field populated and ``rows`` synthesized via
``populate_rows_from_lines()`` so all existing overlay / drawing code
continues to work without modification.

JSON layout
-----------
::

    {
      "version": 1,
      "page": 2,
      "page_width": 2448.0,
      "page_height": 1584.0,
      "tokens": [ {GlyphBox.to_dict()}, ... ],
      "blocks": [ {BlockCluster.to_dict()}, ... ],
      "notes_columns": [ {NotesColumn.to_dict()}, ... ]
    }
"""

from __future__ import annotations

from typing import Any

from plancheck.models import BlockCluster, GlyphBox, NotesColumn

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def serialize_page(
    page: int,
    page_width: float,
    page_height: float,
    tokens: list[GlyphBox],
    blocks: list[BlockCluster],
    notes_columns: list[NotesColumn],
) -> dict[str, Any]:
    """Serialize a single page's pipeline output to a JSON-friendly dict."""
    return {
        "version": 1,
        "page": page,
        "page_width": round(page_width, 3),
        "page_height": round(page_height, 3),
        "tokens": [t.to_dict() for t in tokens],
        "blocks": [b.to_dict() for b in blocks],
        "notes_columns": [nc.to_dict(blocks) for nc in notes_columns],
    }


def deserialize_page(
    data: dict[str, Any],
) -> tuple[list[GlyphBox], list[BlockCluster], list[NotesColumn], float, float]:
    """Deserialize a page extraction JSON dict.

    Returns
    -------
    tokens : list[GlyphBox]
    blocks : list[BlockCluster]
        Each block has ``_tokens`` set and ``rows`` populated so bbox() and
        get_all_boxes() work the same as at pipeline runtime.
    notes_columns : list[NotesColumn]
    page_width : float
    page_height : float
    """
    tokens = [GlyphBox.from_dict(t) for t in data["tokens"]]
    blocks = [BlockCluster.from_dict(b, tokens) for b in data["blocks"]]
    notes_columns = [NotesColumn.from_dict(nc, blocks) for nc in data["notes_columns"]]
    return tokens, blocks, notes_columns, data["page_width"], data["page_height"]
