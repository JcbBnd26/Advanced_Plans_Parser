"""Shared test fixtures for Advanced Plan Parser."""

import pytest

from plancheck.config import GroupingConfig
from plancheck.models import BlockCluster, GlyphBox, Line, RowBand, Span

# ── Helpers ────────────────────────────────────────────────────────────


def make_box(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    text: str = "",
    page: int = 0,
    origin: str = "text",
) -> GlyphBox:
    """Create a GlyphBox with sane defaults."""
    return GlyphBox(page=page, x0=x0, y0=y0, x1=x1, y1=y1, text=text, origin=origin)


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def default_cfg() -> GroupingConfig:
    """Return a default GroupingConfig."""
    return GroupingConfig()


@pytest.fixture
def simple_boxes() -> list[GlyphBox]:
    """Return a small list of horizontally-arranged word boxes on one line.

    Layout (approx):
        "HELLO"   "WORLD"   "TEST"
        (10,100)  (80,100)  (150,100)
    All tokens have height 12 (y0=100, y1=112).
    """
    return [
        make_box(10, 100, 60, 112, "HELLO"),
        make_box(80, 100, 130, 112, "WORLD"),
        make_box(150, 100, 200, 112, "TEST"),
    ]


@pytest.fixture
def two_line_boxes() -> list[GlyphBox]:
    """Return boxes on two distinct lines separated by a baseline gap.

    Line 1 (y=100): "GENERAL" "NOTES:"
    Line 2 (y=120): "1." "ALL" "WORK"
    """
    return [
        make_box(50, 100, 120, 112, "GENERAL"),
        make_box(130, 100, 190, 112, "NOTES:"),
        make_box(50, 120, 65, 132, "1."),
        make_box(70, 120, 100, 132, "ALL"),
        make_box(110, 120, 150, 132, "WORK"),
    ]


@pytest.fixture
def multi_block_boxes() -> list[GlyphBox]:
    """Return boxes forming two distinct blocks with a large vertical gap.

    Block 1 (y~100):
        "GENERAL" "NOTES:"
        "1." "SEE" "PLANS"
    Block 2 (y~300):
        "LEGEND"
        "SYMBOL" "MEANING"
    """
    return [
        # Block 1
        make_box(50, 100, 130, 112, "GENERAL"),
        make_box(140, 100, 200, 112, "NOTES:"),
        make_box(50, 118, 65, 130, "1."),
        make_box(70, 118, 105, 130, "SEE"),
        make_box(110, 118, 160, 130, "PLANS"),
        # Block 2 — large gap
        make_box(50, 300, 120, 312, "LEGEND"),
        make_box(50, 318, 110, 330, "SYMBOL"),
        make_box(150, 318, 230, 330, "MEANING"),
    ]


@pytest.fixture
def overlapping_boxes() -> list[GlyphBox]:
    """Return overlapping boxes for NMS pruning tests."""
    return [
        make_box(10, 10, 50, 30, "A"),  # area 800
        make_box(12, 11, 48, 29, "B"),  # heavily overlaps A
        make_box(100, 100, 150, 120, "C"),  # no overlap
    ]
