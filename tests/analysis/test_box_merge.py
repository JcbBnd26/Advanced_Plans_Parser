"""Tests for plancheck.analysis.box_merge — polygon-based box merging."""

from __future__ import annotations

import pytest

from plancheck.analysis.box_merge import (
    boxes_overlap,
    find_overlap_clusters,
    merge_boxes,
    merge_boxes_multi,
    polygon_bbox,
    simplify_polygon,
)

# ---------------------------------------------------------------------------
# merge_boxes — single polygon output
# ---------------------------------------------------------------------------


class TestMergeBoxes:
    def test_single_box_returns_rectangle(self) -> None:
        coords = merge_boxes([(0, 0, 10, 10)])
        # Shapely returns a closed ring (5 vertices, first == last)
        assert len(coords) == 5
        assert coords[0] == coords[-1]
        # Bounding box should match
        assert polygon_bbox(coords) == (0.0, 0.0, 10.0, 10.0)

    def test_identical_boxes_produce_rectangle(self) -> None:
        coords = merge_boxes([(0, 0, 10, 10), (0, 0, 10, 10)])
        assert polygon_bbox(coords) == (0.0, 0.0, 10.0, 10.0)
        assert len(coords) == 5

    def test_two_overlapping_boxes_l_shape(self) -> None:
        """Two partially overlapping boxes → polygon with >5 vertices."""
        #  ┌──────┐
        #  │  A   │
        #  │   ┌──┼──┐
        #  └───┼──┘  │
        #      │  B  │
        #      └─────┘
        a = (0, 0, 10, 10)
        b = (5, 5, 15, 15)
        coords = merge_boxes([a, b])
        # L-shape has 8 vertices (+ closing = 9, but shapely may vary depending on corner arrangement)
        # At minimum: more than a simple rectangle
        assert len(coords) > 5
        assert polygon_bbox(coords) == (0.0, 0.0, 15.0, 15.0)

    def test_three_boxes_cross_shape(self) -> None:
        """Three boxes forming a cross/plus."""
        #     ┌──┐
        #     │  │
        #  ┌──┼──┼──┐
        #  │  │  │  │
        #  └──┼──┼──┘
        #     │  │
        #     └──┘
        vert = (4, 0, 8, 16)
        horiz = (0, 5, 12, 11)
        coords = merge_boxes([vert, horiz])
        # Cross has 12 vertices + closing = 13
        assert len(coords) > 5
        bb = polygon_bbox(coords)
        assert bb == (0.0, 0.0, 12.0, 16.0)

    def test_t_shape(self) -> None:
        """Two boxes forming a T."""
        #  ┌──────────┐
        #  │   top    │
        #  └──┬────┬──┘
        #     │stem│
        #     └────┘
        top = (0, 0, 20, 5)
        stem = (7, 5, 13, 15)
        coords = merge_boxes([top, stem])
        assert len(coords) > 5
        assert polygon_bbox(coords) == (0.0, 0.0, 20.0, 15.0)

    def test_non_overlapping_returns_largest(self) -> None:
        """Two disjoint boxes → returns the larger one."""
        small = (0, 0, 5, 5)  # area 25
        big = (20, 20, 40, 40)  # area 400
        coords = merge_boxes([small, big])
        bb = polygon_bbox(coords)
        assert bb == (20.0, 20.0, 40.0, 40.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            merge_boxes([])

    def test_contained_box(self) -> None:
        """A box fully inside another → same as the outer box."""
        outer = (0, 0, 20, 20)
        inner = (5, 5, 15, 15)
        coords = merge_boxes([outer, inner])
        assert polygon_bbox(coords) == (0.0, 0.0, 20.0, 20.0)
        # Still a simple rectangle
        assert len(coords) == 5

    def test_adjacent_boxes_merge(self) -> None:
        """Two boxes sharing an edge → merged into one polygon."""
        left = (0, 0, 10, 10)
        right = (10, 0, 20, 10)
        coords = merge_boxes([left, right])
        assert polygon_bbox(coords) == (0.0, 0.0, 20.0, 10.0)
        # Shapely may keep the shared-edge vertex; simplify cleans it
        simplified = simplify_polygon(coords, tolerance=0.5)
        assert len(simplified) == 5

    def test_many_boxes_staircase(self) -> None:
        """Several boxes forming a staircase → complex polygon."""
        boxes = [(i * 5, i * 5, i * 5 + 10, i * 5 + 10) for i in range(5)]
        coords = merge_boxes(boxes)
        assert len(coords) > 5
        assert polygon_bbox(coords) == (0.0, 0.0, 30.0, 30.0)


# ---------------------------------------------------------------------------
# merge_boxes_multi — preserves disjoint groups
# ---------------------------------------------------------------------------


class TestMergeBoxesMulti:
    def test_disjoint_returns_two(self) -> None:
        polys = merge_boxes_multi([(0, 0, 5, 5), (20, 20, 30, 30)])
        assert len(polys) == 2

    def test_overlapping_returns_one(self) -> None:
        polys = merge_boxes_multi([(0, 0, 10, 10), (5, 5, 15, 15)])
        assert len(polys) == 1

    def test_empty_returns_empty(self) -> None:
        assert merge_boxes_multi([]) == []

    def test_three_groups(self) -> None:
        polys = merge_boxes_multi(
            [
                (0, 0, 5, 5),
                (20, 20, 25, 25),
                (50, 50, 55, 55),
            ]
        )
        assert len(polys) == 3


# ---------------------------------------------------------------------------
# polygon_bbox
# ---------------------------------------------------------------------------


class TestPolygonBbox:
    def test_basic(self) -> None:
        coords = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        assert polygon_bbox(coords) == (0, 0, 10, 10)

    def test_irregular(self) -> None:
        coords = [(1, 2), (5, 1), (8, 4), (3, 7), (1, 2)]
        assert polygon_bbox(coords) == (1, 1, 8, 7)


# ---------------------------------------------------------------------------
# simplify_polygon
# ---------------------------------------------------------------------------


class TestSimplifyPolygon:
    def test_rectangle_unchanged(self) -> None:
        coords = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        result = simplify_polygon(coords, tolerance=1.0)
        assert len(result) == 5

    def test_collinear_points_removed(self) -> None:
        # Rectangle with extra collinear midpoint on top edge
        coords = [(0, 0), (5, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        result = simplify_polygon(coords, tolerance=0.5)
        # Midpoint (5,0) is collinear and should be removed
        assert len(result) <= 5


# ---------------------------------------------------------------------------
# boxes_overlap
# ---------------------------------------------------------------------------


class TestBoxesOverlap:
    def test_overlap(self) -> None:
        assert boxes_overlap((0, 0, 10, 10), (5, 5, 15, 15)) is True

    def test_no_overlap(self) -> None:
        assert boxes_overlap((0, 0, 10, 10), (20, 20, 30, 30)) is False

    def test_touching_edge_no_overlap(self) -> None:
        # Touching edges only (no interior overlap)
        assert boxes_overlap((0, 0, 10, 10), (10, 0, 20, 10)) is False

    def test_contained(self) -> None:
        assert boxes_overlap((0, 0, 20, 20), (5, 5, 15, 15)) is True


# ---------------------------------------------------------------------------
# find_overlap_clusters
# ---------------------------------------------------------------------------


class TestFindOverlapClusters:
    def test_two_overlapping(self) -> None:
        clusters = find_overlap_clusters([(0, 0, 10, 10), (5, 5, 15, 15)])
        assert len(clusters) == 1
        assert sorted(clusters[0]) == [0, 1]

    def test_two_disjoint(self) -> None:
        clusters = find_overlap_clusters([(0, 0, 5, 5), (20, 20, 30, 30)])
        assert len(clusters) == 2
        assert [0] in clusters
        assert [1] in clusters

    def test_transitive(self) -> None:
        # A overlaps B, B overlaps C, but A doesn't overlap C
        clusters = find_overlap_clusters(
            [
                (0, 0, 10, 10),  # A
                (5, 5, 15, 15),  # B: overlaps A
                (10, 10, 20, 20),  # C: overlaps B but not A (touching corner only)
            ]
        )
        # A-B overlap, but B-C: (5,5,15,15) vs (10,10,20,20) → overlap
        # So all three should be in one cluster
        assert len(clusters) == 1
        assert sorted(clusters[0]) == [0, 1, 2]

    def test_empty(self) -> None:
        assert find_overlap_clusters([]) == []

    def test_single(self) -> None:
        clusters = find_overlap_clusters([(0, 0, 10, 10)])
        assert clusters == [[0]]

    def test_three_groups(self) -> None:
        clusters = find_overlap_clusters(
            [
                (0, 0, 5, 5),
                (3, 3, 8, 8),  # overlaps idx 0
                (50, 50, 60, 60),  # alone
                (100, 100, 110, 110),
                (105, 105, 115, 115),  # overlaps idx 3
            ]
        )
        assert len(clusters) == 3
