"""Tests for plancheck.preprocess — NMS, skew, rotation, line clustering."""

import math

from conftest import make_box

from plancheck.preprocess import (
    _cluster_boxes_into_lines,
    _line_angle,
    _weighted_median,
    estimate_skew_degrees,
    intersection_over_union,
    nms_prune,
    rotate_boxes,
)


class TestIoU:
    def test_identical_boxes(self):
        a = make_box(0, 0, 10, 10)
        b = make_box(0, 0, 10, 10)
        assert intersection_over_union(a, b) == 1.0

    def test_no_overlap(self):
        a = make_box(0, 0, 10, 10)
        b = make_box(20, 20, 30, 30)
        assert intersection_over_union(a, b) == 0.0

    def test_partial_overlap(self):
        a = make_box(0, 0, 10, 10)
        b = make_box(5, 5, 15, 15)
        iou = intersection_over_union(a, b)
        # Intersection: 5x5=25, Union: 100+100-25=175
        assert abs(iou - 25 / 175) < 1e-9

    def test_contained_box(self):
        a = make_box(0, 0, 20, 20)  # area 400
        b = make_box(5, 5, 10, 10)  # area 25, fully inside
        iou = intersection_over_union(a, b)
        assert abs(iou - 25 / 400) < 1e-9


class TestNmsPrune:
    def test_no_overlap_keeps_all(self):
        boxes = [make_box(0, 0, 10, 10, "A"), make_box(20, 20, 30, 30, "B")]
        result = nms_prune(boxes, iou_threshold=0.5)
        assert len(result) == 2

    def test_overlapping_removes_smaller(self, overlapping_boxes):
        result = nms_prune(overlapping_boxes, iou_threshold=0.5)
        # A and B heavily overlap; larger is kept, smaller suppressed; C unaffected
        assert len(result) == 2
        texts = {b.text for b in result}
        assert "A" in texts
        assert "C" in texts

    def test_empty_input(self):
        assert nms_prune([], 0.5) == []

    def test_single_box(self):
        boxes = [make_box(0, 0, 10, 10, "X")]
        result = nms_prune(boxes, 0.5)
        assert len(result) == 1


class TestClusterBoxesIntoLines:
    def test_single_line(self):
        boxes = [
            make_box(10, 100, 30, 110),
            make_box(40, 100, 60, 110),
            make_box(70, 100, 90, 110),
        ]
        lines = _cluster_boxes_into_lines(boxes, y_tolerance=5.0)
        assert len(lines) == 1
        assert len(lines[0]) == 3

    def test_two_lines(self):
        boxes = [
            make_box(10, 100, 30, 110),
            make_box(40, 100, 60, 110),
            make_box(10, 130, 30, 140),
            make_box(40, 130, 60, 140),
        ]
        lines = _cluster_boxes_into_lines(boxes, y_tolerance=5.0)
        assert len(lines) == 2
        assert len(lines[0]) == 2
        assert len(lines[1]) == 2

    def test_empty(self):
        assert _cluster_boxes_into_lines([], y_tolerance=5.0) == []

    def test_sorted_by_x(self):
        """Within a line, boxes should be sorted by x0."""
        boxes = [
            make_box(80, 100, 100, 110),
            make_box(10, 100, 30, 110),
            make_box(40, 100, 60, 110),
        ]
        lines = _cluster_boxes_into_lines(boxes, y_tolerance=5.0)
        assert lines[0][0].x0 == 10
        assert lines[0][-1].x0 == 80


class TestLineAngle:
    def test_horizontal_line(self):
        boxes = [
            make_box(10, 100, 30, 110),
            make_box(40, 100, 60, 110),
            make_box(70, 100, 90, 110),
        ]
        angle = _line_angle(boxes)
        assert angle is not None
        assert abs(angle) < 0.01

    def test_tilted_line(self):
        # y increases with x → positive slope
        boxes = [
            make_box(10, 100, 30, 110),  # centre y ≈ 105
            make_box(60, 105, 80, 115),  # centre y ≈ 110
            make_box(110, 110, 130, 120),  # centre y ≈ 115
        ]
        angle = _line_angle(boxes)
        assert angle is not None
        assert angle > 0

    def test_too_few_boxes(self):
        boxes = [make_box(10, 100, 30, 110), make_box(40, 100, 60, 110)]
        assert _line_angle(boxes) is None

    def test_vertical_boxes(self):
        """All centres same x → returns None (zero denominator)."""
        boxes = [
            make_box(50, 100, 60, 110),
            make_box(50, 130, 60, 140),
            make_box(50, 160, 60, 170),
        ]
        assert _line_angle(boxes) is None


class TestWeightedMedian:
    def test_simple(self):
        assert _weighted_median([1, 2, 3], [1, 1, 1]) == 2

    def test_weighted(self):
        # weight on 3 is so heavy it becomes the median
        assert _weighted_median([1, 2, 3], [1, 1, 10]) == 3

    def test_empty(self):
        assert _weighted_median([], []) == 0.0

    def test_single(self):
        assert _weighted_median([5.0], [1.0]) == 5.0


class TestEstimateSkew:
    def test_zero_skew(self):
        # All boxes at same y → no skew
        boxes = [
            make_box(10, 100, 50, 112),
            make_box(60, 100, 110, 112),
            make_box(120, 100, 170, 112),
        ]
        skew = estimate_skew_degrees(boxes, max_degrees=5.0)
        assert abs(skew) < 0.01

    def test_clamped_to_max(self):
        # Create steeply tilted boxes
        boxes = [
            make_box(10, 100, 50, 112),
            make_box(60, 200, 110, 212),
            make_box(120, 300, 170, 312),
        ]
        skew = estimate_skew_degrees(boxes, max_degrees=3.0)
        assert abs(skew) <= 3.0

    def test_single_box_returns_zero(self):
        boxes = [make_box(10, 100, 50, 112)]
        assert estimate_skew_degrees(boxes, 5.0) == 0.0

    def test_regression_method_explicit(self):
        boxes = [
            make_box(10, 100, 50, 112),
            make_box(60, 100, 110, 112),
        ]
        skew = estimate_skew_degrees(boxes, 5.0, method="regression")
        assert abs(skew) < 0.1

    def test_line_median_method(self):
        """Build enough lines for line_median to engage."""
        all_boxes = []
        for row_idx in range(6):
            y = 100 + row_idx * 20
            for col_idx in range(5):
                x = 10 + col_idx * 40
                all_boxes.append(make_box(x, y, x + 30, y + 10))
        skew = estimate_skew_degrees(all_boxes, 5.0, method="line_median")
        assert abs(skew) < 0.1

    def test_ransac_method(self):
        boxes = [
            make_box(10, 100, 50, 112),
            make_box(60, 100, 110, 112),
            make_box(120, 100, 170, 112),
        ]
        skew = estimate_skew_degrees(boxes, 5.0, method="ransac", seed=42)
        assert abs(skew) < 0.5

    def test_auto_selects_line_median_when_enough_lines(self):
        """Auto method should switch to line_median with 5+ good lines."""
        all_boxes = []
        for row_idx in range(6):
            y = 100 + row_idx * 20
            for col_idx in range(4):
                x = 10 + col_idx * 60
                all_boxes.append(make_box(x, y, x + 40, y + 10))
        skew = estimate_skew_degrees(all_boxes, 5.0, method="auto")
        assert abs(skew) < 0.1

    def test_auto_falls_back_to_regression(self):
        """Auto method should fallback to regression with few boxes."""
        boxes = [
            make_box(10, 100, 50, 112),
            make_box(60, 100, 110, 112),
        ]
        skew = estimate_skew_degrees(boxes, 5.0, method="auto")
        assert abs(skew) < 0.1

    def test_outlier_robustness_line_median(self):
        """Line-median should ignore an outlier cluster."""
        # 5 straight lines
        all_boxes = []
        for row_idx in range(5):
            y = 100 + row_idx * 20
            for col_idx in range(4):
                x = 10 + col_idx * 60
                all_boxes.append(make_box(x, y, x + 40, y + 10))
        # Add outlier boxes with steep angle (stamp-like)
        all_boxes.append(make_box(400, 50, 430, 60))
        all_boxes.append(make_box(410, 90, 440, 100))
        all_boxes.append(make_box(420, 130, 450, 140))
        skew = estimate_skew_degrees(all_boxes, 5.0, method="line_median")
        # Should still be near zero — outliers are too few to form a heavy line
        assert abs(skew) < 0.5


class TestRotateBoxes:
    def test_zero_rotation_no_change(self):
        boxes = [make_box(10, 10, 50, 30, "A")]
        result = rotate_boxes(boxes, 0.0, 100, 100)
        assert len(result) == 1
        assert result[0].x0 == 10
        assert result[0].y0 == 10

    def test_small_rotation_below_threshold(self):
        boxes = [make_box(10, 10, 50, 30, "A")]
        result = rotate_boxes(boxes, 0.005, 100, 100, min_rotation=0.01)
        assert result[0].x0 == 10  # No rotation applied

    def test_180_rotation(self):
        boxes = [make_box(10, 10, 50, 30, "A")]
        result = rotate_boxes(boxes, 180, 100, 100)
        assert len(result) == 1
        # After 180° rotation around (50,50), (10,10,50,30) → (50,70,90,90)
        assert abs(result[0].x0 - 50) < 0.1
        assert abs(result[0].y0 - 70) < 0.1
