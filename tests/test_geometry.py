from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridge_edge.runtime.geometry import class_aware_nms, decode_yolo, iou, letterbox, tile_origins


class GeometryTests(unittest.TestCase):
    def test_letterbox_portrait_and_landscape(self):
        for height, width in ((100, 200), (200, 100), (101, 301)):
            tensor, ratio, padding = letterbox(np.zeros((height, width, 3), dtype=np.uint8), 384)
            self.assertEqual(tensor.shape, (1, 3, 384, 384))
            self.assertEqual(tensor.dtype, np.float32)
            self.assertGreater(ratio, 0)
            self.assertTrue(all(number >= 0 for number in padding))

    def test_nms_preserves_overlapping_different_classes(self):
        boxes = np.array([[0, 0, 20, 20], [1, 1, 21, 21], [0, 0, 20, 20]], dtype=float)
        keep = class_aware_nms(boxes, np.array([0.9, 0.8, 0.7]), np.array([0, 0, 1]), 0.5)
        self.assertEqual(keep, [0, 2])

    def test_nms_empty_and_non_overlapping(self):
        self.assertEqual(class_aware_nms(np.empty((0, 4)), np.array([]), np.array([])), [])
        self.assertEqual(class_aware_nms(np.array([[0, 0, 1, 1], [4, 4, 5, 5]]), np.array([0.5, 0.9]), np.array([0, 0])), [1, 0])

    def test_iou_reference(self):
        values = iou(np.array([0, 0, 2, 2]), np.array([[0, 0, 2, 2], [1, 1, 3, 3], [3, 3, 4, 4]]))
        np.testing.assert_allclose(values, [1, 1 / 7, 0])

    def test_decode_unpads_coordinates(self):
        # Original 200x100 image letterboxed to 400x400: scale 2, top pad 100.
        raw = np.zeros((1, 9, 2), dtype=np.float32)
        raw[0, :4, 0] = [200, 200, 200, 100]
        raw[0, 4, 0] = 0.9
        boxes = decode_yolo(raw, (100, 200, 3), 2, (0, 100))
        self.assertEqual(len(boxes), 1)
        np.testing.assert_allclose(boxes[0, :4], [50, 25, 150, 75])

    def test_decode_empty(self):
        self.assertEqual(decode_yolo(np.zeros((1, 9, 100)), (100, 100), 1, (0, 0)).shape, (0, 6))

    def test_reject_unexpected_export(self):
        with self.assertRaises(ValueError):
            decode_yolo(np.zeros((1, 84, 10)), (100, 100), 1, (0, 0))

    def test_nonfinite_predictions_are_not_reported_as_no_defect(self):
        raw = np.zeros((1, 9, 10)); raw[0, 4, 0] = np.nan
        with self.assertRaises(ValueError):
            decode_yolo(raw, (100, 100), 1, (0, 0))

    def test_tile_coverage_and_bounds(self):
        for length in (20, 384, 385, 1000, 1921):
            positions = tile_origins(length, 384, 0.2)
            self.assertEqual(positions[0], 0)
            self.assertEqual(positions[-1] + min(length, 384), length)
            covered = np.zeros(length, dtype=bool)
            for start in positions:
                covered[start:start + 384] = True
            self.assertTrue(covered.all())

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            tile_origins(1000, 384, 1)


if __name__ == "__main__":
    unittest.main()
