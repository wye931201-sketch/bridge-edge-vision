from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridge_edge.evaluation import match_detections, read_yolo_boxes, summarize_detection_records
from bridge_edge.presentation import render_result_card, result_state
from bridge_edge.runtime.inspection import InspectionProfile, inference_views, inspect_image


def finding(class_id=0, box=(0, 0, 10, 10), confidence=.9):
    return {"class_id": class_id, "label": "crack" if class_id == 0 else "spalling", "chinese": "裂缝" if class_id == 0 else "剥落",
            "xyxy": list(box), "confidence": confidence}


class ViewTests(unittest.TestCase):
    def test_large_images_adapt_scale_to_a_fixed_budget_and_keep_full_coverage(self):
        profile = InspectionProfile("bounded", (384,), max_passes=16)
        views = inference_views((8000, 10000, 3), profile)
        tiles = [v for v in views if v["kind"] == "tile"]
        self.assertLessEqual(len(views), 16)
        self.assertGreater(tiles[0]["effective_size"], 384)
        self.assertEqual(min(v["x"] for v in tiles), 0)
        self.assertEqual(min(v["y"] for v in tiles), 0)
        self.assertEqual(max(v["x"] + v["width"] for v in tiles), 10000)
        self.assertEqual(max(v["y"] + v["height"] for v in tiles), 8000)

    def test_tiny_image_does_not_duplicate_the_global_view(self):
        views = inference_views((50, 60, 3), InspectionProfile("tiny", (768, 384)))
        self.assertEqual(len(views), 1)

    def test_invalid_profiles_are_rejected(self):
        for options in ({"tile_sizes": (100,)}, {"tile_sizes": (384, 384)},
                        {"tile_sizes": (), "include_global": False}, {"tile_sizes": (384,), "contrast_global": True, "max_passes": 1}):
            with self.assertRaises(ValueError):
                InspectionProfile("invalid", **options)

    def test_crop_coordinates_are_restored_and_duplicate_global_box_is_removed(self):
        class Engine:
            def predict(self, image, **kwargs):
                return {"detections": [finding()], "finding_count": 1}
        profile = InspectionProfile("tiles", (256,), overlap=.5, max_passes=8)
        image = np.zeros((300, 450, 3), dtype=np.uint8)
        result = inspect_image(Engine(), image, profile=profile)
        views = inference_views(image.shape, profile)
        expected = {(v["x"], v["y"], v["x"] + 10, v["y"] + 10) for v in views}
        self.assertEqual({tuple(x["xyxy"]) for x in result["detections"]}, expected)
        self.assertEqual(result["image_size"], {"width": 450, "height": 300})


class MatchingTests(unittest.TestCase):
    def test_duplicate_wrong_class_and_missed_target_have_separate_counts(self):
        targets = [finding(), finding(1, (20, 20, 30, 30))]
        predictions = [finding(), finding(confidence=.8), finding(0, (20, 20, 30, 30), .7)]
        report = summarize_detection_records([{"detections": predictions, "ground_truth": targets}])
        self.assertEqual(report["micro"]["true_positives"], 1)
        self.assertEqual(report["micro"]["false_positives"], 2)
        self.assertEqual(report["micro"]["false_negatives"], 1)
        self.assertAlmostEqual(report["micro"]["f1"], .4)

    def test_negative_images_contribute_false_positives(self):
        report = summarize_detection_records([{"detections": [finding()], "ground_truth": []}])
        self.assertEqual(report["micro"]["false_positives"], 1)

    def test_matching_is_score_ordered_and_one_to_one(self):
        matched = match_detections([finding(confidence=.3), finding(confidence=.9)], [finding()])
        self.assertEqual(matched["matches"][0]["prediction"], 1)
        self.assertEqual(matched["false_positive_indices"], [0])

    def test_label_roundoff_at_boundary_is_clipped_but_invalid_box_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.txt"
            path.write_text("0 0.1 0.5 0.20000001 0.5\n", "utf-8")
            self.assertEqual(read_yolo_boxes(path, 100, 100)[0]["xyxy"][0], 0)
            path.write_text("0 -0.2 0.5 0.2 0.5\n", "utf-8")
            with self.assertRaises(ValueError):
                read_yolo_boxes(path, 100, 100)


class PresentationTests(unittest.TestCase):
    def test_empty_result_has_visible_status_without_inventing_a_box(self):
        image = np.zeros((60, 100, 3), dtype=np.uint8)
        before = image.copy()
        rendered, metadata = render_result_card(image, [], category="裂缝")
        self.assertEqual(metadata["status"], "empty")
        self.assertEqual(metadata["finding_count"], 0)
        self.assertFalse(metadata["target_detected"])
        self.assertGreater(rendered.height, metadata["source_size"][1])
        self.assertIsNotNone(ImageChops.difference(rendered.crop((0, 0, rendered.width, metadata["header_height"])),
                                                Image.new("RGB", (rendered.width, metadata["header_height"]))).getbbox())
        photograph = rendered.crop((0, metadata["header_height"], rendered.width, rendered.height))
        self.assertIsNone(ImageChops.difference(photograph, Image.new("RGB", photograph.size)).getbbox())
        np.testing.assert_array_equal(before, image)

    def test_other_class_is_not_presented_as_a_successful_crack_detection(self):
        state = result_state([finding(1)], "crack")
        self.assertEqual(state["status"], "target_missed")
        self.assertFalse(state["target_detected"])
        self.assertEqual(state["detected_labels"], ["spalling"])

    def test_real_box_is_drawn_in_the_photo_coordinate_system(self):
        image = np.zeros((60, 100, 3), dtype=np.uint8)
        rendered, metadata = render_result_card(image, [finding(box=(20, 20, 40, 40))], category="裂缝")
        self.assertEqual(metadata["status"], "detected")
        self.assertNotEqual(rendered.getpixel((round(20 * metadata["display_scale"]),
                                             metadata["header_height"] + round(40 * metadata["display_scale"]))), (0, 0, 0))
