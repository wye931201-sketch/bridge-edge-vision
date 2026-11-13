from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridge_edge.crack_segmentation import CrackSegmenter, component_detections, replace_crack_candidates
from bridge_edge.runtime.inspection import BridgeInspector
from bridge_edge.utils import sha256, write_json


class CrackSegmentationTests(unittest.TestCase):
    def test_component_box_maps_to_original_pixels_and_mask_keeps_only_valid_regions(self):
        probability = np.zeros((10, 20), dtype=np.float32)
        probability[2:8, 4] = .9
        probability[1, 18] = .8
        detections, mask = component_detections(probability, (100, 200, 3), .4, 4, return_mask=True)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["xyxy"], [40., 20., 50., 80.])
        self.assertEqual(int(mask.sum()), 6)
        self.assertEqual(mask[1, 18], 0)
        self.assertAlmostEqual(detections[0]["confidence"], .9)

    def test_dense_regions_can_be_filtered_without_source_labels(self):
        probability = np.zeros((20, 20), dtype=np.float32)
        probability[2:8, 2:8] = .9
        self.assertEqual(component_detections(probability, (20, 20), .4, 4, max_fill_ratio=.6), [])

    def test_nonfinite_probability_is_rejected(self):
        with self.assertRaises(ValueError):
            component_detections(np.full((2, 2), np.nan), (20, 20))

    def test_replacement_retains_other_classes_and_does_not_serialize_a_mask_as_metadata(self):
        original = {"detections": [{"class_id": 0, "label": "crack", "xyxy": [0, 0, 10, 10], "confidence": .2},
                                    {"class_id": 3, "label": "rust", "xyxy": [20, 20, 30, 30], "confidence": .6}],
                    "total_latency_ms": 5}
        predicted = {"detections": [], "_native_crack_mask": np.zeros((2, 2), dtype=np.uint8), "total_latency_ms": 3}
        result = replace_crack_candidates(original, predicted)
        self.assertEqual([item["label"] for item in result["detections"]], ["rust"])
        self.assertNotIn("_native_crack_mask", result["crack_auxiliary"])
        self.assertEqual(result["total_latency_ms"], 8)

    def test_different_model_bytes_cannot_reuse_the_old_auxiliary_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def locate(name):
                path = root / name
                path.mkdir(exist_ok=True)
                return path
            manifest = write_json(locate("prepared") / "pilot/manifest.json", {"records": []})
            write_json(locate("reports") / "crack_auxiliary_evaluation.json",
                       {"enabled": True, "bridge_weights_sha256": "old-model", "dataset_manifest_sha256": sha256(manifest), "base_profile": "legacy"})
            class Engine:
                path = Path("new.onnx")
                weights_sha256 = "new-model"
            with patch("bridge_edge.paths.location", side_effect=locate), patch("bridge_edge.crack_segmentation.CrackSegmenter") as loader:
                inspector = BridgeInspector(Engine(), profile="legacy")
                self.assertIsNone(inspector.segmenter)
                loader.assert_not_called()
