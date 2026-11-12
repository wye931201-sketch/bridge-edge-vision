from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridge_edge.store import get_prediction, list_predictions, record_prediction
from bridge_edge.runtime.engine import decode_image, available_models, Detector
from bridge_edge.utils import write_json, sha256


class LocalStorageTests(unittest.TestCase):
    def test_history_round_trip_with_sql_characters(self):
        with tempfile.TemporaryDirectory() as temporary, patch("bridge_edge.store.workspace", return_value=Path(temporary)):
            data = {"id": "a" * 32, "created_utc": "2026-01-01T00:00:00+00:00", "model": "pilot.pt",
                    "source_filename": "bridge'; DROP TABLE predictions;--.jpg", "finding_count": 2,
                    "total_latency_ms": 20.5, "detections": []}
            record_prediction(data)
            self.assertEqual(get_prediction(data["id"]), data)
            self.assertEqual(len(list_predictions()), 1)
            self.assertEqual(list_predictions()[0]["finding_count"], 2)
            self.assertIsNone(get_prediction("missing"))

    def test_invalid_image_content(self):
        with self.assertRaises(ValueError):
            decode_image(b"not an image")
        with self.assertRaises(ValueError):
            decode_image(b"")
        with self.assertRaises(ValueError):
            decode_image(b"\xff\xd8\xffbroken JPEG")

    def test_pagination_bounds(self):
        for limit, offset in ((0, 0), (101, 0), (10, -1)):
            with self.assertRaises(ValueError):
                list_predictions(limit, offset)

    def test_model_registry_hides_previous_dataset_and_unpaired_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def storage(name):
                path = root / name
                path.mkdir(exist_ok=True)
                return path
            manifest = write_json(storage("prepared") / "pilot" / "manifest.json", {"records": []})
            for name, digest in (("current", sha256(manifest)), ("old", "previous-dataset")):
                (storage("models") / (name + ".pt")).write_bytes(b"fixture, never loaded as weights")
                (storage("models") / (name + ".onnx")).write_bytes(b"fixture")
                write_json(storage("models") / (name + ".json"), {"status": "passed", "run_name": "pilot-" + name,
                           "dataset_scope": "pilot", "dataset_manifest_sha256": digest})
                write_json(storage("models") / (name + ".export.json"), {"passed": True})
            (storage("models") / "unpaired.onnx").write_bytes(b"fixture")
            with patch("bridge_edge.runtime.engine.location", side_effect=storage):
                self.assertEqual({item["id"] for item in available_models()}, {"current.pt", "current.onnx"})

    def test_changed_checkpoint_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "model.pt"
            weights.write_bytes(b"changed checkpoint")
            write_json(weights.with_suffix(".json"), {"weights": {"sha256": "recorded-original-hash"}})
            with patch("bridge_edge.runtime.engine.load_detector") as loader:
                with self.assertRaisesRegex(ValueError, "Checkpoint bytes differ"):
                    Detector(weights, device="cpu")
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
