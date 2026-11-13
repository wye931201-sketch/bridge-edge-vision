from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridge_edge.comparisons import export_comparisons
from bridge_edge.utils import read_json, sha256, write_json


class FixtureDetector:
    path = Path("fixture.onnx")
    backend = "onnx"
    weights_sha256 = "fixture-hash"

    def __init__(self, *args, **kwargs):
        pass

    def predict(self, image, confidence, tile_size=0):
        return {"finding_count": 1, "detections": [{"class_id": 1, "label": "spalling", "chinese": "剥落",
                "confidence": .5, "xyxy": [2, 3, 20, 22]}]}


class ComparisonExportTests(unittest.TestCase):
    def storage(self, root):
        def locate(name):
            path = root / name
            path.mkdir(exist_ok=True)
            return path
        return locate

    def fixture(self, root, wrong_hash=False):
        locate = self.storage(root)
        records = []
        for name, split, color in (("b.png", "val", "gray"), ("train.png", "train", "blue"), ("a.png", "val", "white")):
            image = locate("raw") / name
            Image.new("RGB", (40, 30), color).save(image)
            records.append({"source_path": name, "image": "images/" + name, "split": split, "source_labels": ["Crack"] if name == "a.png" else ["Spalling"],
                            "sha256": "wrong-hash" if wrong_hash else sha256(image)})
        write_json(locate("prepared") / "pilot/manifest.json", {"records": records})
        return locate

    def test_numbered_pairs_keep_original_bytes_and_use_validation_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            locate = self.fixture(root)
            output = root / "对比"
            with patch("bridge_edge.comparisons.location", side_effect=locate), patch("bridge_edge.comparisons.Detector", FixtureDetector):
                report = export_comparisons(output=output, weights="fixture.onnx")
            self.assertEqual(report["pairs"], 2)
            self.assertEqual([item["source_filename"] for item in report["items"]], ["a.png", "b.png"])
            for number, filename in ((1, "a.png"), (2, "b.png")):
                folder = output / str(number)
                self.assertEqual({path.name for path in folder.iterdir()}, {"原始图片.png", "识别结果.jpg"})
                self.assertEqual((folder / "原始图片.png").read_bytes(), (root / "raw" / filename).read_bytes())
                with Image.open(folder / "识别结果.jpg") as image:
                    self.assertEqual(image.size, (40, 30))
                    image.verify()
            self.assertIn("第 1 组", (output / "打开对比.html").read_text("utf-8"))
            self.assertEqual(read_json(root / "reports/latest_comparison_export.json")["pairs"], 2)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "对比"
            output.mkdir()
            note = output / "keep.txt"
            note.write_text("existing user output", "utf-8")
            with patch("bridge_edge.comparisons.Detector") as detector:
                with self.assertRaises(FileExistsError):
                    export_comparisons(output=output)
                detector.assert_not_called()
            self.assertEqual(note.read_text("utf-8"), "existing user output")

    def test_crack_filter_uses_source_labels_not_model_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            locate = self.fixture(root)
            with patch("bridge_edge.comparisons.location", side_effect=locate), patch("bridge_edge.comparisons.Detector", FixtureDetector):
                result = export_comparisons(output=root / "裂缝对比", weights="fixture.onnx", classes=["crack"], tile_size=768)
            self.assertEqual(result["pairs"], 1)
            self.assertEqual(result["items"][0]["source_filename"], "a.png")
            self.assertEqual(result["items"][0]["detections"][0]["label"], "spalling")
            self.assertEqual(result["source_class_filters"], ["crack"])
            self.assertEqual(result["tile_size"], 768)

    def test_changed_original_does_not_publish_partial_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            locate = self.fixture(root, wrong_hash=True)
            original_hash = sha256(root / "raw/a.png")
            with patch("bridge_edge.comparisons.location", side_effect=locate), patch("bridge_edge.comparisons.Detector", FixtureDetector):
                with self.assertRaisesRegex(ValueError, "Original dataset image changed"):
                    export_comparisons(output=root / "对比", weights="fixture.onnx")
            self.assertFalse((root / "对比").exists())
            self.assertFalse(list(root.glob(".comparison-*")))
            self.assertEqual(sha256(root / "raw/a.png"), original_hash)


if __name__ == "__main__":
    unittest.main()
