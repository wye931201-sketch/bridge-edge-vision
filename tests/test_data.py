from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridge_edge.data.download import _safe_relative, select_samples
from bridge_edge.data.prepare import convert_box, _copy_image, prepare_dataset
from bridge_edge.utils import sha256, write_json


class DatasetTests(unittest.TestCase):
    def test_box_conversion_reference(self):
        box, state = convert_box([0.1, 0.2, 0.4, 0.6], 1000, 500)
        self.assertEqual(state, "valid")
        for actual, expected in zip(box, [0.3, 0.5, 0.4, 0.6]):
            self.assertAlmostEqual(actual, expected)

    def test_clipped_box_is_recorded(self):
        box, state = convert_box([-0.2, 0.2, 0.5, 1.0], 100, 100)
        self.assertEqual(state, "clipped")
        self.assertAlmostEqual(box[2], 0.3)
        self.assertAlmostEqual(box[3], 0.8)

    def test_invalid_boxes_are_rejected(self):
        for value in ([0, 0, -1, 1], [0, 0, float("nan"), 1], [2, 2, 1, 1], [0, 0, 0.0001, 1]):
            self.assertIsNone(convert_box(value, 100, 100)[0])

    def test_path_traversal_is_rejected(self):
        for value in ("../keys.txt", "/tmp/secret.jpg", "data/../../secret.jpg", "other/a.jpg"):
            with self.assertRaises(ValueError):
                _safe_relative(value)
        self.assertEqual(_safe_relative("data/data_0/image.jpg"), Path("data/data_0/image.jpg"))

    def test_selection_is_order_independent_and_preserves_splits(self):
        source = [{"filepath": f"data/x/{split}{index}.jpg", "tags": [split]} for split in ("train", "validation", "testdev") for index in range(10)]
        selected, counts = select_samples(source, 5, 3, 42)
        reverse, _ = select_samples(list(reversed(source)), 5, 3, 42)
        self.assertEqual([(split, item["filepath"]) for split, item in selected], [(split, item["filepath"]) for split, item in reverse])
        self.assertEqual(counts["train"]["selected"], 5)
        self.assertEqual(counts["val"]["selected"], 3)
        self.assertFalse(any("testdev" in item["tags"] for _, item in selected))

    def test_full_split_selection(self):
        selected, _ = select_samples([{"filepath": "data/a.jpg", "tags": ["train"]}, {"filepath": "data/b.jpg", "tags": ["validation"]}], 0, 0, 1)
        self.assertEqual(len(selected), 2)

    def test_training_copy_preserves_original_and_normalizes_jpeg_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, prepared = root / "raw.jpg", root / "train.jpg"
            Image.new("RGB", (20, 12), (27, 43, 91)).save(raw)
            raw.write_bytes(raw.read_bytes() + b"trailing padding")
            original = raw.read_bytes()
            record = _copy_image(raw, prepared)
            self.assertEqual(raw.read_bytes(), original)
            self.assertTrue(prepared.read_bytes().endswith(b"\xff\xd9"))
            self.assertEqual(record["image_transformation"], "trim_bytes_after_jpeg_eoi")
            with Image.open(raw) as left, Image.open(prepared) as right:
                self.assertEqual(left.tobytes(), right.tobytes())
            prepared.write_bytes(b"third-party mutation")
            self.assertEqual(raw.read_bytes(), original)

    def test_ambiguous_exif_images_are_quarantined_with_original_splits_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def storage(name):
                path = root / name
                path.mkdir(exist_ok=True)
                return path
            records, samples = [], []
            for filename, split, orientation in (("a.jpg", "train", 1), ("rotated.jpg", "train", 6), ("b.jpg", "val", 1)):
                path = storage("raw") / "data" / filename
                path.parent.mkdir(exist_ok=True)
                exif = Image.Exif()
                exif[274] = orientation
                Image.new("RGB", (20, 12), (10 + len(records) * 40, 20, 30)).save(path, exif=exif)
                records.append({"source_path": f"data/{filename}", "split": split, "width": 20, "height": 12, "sha256": sha256(path)})
                samples.append({"filepath": f"data/{filename}"})
            write_json(storage("raw") / "download_manifest.json", {"records": records, "scope": "fixture", "source": {}, "selection": "fixture", "seed": 1})
            with patch("bridge_edge.data.prepare.location", side_effect=storage), patch("bridge_edge.data.prepare.metadata", return_value=samples):
                result = prepare_dataset("fixture")
            # Keep the annotation frame explicit instead of guessing box rotations.
            self.assertEqual(result["quarantined"][0]["reason"], "non_identity_exif_orientation")
            self.assertEqual(result["quarantined"][0]["source_path"], "data/rotated.jpg")
            self.assertEqual({item["split"] for item in result["records"]}, {"train", "val"})


if __name__ == "__main__":
    unittest.main()
