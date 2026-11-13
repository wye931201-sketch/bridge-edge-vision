from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridge_edge.damage_gallery import select_bridge_groups
from bridge_edge.industrial import IndustrialDetector, PALETTE
from bridge_edge.runtime.engine import annotate
from bridge_edge.utils import write_json


class DamageGalleryTests(unittest.TestCase):
    def test_group_selection_uses_annotations_and_keeps_images_distinct(self):
        records = [{"image": name, "source_path": name, "source_labels": labels, "split": split}
                   for name, labels, split in (("c.jpg", ["Rust"], "val"), ("a.jpg", ["Crack", "Spalling", "Rust"], "val"),
                                               ("b.jpg", ["Spalling", "Rust"], "val"), ("0.jpg", ["Crack"], "train"))]
        groups = select_bridge_groups(records, 1)
        self.assertEqual([(name, items[0]["image"]) for name, items in groups], [("裂缝", "a.jpg"), ("剥落", "b.jpg"), ("锈蚀", "c.jpg")])
        with self.assertRaises(ValueError):
            select_bridge_groups(records, 2)

    def test_changed_industrial_weight_is_rejected_before_runtime_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "models/external").mkdir(parents=True)
            (root / "models/external/model.onnx").write_bytes(b"changed model")
            write_json(root / "industrial_sources.json", {"model": {"filename": "model.onnx", "sha256": "expected-hash"}})
            with patch("bridge_edge.industrial.CONFIG_ROOT", root), patch("bridge_edge.industrial.location", return_value=root / "models"):
                with self.assertRaisesRegex(ValueError, "Industrial model bytes differ"):
                    IndustrialDetector()

    def test_scratch_annotation_uses_its_own_taxonomy_palette(self):
        finding = {"class_id": 5, "label": "scratches", "confidence": .9, "xyxy": [10, 10, 30, 30]}
        rendered = annotate(np.zeros((50, 50, 3), dtype=np.uint8), [finding], palette=PALETTE)
        self.assertEqual(rendered.getpixel((10, 30)), PALETTE[5])


if __name__ == "__main__":
    unittest.main()
