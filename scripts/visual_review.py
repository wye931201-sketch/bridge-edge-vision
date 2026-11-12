"""Render fixed validation examples beside source boxes for local visual review."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_edge.paths import location
from bridge_edge.runtime.engine import Detector, annotate, decode_image, latest_weights
from bridge_edge.taxonomy import BY_ID
from bridge_edge.utils import read_json, write_json


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.count <= 12:
        raise ValueError("Review between 1 and 12 fixed samples")
    base = location("prepared") / "pilot"
    records = sorted((item for item in read_json(base / "manifest.json")["records"]
                      if item["split"] == "val" and item["target_boxes"]), key=lambda item: item["image"])[:args.count]
    weights = latest_weights()
    if weights.with_suffix(".onnx").exists():
        weights = weights.with_suffix(".onnx")
    engine = Detector(weights, device="cpu")
    figure, axes = plt.subplots(len(records), 2, figsize=(12, len(records) * 4.2), squeeze=False, layout="constrained")
    for index, record in enumerate(records):
        image = decode_image((base / record["image"]).read_bytes())
        height, width = image.shape[:2]
        source = []
        for line in (base / record["label"]).read_text("utf-8").splitlines():
            label, x, y, w, h = map(float, line.split())
            source.append({"class_id": int(label), "label": BY_ID[int(label)].name, "confidence": 1,
                           "xyxy": [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]})
        result = engine.predict(image, confidence=.15)
        for column, detections, title in ((0, source, "Source boxes"), (1, result["detections"], "Predictions @ 0.15")):
            axes[index, column].imshow(annotate(image, detections))
            axes[index, column].set_title(f"{Path(record['image']).name} | {title} ({len(detections)})", fontsize=10)
            axes[index, column].axis("off")
    destination = location("reports") / "fixed_validation_visual_review.png"
    figure.savefig(destination, dpi=130)
    plt.close(figure)
    write_json(destination.with_suffix(".json"), {"selection": "first fixed validation filenames with any source box; not selected by prediction quality",
               "images": [Path(item["image"]).name for item in records], "model": weights.name,
               "scope": "local source-box and prediction visualization; not human-certified labels or a dataset accuracy estimate"})
    print(destination)


if __name__ == "__main__":
    main()
