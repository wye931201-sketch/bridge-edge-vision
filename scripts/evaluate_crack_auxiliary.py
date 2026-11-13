"""Validate a crack auxiliary on all images, including images without cracks."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_edge.crack_segmentation import CrackSegmenter, component_detections, replace_crack_candidates
from bridge_edge.evaluation import summarize_detection_records
from bridge_edge.paths import location
from bridge_edge.runtime.engine import decode_image, latest_weights
from bridge_edge.utils import read_json, sha256, utc_now, write_json


def main():
    comparison = read_json(location("reports") / "inference_profile_comparison.json")
    baseline = read_json(location("reports") / "inference-profile-legacy.json")
    chosen = comparison["selected_profile"]
    detector = read_json(location("reports") / ("inference-profile-" + chosen + ".json"))
    manifest = location("prepared") / "pilot/manifest.json"
    if detector["dataset_manifest_sha256"] != sha256(manifest):
        raise ValueError("Cached detector evaluation belongs to another dataset")
    current_weights = sha256(latest_weights().with_suffix(".onnx"))
    if baseline["weights_sha256"] != detector["weights_sha256"] or current_weights != detector["weights_sha256"]:
        raise ValueError("Cached evaluations belong to different model bytes")
    entries = {Path(item["image"]).name: item for item in read_json(manifest)["records"] if item["split"] == "val"}
    segmenter = CrackSegmenter()
    settings = [(threshold, area, fill) for threshold in (.4, .55, .7, .85)
                for area in (64, 128, 256, 512, 1024) for fill in (.25, .6, 1.)]
    records = {str(setting): [] for setting in settings}
    cache = location("cache") / ("crack-probability-rgb-resize-v1-" + segmenter.weights_sha256)
    cache.mkdir(exist_ok=True)
    for index, record in enumerate(detector["records"], 1):
        entry = entries[record["image"]]
        source = location("raw") / entry["source_path"]
        if sha256(source) != entry["sha256"]:
            raise ValueError("Source image changed")
        image = decode_image(source.read_bytes())
        cached = cache / (entry["sha256"] + ".npy")
        probability = np.load(cached, allow_pickle=False) if cached.is_file() else segmenter.probability(image)
        if not cached.exists():
            np.save(cached, probability, allow_pickle=False)
        threshold_boxes = {threshold: component_detections(probability, image.shape, threshold, 1)
                           for threshold in (.4, .55, .7, .85)}
        for threshold, area, fill in settings:
            detections = [d for d in threshold_boxes[threshold] if d["native_mask_area"] >= area and d["native_fill_ratio"] <= fill]
            auxiliary = {"detections": detections, "finding_count": len(detections), "total_latency_ms": 0}
            result = replace_crack_candidates(record, auxiliary)
            records[str((threshold, area, fill))].append({"image": record["image"], "detections": result["detections"],
                                                    "ground_truth": record["ground_truth"]})
        print(f"crack auxiliary: {index}/{len(detector['records'])}", flush=True)
    candidates = []
    for threshold, area, fill in settings:
        summary = summarize_detection_records(records[str((threshold, area, fill))])
        summary.update({"probability_threshold": threshold, "min_native_area": area, "max_fill_ratio": fill})
        candidates.append(summary)
    legacy = baseline["summary"]
    acceptable = [row for row in candidates if row["micro"]["f1"] >= legacy["micro"]["f1"]
                  and row["per_class"]["crack"]["f1"] > legacy["per_class"]["crack"]["f1"]
                  and row["per_class"]["crack"]["precision"] >= legacy["per_class"]["crack"]["precision"]]
    best = max(acceptable, key=lambda row: (row["per_class"]["crack"]["f1"], row["micro"]["f1"])) if acceptable else None
    report = {"created_utc": utc_now(), "passed": True, "enabled": best is not None,
              "dataset_manifest_sha256": sha256(manifest), "bridge_weights_sha256": detector["weights_sha256"],
              "auxiliary_weights_sha256": segmenter.weights_sha256, "base_profile": chosen,
              "legacy_summary": legacy, "detector_only_summary": detector["summary"], "candidates": candidates,
              "selected": best, "selection_metric": "highest validation crack F1 with micro F1 and crack precision at least as high as legacy; all parameter candidates retained",
              "evaluation_scope": "fixed pilot validation boxes at IoU 0.5; tuning, not independent test or segmentation mIoU",
              "model_origin": "single upstream ResNet34 U-Net seed42; not trained here and not the six-model ensemble"}
    write_json(location("reports") / "crack_auxiliary_evaluation.json", report)
    write_json(ROOT / "artifacts/experiments/crack_auxiliary.json", report)
    print(report)


if __name__ == "__main__":
    main()
