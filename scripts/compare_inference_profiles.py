"""Evaluate every validation image before choosing a multiscale inference plan."""
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_edge.evaluation import read_yolo_boxes, summarize_detection_records
from bridge_edge.paths import location
from bridge_edge.runtime.engine import Detector, decode_image, latest_weights
from bridge_edge.runtime.inspection import inspect_image, PROFILES
from bridge_edge.utils import read_json, sha256, utc_now, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=["legacy", "detail", "detail-contrast"])
    parser.add_argument("--confidence", type=float, default=.15)
    args = parser.parse_args()
    manifest = location("prepared") / "pilot/manifest.json"
    entries = sorted((r for r in read_json(manifest)["records"] if r["split"] == "val"), key=lambda r: r["image"])
    engine = Detector(latest_weights().with_suffix(".onnx"), device="cpu")
    results = []
    for profile in args.profiles:
        records = []
        for index, entry in enumerate(entries, 1):
            source = location("raw") / entry["source_path"]
            if sha256(source) != entry["sha256"]:
                raise ValueError("Validation source bytes changed")
            image = decode_image(source.read_bytes())
            result = inspect_image(engine, image, profile=profile, confidence=args.confidence)
            labels = location("prepared") / "pilot" / entry["label"]
            if sha256(labels) != entry["label_sha256"]:
                raise ValueError("Validation label bytes changed")
            records.append({"image": source.name, "detections": result["detections"],
                            "ground_truth": read_yolo_boxes(labels, image.shape[1], image.shape[0]),
                            "latency_ms": result["total_latency_ms"], "passes": result["tile_count"]})
            write_json(location("reports") / "inference_profile_progress.json",
                       {"profile": profile, "completed": index, "total": len(entries)})
            print(f"{profile}: {index}/{len(entries)}", flush=True)
        summary = summarize_detection_records(records)
        summary.update({"profile": profile, "latency_ms_p50": float(np.median([r["latency_ms"] for r in records])),
                        "mean_inference_passes": float(np.mean([r["passes"] for r in records]))})
        results.append(summary)
        write_json(location("reports") / ("inference-profile-" + profile + ".json"),
                   {"created_utc": utc_now(), "weights_sha256": engine.weights_sha256,
                    "dataset_manifest_sha256": sha256(manifest), "confidence": args.confidence,
                    "summary": summary, "records": records})
    best = max(results, key=lambda r: (r["micro"]["f1"], -r["latency_ms_p50"]))
    report = {"created_utc": utc_now(), "passed": True, "weights_sha256": engine.weights_sha256,
              "dataset_manifest_sha256": sha256(manifest), "confidence": args.confidence, "profiles": results,
              "selected_profile": best["profile"], "selection_metric": "micro F1 on the fixed validation subset",
              "independent_test_evaluated": False}
    write_json(location("reports") / "inference_profile_comparison.json", report)
    write_json(ROOT / "artifacts/experiments/inference_profiles.json", report)
    print(report)


if __name__ == "__main__":
    main()
