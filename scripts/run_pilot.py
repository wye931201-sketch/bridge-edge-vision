"""Run the same-budget pilot comparison and retain all measured evidence."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bridge_edge.data.audit import audit_dataset
from bridge_edge.data.prepare import prepare_dataset
from bridge_edge.deployment import benchmark, export_onnx
from bridge_edge.paths import location
from bridge_edge.training import train
from bridge_edge.utils import read_json, sha256, utc_now, write_json, training_fingerprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-data", type=int, default=0, help="maximum seconds to await the active download")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--variants", nargs="+", default=["yolov8n", "repvit", "repvit-amsf"])
    parser.add_argument("--device", default="auto", help="auto, cpu, or CUDA device index")
    args = parser.parse_args()
    from bridge_edge.models.registry import configure_runtime
    configure_runtime()
    import torch
    device = ("0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    deadline = time.monotonic() + args.wait_for_data
    manifest_path = location("raw") / "download_manifest.json"
    progress_path = location("reports") / "pilot_progress.json"
    progress = {"started_at": utc_now(), "status": "awaiting_data", "runs": []}
    write_json(progress_path, progress)
    while not manifest_path.exists():
        if time.monotonic() > deadline:
            raise RuntimeError("Dataset download is not complete")
        time.sleep(2)
    prepared = location("prepared") / "pilot" / "manifest.json"
    if not prepared.exists() or read_json(prepared)["download_manifest_sha256"] != sha256(manifest_path):
        prepare_dataset("pilot")
    audit = audit_dataset("pilot")
    if not audit["passed"]:
        raise RuntimeError("Dataset audit failed; training was not started")
    progress["dataset"] = {"manifest_sha256": sha256(prepared), "audit_passed": True,
                           "stats": audit["stats"], "perceptual_review_candidates": len(audit["perceptual_cross_split_candidates"])}
    records = []
    for variant in args.variants:
        name = f"pilot-{variant}-s2026"
        recipe = ROOT / "configs" / "experiments" / f"pilot-{variant}.yaml"
        candidates = [read_json(path) for path in location("reports").glob(name + "*.json")]
        matching = [item for item in candidates if item.get("status") == "passed"
                    and item.get("architecture") == variant
                    and item.get("dataset_manifest_sha256") == sha256(prepared)
                    and item.get("configuration_sha256") == sha256(recipe)
                    and item.get("training_code_sha256") == training_fingerprint(ROOT)
                    and item.get("configuration", {}).get("epochs") == args.epochs
                    and str(item.get("configuration", {}).get("device")) == device
                    and item.get("execution", {}).get("completed_epochs") == args.epochs
                    and (location("models") / item.get("weights", {}).get("filename", "missing")).is_file()]
        existing = max(matching, key=lambda item: item["created_utc"]) if matching else {}
        if existing:
            record = existing
        else:
            attempt = 1
            while (location("runs") / name).exists():
                attempt += 1
                name = f"pilot-{variant}-s2026-r{attempt}"
            progress.update({"status": "training", "active_variant": variant, "active_run": name})
            write_json(progress_path, progress)
            record = train(f"experiments/pilot-{variant}.yaml", run_name=name, epochs=args.epochs, device=device)
        records.append(record)
        progress["runs"].append({"name": record["run_name"], "status": record["status"], "metrics": record["metrics"]["aggregate"]})
        write_json(progress_path, progress)
    best = max(records, key=lambda record: record["metrics"]["aggregate"]["metrics/mAP50-95(B)"])
    best_weights = location("models") / best["weights"]["filename"]
    write_json(location("models") / "latest.json", {"path": best_weights.name, "run_name": best["run_name"],
               "variant": best["architecture"], "scope": best["dataset_scope"], "selection_metric": "pilot validation mAP50-95"})
    validation = [item for item in read_json(prepared)["records"] if item["split"] == "val"]
    sample = location("prepared") / "pilot" / validation[0]["image"]
    progress.update({"status": "deployment_verification", "best_run": best["run_name"]})
    write_json(progress_path, progress)
    exported = export_onnx(best_weights, sample=sample)
    benchmarks = [benchmark(best_weights, sample, device=device), benchmark(location("models") / exported["onnx"], sample, device="cpu")]
    public_records = [{"run": record["run_name"], "architecture": record["architecture"], "epochs": record["configuration"]["epochs"],
                       "image_size": record["configuration"]["imgsz"], "seed": record["configuration"]["seed"],
                       "parameters": record["parameters"], "training_seconds": record["training_seconds"], "execution": record["execution"],
                       "metrics": record["metrics"], "initialization": record["initialization"], "weights": record["weights"]} for record in records]
    for public, record in zip(public_records, records):
        public["hardware"] = record["hardware"]
        public["data_transfer"] = record.get("data_transfer")
        public["optimizer_implementation"] = record.get("optimizer_implementation")
        public["configuration"] = record["configuration"]
        public["training_code_sha256"] = record.get("training_code_sha256")
        public["code_revision"] = record.get("code_revision")
    report = {"created_utc": utc_now(), "scope": f"single-seed pilot, {audit['stats']['train']['images']} train / {audit['stats']['val']['images']} validation images after quarantine",
              "independent_test_evaluated": False, "dataset": progress["dataset"], "experiments": public_records,
              "selected_by_validation": best["run_name"], "export": exported, "benchmarks": benchmarks,
              "jetson_hardware_verified": False, "paper_novelty_established": False}
    write_json(ROOT / "artifacts" / "experiments" / "pilot_results.json", report)
    progress.update({"status": "passed", "finished_at": utc_now()})
    write_json(progress_path, progress)
    print("Pilot workflow completed; measured results saved to artifacts/experiments/pilot_results.json", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        path = location("reports") / "pilot_progress.json"
        progress = read_json(path) if path.exists() else {}
        progress.update({"status": "failed", "error_type": type(error).__name__, "error": str(error), "finished_at": utc_now()})
        write_json(path, progress)
        raise
