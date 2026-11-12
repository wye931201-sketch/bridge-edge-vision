"""Tracked training runs with explicit data, initialization and evaluation scope."""
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import yaml

from .models.registry import build_detector, load_detector, configure_runtime
from .paths import ROOT, location, resolve_config
from .utils import read_json, revision, sha256, utc_now, write_json, training_fingerprint


def _metrics(result):
    values = {str(key): float(value) for key, value in result.results_dict.items()}
    per_class = []
    for index, class_id in enumerate(result.box.ap_class_index):
        precision, recall, ap50, ap = result.box.class_result(index)
        per_class.append({"class_id": int(class_id), "precision": float(precision), "recall": float(recall),
                          "AP50": float(ap50), "AP50_95": float(ap)})
    return {"aggregate": values, "per_class_with_ground_truth": per_class, "timings_ms": result.speed}


def train(config_path="experiments/pilot-yolov8n.yaml", run_name=None, **overrides):
    config_path = resolve_config(config_path)
    config = yaml.safe_load(config_path.read_text("utf-8"))
    config.update({key: value for key, value in overrides.items() if value is not None})
    if "cuda_launch_blocking" in config:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1" if config["cuda_launch_blocking"] else "0"
    configure_runtime()
    import torch
    from .models.trainer import CheckedDetectionTrainer
    if config.get("device") == "auto":
        config["device"] = "0" if torch.cuda.is_available() else "cpu"
    seed = int(config.get("seed", 2026))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config.get("cpu_threads", 4)))
    torch.backends.cudnn.enabled = bool(config.get("cudnn_enabled", True))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    profile = config["dataset_profile"]
    data_root = location("prepared") / profile
    dataset_manifest = data_root / "manifest.json"
    audit = read_json(data_root / "audit.json")
    if not audit["passed"] or audit["manifest_sha256"] != sha256(dataset_manifest):
        raise RuntimeError("Dataset audit is missing, failed, or out of date; run data audit")
    data = read_json(dataset_manifest)
    name = run_name or (time.strftime("%Y%m%d-%H%M%S") + "-" + config["variant"])
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in name):
        raise ValueError("Run name must be a simple identifier")
    run_dir = location("runs") / name
    if run_dir.exists():
        raise FileExistsError(f"Run already exists: {name}. Use a new name to preserve experiments.")
    pretrained = bool(config.get("pretrained", True))
    detector = build_detector(config["variant"], pretrained=pretrained)
    initialization = {"pretrained": pretrained, "variant": config["variant"]}
    if pretrained:
        initial_name = "yolov8n.pt" if config["variant"] == "yolov8n" else "repvit_m0_9_imagenet.pth"
        initialization["initial_weights_sha256"] = sha256(location("models") / "initial" / initial_name)
    if config["variant"] == "repvit-amsf":
        # Same indices/shapes in both custom YAMLs allow exact shared-weight transfer.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            reference = build_detector("repvit", pretrained=pretrained)
        missing, unexpected = detector.model.load_state_dict(reference.model.state_dict(), strict=False)
        if unexpected:
            raise RuntimeError("Unexpected keys while aligning ablation initialization")
        initialization["shared_repvit_state_aligned"] = True
        initialization["new_amsf_state_entries"] = len(missing)
        del reference
    if config["variant"] != "yolov8n":
        # Ultralytics reconstructs YAML models in the trainer. Keep the initialized
        # in-memory state as its weight source, and verify it before the first step.
        detector.ckpt = {"bridge_initialized_in_memory": True}
    first_name, first_parameter = next((name, value) for name, value in detector.model.state_dict().items() if value.ndim == 4)
    expected_initial = first_parameter.detach().cpu().clone()

    def verify_initialization(trainer):
        actual = trainer.model.state_dict()[first_name].detach().float().cpu()
        if not torch.equal(actual, expected_initial.float()):
            raise RuntimeError("Trainer discarded or changed initialized backbone weights")
        initialization["trainer_initial_weights_verified"] = True

    observed = {"completed_epochs": 0, "batches": 0}

    def check_batch(trainer):
        observed["batches"] += 1
        if not torch.isfinite(trainer.loss).all():
            raise FloatingPointError(f"Non-finite training loss at epoch {trainer.epoch + 1}, batch {observed['batches']}")

    def count_epoch(trainer):
        observed["completed_epochs"] += 1

    detector.add_callback("on_train_start", verify_initialization)
    detector.add_callback("on_train_batch_end", check_batch)
    detector.add_callback("on_train_epoch_end", count_epoch)
    record = {"created_utc": utc_now(), "status": "running", "run_name": name, "configuration": config,
              "optimizer_implementation": "AdamW fused=False foreach=False; finite gradient norm required",
              "data_transfer": {"pin_memory": False, "non_blocking": False},
              "configuration_sha256": sha256(config_path), "code_revision": revision(ROOT),
              "training_code_sha256": training_fingerprint(ROOT),
              "dataset_manifest_sha256": sha256(dataset_manifest), "dataset_scope": data["scope"],
              "evaluation_scope": "source validation subset; not independent test or challenge score",
              "initialization": initialization, "architecture": config["variant"],
              "hardware": {"torch": torch.__version__, "cuda": torch.version.cuda,
                           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                           "cudnn": torch.backends.cudnn.version(),
                           "cudnn_enabled": torch.backends.cudnn.enabled,
                           "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING", "0")},
              "started_at": utc_now()}
    evidence = location("reports") / f"{name}.json"
    write_json(evidence, record)
    started = time.perf_counter()
    try:
        result = detector.train(
            trainer=CheckedDetectionTrainer,
            data=str(data_root / "dataset.yaml"), epochs=int(config["epochs"]), imgsz=int(config["imgsz"]),
            batch=int(config["batch"]), device=config.get("device", "0"), workers=int(config.get("workers", 0)),
            seed=seed, deterministic=True, optimizer="AdamW", lr0=float(config.get("lr0", 0.001)),
            warmup_bias_lr=float(config.get("warmup_bias_lr", 0.0)),
            weight_decay=float(config.get("weight_decay", 0.0005)), cos_lr=True, patience=0,
            pretrained=pretrained, amp=False, plots=False, cache=False, val=True, save=True,
            mosaic=float(config.get("mosaic", 0.5)), close_mosaic=min(5, int(config["epochs"])),
            mixup=0.0, degrees=0.0, flipud=0.0, fliplr=0.5,
            project=str(location("runs")), name=name, exist_ok=False,
        )
        if observed["completed_epochs"] != int(config["epochs"]):
            raise RuntimeError("Training stopped before the configured epoch budget was completed")
        if not all(np.isfinite(value) for value in result.results_dict.values()):
            raise FloatingPointError("Validation returned non-finite metrics")
        record.update({"status": "passed", "execution": observed, "training_seconds": round(time.perf_counter() - started, 3),
                       "metrics": _metrics(result), "finished_at": utc_now(),
                       "parameters": sum(parameter.numel() for parameter in detector.model.parameters())})
        best = run_dir / "weights" / "best.pt"
        if not best.exists():
            raise RuntimeError("Training did not produce best.pt")
        published = location("models") / f"{name}.pt"
        shutil.copyfile(best, published)
        record["weights"] = {"filename": published.name, "sha256": sha256(published), "bytes": published.stat().st_size}
        write_json(published.with_suffix(".json"), record)
        write_json(location("models") / "latest.json", {"path": published.name, "run_name": name, "variant": config["variant"], "scope": data["scope"]})
    except Exception as error:
        record.update({"status": "failed", "execution": observed, "error_type": type(error).__name__, "error": str(error), "finished_at": utc_now()})
        write_json(evidence, record)
        raise
    write_json(evidence, record)
    return record


def evaluate(weights, profile="pilot", device="0", image_size=384):
    if device == "auto":
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"
    detector = load_detector(weights)
    result = detector.val(data=str(location("prepared") / profile / "dataset.yaml"),
                          imgsz=image_size, device=device, workers=0, plots=False,
                          project=str(location("runs")), name=time.strftime("eval-%Y%m%d-%H%M%S"))
    record = {"created_utc": utc_now(), "profile": profile, "weights": str(weights),
              "evaluation_scope": "validation; no independent test claim", "metrics": _metrics(result)}
    write_json(location("reports") / "latest_evaluation.json", record)
    return record
