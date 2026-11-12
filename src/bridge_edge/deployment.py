"""Export validation and device-specific latency measurements."""
from pathlib import Path
import os
import statistics
import time

import numpy as np

from .models.registry import load_detector
from .paths import location
from .runtime.engine import Detector, decode_image
from .runtime.geometry import iou, letterbox
from .utils import sha256, utc_now, write_json


def export_onnx(weights, image_size=384, device="cpu", sample=None):
    import onnx
    import onnxruntime as ort
    import torch
    detector = load_detector(weights)
    exported = Path(detector.export(format="onnx", imgsz=image_size, opset=17, dynamic=False,
                                    simplify=False, nms=False, device=device, half=False))
    onnx.checker.check_model(onnx.load(str(exported)))
    if exported.parent != location("models"):
        import shutil
        destination = location("models") / exported.name
        shutil.copyfile(exported, destination)
        exported = destination
    generator = np.random.default_rng(2026)
    tensor = generator.random((1, 3, image_size, image_size), dtype=np.float32)
    model = detector.model.to("cpu").eval()
    with torch.inference_mode():
        expected = model(torch.from_numpy(tensor))
        if isinstance(expected, tuple):
            expected = expected[0]
        expected = expected.detach().cpu().numpy()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(exported), sess_options=options, providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)
    report = {"created_utc": utc_now(), "passed": True, "source_weights": Path(weights).name,
              "onnx": exported.name, "sha256": sha256(exported), "image_size": image_size,
              "provider": "CPUExecutionProvider", "max_raw_absolute_difference": float(np.max(np.abs(actual - expected))),
              "verification": "FP32 raw predictions; not Jetson/TensorRT validation"}
    if sample:
        write_json(exported.with_suffix(".export.json"), {**report, "passed": False,
                   "verification_stage": "raw predictions verified; sample detection parity pending"})
        image = decode_image(Path(sample).read_bytes())
        torch_engine = Detector(weights, device="cpu", image_size=image_size)
        ort_engine = Detector(exported, device="cpu", image_size=image_size)
        left, right = torch_engine.predict(image), ort_engine.predict(image)
        report["sample_parity"] = detection_parity(left["detections"], right["detections"])
        if not report["sample_parity"]["passed"]:
            report["passed"] = False
            write_json(exported.with_suffix(".export.json"), report)
            raise RuntimeError("Exported detections differ from PyTorch; inspect export report")
    write_json(exported.with_suffix(".export.json"), report)
    return report


def detection_parity(left, right, min_iou=0.97, score_tolerance=0.01):
    used = set()
    matches = []
    for detection in left:
        candidates = [(index, candidate) for index, candidate in enumerate(right)
                      if index not in used and candidate["class_id"] == detection["class_id"]]
        if not candidates:
            continue
        overlaps = iou(np.asarray(detection["xyxy"]), np.asarray([candidate["xyxy"] for _, candidate in candidates]))
        best = int(np.argmax(overlaps))
        index, candidate = candidates[best]
        score_gap = abs(detection["confidence"] - candidate["confidence"])
        if overlaps[best] >= min_iou and score_gap <= score_tolerance:
            used.add(index)
            matches.append({"iou": float(overlaps[best]), "score_gap": score_gap})
    return {"passed": len(matches) == len(left) == len(right), "pytorch_detections": len(left),
            "onnx_detections": len(right), "matched": len(matches), "min_iou_required": min_iou,
            "max_score_gap_allowed": score_tolerance}


def benchmark(weights, sample, device="0", image_size=384, warmup=5, iterations=30):
    if warmup < 0 or iterations < 5:
        raise ValueError("Use nonnegative warmup and at least five timed iterations")
    image = decode_image(Path(sample).read_bytes())
    engine = Detector(weights, device=device, image_size=image_size)
    device = engine.device
    if engine.backend == "pytorch":
        import torch
        torch.set_num_threads(4)
        synchronize = lambda: torch.cuda.synchronize() if str(device) not in ("cpu", "mps") and torch.cuda.is_available() else None
        hardware = torch.cuda.get_device_name(0) if str(device) not in ("cpu", "mps") and torch.cuda.is_available() else "CPU"
    else:
        synchronize = lambda: None
        hardware = "CPU / ONNX Runtime"
    # Initialize the prediction backend before fixing the timed CPU thread budget:
    # Ultralytics may change the global thread count while selecting a device.
    engine.predict(image)
    if engine.backend == "pytorch":
        torch.set_num_threads(4)
    for _ in range(warmup):
        engine.predict(image)
    samples = []
    for _ in range(iterations):
        synchronize()
        started = time.perf_counter()
        engine.predict(image)
        synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    report = {"created_utc": utc_now(), "model": engine.path.name, "weights_sha256": engine.weights_sha256,
              "backend": engine.backend, "device": device, "hardware": hardware, "image_size": engine.image_size,
              "precision": "FP32", "warmup": warmup, "iterations": iterations,
              "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING", "0"),
              "cpu_threads": torch.get_num_threads() if engine.backend == "pytorch" else 4,
              "initialization_predictions_before_warmup": 1,
              "scope": "single-process image prediction including preprocessing and NMS; excludes decoding, model loading, annotation, persistence, HTTP and camera capture",
              "latency_ms": {"mean": statistics.mean(samples), "p50": float(np.percentile(samples, 50)),
                             "p95": float(np.percentile(samples, 95)), "min": min(samples), "max": max(samples)},
              "throughput_estimate_fps": 1000 / statistics.mean(samples), "jetson_hardware_verified": False}
    write_json(location("reports") / (engine.path.stem + f"-benchmark-{engine.backend}.json"), report)
    return report
