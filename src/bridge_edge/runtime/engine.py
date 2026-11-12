"""One result schema for PyTorch and ONNX Runtime, with no network dependency."""
from dataclasses import asdict, dataclass
import ast
import hashlib
import io
from pathlib import Path
import time
import uuid

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

from .geometry import class_aware_nms, decode_yolo, letterbox, tile_origins
from ..models.registry import load_detector
from ..paths import location
from ..taxonomy import BY_ID, NAMES
from ..utils import read_json, sha256, utc_now, write_json


@dataclass
class Detection:
    class_id: int
    label: str
    chinese: str
    confidence: float
    xyxy: list[float]


def available_models():
    entries = []
    data_manifest = location("prepared") / "pilot" / "manifest.json"
    current_pilot_hash = sha256(data_manifest) if data_manifest.is_file() else None
    for path in sorted(location("models").glob("*.pt")):
        card = path.with_suffix(".json")
        if card.exists():
            data = read_json(card)
            if data.get("status") != "passed" or data.get("run_name", "").startswith("diagnose"):
                continue
            if data.get("dataset_scope") == "pilot" and current_pilot_hash and data.get("dataset_manifest_sha256") != current_pilot_hash:
                continue
            entries.append({"id": path.name, "variant": data.get("architecture"), "scope": data.get("dataset_scope"),
                            "map50_95": data.get("metrics", {}).get("aggregate", {}).get("metrics/mAP50-95(B)")})
    for path in sorted(location("models").glob("*.onnx")):
        if path.with_suffix(".pt").name in {entry["id"] for entry in entries}:
            entries.append({"id": path.name, "variant": "onnx", "scope": "export of trained detector"})
    return entries


def latest_weights():
    return location("models") / read_json(location("models") / "latest.json")["path"]


def decode_image(content: bytes):
    if not content or len(content) > 25 * 1024 * 1024:
        raise ValueError("Image must be non-empty and at most 25 MiB")
    supported = content.startswith(b"\xff\xd8\xff") or content.startswith(b"\x89PNG\r\n\x1a\n")
    supported = supported or (content.startswith(b"RIFF") and content[8:12] == b"WEBP")
    if not supported:
        raise ValueError("Only JPEG, PNG and WEBP image content is supported")
    try:
        with Image.open(io.BytesIO(content)) as opened:
            if opened.width * opened.height > 25_000_000:
                raise ValueError("Image exceeds the 25 megapixel limit")
            rgb = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
    except (UnidentifiedImageError, OSError, ImportError, Image.DecompressionBombError) as error:
        raise ValueError("Invalid, truncated or unsupported image") from error
    return np.ascontiguousarray(rgb[:, :, ::-1])


class Detector:
    def __init__(self, weights=None, device="auto", image_size=384):
        self.path = Path(weights) if weights else latest_weights()
        if not self.path.is_file():
            self.path = location("models") / self.path
        if not self.path.is_file():
            raise FileNotFoundError("No trained model is available. Run training first.")
        if device == "auto":
            if self.path.suffix == ".onnx":
                device = "cpu"
            else:
                import torch
                device = "0" if torch.cuda.is_available() else "cpu"
        self.image_size = image_size
        self.device = device
        self.backend = "onnx" if self.path.suffix == ".onnx" else "pytorch"
        self.weights_sha256 = sha256(self.path)
        if self.backend == "pytorch":
            card = self.path.with_suffix(".json")
            if card.exists() and read_json(card).get("hardware", {}).get("cudnn_enabled") is False:
                import torch
                torch.backends.cudnn.enabled = False
            self.model = load_detector(self.path)
            if dict(self.model.names) != NAMES:
                raise ValueError("Checkpoint taxonomy differs from this five-class bridge detector")
        else:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            self.session = ort.InferenceSession(str(self.path), sess_options=options, providers=["CPUExecutionProvider"])
            try:
                names = ast.literal_eval(self.session.get_modelmeta().custom_metadata_map.get("names", "{}"))
            except (ValueError, SyntaxError) as error:
                raise ValueError("ONNX taxonomy metadata is malformed") from error
            if names != NAMES:
                raise ValueError("ONNX taxonomy differs from this five-class bridge detector")
            self.input_name = self.session.get_inputs()[0].name
            shape = self.session.get_inputs()[0].shape
            if isinstance(shape[-1], int):
                self.image_size = int(shape[-1])

    def _infer(self, image, confidence, iou_threshold):
        if self.backend == "pytorch":
            result = self.model.predict(image, device=self.device, imgsz=self.image_size, conf=confidence,
                                        iou=iou_threshold, rect=False, verbose=False, save=False, max_det=300)[0]
            return result.boxes.data.detach().float().cpu().numpy()
        tensor, ratio, padding = letterbox(image, self.image_size)
        raw = self.session.run(None, {self.input_name: tensor})[0]
        return decode_yolo(raw, image.shape, ratio, padding, confidence, iou_threshold, len(NAMES))

    def predict(self, image, confidence=0.25, iou_threshold=0.5, tile_size=0, overlap=0.2):
        if not 0 <= confidence <= 1 or not 0 < iou_threshold <= 1:
            raise ValueError("Invalid confidence or IoU threshold")
        started = time.perf_counter()
        tile_count = 1
        if tile_size:
            origins = [(x, y) for y in tile_origins(image.shape[0], tile_size, overlap)
                       for x in tile_origins(image.shape[1], tile_size, overlap)]
            if len(origins) > 150:
                raise ValueError("Too many tiles; increase tile size or resize the input")
            parts = []
            for x, y in origins:
                boxes = self._infer(image[y:y + tile_size, x:x + tile_size], confidence, iou_threshold).copy()
                if len(boxes):
                    boxes[:, [0, 2]] += x
                    boxes[:, [1, 3]] += y
                    parts.append(boxes)
            merged = np.concatenate(parts) if parts else np.empty((0, 6), dtype=np.float32)
            keep = class_aware_nms(merged[:, :4], merged[:, 4], merged[:, 5], iou_threshold) if len(merged) else []
            raw = merged[keep]
            tile_count = len(origins)
        else:
            raw = self._infer(image, confidence, iou_threshold)
        findings = []
        for row in raw:
            class_id = int(row[5])
            if class_id not in BY_ID:
                raise ValueError("Unexpected model class")
            spec = BY_ID[class_id]
            findings.append(asdict(Detection(class_id, spec.name, spec.chinese, round(float(row[4]), 6),
                                             [round(float(value), 2) for value in row[:4]])))
        elapsed = (time.perf_counter() - started) * 1000
        return {"created_utc": utc_now(), "model": self.path.name, "weights_sha256": self.weights_sha256,
                "backend": self.backend, "device": self.device if self.backend == "pytorch" else "CPUExecutionProvider",
                "image_size": {"width": image.shape[1], "height": image.shape[0]}, "inference_size": self.image_size,
                "confidence_threshold": confidence, "iou_threshold": iou_threshold, "tile_count": tile_count,
                "total_latency_ms": round(elapsed, 3), "detections": findings,
                "latency_scope": "image preprocessing, model prediction and NMS; excludes image decoding, annotation, persistence and HTTP transfer",
                "finding_count": len(findings), "physical_dimensions_available": False,
                "interpretation": "Visual findings only; confidence is not a calibrated safety probability"}


def annotate(image, detections):
    canvas = Image.fromarray(image[:, :, ::-1])
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(13, min(image.shape[:2]) // 45))
    except OSError:
        font = ImageFont.load_default()
    line = max(2, min(image.shape[:2]) // 350)
    for finding in detections:
        box = finding["xyxy"]
        color = BY_ID[finding["class_id"]].color
        draw.rectangle(box, outline=color, width=line)
        text = f"{finding['label']} {finding['confidence']:.2f}"
        bounds = draw.textbbox((0, 0), text, font=font)
        x, y = max(0, box[0]), max(0, box[1] - (bounds[3] - bounds[1]) - 7)
        draw.rectangle((x, y, x + bounds[2] + 8, y + bounds[3] - bounds[1] + 7), fill=color)
        draw.text((x + 4, y + 2 - bounds[1]), text, font=font, fill=(15, 23, 42))
    return canvas


def save_prediction(image, result, original_name="image"):
    run_id = uuid.uuid4().hex
    folder = location("predictions") / run_id
    folder.mkdir()
    result = {**result, "id": run_id, "source_filename": Path(original_name).name}
    annotate(image, result["detections"]).save(folder / "annotated.jpg", quality=92)
    write_json(folder / "result.json", result)
    from ..store import record_prediction
    record_prediction(result)
    return result
