"""Optional upstream crack segmentation, converted to reviewable component boxes."""
import time

import cv2
import numpy as np

from .paths import CONFIG_ROOT, location
from .runtime.geometry import class_aware_nms
from .utils import read_json, sha256


def component_detections(probability, image_shape, threshold=.4, min_area=24, max_fill_ratio=1., return_mask=False):
    if probability.ndim != 2 or not np.isfinite(probability).all() or not 0 <= probability.min() <= probability.max() <= 1:
        raise ValueError("Expected a finite two-dimensional probability map")
    if not 0 < threshold < 1 or min_area < 1 or not 0 < max_fill_ratio <= 1:
        raise ValueError("Invalid segmentation thresholds")
    binary = (probability >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = image_shape[:2]
    ph, pw = probability.shape
    detections = []
    accepted_mask = np.zeros(binary.shape, dtype=np.uint8)
    for index in range(1, count):
        x, y, bw, bh, area = stats[index]
        fill_ratio = float(area) / (int(bw) * int(bh))
        if area < min_area or fill_ratio > max_fill_ratio:
            continue
        if return_mask:
            accepted_mask[labels == index] = 1
        score = float(probability[labels == index].mean())
        detections.append({"class_id": 0, "label": "crack", "chinese": "裂缝",
                           "confidence": round(score, 6),
                           "xyxy": [round(float(value), 2) for value in (x * width / pw, y * height / ph,
                                                                        (x + bw) * width / pw, (y + bh) * height / ph)],
                           "proposal_source": "upstream_crack_segmentation", "native_mask_area": int(area),
                           "native_fill_ratio": fill_ratio,
                           "score_kind": "mean segmentation probability in a connected component"})
    return (detections, accepted_mask) if return_mask else detections


class CrackSegmenter:
    def __init__(self):
        config = read_json(CONFIG_ROOT / "crack_segmentation.json")
        self.path = location("models") / "external/crack_segmentation" / config["filename"]
        self.weights_sha256 = sha256(self.path)
        if self.weights_sha256 != config["sha256"]:
            raise ValueError("Crack model bytes differ from the pinned release")
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(self.path), sess_options=options, providers=["CPUExecutionProvider"])
        spec, output = self.session.get_inputs()[0], self.session.get_outputs()[0]
        if spec.shape != config["input_shape"] or output.shape != config["output_shape"]:
            raise ValueError("Unexpected crack segmentation graph shape")
        self.input_name = spec.name

    def probability(self, image):
        resized = cv2.resize(image, (672, 378), interpolation=cv2.INTER_LINEAR)
        tensor = np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
        logits = self.session.run(None, {self.input_name: tensor})[0][0, 0]
        if not np.isfinite(logits).all():
            raise ValueError("Crack segmenter produced non-finite logits")
        return 1 / (1 + np.exp(-np.clip(logits, -80, 80)))

    def predict(self, image, threshold=.4, min_area=24, max_fill_ratio=1.):
        started = time.perf_counter()
        probability = self.probability(image)
        detections, mask = component_detections(probability, image.shape, threshold, min_area, max_fill_ratio, return_mask=True)
        return {"detections": detections, "finding_count": len(detections), "threshold": threshold,
                "min_native_mask_area": min_area, "max_fill_ratio": max_fill_ratio, "_native_crack_mask": mask,
                "model": self.path.name, "weights_sha256": self.weights_sha256,
                "total_latency_ms": (time.perf_counter() - started) * 1000,
                "scope": "one upstream graph on resized photos; no ensemble score or physical dimensions claimed"}


def replace_crack_candidates(base_result, segmentation_result):
    detections = [item for item in base_result["detections"] if item["label"] != "crack"] + segmentation_result["detections"]
    if detections:
        keep = class_aware_nms(np.asarray([item["xyxy"] for item in detections]),
                               np.asarray([item["confidence"] for item in detections]),
                               np.asarray([item["class_id"] for item in detections]), .5)
        detections = [detections[index] for index in keep]
    return {**base_result, "detections": detections, "finding_count": len(detections),
            "total_latency_ms": base_result.get("total_latency_ms", 0) + segmentation_result["total_latency_ms"],
            "crack_auxiliary": {key: value for key, value in segmentation_result.items() if key != "detections" and not key.startswith("_")},
            "_native_crack_mask": segmentation_result.get("_native_crack_mask"),
            "score_interpretation": "detector confidence for non-crack boxes; mean segmentation probability for crack components"}
