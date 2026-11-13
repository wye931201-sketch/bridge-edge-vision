"""Measured multiscale inspection without using image labels at inference time."""
from dataclasses import asdict, dataclass
import time

import cv2
import numpy as np

from .geometry import class_aware_nms, tile_origins


@dataclass(frozen=True)
class InspectionProfile:
    name: str
    tile_sizes: tuple[int, ...]
    include_global: bool = True
    overlap: float = .25
    max_passes: int = 128
    contrast_global: bool = False

    def __post_init__(self):
        if not self.name or not 0 <= self.overlap < .75 or self.max_passes < 1:
            raise ValueError("Invalid inspection profile")
        if not self.tile_sizes and not self.include_global:
            raise ValueError("The profile must contain an inference view")
        if any(size < 128 or size % 32 for size in self.tile_sizes):
            raise ValueError("Tile sizes must be multiples of 32 and at least 128")
        if len(set(self.tile_sizes)) != len(self.tile_sizes):
            raise ValueError("Repeated tile scales are not allowed")
        if int(self.include_global) + int(self.contrast_global) > self.max_passes:
            raise ValueError("Global views exceed the pass budget")


PROFILES = {
    "legacy": InspectionProfile("legacy", (768,), include_global=False, overlap=.2, max_passes=150),
    "detail": InspectionProfile("detail", (768, 384)),
    "detail-contrast": InspectionProfile("detail-contrast", (768, 384), contrast_global=True),
}


def resolve_profile(engine, requested="auto"):
    if requested != "auto":
        if requested not in PROFILES:
            raise ValueError("Unknown inspection profile")
        return requested
    from ..paths import location
    from ..utils import read_json, sha256
    report_path = location("reports") / "inference_profile_comparison.json"
    manifest = location("prepared") / "pilot/manifest.json"
    if report_path.is_file() and manifest.is_file():
        report = read_json(report_path)
        if (report.get("passed") and report.get("weights_sha256") == engine.weights_sha256
                and report.get("dataset_manifest_sha256") == sha256(manifest)
                and report.get("selected_profile") in PROFILES):
            return report["selected_profile"]
    return "legacy"


class BridgeInspector:
    """Use only settings validated for these model bytes and this data version."""
    def __init__(self, engine, profile="auto", crack_mode="auto"):
        if crack_mode not in {"auto", "off"}:
            raise ValueError("Use auto or off for the crack auxiliary")
        self.engine = engine
        self.path, self.weights_sha256 = engine.path, engine.weights_sha256
        self.profile = resolve_profile(engine, profile)
        self.segmenter, self.crack_settings = None, {}
        if crack_mode == "auto":
            from ..paths import location
            from ..utils import read_json, sha256
            evidence = location("reports") / "crack_auxiliary_evaluation.json"
            manifest = location("prepared") / "pilot/manifest.json"
            if evidence.is_file() and manifest.is_file():
                record = read_json(evidence)
                if (record.get("enabled") and record.get("bridge_weights_sha256") == self.weights_sha256
                        and record.get("dataset_manifest_sha256") == sha256(manifest)
                        and record.get("base_profile") == self.profile):
                    from ..crack_segmentation import CrackSegmenter
                    segmenter = CrackSegmenter()
                    if segmenter.weights_sha256 != record.get("auxiliary_weights_sha256"):
                        raise ValueError("Validated auxiliary model changed")
                    selected = record["selected"]
                    self.segmenter = segmenter
                    self.crack_settings = {"threshold": selected["probability_threshold"],
                                           "min_area": selected["min_native_area"],
                                           "max_fill_ratio": selected.get("max_fill_ratio", 1.)}
        self.display_name = self.path.name + (" + upstream crack segmenter" if self.segmenter else "")

    def predict(self, image, confidence=.15, tile_size=768):
        result = (self.engine.predict(image, confidence=confidence, tile_size=tile_size) if self.profile == "legacy"
                  else inspect_image(self.engine, image, self.profile, confidence))
        if self.segmenter:
            from ..crack_segmentation import replace_crack_candidates
            result = replace_crack_candidates(result, self.segmenter.predict(image, **self.crack_settings))
        result["inspection_profile_name"] = self.profile + ("+crack-segmentation" if self.segmenter else "")
        return result


def inference_views(image_shape, profile):
    """Cover the complete image at every retained scale within a pass budget."""
    height, width = image_shape[:2]
    if min(height, width) < 1:
        raise ValueError("Empty image")
    views = []
    if profile.include_global:
        views.append({"x": 0, "y": 0, "width": width, "height": height, "kind": "global"})
    if profile.contrast_global:
        views.append({"x": 0, "y": 0, "width": width, "height": height, "kind": "contrast"})
    for requested in profile.tile_sizes:
        available = profile.max_passes - len(views)
        if available < 1:
            break
        size = requested
        while True:
            xs, ys = tile_origins(width, size, profile.overlap), tile_origins(height, size, profile.overlap)
            if len(xs) * len(ys) <= available:
                break
            size += 32
        for y in ys:
            for x in xs:
                view = {"x": x, "y": y, "width": min(size, width - x),
                        "height": min(size, height - y), "kind": "tile", "requested_size": requested, "effective_size": size}
                # A full-image tile duplicates the unmodified global pass.
                if profile.include_global and x == y == 0 and view["width"] == width and view["height"] == height:
                    continue
                views.append(view)
    if not views or len(views) > profile.max_passes:
        raise ValueError("No valid inference plan fits the pass budget")
    return views


def _contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def inspect_image(engine, image, profile="detail", confidence=.15, iou_threshold=.5):
    if isinstance(profile, str):
        if profile not in PROFILES:
            raise ValueError("Unknown inspection profile")
        profile = PROFILES[profile]
    if not 0 <= confidence <= 1 or not 0 < iou_threshold <= 1:
        raise ValueError("Invalid detection thresholds")
    started = time.perf_counter()
    views = inference_views(image.shape, profile)
    detections, template = [], None
    for view in views:
        x, y, width, height = (view[key] for key in ("x", "y", "width", "height"))
        crop = image[y:y + height, x:x + width]
        if view["kind"] == "contrast":
            crop = _contrast(crop)
        result = engine.predict(crop, confidence=confidence, iou_threshold=iou_threshold)
        template = result
        for finding in result["detections"]:
            translated = {**finding, "xyxy": [finding["xyxy"][0] + x, finding["xyxy"][1] + y,
                                               finding["xyxy"][2] + x, finding["xyxy"][3] + y]}
            box = np.asarray(translated["xyxy"], dtype=np.float64)
            if not np.isfinite(box).all() or not np.isfinite(finding["confidence"]):
                raise ValueError("Non-finite candidate from an inference view")
            box[[0, 2]] = box[[0, 2]].clip(0, image.shape[1])
            box[[1, 3]] = box[[1, 3]].clip(0, image.shape[0])
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            translated["xyxy"] = [round(float(value), 2) for value in box]
            detections.append(translated)
    if detections:
        keep = class_aware_nms(np.asarray([d["xyxy"] for d in detections]),
                               np.asarray([d["confidence"] for d in detections]),
                               np.asarray([d["class_id"] for d in detections]), iou_threshold)
        detections = [detections[index] for index in keep]
    return {**template, "image_size": {"width": image.shape[1], "height": image.shape[0]},
            "detections": detections, "finding_count": len(detections), "inspection_profile": asdict(profile),
            "tile_count": len(views), "inference_views": views,
            "total_latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "latency_scope": "all selected views, preprocessing, prediction and cross-view NMS; excludes decode and saving"}
