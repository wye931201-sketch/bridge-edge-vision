"""Convert mirror bounding boxes into a five-class YOLO research subset."""
from collections import Counter
import math
from pathlib import Path
import shutil

import yaml
from PIL import Image

from .download import metadata, _safe_relative
from ..paths import location
from ..taxonomy import NAMES, SOURCE_TO_ID
from ..utils import read_json, sha256, utc_now, write_json


def convert_box(box, width: int, height: int):
    """FiftyOne xywh -> YOLO center-xywh, with explicit clipping diagnostics."""
    if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
        return None, "non_finite_or_malformed"
    x, y, w, h = map(float, box)
    if w <= 0 or h <= 0:
        return None, "non_positive_extent"
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(1.0, x + w), min(1.0, y + h)
    if x2 <= x1 or y2 <= y1 or (x2 - x1) * width < 1 or (y2 - y1) * height < 1:
        return None, "outside_or_subpixel"
    clipped = any(abs(a - b) > 1e-7 for a, b in zip((x, y, x + w, y + h), (x1, y1, x2, y2)))
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), "clipped" if clipped else "valid"


def _copy_image(source: Path, destination: Path):
    """Isolate originals from third-party loaders that repair JPEGs in place."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Untracked prepared image exists: {destination.name}")
    shutil.copyfile(source, destination)
    raw = destination.read_bytes()
    repaired = destination.suffix.lower() in {".jpg", ".jpeg"} and not raw.endswith(b"\xff\xd9")
    if repaired:
        # A decoded, valid image may contain trailing padding after its final EOI.
        # Trim only that padding; refuse truly truncated JPEGs rather than re-encode.
        end = raw.rfind(b"\xff\xd9")
        if end < 0:
            raise ValueError(f"JPEG has no end-of-image marker: {source.name}")
        destination.write_bytes(raw[:end + 2])
    return {"prepared_sha256": sha256(destination),
            "image_transformation": "trim_bytes_after_jpeg_eoi" if repaired else "byte_identical_copy"}


def prepare_dataset(profile="pilot"):
    if not profile.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Profile must contain only letters, numbers, '-' and '_'")
    download_path = location("raw") / "download_manifest.json"
    downloaded = read_json(download_path)
    samples = {sample["filepath"]: sample for sample in metadata()}
    destination = location("prepared") / profile
    destination.mkdir(parents=True, exist_ok=True)
    selection_path = destination / "selection.json"
    manifest_hash = sha256(download_path)
    if selection_path.exists() and read_json(selection_path)["download_manifest_sha256"] != manifest_hash:
        raise FileExistsError("This profile belongs to a different download selection. Choose a new profile name.")
    write_json(selection_path, {"download_manifest_sha256": manifest_hash})
    if (destination / "manifest.json").exists():
        existing = read_json(destination / "manifest.json")
        if existing.get("preparation_version") != 2:
            raise FileExistsError("Existing preparation uses an older image-orientation policy. Preserve it and use a new profile.")
        return existing
    counts = {split: Counter() for split in ("train", "val")}
    diagnostics = Counter()
    records = []
    quarantine = []
    seen_hashes = {}
    # Keep the original validation image in preference to an identical training copy.
    ordered = sorted(downloaded["records"], key=lambda item: (item["split"] != "val", item["source_path"]))
    for item in ordered:
        if item["sha256"] in seen_hashes:
            quarantine.append({"source_path": item["source_path"], "reason": "exact_duplicate", "kept": seen_hashes[item["sha256"]]})
            continue
        seen_hashes[item["sha256"]] = item["source_path"]
        source = location("raw") / _safe_relative(item["source_path"])
        if sha256(source) != item["sha256"]:
            raise ValueError(f"Raw image hash changed: {source.name}")
        with Image.open(source) as image:
            orientation = image.getexif().get(274, 1)
        if orientation != 1:
            quarantine.append({"source_path": item["source_path"], "reason": "non_identity_exif_orientation",
                               "orientation": orientation, "detail": "Annotation coordinate frame requires confirmation; do not guess a box rotation"})
            continue
        sample = samples[item["source_path"]]
        labels = []
        for detection in (sample.get("od_ground_truth") or {}).get("detections", []):
            if detection["label"] not in SOURCE_TO_ID:
                continue
            box, state = convert_box(detection["bounding_box"], item["width"], item["height"])
            diagnostics[state] += 1
            if box is None:
                continue
            class_id = SOURCE_TO_ID[detection["label"]]
            labels.append((class_id, *box))
        # Identical annotations may occur in the mirror; keep one copy of each box.
        labels = sorted(set(labels))
        for label in labels:
            counts[item["split"]][NAMES[label[0]]] += 1
        filename = Path(item["source_path"]).name
        prepared_image = destination / "images" / item["split"] / filename
        label_path = destination / "labels" / item["split"] / (Path(filename).stem + ".txt")
        image_integrity = _copy_image(source, prepared_image)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("".join(f"{label[0]} " + " ".join(f"{value:.8f}" for value in label[1:]) + "\n" for label in labels), encoding="utf-8")
        records.append({**item, **image_integrity, "image": prepared_image.relative_to(destination).as_posix(),
                        "label": label_path.relative_to(destination).as_posix(), "label_sha256": sha256(label_path),
                        "target_boxes": len(labels), "negative_for_selected_classes": not labels})
    for split in ("train", "val"):
        if not any(item["split"] == split for item in records):
            raise ValueError(f"No images in {split}")
    yaml_path = destination / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump({"path": str(destination.resolve()), "train": "images/train", "val": "images/val", "names": NAMES}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    report = {
        "created_utc": utc_now(), "preparation_version": 2, "profile": profile, "scope": downloaded["scope"], "source": downloaded["source"],
        "download_manifest_sha256": sha256(download_path), "selection": downloaded["selection"], "seed": downloaded["seed"],
        "annotation_origin": "Voxel51 od_ground_truth bounding boxes derived from dacl10k semantic polygons; not an official instance-detection benchmark",
        "split_policy": "Original train and validation tags preserved; exact duplicate images quarantined",
        "class_box_counts": {split: dict(value) for split, value in counts.items()}, "box_diagnostics": dict(diagnostics),
        "records": sorted(records, key=lambda item: (item["split"], item["image"])), "quarantined": quarantine,
        "test_split": None, "evaluation_scope": "validation only; no independent test claim",
    }
    write_json(destination / "manifest.json", report)
    print(f"Prepared {len(records)} images: {yaml_path}", flush=True)
    return report
