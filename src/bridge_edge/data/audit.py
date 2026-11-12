"""Audit annotation validity, exact leakage and perceptual review candidates."""
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from ..paths import location
from ..taxonomy import CLASSES
from ..utils import read_json, sha256, utc_now, write_json


def difference_hash(image_path: Path) -> int:
    with Image.open(image_path) as opened:
        pixels = np.asarray(opened.convert("L").resize((9, 8)))
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def audit_dataset(profile="pilot", check_hashes=True):
    root = location("prepared") / profile
    manifest = read_json(root / "manifest.json")
    errors = []
    seen = {}
    fingerprints = {"train": [], "val": []}
    stats = {split: {"images": 0, "negative_images": 0, "boxes": 0, "per_class": Counter()} for split in ("train", "val")}
    for record in manifest["records"]:
        image, label = root / record["image"], root / record["label"]
        split = record["split"]
        stats[split]["images"] += 1
        if not image.is_file() or not label.is_file():
            errors.append({"image": record["image"], "reason": "missing_file"})
            continue
        expected_hash = record.get("prepared_sha256", record["sha256"])
        actual_hash = sha256(image) if check_hashes else expected_hash
        if actual_hash != expected_hash or sha256(label) != record["label_sha256"]:
            errors.append({"image": record["image"], "reason": "hash_changed"})
        with Image.open(image) as opened:
            if opened.getexif().get(274, 1) != 1 or opened.size != (record["width"], record["height"]):
                errors.append({"image": record["image"], "reason": "ambiguous_orientation_or_dimensions"})
        if actual_hash in seen:
            errors.append({"image": record["image"], "reason": "exact_image_duplicate", "other": seen[actual_hash]})
        seen[actual_hash] = record["image"]
        rows = label.read_text("utf-8").splitlines()
        if not rows:
            stats[split]["negative_images"] += 1
        for row in rows:
            try:
                values = list(map(float, row.split()))
            except ValueError:
                errors.append({"image": record["image"], "reason": "unparseable_label_row"})
                continue
            valid = len(values) == 5 and values[0].is_integer() and 0 <= values[0] < len(CLASSES)
            valid = valid and all(np.isfinite(values)) and all(0 <= number <= 1 for number in values[1:])
            if not valid:
                errors.append({"image": record["image"], "reason": "invalid_label_row"})
                continue
            _, x, y, w, h = values
            if min(w, h) <= 0 or min(x - w / 2, y - h / 2) < -1e-6 or max(x + w / 2, y + h / 2) > 1 + 1e-6:
                errors.append({"image": record["image"], "reason": "box_out_of_bounds"})
            stats[split]["boxes"] += 1
            stats[split]["per_class"][CLASSES[int(values[0])].name] += 1
        fingerprints[split].append((record["image"], difference_hash(image)))
    for split in ("train", "val"):
        expected = {Path(item["image"]).name for item in manifest["records"] if item["split"] == split}
        actual = {path.name for path in (root / "images" / split).glob("*.jpg")}
        if actual != expected:
            errors.append({"split": split, "reason": "image_directory_differs_from_manifest", "extra_count": len(actual - expected), "missing_count": len(expected - actual)})
    review = []
    for train_name, train_hash in fingerprints["train"]:
        for val_name, val_hash in fingerprints["val"]:
            distance = (train_hash ^ val_hash).bit_count()
            if distance <= 2:
                review.append({"train": train_name, "val": val_name, "dhash_distance": distance, "status": "manual_review_candidate"})
    report = {"created_utc": utc_now(), "passed": not errors, "profile": profile,
              "manifest_sha256": sha256(root / "manifest.json"), "stats": stats, "errors": errors,
              "perceptual_cross_split_candidates": review,
              "limits": ["Perceptual similarity is a review hint, not confirmed duplication", "Bridge-identity and capture-session separation are not established by image hashes", "This audit does not manually certify annotation correctness"]}
    write_json(root / "audit.json", report)
    print(f"Dataset audit: {'passed' if report['passed'] else 'failed'}; {len(review)} perceptual review candidates", flush=True)
    return report
