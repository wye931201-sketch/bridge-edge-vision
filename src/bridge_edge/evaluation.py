"""Fixed-threshold detection evaluation with explicit false positives and misses."""
from collections import defaultdict

import numpy as np

from .runtime.geometry import iou
from .taxonomy import NAMES


def read_yolo_boxes(path, width, height):
    boxes = []
    for line in path.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError("YOLO label must contain a class and four coordinates")
        raw_id, cx, cy, bw, bh = map(float, fields)
        if raw_id != int(raw_id) or int(raw_id) not in NAMES or not np.isfinite([cx, cy, bw, bh]).all():
            raise ValueError("Invalid class or coordinates")
        box = [(cx - bw / 2) * width, (cy - bh / 2) * height,
               (cx + bw / 2) * width, (cy + bh / 2) * height]
        if not -.01 <= box[0] < box[2] <= width + .01 or not -.01 <= box[1] < box[3] <= height + .01:
            raise ValueError("Ground-truth box is outside its image")
        box = [max(0., box[0]), max(0., box[1]), min(float(width), box[2]), min(float(height), box[3])]
        boxes.append({"class_id": int(raw_id), "xyxy": box})
    return boxes


def match_detections(predictions, ground_truth, min_iou=.5):
    if not 0 < min_iou <= 1:
        raise ValueError("Invalid matching IoU")
    used = set()
    matches, false_positives = [], []
    order = sorted(range(len(predictions)), key=lambda index: (-predictions[index]["confidence"], index))
    for index in order:
        prediction = predictions[index]
        candidates = [i for i, target in enumerate(ground_truth)
                      if i not in used and target["class_id"] == prediction["class_id"]]
        if candidates:
            overlaps = iou(np.asarray(prediction["xyxy"]), np.asarray([ground_truth[i]["xyxy"] for i in candidates]))
            best = int(np.argmax(overlaps))
            if overlaps[best] >= min_iou:
                target_index = candidates[best]
                used.add(target_index)
                matches.append({"prediction": index, "ground_truth": target_index,
                                "class_id": prediction["class_id"], "iou": float(overlaps[best])})
                continue
        false_positives.append(index)
    missed = [i for i in range(len(ground_truth)) if i not in used]
    return {"matches": matches, "false_positive_indices": false_positives, "missed_ground_truth_indices": missed}


def rates(tp, fp, fn):
    return {"true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0.,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}


def summarize_detection_records(records, min_iou=.5):
    counts = defaultdict(lambda: [0, 0, 0])
    images_without_candidates = 0
    for record in records:
        predictions, targets = record["detections"], record["ground_truth"]
        images_without_candidates += not predictions
        matched = match_detections(predictions, targets, min_iou)
        for finding in matched["matches"]:
            counts[finding["class_id"]][0] += 1
        for index in matched["false_positive_indices"]:
            counts[predictions[index]["class_id"]][1] += 1
        for index in matched["missed_ground_truth_indices"]:
            counts[targets[index]["class_id"]][2] += 1
    per_class = {name: rates(*counts[class_id]) for class_id, name in NAMES.items()}
    totals = [sum(value[index] for value in counts.values()) for index in range(3)]
    return {"images": len(records), "images_without_candidates": images_without_candidates,
            "iou_threshold": min_iou, "micro": rates(*totals), "per_class": per_class,
            "scope": "fixed-threshold validation precision/recall/F1, not mAP or an independent test score"}
