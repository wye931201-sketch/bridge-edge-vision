import cv2
import numpy as np


def letterbox(image, size: int):
    height, width = image.shape[:2]
    ratio = min(size / width, size / height)
    resized_width, resized_height = round(width * ratio), round(height * ratio)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    left = round((size - resized_width) / 2 - 0.1)
    top = round((size - resized_height) / 2 - 0.1)
    padded = cv2.copyMakeBorder(resized, top, size - resized_height - top, left, size - resized_width - left,
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor, ratio, (left, top)


def iou(box, boxes):
    lo = np.maximum(box[:2], boxes[:, :2])
    hi = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(hi - lo, 0).prod(axis=1)
    area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    other = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
    return intersection / np.maximum(area + other - intersection, 1e-9)


def class_aware_nms(boxes, scores, classes, threshold=0.5, max_detections=300):
    kept = []
    for class_id in np.unique(classes):
        indices = np.where(classes == class_id)[0]
        indices = indices[np.argsort(-scores[indices], kind="stable")]
        while indices.size:
            best = int(indices[0])
            kept.append(best)
            indices = indices[1:]
            if indices.size:
                indices = indices[iou(boxes[best], boxes[indices]) <= threshold]
    return sorted(kept, key=lambda index: (-float(scores[index]), index))[:max_detections]


def decode_yolo(raw, image_shape, ratio, padding, confidence=0.25, iou_threshold=0.5, class_count=5):
    predictions = np.asarray(raw)[0]
    if predictions.shape[0] == class_count + 4:
        predictions = predictions.T
    if predictions.ndim != 2 or predictions.shape[1] != class_count + 4:
        raise ValueError(f"Unsupported raw YOLO output: {np.asarray(raw).shape}")
    if not np.isfinite(predictions).all():
        raise ValueError("Model produced non-finite predictions")
    class_scores = predictions[:, 4:]
    classes = class_scores.argmax(axis=1)
    scores = class_scores.max(axis=1)
    mask = scores >= confidence
    xywh, scores, classes = predictions[mask, :4], scores[mask], classes[mask]
    if not len(xywh):
        return np.empty((0, 6), dtype=np.float32)
    boxes = np.column_stack((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2))
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - padding[0]) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - padding[1]) / ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, image_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, image_shape[0])
    keep = class_aware_nms(boxes, scores, classes, iou_threshold)
    return np.column_stack((boxes[keep], scores[keep], classes[keep])).astype(np.float32)


def tile_origins(length: int, tile_size: int, overlap: float):
    if not 0 <= overlap < 0.75 or tile_size <= 0:
        raise ValueError("Invalid tiling parameters")
    if length <= tile_size:
        return [0]
    step = max(1, round(tile_size * (1 - overlap)))
    return sorted(set(range(0, length - tile_size + 1, step)) | {length - tile_size})
