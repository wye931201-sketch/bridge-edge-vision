from pathlib import Path

import cv2

from .paths import location
from .runtime.engine import Detector, decode_image, save_prediction
from .utils import utc_now, write_json


def predict_images(source, weights=None, device="0", confidence=0.25, image_size=384, tile_size=0):
    source = Path(source)
    paths = sorted(path for path in source.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}) if source.is_dir() else [source]
    if not paths:
        raise ValueError("No supported images found")
    engine = Detector(weights, device=device, image_size=image_size)
    results = []
    for path in paths:
        image = decode_image(path.read_bytes())
        result = save_prediction(image, engine.predict(image, confidence=confidence, tile_size=tile_size), path.name)
        results.append({"id": result["id"], "source": path.name, "finding_count": result["finding_count"], "latency_ms": result["total_latency_ms"]})
        print(f"{path.name}: {result['finding_count']} findings -> {result['id']}", flush=True)
    write_json(location("reports") / "latest_batch_prediction.json", {"created_utc": utc_now(), "items": results})
    return results


def predict_video(source, weights=None, device="0", confidence=0.25, every=5, max_frames=300):
    if every < 1 or max_frames < 1:
        raise ValueError("Frame interval and maximum frame count must be positive")
    capture = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not capture.isOpened():
        raise ValueError("Video or camera source could not be opened")
    engine = Detector(weights, device=device)
    results = []
    index = 0
    try:
        while index < max_frames:
            available, frame = capture.read()
            if not available:
                break
            if index % every == 0:
                result = engine.predict(frame, confidence=confidence)
                result["frame_index"] = index
                result["video_time_ms"] = capture.get(cv2.CAP_PROP_POS_MSEC)
                results.append(save_prediction(frame, result, f"frame-{index:06d}.jpg"))
            index += 1
    finally:
        capture.release()
    record = {"created_utc": utc_now(), "source": str(source), "frames_read": index, "sample_interval": every, "results": results}
    write_json(location("reports") / "latest_video_prediction.json", record)
    return {"frames_read": index, "frames_processed": len(results)}
