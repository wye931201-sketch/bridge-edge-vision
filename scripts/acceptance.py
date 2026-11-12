"""Exercise real local HTTP prediction, tiling, persistence and backend parity."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx
import cv2
from bridge_edge.deployment import detection_parity
from bridge_edge.paths import location
from bridge_edge.prediction import predict_video
from bridge_edge.runtime.engine import Detector, decode_image, latest_weights
from bridge_edge.utils import read_json, sha256, utc_now, write_json


def validate_result(result):
    width, height = result["image_size"]["width"], result["image_size"]["height"]
    assert result["finding_count"] == len(result["detections"])
    assert not result["physical_dimensions_available"]
    for finding in result["detections"]:
        x1, y1, x2, y2 = finding["xyxy"]
        assert 0 <= x1 <= x2 <= width and 0 <= y1 <= y2 <= height
        assert 0 <= finding["class_id"] < 5 and 0 <= finding["confidence"] <= 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    args = parser.parse_args()
    if not args.url.startswith("http://127.0.0.1:"):
        raise ValueError("Acceptance uploads are restricted to a loopback server")
    weights = latest_weights()
    model_card = read_json(weights.with_suffix(".json"))
    dataset_hash = sha256(location("prepared") / "pilot" / "manifest.json")
    if model_card["dataset_manifest_sha256"] != dataset_hash:
        raise RuntimeError("The default model belongs to a different dataset preparation; complete the current pilot first")
    with httpx.Client(base_url=args.url, timeout=120, trust_env=False) as client:
        health = client.get("/api/health").raise_for_status().json()
        assert health["status"] == "ready" and health["external_api_required"] is False
        samples = client.get("/api/samples").raise_for_status().json()["samples"]
        assert len(samples) == 12
        sample = samples[0]
        content = client.get(sample["url"]).raise_for_status().content
        response = client.post("/api/predict", params={"filename": "验收桥梁.jpg", "confidence": .15}, content=content)
        result = response.raise_for_status().json()
        validate_result(result)
        saved = client.get(result["json_url"]).raise_for_status().json()
        assert saved["id"] == result["id"] and saved["weights_sha256"] == sha256(weights)
        assert client.get(result["annotated_url"]).raise_for_status().headers["content-type"] == "image/jpeg"
        assert client.get("/api/history/" + result["id"]).raise_for_status().json()["id"] == result["id"]
        assert any(item["id"] == result["id"] for item in client.get("/api/history").raise_for_status().json()["items"])
        bad = client.post("/api/predict", content=b"not an image")
        assert bad.status_code == 400
    # Use the same fixed, unselected first 12 validation images for parity.
    records = sorted((item for item in read_json(location("prepared") / "pilot" / "manifest.json")["records"]
                      if item["split"] == "val"), key=lambda item: item["image"])[:12]
    torch_engine = Detector(weights, device="cpu")
    onnx_engine = Detector(weights.with_suffix(".onnx"), device="cpu")
    parity = []
    for record in records:
        image = decode_image((location("prepared") / "pilot" / record["image"]).read_bytes())
        left, right = torch_engine.predict(image, confidence=.15), onnx_engine.predict(image, confidence=.15)
        validate_result(left)
        validate_result(right)
        parity.append({"image": Path(record["image"]).name, **detection_parity(left["detections"], right["detections"])})
    tiled = torch_engine.predict(decode_image(content), confidence=.15, tile_size=768)
    validate_result(tiled)
    assert tiled["tile_count"] > 1
    video = location("reports") / "acceptance-video.avi"
    frame = cv2.resize(decode_image(content), (640, 480))
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 480))
    if not writer.isOpened():
        raise RuntimeError("The local MJPEG video fixture could not be created")
    try:
        for _ in range(12):
            writer.write(frame)
    finally:
        writer.release()
    video_result = predict_video(str(video), weights=weights, device="cpu", every=4, max_frames=12)
    assert video_result == {"frames_read": 12, "frames_processed": 3}
    report = {"created_utc": utc_now(), "passed": all(item["passed"] for item in parity),
              "weights": weights.name, "weights_sha256": sha256(weights), "dataset_manifest_sha256": dataset_hash,
              "http_prediction_history_and_downloads": True, "invalid_image_returns_400": True,
              "unicode_filename_verified": True, "tile_count": tiled["tile_count"],
              "parity": parity, "pytorch_detection_total": sum(item["pytorch_detections"] for item in parity),
              "sample_policy": "first 12 fixed validation filenames; confidence 0.15; no result-based selection",
              "video": {**video_result, "sample_interval": 4, "fixture_sha256": sha256(video),
                        "scope": "12-frame repeated validation image in MJPEG; decoding and sampling only, not real camera validation"},
              "web_visual_inspection_performed": False}
    if report["pytorch_detection_total"] == 0:
        report["passed"] = False
        report["failure"] = "Empty-only detection parity is insufficient acceptance evidence"
    write_json(ROOT / "artifacts" / "experiments" / "acceptance.json", report)
    if not report["passed"]:
        raise RuntimeError("Acceptance failed; inspect artifacts/experiments/acceptance.json")
    print("Real HTTP, storage, tiling and 12-image backend parity passed")


if __name__ == "__main__":
    main()
