"""Loopback-only by default, fully local HTTP application."""
from functools import lru_cache
from pathlib import Path
import re
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .paths import location
from .runtime.engine import Detector, available_models, decode_image, latest_weights, save_prediction
from .store import get_prediction, list_predictions
from .taxonomy import class_metadata
from .utils import read_json

WEB = Path(__file__).parent / "web"
app = FastAPI(title="Bridge Edge Vision", version="0.1.0", docs_url=None, redoc_url=None)
app.mount("/assets", StaticFiles(directory=WEB), name="assets")
_models = {}
_inference_lock = Lock()


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@lru_cache(maxsize=1)
def hardware():
    import torch
    return {"cuda_available": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}


@app.get("/api/health")
def health():
    entries = available_models()
    try:
        default_model = latest_weights().name
    except (FileNotFoundError, KeyError):
        default_model = None
    if default_model not in {entry["id"] for entry in entries}:
        default_model = entries[0]["id"] if entries else None
    return {"status": "ready" if entries else "model_not_ready", "mode": "local", "hardware": hardware(),
            "models": entries, "default_model": default_model, "classes": class_metadata(), "external_api_required": False}


def _sample_map():
    manifest_path = location("prepared") / "pilot" / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = read_json(manifest_path)
    return {Path(item["image"]).name: location("prepared") / "pilot" / item["image"]
            for item in manifest["records"] if item["split"] == "val"}


@app.get("/api/samples")
def samples():
    return {"samples": [{"id": name, "url": f"/api/samples/{name}"} for name in sorted(_sample_map())[:12]],
            "selection": "first 12 filenames of the fixed validation subset; no performance-based selection"}


@app.get("/api/samples/{name}")
def sample_image(name: str):
    path = _sample_map().get(name)
    if path is None:
        raise HTTPException(404, "样例不存在")
    return FileResponse(path, media_type="image/jpeg")


def _predict(content, filename, model_id, confidence, tile):
    registry = {entry["id"] for entry in available_models()}
    model_id = model_id or latest_weights().name
    if model_id not in registry:
        raise ValueError("请从已训练模型列表选择模型")
    image = decode_image(content)
    device = "0" if hardware()["cuda_available"] and not model_id.endswith(".onnx") else "cpu"
    with _inference_lock:
        model_path = location("models") / model_id
        state = model_path.stat()
        key = (model_id, device, state.st_mtime_ns, state.st_size)
        if key not in _models:
            # Avoid keeping every experimental model resident on an 8GB GPU.
            _models.clear()
            _models[key] = Detector(model_path, device=device)
        result = _models[key].predict(image, confidence=confidence, tile_size=768 if tile else 0)
    saved = save_prediction(image, result, filename)
    saved["annotated_url"] = f"/api/artifacts/{saved['id']}/annotated.jpg"
    saved["json_url"] = f"/api/artifacts/{saved['id']}/result.json"
    return saved


@app.post("/api/predict")
async def predict(request: Request, filename: str = "image.jpg", model: str = "", confidence: float = 0.25, tile: bool = False):
    content = bytearray()
    async for block in request.stream():
        content.extend(block)
        if len(content) > 25 * 1024 * 1024:
            raise HTTPException(413, "图片超过 25 MiB")
    try:
        return await run_in_threadpool(_predict, bytes(content), Path(filename).name, model, confidence, tile)
    except FileNotFoundError as error:
        raise HTTPException(503, "模型尚未就绪，请先完成训练") from error
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.get("/api/history")
def history(limit: int = 20, offset: int = 0):
    try:
        return {"items": list_predictions(limit, offset)}
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.get("/api/history/{identifier}")
def history_item(identifier: str):
    if not re.fullmatch(r"[a-f0-9]{32}", identifier):
        raise HTTPException(404)
    item = get_prediction(identifier)
    if item is None:
        raise HTTPException(404)
    return {**item, "annotated_url": f"/api/artifacts/{identifier}/annotated.jpg", "json_url": f"/api/artifacts/{identifier}/result.json"}


@app.get("/api/artifacts/{identifier}/{filename}")
def artifact(identifier: str, filename: str):
    if not re.fullmatch(r"[a-f0-9]{32}", identifier) or filename not in {"annotated.jpg", "result.json"}:
        raise HTTPException(404)
    path = location("predictions") / identifier / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, filename=f"{identifier}-{filename}")
