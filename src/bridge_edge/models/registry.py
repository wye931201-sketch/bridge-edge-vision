import os
import sys
from pathlib import Path

from ..paths import location, resolve_config

VARIANTS = ("yolov8n", "repvit", "repvit-amsf")


def configure_runtime():
    if sys.platform == "win32":
        # This development stack exhibited asynchronous CUDA failures during
        # long training. Keep the tested conservative path explicit by default.
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(location("cache") / "ultralytics"))
    os.environ.setdefault("TORCH_HOME", str(location("cache") / "torch"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("WANDB_MODE", "disabled")


def register_modules():
    configure_runtime()
    import ultralytics.nn.tasks as tasks
    from .layers import AdaptiveMultiScale, RepViTBackbone
    tasks.RepViTBackbone = RepViTBackbone
    tasks.AdaptiveMultiScale = AdaptiveMultiScale


def build_detector(variant="yolov8n", pretrained=True):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown detector variant: {variant}")
    register_modules()
    from ultralytics import YOLO
    if variant == "yolov8n":
        weights = location("models") / "initial" / "yolov8n.pt"
        if pretrained:
            if not weights.exists():
                raise FileNotFoundError("Download initial weights before training")
            return YOLO(str(weights), task="detect")
        return YOLO("yolov8n.yaml", task="detect")
    detector = YOLO(str(resolve_config(Path("models") / f"{variant}.yaml")), task="detect")
    if pretrained:
        weights = location("models") / "initial" / "repvit_m0_9_imagenet.pth"
        if not weights.exists():
            raise FileNotFoundError("RepViT ImageNet initial weights were not found")
        detector.model.model[0].load_imagenet(weights)
    return detector


def load_detector(weights: str | Path):
    register_modules()
    from ultralytics import YOLO
    path = Path(weights)
    if not path.is_file():
        candidate = location("models") / weights
        if not candidate.is_file():
            raise FileNotFoundError(f"Model file not found: {weights}")
        path = candidate
    return YOLO(str(path), task="detect")


def smoke_models(device="cpu", image_size=256):
    import torch
    from ..utils import utc_now, write_json
    results = []
    for variant in VARIANTS:
        detector = build_detector(variant, pretrained=variant != "yolov8n")
        model = detector.model.to(device).train()
        image = torch.randn(2, 3, image_size, image_size, device=device)
        prediction = model(image)
        tensors = prediction if isinstance(prediction, list) else list(prediction.values()) if isinstance(prediction, dict) else [prediction]
        tensors = [item for item in tensors if isinstance(item, torch.Tensor)]
        loss = sum(item.square().mean() for item in tensors)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite model smoke objective")
        loss.backward()
        results.append({"variant": variant, "parameters": sum(p.numel() for p in model.parameters()),
                        "device": str(next(model.parameters()).device), "feature_shapes": [list(item.shape) for item in tensors],
                        "forward_backward_passed": True, "purpose": "synthetic model integrity; not detection accuracy"})
        del model, detector, image, prediction, tensors, loss
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    report = {"created_utc": utc_now(), "models": results, "passed": True}
    write_json(location("reports") / "model_smoke.json", report)
    return report
