"""Export numbered original/prediction image pairs for direct visual review."""
import json
from pathlib import Path
import shutil
import tempfile

from PIL import Image

from .paths import location, workspace
from .presentation import render_result_card, result_state, EXPECTED_LABELS
from .runtime.engine import Detector, decode_image, latest_weights
from .runtime.inspection import BridgeInspector
from .taxonomy import CLASSES, NAMES
from .utils import read_json, sha256, utc_now, write_json

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _inputs(source, classes):
    if source is not None:
        if classes:
            raise ValueError("Class filtering requires the annotated validation dataset, not an unlabelled source folder")
        source = Path(source).resolve()
        paths = sorted((item for item in source.iterdir() if item.is_file() and item.suffix.lower() in EXTENSIONS),
                       key=lambda item: item.name) if source.is_dir() else [source]
        if not paths or any(not item.is_file() or item.suffix.lower() not in EXTENSIONS for item in paths):
            raise ValueError("Source must contain supported image files")
        return [(item, None) for item in paths], None
    manifest = location("prepared") / "pilot" / "manifest.json"
    records = sorted((item for item in read_json(manifest)["records"] if item["split"] == "val"),
                     key=lambda item: item["image"])
    source_labels = {item.source_label for item in CLASSES if item.name in classes}
    if source_labels:
        records = [item for item in records if source_labels.intersection(item.get("source_labels", []))]
    if not records:
        raise ValueError("No validation images are available")
    raw = location("raw").resolve()
    paths = []
    for item in records:
        path = (raw / item["source_path"]).resolve()
        if not path.is_relative_to(raw):
            raise ValueError("Source image is outside the dataset directory")
        paths.append((path, item["sha256"]))
    return paths, sha256(manifest)


def _gallery(destination, items, confidence, model, classes, tile_size, description=""):
    display = []
    for item in items:
        state = result_state(item["detections"], EXPECTED_LABELS.get(item.get("category")))
        display.append({"number": item["number"], "category": item.get("category", ""),
                        "source_filename": item["source_filename"], "original": item["original"],
                        "finding_count": item["finding_count"], "status": state["status"],
                        "detected_classes": sorted({d.get("chinese", d["label"]) for d in item["detections"]}),
                        "model": item.get("model_display_name", item.get("model", model)), "profile": item.get("inspection_profile", "standard")})
    content = json.dumps({"description": description or f"模型：{model}；阈值：{confidence:g}。按文件名排序的真实预测结果。",
                          "items": display}, ensure_ascii=False)
    # Filenames and source metadata are data, never executable HTML.
    content = content.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    web = Path(__file__).parent / "web"
    document = (web / "gallery.html").read_text("utf-8").replace("__GALLERY_DATA__", content)
    (destination / "打开对比.html").write_text(document, encoding="utf-8")
    for filename in ("gallery.css", "gallery.js"):
        shutil.copyfile(web / filename, destination / filename)


def export_comparisons(source=None, output=None, weights=None, device="cpu", confidence=0.15, limit=0, classes=None, tile_size=0, profile=None, crack_mode="auto"):
    if limit < 0 or not 0 <= confidence <= 1 or tile_size < 0:
        raise ValueError("Use a nonnegative limit and a confidence threshold between zero and one")
    classes = list(classes or [])
    if set(classes) - set(NAMES.values()):
        raise ValueError("Unknown class filter")
    destination = Path(output).expanduser().resolve() if output else workspace() / "对比"
    if destination.exists():
        raise FileExistsError("The comparison folder already exists; choose a new output folder")
    paths, manifest_hash = _inputs(source, classes)
    if limit:
        paths = paths[:limit]
    selected = Path(weights) if weights else latest_weights()
    if not weights and selected.with_suffix(".onnx").is_file():
        card = selected.with_suffix(".export.json")
        if card.is_file() and read_json(card).get("passed") is True:
            selected = selected.with_suffix(".onnx")
    engine = Detector(selected, device=device)
    inspector = BridgeInspector(engine, profile, crack_mode) if profile else None
    category = next((item.chinese for item in CLASSES if len(classes) == 1 and item.name == classes[0]), None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    items = []
    # Only this newly created staging directory is eligible for automatic cleanup.
    # An existing output directory, dataset or model is never overwritten/deleted.
    with tempfile.TemporaryDirectory(prefix=".comparison-", dir=destination.parent) as temporary:
        staging = Path(temporary).resolve()
        if staging.parent != destination.parent.resolve() or not staging.name.startswith(".comparison-"):
            raise RuntimeError("Unexpected comparison staging directory")
        for number, (path, expected_hash) in enumerate(paths, 1):
            original_hash = sha256(path)
            if expected_hash and original_hash != expected_hash:
                raise ValueError("Original dataset image changed: " + path.name)
            image = decode_image(path.read_bytes())
            result = inspector.predict(image, confidence=confidence, tile_size=tile_size) if inspector else engine.predict(image, confidence=confidence, tile_size=tile_size)
            folder = staging / str(number)
            folder.mkdir()
            original = folder / ("原始图片" + path.suffix.lower())
            annotated = folder / "识别结果.jpg"
            shutil.copyfile(path, original)
            profile_name = result.get("inspection_profile_name", "standard")
            card, rendering = render_result_card(image, result["detections"], model=inspector.display_name if inspector else engine.path.name,
                                                category=category, confidence=confidence, profile=profile_name,
                                                crack_mask=result.get("_native_crack_mask"))
            card.save(annotated, quality=95)
            if sha256(original) != original_hash:
                raise RuntimeError("Original image copy is not byte-identical")
            with Image.open(annotated) as rendered:
                if list(rendered.size) != rendering["rendered_size"]:
                    raise RuntimeError("Result card size differs from its rendering record")
                rendered.verify()
            items.append({"number": number, "source_filename": path.name, "original": original.name,
                          "original_sha256": original_hash, "annotated_sha256": sha256(annotated),
                          "finding_count": result["finding_count"], "detections": result["detections"],
                          "rendering": rendering, "inspection_profile": profile_name, "category": category or "",
                          "crack_auxiliary": result.get("crack_auxiliary")})
            print(f"{number}/{len(paths)}: {path.name} -> {result['finding_count']} candidates", flush=True)
        _gallery(staging, items, confidence, engine.path.name, classes, tile_size)
        staging.rename(destination)
    report = {"created_utc": utc_now(), "passed": True, "output": str(destination), "pairs": len(items),
              "model": engine.path.name, "weights_sha256": engine.weights_sha256,
              "backend": engine.backend, "confidence_threshold": confidence,
              "source_class_filters": classes, "tile_size": tile_size, "inspection_profile": inspector.profile if inspector else "standard",
              "root_files": ["打开对比.html", "gallery.css", "gallery.js"],
              "dataset_manifest_sha256": manifest_hash,
              "selection": "fixed validation filenames in order" if source is None else "source filenames in order",
              "originals_byte_identical": True, "items": items}
    write_json(location("reports") / "latest_comparison_export.json", report)
    return report
