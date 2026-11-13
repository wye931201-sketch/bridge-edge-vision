"""Export numbered original/prediction image pairs for direct visual review."""
from html import escape
from pathlib import Path
import shutil
import tempfile

from PIL import Image

from .paths import location, workspace
from .runtime.engine import Detector, annotate, decode_image, latest_weights
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
    cards = []
    for item in items:
        number = item["number"]
        cards.append(f'<section id="group-{number}"><h2>第 {number} 组 · {escape(item.get("category", ""))} · {escape(item["source_filename"])}</h2>'
                     f'<p>检出候选：{item["finding_count"]} 个 · 模型：{escape(item.get("model", model))} · 切片：{item.get("tile_size", tile_size) or "无"}</p><div class="pair">'
                     f'<figure><figcaption>原始图片</figcaption><a href="{number}/{item["original"]}">'
                     f'<img loading="lazy" src="{number}/{item["original"]}" alt="第{number}组原始图片"></a></figure>'
                     f'<figure><figcaption>识别结果</figcaption><a href="{number}/识别结果.jpg">'
                     f'<img loading="lazy" src="{number}/识别结果.jpg" alt="第{number}组模型识别结果"></a></figure></div></section>')
    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1"><title>原始图片与识别结果对比</title>'
                '<style>body{font:16px system-ui,sans-serif;background:#eef2f5;color:#172b36;margin:24px}'
                'main{max-width:1500px;margin:auto}section{background:white;padding:20px;margin:22px 0;border-radius:10px}'
                'h1{font-size:26px}h2{font-size:18px;overflow-wrap:anywhere}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}'
                'figure{margin:0}figcaption{font-weight:bold;margin-bottom:8px}img{width:100%;height:auto;display:block}'
                '@media(max-width:700px){.pair{grid-template-columns:1fr}body{margin:12px}}</style><main>'
                f'<h1>原始图片与识别结果对比 · {len(items)} 组</h1>'
                f'<p>模型：{escape(model)}　置信度阈值：{confidence:g}。点击图片可查看原尺寸。</p>'
                f'<p>来源标注筛选：{escape("、".join(item.chinese for item in CLASSES if item.name in classes) or "全部类别")}；'
                f'切片尺寸：{tile_size or "不切片"}。来源标注用于选择照片，图片上的框仍由模型实际预测。</p>'
                f'<p>{escape(description)}</p>'
                '<p>数字文件夹内各有一张原始图片和一张模型识别结果。默认样本来自固定验证集，按文件名排序，不按检测效果挑选。</p>'
                '<p>crack：裂缝；spalling：剥落；exposed_rebar：露筋；rust：锈迹；rockpocket：蜂窝麻面。'
                'scratches：工业划痕。框是模型候选，尚需复核；未画框表示当前阈值下未检出，不能据此判断没有病害。</p>'
                + ''.join(cards) + '</main></html>')
    (destination / "打开对比.html").write_text(document, encoding="utf-8")


def export_comparisons(source=None, output=None, weights=None, device="cpu", confidence=0.15, limit=0, classes=None, tile_size=0):
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
            result = engine.predict(image, confidence=confidence, tile_size=tile_size)
            folder = staging / str(number)
            folder.mkdir()
            original = folder / ("原始图片" + path.suffix.lower())
            annotated = folder / "识别结果.jpg"
            shutil.copyfile(path, original)
            annotate(image, result["detections"]).save(annotated, quality=95)
            if sha256(original) != original_hash:
                raise RuntimeError("Original image copy is not byte-identical")
            with Image.open(annotated) as rendered:
                if rendered.size != (image.shape[1], image.shape[0]):
                    raise RuntimeError("Annotated image size differs from its original")
                rendered.verify()
            items.append({"number": number, "source_filename": path.name, "original": original.name,
                          "original_sha256": original_hash, "annotated_sha256": sha256(annotated),
                          "finding_count": result["finding_count"], "detections": result["detections"]})
            print(f"{number}/{len(paths)}: {path.name} -> {result['finding_count']} candidates", flush=True)
        _gallery(staging, items, confidence, engine.path.name, classes, tile_size)
        staging.rename(destination)
    report = {"created_utc": utc_now(), "passed": True, "output": str(destination), "pairs": len(items),
              "model": engine.path.name, "weights_sha256": engine.weights_sha256,
              "backend": engine.backend, "confidence_threshold": confidence,
              "source_class_filters": classes, "tile_size": tile_size,
              "dataset_manifest_sha256": manifest_hash,
              "selection": "fixed validation filenames in order" if source is None else "source filenames in order",
              "originals_byte_identical": True, "items": items}
    write_json(location("reports") / "latest_comparison_export.json", report)
    return report
