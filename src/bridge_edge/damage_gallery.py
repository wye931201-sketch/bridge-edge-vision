"""Balanced photographs of cracks, spalling, rust and industrial scratches."""
from pathlib import Path
import shutil
import tempfile

from PIL import Image

from .comparisons import _gallery
from .industrial import IndustrialDetector, PALETTE
from .paths import location, workspace
from .runtime.engine import Detector, annotate, decode_image, latest_weights
from .utils import read_json, sha256, utc_now, write_json


def select_bridge_groups(records, count):
    groups, used = [], set()
    for source_label, chinese in (("Crack", "裂缝"), ("Spalling", "剥落"), ("Rust", "锈蚀")):
        chosen = [item for item in sorted(records, key=lambda item: item["image"])
                  if item["split"] == "val" and source_label in item.get("source_labels", []) and item["source_path"] not in used][:count]
        if len(chosen) != count:
            raise ValueError("Not enough distinct validation images for " + chinese)
        used.update(item["source_path"] for item in chosen)
        groups.append((chinese, chosen))
    return groups


def export_damage_gallery(output=None, per_category=24, confidence=.15, tile_size=768):
    if not 1 <= per_category <= 24 or not 0 <= confidence <= 1 or tile_size < 0:
        raise ValueError("Use 1–24 images per category and valid inference settings")
    destination = Path(output).expanduser().resolve() if output else workspace() / "对比"
    if destination.exists():
        raise FileExistsError("The output already exists; choose a new folder")
    manifest = location("prepared") / "pilot/manifest.json"
    groups = select_bridge_groups(read_json(manifest)["records"], per_category)
    assets = read_json(location("reports") / "industrial_assets.json")
    if not assets["passed"]:
        raise ValueError("Run scripts/download_industrial.py first")
    bridge_weights = latest_weights().with_suffix(".onnx")
    bridge, industrial = Detector(bridge_weights, device="cpu"), IndustrialDetector()
    tasks = []
    for chinese, records in groups:
        tasks += [(chinese, location("raw") / item["source_path"], item["sha256"], bridge, None) for item in records]
    scratches = sorted(assets["records"], key=lambda item: item["image"])[:per_category]
    if len(scratches) != per_category:
        raise ValueError("Not enough verified scratch images")
    tasks += [("划痕", location("raw") / "industrial" / item["image"], item["sha256"], industrial, PALETTE) for item in scratches]
    destination.parent.mkdir(parents=True, exist_ok=True)
    items = []
    with tempfile.TemporaryDirectory(prefix=".damage-gallery-", dir=destination.parent) as temporary:
        staging = Path(temporary).resolve()
        if staging.parent != destination.parent.resolve() or not staging.name.startswith(".damage-gallery-"):
            raise RuntimeError("Unexpected staging directory")
        for number, (category, source, expected_hash, engine, palette) in enumerate(tasks, 1):
            if sha256(source) != expected_hash:
                raise ValueError("Source photograph changed: " + source.name)
            image = decode_image(source.read_bytes())
            settings = {"tile_size": tile_size} if engine is bridge else {}
            result = engine.predict(image, confidence=confidence, **settings)
            folder = staging / str(number)
            folder.mkdir()
            original = folder / ("原始图片" + source.suffix.lower())
            rendered = folder / "识别结果.jpg"
            shutil.copyfile(source, original)
            annotate(image, result["detections"], palette=palette).save(rendered, quality=95)
            assert sha256(original) == expected_hash
            with Image.open(rendered) as opened:
                assert opened.size == (image.shape[1], image.shape[0])
                opened.verify()
            items.append({"number": number, "category": category, "source_filename": source.name,
                          "source_path": str(source), "original": original.name, "original_sha256": expected_hash,
                          "annotated_sha256": sha256(rendered), "model": engine.path.name,
                          "weights_sha256": engine.weights_sha256, "tile_size": tile_size if engine is bridge else 0,
                          "finding_count": result["finding_count"], "detections": result["detections"]})
            print(f"{number}/{len(tasks)} {category}: {source.name} -> {result['finding_count']} candidates", flush=True)
        description = (f"四组各 {per_category} 张，按来源标注和文件名选择，照片不重复，不按预测效果挑选。"
                       "前三组使用本项目桥梁模型；工业划痕组使用 winthegame259 的公开 NEU-DET 预训练模型。"
                       "划痕图片与该模型训练数据是否重叠尚未确认，本页用于展示，不作为独立测试成绩。")
        _gallery(staging, items, confidence, "桥梁 ONNX + 工业 ONNX", [], tile_size, description)
        staging.rename(destination)
    report = {"created_utc": utc_now(), "passed": True, "output": str(destination), "pairs": len(items),
              "per_category": per_category, "confidence_threshold": confidence, "dataset_manifest_sha256": sha256(manifest),
              "bridge_weights_sha256": bridge.weights_sha256, "industrial_weights_sha256": industrial.weights_sha256,
              "industrial_model_origin": "upstream pretrained release, not trained in this project",
              "independent_test_claimed": False, "items": items}
    write_json(location("reports") / "latest_damage_gallery.json", report)
    return report
