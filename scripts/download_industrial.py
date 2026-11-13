"""Cache pinned scratch photographs, annotations and an upstream ONNX model."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import sys
from urllib.request import getproxies
import xml.etree.ElementTree as ET

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_edge.paths import CONFIG_ROOT, location
from bridge_edge.utils import read_json, sha256, utc_now, write_json


def main():
    config = read_json(CONFIG_ROOT / "industrial_sources.json")
    raw = location("raw") / "industrial"
    raw.mkdir(exist_ok=True)
    model = location("models") / "external" / config["model"]["filename"]
    model.parent.mkdir(exist_ok=True)
    proxy = getproxies().get("https") or getproxies().get("http")
    tasks = []
    for item in config["images"]:
        for key in ("image", "annotation"):
            relative = item[key]
            url = f"https://raw.githubusercontent.com/{config['image_repository']}/{config['image_commit']}/{relative}"
            tasks.append((url, raw / Path(relative).name, item[key + "_git_oid"], "git"))
    tasks.append((config["model"]["url"], model, config["model"]["sha256"], "sha256"))
    with httpx.Client(proxy=proxy, follow_redirects=True, timeout=90) as client:
        def fetch(task):
            url, target, expected, kind = task
            content = target.read_bytes() if target.is_file() else client.get(url).raise_for_status().content
            digest = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest() if kind == "git" else hashlib.sha256(content).hexdigest()
            if digest != expected:
                raise ValueError("Downloaded bytes do not match the pinned source: " + target.name)
            if not target.exists():
                temporary = target.with_suffix(target.suffix + ".part")
                temporary.write_bytes(content)
                temporary.replace(target)
            return target.name
        for filename in ThreadPoolExecutor(max_workers=4).map(fetch, tasks):
            print("Verified " + filename, flush=True)
    records = []
    for item in config["images"]:
        image, annotation = raw / Path(item["image"]).name, raw / Path(item["annotation"]).name
        xml = ET.parse(annotation).getroot()
        with Image.open(image) as opened:
            size = opened.size
            opened.verify()
        assert size == (int(xml.findtext("size/width")), int(xml.findtext("size/height"))) == (200, 200)
        boxes, labels = [], set()
        for obj in xml.findall("object"):
            label = obj.findtext("name")
            assert label in config["names"]
            labels.add(label)
            box = [int(obj.findtext("bndbox/" + key)) for key in ("xmin", "ymin", "xmax", "ymax")]
            assert 0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1]
            boxes.append({"label": label, "xyxy": box})
        assert boxes and "scratches" in labels
        records.append({"image": image.name, "annotation": annotation.name,
                        "sha256": sha256(image), "annotation_sha256": sha256(annotation), "source_boxes": boxes,
                        "source_labels": sorted(labels)})
    write_json(location("reports") / "industrial_assets.json", {"created_utc": utc_now(), "passed": True,
               "config_sha256": sha256(CONFIG_ROOT / "industrial_sources.json"), "model_sha256": sha256(model),
               "records": records, "independent_test_claimed": False})
    print("Industrial assets verified; scratch images are cached locally")


if __name__ == "__main__":
    main()
