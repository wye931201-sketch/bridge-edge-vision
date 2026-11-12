"""Pinned public data acquisition, without requiring FiftyOne or MongoDB."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path, PurePosixPath
import time
from urllib.parse import quote

import httpx
from PIL import Image

from ..paths import location, resolve_config
from ..taxonomy import SOURCE_TO_ID
from ..utils import read_json, sha256, stable_order, utc_now, write_json


def source_spec():
    return read_json(resolve_config("sources.json"))


def _url(spec, path: str) -> str:
    return f"https://huggingface.co/datasets/{spec['mirror']}/resolve/{spec['revision']}/{quote(path, safe='/')}?download=true"


def _safe_relative(value: str) -> Path:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "data":
        raise ValueError(f"Unexpected source image path: {value}")
    return Path(*path.parts)


def _fetch(client, url: str, target: Path, expected_sha: str | None = None):
    if target.exists() and expected_sha and sha256(target) == expected_sha:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    for attempt in range(4):
        try:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_bytes(1024 * 1024):
                        handle.write(chunk)
            if expected_sha and sha256(temporary) != expected_sha:
                raise ValueError("Downloaded content did not match the pinned SHA256")
            temporary.replace(target)
            return
        except (httpx.HTTPError, OSError, ValueError):
            if attempt == 3:
                raise
            time.sleep(min(8, 2 ** attempt))


def metadata(client=None):
    spec = source_spec()
    path = location("cache") / "samples.json"
    if not path.exists() or sha256(path) != spec["samples_sha256"]:
        if client is None:
            with httpx.Client(follow_redirects=True, timeout=90) as temporary_client:
                _fetch(temporary_client, _url(spec, "samples.json"), path, spec["samples_sha256"])
        else:
            _fetch(client, _url(spec, "samples.json"), path, spec["samples_sha256"])
    content = read_json(path)
    return content["samples"] if isinstance(content, dict) else content


def select_samples(samples, train_limit: int, val_limit: int, seed: int):
    """Uniform, deterministic image selection inside the unchanged source splits."""
    chosen = []
    counts = {}
    for source_split, split, limit in (("train", "train", train_limit), ("validation", "val", val_limit)):
        pool = [sample for sample in samples if source_split in sample.get("tags", [])]
        by_name = {sample["filepath"]: sample for sample in pool}
        names = stable_order(by_name, seed)
        if limit > 0:
            names = names[:limit]
        counts[split] = {"source_count": len(pool), "selected": len(names)}
        chosen.extend((split, by_name[name]) for name in names)
    return chosen, counts


def _file_index(client, spec, chosen):
    """Git blob or LFS hashes make image downloads independently verifiable."""
    folder_names = sorted({str(PurePosixPath(sample["filepath"]).parent) for _, sample in chosen})
    result = {}
    for folder in folder_names:
        cache = location("cache") / (folder.replace("/", "_") + "_tree.json")
        if cache.exists():
            files = read_json(cache)
        else:
            files = []
            url = f"https://huggingface.co/api/datasets/{spec['mirror']}/tree/{spec['revision']}/{folder}?limit=1000"
            while url:
                response = client.get(url)
                response.raise_for_status()
                files.extend(response.json())
                url = response.links.get("next", {}).get("url")
            write_json(cache, files)
        for item in files:
            if item.get("type") == "file":
                result[item["path"]] = item
    return result


def _verify_image(path: Path, sample, entry):
    raw = path.read_bytes()
    lfs_sha = (entry.get("lfs") or {}).get("oid")
    if lfs_sha:
        if hashlib.sha256(raw).hexdigest() != lfs_sha:
            raise ValueError("Image LFS SHA256 mismatch")
    else:
        digest = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if digest != entry["oid"]:
            raise ValueError("Image Git blob SHA1 mismatch")
    with Image.open(path) as image:
        image.load()
        width, height = image.size
    expected = sample["metadata"]
    if (width, height) != (expected["width"], expected["height"]):
        raise ValueError("Image dimensions differ from source annotation metadata")
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "width": width, "height": height}


def download_dataset(train_limit=768, val_limit=160, seed=2026, workers=8):
    spec = source_spec()
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(90, connect=15),
                      limits=httpx.Limits(max_connections=workers * 2)) as client:
        chosen, counts = select_samples(metadata(client), train_limit, val_limit, seed)
        index = _file_index(client, spec, chosen)

        def acquire(pair):
            split, sample = pair
            relative = _safe_relative(sample["filepath"])
            target = location("raw") / relative
            entry = index.get(sample["filepath"])
            if entry is None:
                raise ValueError("The pinned repository index does not include a selected image")
            if not target.exists():
                _fetch(client, _url(spec, sample["filepath"]), target, (entry.get("lfs") or {}).get("oid"))
            verified = _verify_image(target, sample, entry)
            return {"split": split, "source_split": sample["tags"][0], "source_path": sample["filepath"],
                    **verified, "source_git_oid": entry["oid"],
                    "source_labels": sorted({d["label"] for d in (sample.get("od_ground_truth") or {}).get("detections", [])})}

        records = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(acquire, pair) for pair in chosen]
            for future in as_completed(futures):
                records.append(future.result())
                if len(records) % 50 == 0 or len(records) == len(chosen):
                    print(f"Verified {len(records)}/{len(chosen)} bridge images", flush=True)
    report = {"created_utc": utc_now(), "source": spec, "seed": seed, "selection": "SHA256-ranked filenames within original splits",
              "counts": counts, "records": sorted(records, key=lambda record: (record["split"], record["source_path"])),
              "scope": "pilot" if train_limit > 0 or val_limit > 0 else "full_official_labeled_splits"}
    destination = location("raw") / "download_manifest.json"
    write_json(destination, report)
    print(f"Download manifest: {destination}", flush=True)
    return report
