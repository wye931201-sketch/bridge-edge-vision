"""Download a pinned optional crack graph without installing a training stack."""
from pathlib import Path
import sys
from urllib.request import getproxies
import zipfile

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_edge.paths import CONFIG_ROOT, location
from bridge_edge.utils import read_json, sha256


def main():
    config = read_json(CONFIG_ROOT / "crack_segmentation.json")
    destination = location("models") / "external/crack_segmentation"
    destination.mkdir(parents=True, exist_ok=True)
    model = destination / config["filename"]
    if model.exists():
        if sha256(model) != config["sha256"]:
            raise ValueError("Cached crack model differs from the pinned graph")
        print("Crack model is already cached and verified")
        return
    archive = location("cache") / "uavcrack_r10_onnx.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        proxy = getproxies().get("https") or getproxies().get("http")
        with httpx.Client(proxy=proxy, follow_redirects=True, timeout=120) as client:
            with client.stream("GET", config["archive_url"]) as response:
                response.raise_for_status()
                received = 0
                with temporary.open("wb") as handle:
                    for block in response.iter_bytes(2**20):
                        handle.write(block)
                        received += len(block)
                        if received % (32 * 2**20) < 2**20:
                            print(f"Downloaded {received // 2**20} MiB", flush=True)
        temporary.replace(archive)
    if sha256(archive) != config["archive_sha256"]:
        raise ValueError("Crack model archive checksum mismatch")
    # Read one named graph; do not extract arbitrary archive paths.
    with zipfile.ZipFile(archive) as package:
        content = package.read(config["archive_member"])
    import hashlib
    if hashlib.sha256(content).hexdigest() != config["sha256"]:
        raise ValueError("Crack graph checksum mismatch")
    temporary = model.with_suffix(".part")
    temporary.write_bytes(content)
    temporary.replace(model)
    print("Verified optional upstream crack graph; no ensemble score is claimed")


if __name__ == "__main__":
    main()
