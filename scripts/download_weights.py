"""Acquire immutable upstream initialization files and verify their hashes."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx
from bridge_edge.data.download import _fetch
from bridge_edge.paths import location
from bridge_edge.utils import read_json, sha256, utc_now, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("all", "yolov8n", "repvit"), default="all")
    args = parser.parse_args()
    specifications = read_json(ROOT / "configs" / "weights.json")
    records = []
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        for name, spec in specifications.items():
            if args.only not in ("all", name):
                continue
            target = location("models") / "initial" / spec["target"]
            digest = sha256(target) if target.exists() else None
            equivalence = None
            if digest == spec.get("accepted_local_reserialization_sha256"):
                # Existing environment used torch.save(timm.state_dict()). Verify
                # its numerical contents against the exact original checkpoint.
                official = location("cache") / "initial_weights" / spec["filename"]
                url = f"https://huggingface.co/{spec['repository']}/resolve/{spec['revision']}/{spec['filename']}"
                _fetch(client, url, official, spec["sha256"])
                import torch
                left = torch.load(target, map_location="cpu", weights_only=True)
                right = torch.load(official, map_location="cpu", weights_only=True)
                equivalence = left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)
                if not equivalence:
                    raise ValueError("Existing RepViT state differs from pinned upstream tensors")
            else:
                if digest is not None and digest != spec["sha256"]:
                    raise FileExistsError(f"Unexpected existing weights: {target.name}; preserve and inspect them first")
                url = f"https://huggingface.co/{spec['repository']}/resolve/{spec['revision']}/{spec['filename']}"
                _fetch(client, url, target, spec["sha256"])
            records.append({"name": name, "repository": spec["repository"], "revision": spec["revision"],
                            "upstream_sha256": spec["sha256"], "local_sha256": sha256(target),
                            "target": target.name, "local_tensors_equal_upstream": equivalence})
            print(f"Verified initial weights: {name}", flush=True)
    write_json(location("reports") / "initial_weights.json", {"created_utc": utc_now(), "weights": records})


if __name__ == "__main__":
    main()
