"""Small, shared I/O primitives with deterministic manifests."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import tempfile


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: str | Path, value) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)
    return path


def revision(root: Path) -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def training_fingerprint(root: Path) -> str:
    """Core training/model sources, independent of local paths and result files."""
    files = list((root / "src" / "bridge_edge" / "models").glob("*.py"))
    files += [root / "src" / "bridge_edge" / "training.py"]
    files += list((root / "configs" / "models").glob("*.yaml"))
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def stable_order(values, seed: int = 2026):
    return sorted(values, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).digest())


def portable_path(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()
