"""Machine-specific storage stays outside version-controlled configuration."""
import json
import os
import sys
from pathlib import Path

_development_root = Path(__file__).resolve().parents[2]
ROOT = _development_root if (_development_root / "pyproject.toml").is_file() else Path.cwd()
CONFIG_ROOT = _development_root / "configs" if (_development_root / "configs").is_dir() else Path(sys.prefix) / "share" / "bridge-edge-vision" / "configs"


def workspace() -> Path:
    value = os.environ.get("BRIDGE_EDGE_WORKSPACE")
    local = ROOT / ".local" / "settings.json"
    if not value and local.is_file():
        value = json.loads(local.read_text("utf-8")).get("workspace")
    target = Path(value).expanduser().resolve() if value else ROOT / "data"
    target.mkdir(parents=True, exist_ok=True)
    return target


def location(name: str) -> Path:
    if name not in {"cache", "raw", "prepared", "runs", "models", "predictions", "reports"}:
        raise ValueError(f"Unknown storage area: {name}")
    result = workspace() / name
    result.mkdir(parents=True, exist_ok=True)
    return result


def resolve_config(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path.resolve()
    candidate = CONFIG_ROOT / path
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Configuration not found: {path}")
