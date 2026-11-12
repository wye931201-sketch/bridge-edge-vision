"""PyCharm-friendly entry point; the package can also be installed with pip -e ."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from bridge_edge.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
