"""Cold-process inference with all outbound socket connections denied."""
import argparse
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    attempts = []

    def reject_connection(event, parameters):
        if event == "socket.connect":
            attempts.append("blocked socket connection")
            raise OSError("Offline acceptance test denies all network connections")

    sys.addaudithook(reject_connection)
    from bridge_edge.runtime.engine import Detector, decode_image
    from bridge_edge.utils import sha256, utc_now, write_json
    from bridge_edge.paths import location
    image = decode_image(Path(args.source).read_bytes())
    engine = Detector(args.weights, device=args.device)
    result = engine.predict(image)
    report = {"created_utc": utc_now(), "passed": not attempts, "socket_connections_attempted": len(attempts),
              "outbound_sockets_denied": True, "backend": engine.backend, "device": args.device,
              "weights": engine.path.name, "weights_sha256": engine.weights_sha256,
              "source_sha256": sha256(Path(args.source)), "finding_count": result["finding_count"],
              "scope": "cold Python process, model loading and one real image prediction; not an OS-wide network isolation claim"}
    write_json(location("reports") / ("offline-" + engine.backend + ".json"), report)
    if not report["passed"]:
        raise RuntimeError("Inference attempted a network connection")
    print(report)


if __name__ == "__main__":
    main()
