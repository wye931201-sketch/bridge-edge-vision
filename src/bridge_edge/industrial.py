"""A separate, pinned six-class ONNX detector for steel-surface examples."""
import ast

from .paths import CONFIG_ROOT, location
from .runtime.geometry import decode_yolo, letterbox
from .utils import read_json, sha256

CHINESE = ("龟裂", "夹杂", "斑块", "麻点", "轧入氧化皮", "划痕")
PALETTE = {0: (245, 99, 77), 1: (170, 123, 246), 2: (245, 177, 66),
           3: (67, 190, 184), 4: (88, 158, 247), 5: (236, 72, 153)}


class IndustrialDetector:
    def __init__(self):
        config = read_json(CONFIG_ROOT / "industrial_sources.json")
        self.path = location("models") / "external" / config["model"]["filename"]
        self.weights_sha256 = sha256(self.path)
        if self.weights_sha256 != config["model"]["sha256"]:
            raise ValueError("Industrial model bytes differ from the pinned release")
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(self.path), sess_options=options, providers=["CPUExecutionProvider"])
        self.names = {i: name for i, name in enumerate(config["names"])}
        metadata = ast.literal_eval(self.session.get_modelmeta().custom_metadata_map.get("names", "{}"))
        if metadata != self.names:
            raise ValueError("Industrial model taxonomy differs from its six-class configuration")
        spec = self.session.get_inputs()[0]
        if len(spec.shape) != 4 or not isinstance(spec.shape[-1], int) or spec.shape[-1] != spec.shape[-2]:
            raise ValueError("Industrial example requires a static square ONNX input")
        self.input_name, self.image_size = spec.name, int(spec.shape[-1])
        self.backend = "onnx-industrial"

    def predict(self, image, confidence=.15):
        tensor, ratio, padding = letterbox(image, self.image_size)
        raw = self.session.run(None, {self.input_name: tensor})[0]
        rows = decode_yolo(raw, image.shape, ratio, padding, confidence, .5, len(self.names))
        detections = [{"class_id": int(row[5]), "label": self.names[int(row[5])], "chinese": CHINESE[int(row[5])],
                       "confidence": round(float(row[4]), 6), "xyxy": [round(float(value), 2) for value in row[:4]]} for row in rows]
        return {"detections": detections, "finding_count": len(detections)}
