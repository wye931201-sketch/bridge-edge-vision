"""The first release covers five visible concrete-surface findings."""
from dataclasses import dataclass


@dataclass(frozen=True)
class DefectClass:
    id: int
    name: str
    chinese: str
    source_label: str
    color: tuple[int, int, int]


CLASSES = (
    DefectClass(0, "crack", "裂缝", "Crack", (245, 99, 77)),
    DefectClass(1, "spalling", "混凝土剥落", "Spalling", (245, 177, 66)),
    DefectClass(2, "exposed_rebar", "露筋", "ExposedRebars", (170, 123, 246)),
    DefectClass(3, "rust", "锈蚀 / 锈迹", "Rust", (67, 190, 184)),
    DefectClass(4, "rockpocket", "蜂窝麻面", "Rockpocket", (88, 158, 247)),
)
NAMES = {item.id: item.name for item in CLASSES}
SOURCE_TO_ID = {item.source_label: item.id for item in CLASSES}
BY_ID = {item.id: item for item in CLASSES}


def class_metadata():
    return [{"id": item.id, "name": item.name, "chinese": item.chinese, "color": item.color} for item in CLASSES]
