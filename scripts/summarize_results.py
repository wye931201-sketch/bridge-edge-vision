"""Create shareable figures and a model card from completed measured runs."""
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bridge_edge.paths import location
from bridge_edge.utils import read_json, sha256, write_json


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = read_json(ROOT / "artifacts" / "experiments" / "pilot_results.json")
    order = {"yolov8n": 0, "repvit": 1, "repvit-amsf": 2}
    experiments = sorted(report["experiments"], key=lambda item: order[item["architecture"]])
    if len(experiments) != 3 or any(item.get("execution", {}).get("completed_epochs") != item["epochs"] for item in experiments):
        raise RuntimeError("Only three completed same-budget runs can be summarized")
    budgets = {(item["image_size"], item["epochs"], item["seed"]) for item in experiments}
    if len(budgets) != 1:
        raise RuntimeError("Input size, epochs and seed must agree across the comparison")
    image_size, epochs, seed = next(iter(budgets))
    train_count, val_count = report["dataset"]["stats"]["train"]["images"], report["dataset"]["stats"]["val"]["images"]
    history = []
    for item in experiments:
        path = location("runs") / item["run"] / "results.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = [{key.strip(): float(value) for key, value in row.items() if value.strip()} for row in csv.DictReader(handle)]
        if len(rows) != item["epochs"]:
            raise RuntimeError("Training history does not match completed epoch budget")
        history.append({"run": item["run"], "source_csv_sha256": sha256(path), "epochs": rows})
    write_json(ROOT / "artifacts" / "experiments" / "learning_curves.json", {"history": history})
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.7), layout="constrained")
    colors = ["#087f83", "#4468a3", "#b17442"]
    labels = ["YOLOv8n", "RepViT+YOLO", "+AMSF"]
    scores = [item["metrics"]["aggregate"]["metrics/mAP50-95(B)"] * 100 for item in experiments]
    bars = axes[0].bar(labels, scores, color=colors)
    axes[0].bar_label(bars, fmt="%.2f", padding=3)
    axes[0].set(ylabel="Box mAP50-95 (%)", title=f"Fixed pilot validation ({val_count} images)", ylim=(0, max(scores) * 1.3))
    bars = axes[1].bar(labels, [item["parameters"] / 1e6 for item in experiments], color=colors)
    axes[1].bar_label(bars, fmt="%.2f", padding=3)
    axes[1].set(ylabel="Parameters (million)", title="Measured trained model size")
    for item, label, color in zip(history, labels, colors):
        axes[2].plot([row["epoch"] for row in item["epochs"]],
                     [row["metrics/mAP50-95(B)"] * 100 for row in item["epochs"]], label=label, color=color)
    axes[2].set(xlabel="Epoch", ylabel="Box mAP50-95 (%)", title=f"Single seed: {seed}")
    axes[2].legend(frameon=False)
    folder = ROOT / "artifacts" / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    figure.savefig(folder / "pilot_comparison.png", dpi=180)
    figure.savefig(folder / "pilot_comparison.svg")
    svg = folder / "pilot_comparison.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text("utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(figure)
    lines = ["# 第一版模型卡与实测结果", "", f"固定 {train_count} 训练 / {val_count} 验证图，输入 {image_size}，{epochs} 轮，seed {seed}。单种子预实验，未评估独立测试集。", "",
             "| 模型 | 参数量 | mAP50 (0–1) | mAP50-95 (0–1) | 训练完成轮数 |", "|---|---:|---:|---:|---:|"]
    for item in experiments:
        metrics = item["metrics"]["aggregate"]
        lines.append(f"| {item['architecture']} | {item['parameters']:,} | {metrics['metrics/mAP50(B)']:.4f} | {metrics['metrics/mAP50-95(B)']:.4f} | {item['execution']['completed_epochs']} |")
    lines += ["", f"网页默认模型：`{report['selected_by_validation']}`，按本预实验验证集 mAP50-95 选择，未据此声称独立测试泛化能力。", "",
              "![真实预实验图](../artifacts/figures/pilot_comparison.png)", "", "## 部署实测", "",
              "| 后端 / 设备 | p50 (ms) | p95 (ms) | 平均 (ms) |", "|---|---:|---:|---:|"]
    for item in report["benchmarks"]:
        latency = item["latency_ms"]
        lines.append(f"| {item['backend']} / {item['hardware']} | {latency['p50']:.2f} | {latency['p95']:.2f} | {latency['mean']:.2f} |")
    lines += ["", "测速包含预处理、推理和 NMS，排除图片解码、模型加载、标注保存与 HTTP/相机采集；初始化后 5 次预热、30 次计时、batch 1、FP32。此处没有 Jetson 或 TensorRT 实测。", "",
              "验证证据：[HTTP、切片、视频采样与后端一致性](../artifacts/experiments/acceptance.json)、[PyTorch 离线进程](../artifacts/experiments/offline-pytorch.json)、[ONNX 离线进程](../artifacts/experiments/offline-onnx.json)、[无 PyTorch 的安装包推理](../artifacts/experiments/edge-wheel-onnx.json)。", "",
              "## 使用边界", "", "五类图像可见病害候选；不输出病害深度、毫米尺寸、结构承载力或安全等级。",
              "当前检测精度较低，本轮用于建立可复核的工程基线；需要扩充训练与独立数据验证后再评价实用效果。",
              "数据框来自分割多边形，类别不平衡，拍摄标记与修补痕迹可能造成捷径学习。应补充多种子、跨桥梁/跨来源和独立采集验证。",
              "YOLOv8n 使用 COCO 检测预训练，RepViT 使用 ImageNet 骨干初始化，因此二者比较包含初始化差异；RepViT 与 +AMSF 的共享参数已对齐，用于模块消融。",
              "三组数据与训练预算相同，CUDA 调度设置记录在 JSON 中；训练耗时不作为严格速度对照，部署时延另行测量。",
              "多尺度模块是否有益以数值为准；本项目没有证明论文创新性。源码 AGPL-3.0-only；数据遵守 CC BY-NC 4.0，模型权重不随源码公开。", ""]
    (ROOT / "docs" / "MODEL_CARD.md").write_text("\n".join(lines), encoding="utf-8")
    readme = ROOT / "README.md"
    source = readme.read_text("utf-8")
    start, end = "<!-- experiment-status:start -->", "<!-- experiment-status:end -->"
    rows = [f"已完成三组相同数据与训练预算的 CPU 预实验：**{train_count} 训练 / {val_count} 验证图**，{epochs} 轮，输入 {image_size}，seed {seed}。", "",
            "| 模型 | mAP50 | mAP50–95 |", "|---|---:|---:|"]
    for item in experiments:
        values = item["metrics"]["aggregate"]
        rows.append(f"| {item['architecture']} | {values['metrics/mAP50(B)']:.2%} | {values['metrics/mAP50-95(B)']:.2%} |")
    rows += ["", "这是单种子小样本预实验，当前精度还需要改进。完整数值、限制与设备实测见[模型卡](docs/MODEL_CARD.md)和[实验 JSON](artifacts/experiments/pilot_results.json)。"]
    if start in source and end in source:
        source = source.split(start)[0] + start + "\n" + "\n".join(rows) + "\n" + end + source.split(end, 1)[1]
        readme.write_text(source, encoding="utf-8")
    print("Generated measured comparison figures and model card")


if __name__ == "__main__":
    main()
