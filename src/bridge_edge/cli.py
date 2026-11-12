"""Small command surface covering the whole reproducible local workflow."""
import argparse
import json
from pathlib import Path
import sys


def _parser():
    parser = argparse.ArgumentParser(prog="bridge-edge", description="桥观：本地桥梁表观病害检测与实验工具")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="核对本地运行环境")
    data = commands.add_parser("data", help="数据下载、转换与审计").add_subparsers(dest="data_command", required=True)
    download = data.add_parser("download")
    download.add_argument("--train-limit", type=int, default=768, help="0 表示全部原始训练集")
    download.add_argument("--val-limit", type=int, default=160, help="0 表示全部原始验证集")
    download.add_argument("--seed", type=int, default=2026)
    download.add_argument("--workers", type=int, default=8)
    for name in ("prepare", "audit"):
        item = data.add_parser(name)
        item.add_argument("--profile", default="pilot")
    smoke = commands.add_parser("model-smoke", help="模型结构前后向检查")
    smoke.add_argument("--device", default="cpu")
    training = commands.add_parser("train", help="运行可追踪训练实验")
    training.add_argument("--config", default="experiments/pilot-yolov8n.yaml")
    training.add_argument("--run-name")
    training.add_argument("--epochs", type=int)
    training.add_argument("--device")
    training.add_argument("--batch", type=int)
    training.add_argument("--seed", type=int)
    evaluation = commands.add_parser("evaluate", help="验证集指标")
    evaluation.add_argument("--weights", required=True)
    evaluation.add_argument("--profile", default="pilot")
    evaluation.add_argument("--device", default="auto")
    evaluation.add_argument("--imgsz", type=int, default=384)
    for name in ("predict", "video"):
        item = commands.add_parser(name, help="图片 / 文件夹检测" if name == "predict" else "视频 / 摄像头帧采样检测")
        item.add_argument("--source", required=True)
        item.add_argument("--weights")
        item.add_argument("--device", default="auto")
        item.add_argument("--confidence", type=float, default=0.25)
        if name == "predict":
            item.add_argument("--imgsz", type=int, default=384)
            item.add_argument("--tile-size", type=int, default=0)
        else:
            item.add_argument("--every", type=int, default=5)
            item.add_argument("--max-frames", type=int, default=300)
    exporting = commands.add_parser("export", help="导出 ONNX 并验证原始输出数值")
    exporting.add_argument("--weights", required=True)
    exporting.add_argument("--imgsz", type=int, default=384)
    exporting.add_argument("--sample")
    bench = commands.add_parser("benchmark", help="在当前设备测量真实端到端耗时")
    bench.add_argument("--weights", required=True)
    bench.add_argument("--source", required=True)
    bench.add_argument("--device", default="auto")
    bench.add_argument("--iterations", type=int, default=30)
    server = commands.add_parser("serve", help="启动本地检测界面")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8765)
    return parser


def doctor():
    import importlib.metadata
    import torch
    from .paths import workspace
    from .runtime.engine import available_models
    return {"python": sys.version.split()[0], "interpreter": sys.executable, "workspace": str(workspace()),
            "cuda_available": torch.cuda.is_available(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "packages": {name: importlib.metadata.version(name) for name in ("ultralytics", "timm", "onnxruntime", "fastapi")},
            "trained_models": available_models(), "cloud_api_required": False}


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor()
        elif args.command == "data":
            if args.data_command == "download":
                from .data.download import download_dataset
                result = download_dataset(args.train_limit, args.val_limit, args.seed, args.workers)
                result = {key: value for key, value in result.items() if key != "records"}
            elif args.data_command == "prepare":
                from .data.prepare import prepare_dataset
                result = prepare_dataset(args.profile)
                result = {key: value for key, value in result.items() if key != "records"}
            else:
                from .data.audit import audit_dataset
                result = audit_dataset(args.profile)
                if not result["passed"]:
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                    return 1
        elif args.command == "model-smoke":
            from .models.registry import smoke_models
            result = smoke_models(args.device)
        elif args.command == "train":
            from .training import train
            result = train(args.config, args.run_name, epochs=args.epochs, device=args.device, batch=args.batch, seed=args.seed)
        elif args.command == "evaluate":
            from .training import evaluate
            result = evaluate(args.weights, args.profile, args.device, args.imgsz)
        elif args.command == "predict":
            from .prediction import predict_images
            result = predict_images(args.source, args.weights, args.device, args.confidence, args.imgsz, args.tile_size)
        elif args.command == "video":
            from .prediction import predict_video
            result = predict_video(args.source, args.weights, args.device, args.confidence, args.every, args.max_frames)
        elif args.command == "export":
            from .deployment import export_onnx
            result = export_onnx(args.weights, args.imgsz, sample=args.sample)
        elif args.command == "benchmark":
            from .deployment import benchmark
            result = benchmark(args.weights, args.source, args.device, iterations=args.iterations)
        else:
            import uvicorn
            uvicorn.run("bridge_edge.server:app", host=args.host, port=args.port, workers=1)
            return 0
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
