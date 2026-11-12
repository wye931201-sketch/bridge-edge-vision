# 第一次运行

本项目第一阶段是把真实桥梁图片变成可复核的本地检测结果。无需电路板、百炼密钥或大模型服务。相机与 Jetson 可以在后续阶段接入。

## 已配置的这台电脑

1. 在 PyCharm 打开 `BridgeEdgeVision` 工程文件夹，选择本项目独立的 Python 3.12 CPU 解释器；Windows 启动脚本已在本机配置该路径。
2. 运行 `main.py`，参数填 `doctor`。验收：Python/依赖版本正确，且有训练完成的模型列表。当前采用 CPU 环境，`cuda_available=false` 符合这套运行方式。
3. 把运行参数改成 `serve`。验收：终端显示本机地址 `http://127.0.0.1:8765`。
4. 浏览器访问该地址，点击任意固定验证样例或选择自己的巡检照片，再点击“开始检测”。
5. 查看类别、置信度与像素坐标，下载标注图片和 JSON。历史记录保存在本机数据目录。

网页仅依赖本地静态文件和本机 HTTP 接口。默认使用验证集选出的预实验模型；模型下拉框可以切换各个对照模型及 ONNX CPU 版本。

## 在另一台电脑重建

在工程根目录依次运行（Python 3.12）：

```powershell
uv sync
uv run bridge-edge doctor
uv run python scripts/download_weights.py
uv run bridge-edge data download --train-limit 768 --val-limit 160
uv run bridge-edge data prepare --profile pilot
uv run bridge-edge data audit --profile pilot
uv run python scripts/run_pilot.py
uv run bridge-edge serve
```

验收输出包括：下载来源及 SHA256、数据清单与标签审计、三组训练记录、ONNX 数值核验、真实时延和 `artifacts/experiments/pilot_results.json`。下载需要网络，训练和识别使用本地文件。

如果 CUDA 运行时与驱动不匹配，先用 `--device cpu` 验证推理；不要把环境未通过的运行写成显卡结果。模型和数据不提交到源码仓库，克隆仓库后需要自行下载并训练。

## 常见情况

- **模型尚未就绪**：代码安装成功不等于模型已训练。先完成数据准备与训练。
- **没有检出目标**：可能是阈值、预实验训练量或域差异。未检出不能证明桥梁没有病害。
- **细裂缝漏检**：384 输入会压缩细节。可尝试高分辨率切片；其收益必须在同一验证集单独评估。
- **CPU 较慢**：先选择 ONNX 模型；记录实际设备、输入尺寸与预热次数，不把显卡成绩写成 CPU 成绩。
- **训练 NaN**：运行会中止并记录失败。检查原始数据完整性、学习率和初始化；用新运行名重试，保留失败证据。

物理宽度、深度、承载力和桥梁安全等级均不由这个检测器直接输出。尺寸测量需要标定尺度，结构判断需要工程资料与专业复核。
