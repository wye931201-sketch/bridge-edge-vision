# Jetson 与本地边缘部署

当前已验证设备以实验报告为准。本项目没有接触 Jetson 实物，因此不提供虚构的 Jetson FPS、功耗或 TensorRT 加速比。

## 不依赖硬件的准备

1. 训练得到与五类标签一致的 `.pt` 权重。
2. 使用 `python main.py export --weights <模型路径> --sample <图片路径>` 导出固定输入 FP32 ONNX。
3. 检查模型合法性、原始输出误差和后处理检测结果一致性；导出成功本身不足以证明部署正确。
4. 使用 `benchmark` 在实际运行端测量包括预处理、前向计算和 NMS 的单图时延，明确不包含文件解码和模型加载。

## 纯 ONNX 的 CPU 入口

独立 Python 3.12 环境可只安装 `requirements-edge.txt`，保留源码并复制本机训练导出的 ONNX，运行：

```powershell
python -m pip install -r requirements-edge.txt
python main.py predict --weights model.onnx --source image.jpg --device cpu
```

这条图片/视频命令行路径不加载 PyTorch 或 CUDA。模型文件内的五类名称必须匹配本工程，类别元数据不符会拒绝加载。网页工作台与训练属于完整环境，不能把它们的全部依赖误写成轻量推理依赖。

目标系统必须有对应 Python/架构的 ONNX Runtime 和 OpenCV 发行包；该步骤的 x86 实测不能替代 ARM64/Jetson 实机验证。

## 有 Jetson 后

使用设备对应 JetPack 的官方 Python、PyTorch 与 TensorRT，记录 JetPack、CUDA、TensorRT、设备型号、供电与功耗模式、散热及频率。Windows 的 CUDA 虚拟环境不能复制到 ARM64。

建议先在目标端运行 ONNX 基准，然后用目标端 `trtexec --onnx=model.onnx --saveEngine=model.fp16.engine --fp16` 构建引擎。先查看该版本 `trtexec --help` 确认参数。引擎与设备/软件版本关联，不能承诺跨机器复用。

FP16、INT8 都要对固定验证集重跑精度。INT8 校准集只能来自训练数据，不得取验证/测试图片做校准。保存校准清单、各类别 AP、端到端 p50/p95、功耗、内存与温度；长时间视频运行后检查丢帧和热降频。

摄像头接入后核对曝光、光照、距离、运动模糊和画面尺度。实验室桌面视频结果不能替代桥梁现场泛化测试。

参考：[NVIDIA TensorRT 文档](https://docs.nvidia.com/deeplearning/tensorrt/latest/)、[NVIDIA Jetson 文档](https://docs.nvidia.com/jetson/)。实机执行前按对应版本重新核验。
