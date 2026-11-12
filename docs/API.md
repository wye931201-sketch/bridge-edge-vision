# 本地接口

默认监听 `127.0.0.1:8765`，不要求 API 密钥，不调用外部识别服务。此接口面向单机使用，尚未实现多用户权限体系，不应直接开放到公网。

| 方法与路径 | 用途 |
|---|---|
| GET `/api/health` | 硬件、五类标签、可用模型与默认模型 |
| GET `/api/samples` | 固定验证集按文件名排序的前 12 张样例 |
| POST `/api/predict?filename=image.jpg&confidence=0.25&tile=false` | 请求体为图片原始二进制，可选 `model` 指定注册模型文件名 |
| GET `/api/history?limit=20&offset=0` | 本地检测历史 |
| GET `/api/history/{id}` | 一次检测的结果 |
| GET `/api/artifacts/{id}/annotated.jpg` | 标注图 |
| GET `/api/artifacts/{id}/result.json` | JSON 检测记录 |

图片最多 25 MiB、2500 万像素。模型必须来自本地已训练模型列表。结果包括类别、置信度、原图像素坐标、权重哈希、运行后端及耗时；不表示病害的物理尺寸或结构安全结论。
