# Third-party components and data

| Component | Role | License / source |
|---|---|---|
| Ultralytics | YOLO model/training/export infrastructure | AGPL-3.0, https://github.com/ultralytics/ultralytics |
| RepViT | Backbone design / initial weights | MIT, https://github.com/THU-MIG/RepViT |
| timm | RepViT implementation and feature extraction | Apache-2.0, https://github.com/huggingface/pytorch-image-models |
| PyTorch / torchvision | Training and CUDA operators | BSD-style, https://pytorch.org |
| ONNX / ONNX Runtime | Interchange and local CPU inference | Apache-2.0 / MIT, official ONNX and Microsoft repositories |
| dacl10k | Research image and annotation source | Follow upstream CC BY-NC 4.0; see DATASET_CARD.md |
| Voxel51 dataset mirror | Pinned image/annotation packaging | Original dataset rights retained; source/license discrepancy documented |

This repository's source code is released under AGPL-3.0-only to remain compatible with its Ultralytics integration. Importing a library does not transfer authorship of that library to this project. Third-party source trees, environments, raw data and model binaries are not counted as original project code and are not vendored here.

Initial YOLO and RepViT model files are obtained from their published sources and retain their own terms. Training on noncommercial data does not establish permission for commercial distribution of a trained product. Commercial rights, if required later, must be established from the relevant rights holders.
