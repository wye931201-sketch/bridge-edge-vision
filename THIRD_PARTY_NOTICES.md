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
# Optional industrial gallery

The optional crack auxiliary uses the ResNet34 seed-42 ONNX graph from [Cucom/UAVCrack-R10 v1.0.0](https://github.com/Cucom/UAVCrack-R10/releases/tag/v1.0.0). Its released final weights and project contributions are Apache-2.0, with upstream notices retained; see its [weights license](https://github.com/Cucom/UAVCrack-R10/blob/main/WEIGHTS_LICENSE.md). This project uses one graph, not its six-model ensemble, and claims no upstream leaderboard score. Archive and graph checksums and input conventions are pinned in `configs/crack_segmentation.json`. The graph is cached locally and is not redistributed in this repository.

The optional scratch gallery downloads a pinned ONNX release from [winthegame259/steel-surface-defect-detection](https://github.com/winthegame259/steel-surface-defect-detection/releases/tag/v1.0.0), whose project code is AGPL-3.0. It is an upstream pretrained model, not a model trained by this project. No weights are redistributed here.

NEU-DET scratch photographs and XML annotations are obtained from a pinned [GitHub mirror](https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection) and checked against Git blob hashes. The [dataset maintainer page](https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm) requests citation but does not state a standard dataset license. This project does not grant or imply redistribution/commercial rights over these assets. Source images and annotations remain local. See `docs/INDUSTRIAL_GALLERY.md` and `configs/industrial_sources.json`.
