"""Explicit AdamW implementation and finite-gradient checks for tracked runs."""
import torch
import numpy as np
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data.build import build_dataloader
from ultralytics.utils.torch_utils import torch_distributed_zero_first


class CheckedDetectionTrainer(DetectionTrainer):
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode not in {"train", "val"}:
            raise ValueError("Unknown data split mode")
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            shuffle = False
        return build_dataloader(dataset, batch=batch_size, workers=self.args.workers if mode == "train" else self.args.workers * 2,
                                shuffle=shuffle, rank=rank, drop_last=self.args.compile and mode == "train",
                                pin_memory=False, device=self.device)

    def preprocess_batch(self, batch):
        # Synchronous transfers keep host storage alive until the CUDA copy ends.
        for name, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[name] = value.to(self.device, non_blocking=False)
        return super().preprocess_batch(batch)

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)
        if isinstance(optimizer, torch.optim.AdamW):
            groups = []
            for original in optimizer.param_groups:
                group = dict(original)
                group.pop("fused", None)
                group.pop("foreach", None)
                groups.append(group)
            # Keep the same parameter groups, learning rates and betas, while
            # selecting the scalar implementation explicitly for reproducibility.
            optimizer = torch.optim.AdamW(groups, fused=False, foreach=False)
        return optimizer

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0,
                                       foreach=False, error_if_nonfinite=True)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)
