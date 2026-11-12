"""Experimental modules. Their usefulness must be established by ablation."""
import torch
from torch import nn


class RepViTBackbone(nn.Module):
    """RepViT m0.9 features at strides 8, 16, 32 (96, 192, 384 channels)."""

    def __init__(self, variant="repvit_m0_9"):
        super().__init__()
        import timm
        self.variant = variant
        self.features = timm.create_model(variant, features_only=True, out_indices=(1, 2, 3), pretrained=False)
        self.register_buffer("input_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("input_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, image):
        return self.features((image - self.input_mean) / self.input_std)

    def load_imagenet(self, weights_path):
        import timm
        pretrained = timm.create_model(self.variant, features_only=True, out_indices=(1, 2, 3),
                                      pretrained=True, pretrained_cfg_overlay={"file": str(weights_path)})
        self.features.load_state_dict(pretrained.state_dict(), strict=True)


class AdaptiveMultiScale(nn.Module):
    """Image-conditioned weighting of three inexpensive receptive fields.

    Branch dilation rates are 1, 2, 3. A global gate chooses their weights and a
    learnable residual gain preserves the incoming representation. This is an
    experimental combination of standard operators, not a proven novel method.
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, 3, padding=d, dilation=d, groups=channels, bias=False),
                          nn.BatchNorm2d(channels), nn.SiLU()) for d in (1, 2, 3)
        ])
        hidden = max(8, channels // reduction)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, 3, 1))
        self.project = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels))
        self.gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, feature):
        weights = self.gate(feature).softmax(dim=1)
        fused = sum(branch(feature) * weights[:, index:index + 1] for index, branch in enumerate(self.branches))
        return feature + self.gain * self.project(fused)
