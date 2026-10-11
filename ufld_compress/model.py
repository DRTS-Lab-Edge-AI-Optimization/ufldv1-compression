"""UFLD-v1 network (state-dict compatible with cfzd/Ultra-Fast-Lane-Detection checkpoints) + student variants.

The parameter names are identical to the original `parsingNet`
(`model.conv1 ... model.layer4`, `pool`, `cls.0`, `cls.2`) so `tusimple_18.pth` loads directly.
The auxiliary segmentation branch is only used for training in the original repo and is dropped here.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn
import torchvision


class ResNetBackbone(nn.Module):
    def __init__(self, layers: str = "18", pretrained: bool = False):
        super().__init__()
        fn = {"18": torchvision.models.resnet18, "34": torchvision.models.resnet34}[str(layers)]
        model = fn(weights="DEFAULT" if pretrained else None)
        self.conv1, self.bn1, self.relu, self.maxpool = model.conv1, model.bn1, model.relu, model.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = model.layer1, model.layer2, model.layer3, model.layer4

    @property
    def out_channels(self) -> int:
        return self.layer4[-1].conv2.out_channels

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4


class MobileNetBackbone(nn.Module):
    """MobileNet student backbone exposing the same stride-8/16/32 outputs (`layer2/3/4`) as the ResNet."""

    SPLITS = {  # stem | layer2 (stride 8) | layer3 (stride 16) | layer4 (stride 32)
        "mobilenet_v2": (4, 7, 14, 18),
        "mobilenet_v3_large": (4, 7, 13, 16),
    }

    def __init__(self, arch: str = "mobilenet_v2", pretrained: bool = False):
        super().__init__()
        net = getattr(torchvision.models, arch)(weights="DEFAULT" if pretrained else None)
        f = net.features
        a, b, c, d = self.SPLITS[arch]
        self.stem, self.layer2, self.layer3, self.layer4 = f[:a], f[a:b], f[b:c], f[c:d]
        self.eval()  # don't let the probe pass touch the (ImageNet) BN statistics
        with torch.no_grad():
            self._out_channels = self.forward(torch.zeros(1, 3, 64, 64))[2].shape[1]
        self.train()

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x):
        x2 = self.layer2(self.stem(x))
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4


class UFLDNet(nn.Module):
    """Ultra-Fast-Lane-Detection v1 (row-anchor classification) without the auxiliary seg head.

    output: (N, griding_num + 1, num_row_anchors, num_lanes) logits
    """

    def __init__(self, cls_dim=(101, 56, 4), backbone: str = "18", pretrained: bool = False,
                 input_size=(288, 800), fc_hidden: int = 2048, pool_channels: int = 8):
        super().__init__()
        self.cls_dim = tuple(int(c) for c in cls_dim)
        self.total_dim = int(np.prod(self.cls_dim))
        self.backbone_name = str(backbone)
        self.input_size = tuple(input_size)
        if self.backbone_name in ("18", "34"):
            self.model = ResNetBackbone(self.backbone_name, pretrained)
        elif self.backbone_name.startswith("mobilenet"):
            self.model = MobileNetBackbone(self.backbone_name, pretrained)
        else:
            raise ValueError(f"unknown backbone {backbone}")
        feat_hw = (input_size[0] // 32, input_size[1] // 32)  # 288x800 -> 9x25
        self.pool = nn.Conv2d(self.model.out_channels, pool_channels, 1)
        self.cls = nn.Sequential(
            nn.Linear(pool_channels * feat_hw[0] * feat_hw[1], fc_hidden),  # 1800 -> 2048
            nn.ReLU(),
            nn.Linear(fc_hidden, self.total_dim),
        )
        _init_head(self.cls)

    def forward(self, x):
        _, _, fea = self.model(x)
        fea = self.pool(fea).flatten(1)
        return self.cls(fea).view(-1, *self.cls_dim)


def _init_head(m):
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            mod.weight.data.normal_(0.0, std=0.01)
            if mod.bias is not None:
                nn.init.constant_(mod.bias, 0)


# ------------------------------------------------------------------------------------------------


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        warnings.warn(f"{path}: weights_only load failed, falling back to full unpickling "
                      "(only do this for checkpoints you trust, e.g. the official UFLD release).")
        return torch.load(path, map_location="cpu", weights_only=False)


def load_ufld_checkpoint(model: UFLDNet, path: str, strict: bool = True) -> UFLDNet:
    ckpt = _torch_load(path)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    clean = {}
    for k, v in sd.items():
        k = k[7:] if k.startswith("module.") else k
        if k.startswith("aux_"):
            continue  # auxiliary segmentation branch (train-time only)
        clean[k] = v
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    return model


def build_teacher(cfg, checkpoint: str | None = None) -> UFLDNet:
    """Original UFLD ResNet-18 TuSimple model (FP32), loaded from the official checkpoint."""
    model = UFLDNet(cfg.cls_dim, backbone=cfg.backbone, input_size=(cfg.input_h, cfg.input_w))
    ckpt = checkpoint if checkpoint is not None else cfg.checkpoint
    if cfg.dry_run:
        warnings.warn("dry_run=True -> random weights (accuracy numbers are meaningless).")
    else:
        load_ufld_checkpoint(model, ckpt)
    return model.eval()


def build_student(cfg, arch: str = "mobilenet_v2", pretrained: bool = True, fc_hidden: int = 1024) -> UFLDNet:
    """A smaller UFLD student (MobileNet backbone + slimmer head). ImageNet weights are downloaded by torchvision."""
    try:
        return UFLDNet(cfg.cls_dim, backbone=arch, pretrained=pretrained and not cfg.dry_run,
                       input_size=(cfg.input_h, cfg.input_w), fc_hidden=fc_hidden)
    except Exception as e:  # no internet -> random init
        warnings.warn(f"could not load ImageNet weights for {arch} ({e}); using random init")
        return UFLDNet(cfg.cls_dim, backbone=arch, pretrained=False,
                       input_size=(cfg.input_h, cfg.input_w), fc_hidden=fc_hidden)


def save_model(model: nn.Module, path: str) -> str:
    """Pickle the whole module (keeps pruned/compressed architectures). Load with `load_model`."""
    torch.save(model, path)
    return path


def load_model(path: str) -> nn.Module:
    return torch.load(path, map_location="cpu", weights_only=False)
