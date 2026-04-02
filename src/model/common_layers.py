from __future__ import annotations

from torch import nn


def build_norm(kind: str, channels: int) -> nn.Module:
    norm = str(kind).lower()
    if norm == "bn":
        return nn.BatchNorm2d(channels)
    if norm == "in":
        return nn.InstanceNorm2d(channels, affine=True)
    raise ValueError(f"Unsupported norm type: {kind}")


def build_activation(name: str) -> nn.Module:
    activation = str(name).lower()
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "gelu":
        return nn.GELU()
    if activation == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")