"""Lightweight U-Net style decoder for NGIML feature fusion outputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _build_norm(kind: str, channels: int) -> nn.Module:
    if kind.lower() == "bn":
        return nn.BatchNorm2d(channels)
    if kind.lower() == "in":
        return nn.InstanceNorm2d(channels, affine=True)
    raise ValueError(f"Unsupported norm type: {kind}")


def _build_activation(name: str) -> nn.Module:
    if name.lower() == "relu":
        return nn.ReLU(inplace=True)
    if name.lower() == "gelu":
        return nn.GELU()
    if name.lower() == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str, activation: str) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _build_norm(norm, out_channels),
            _build_activation(activation),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _build_norm(norm, out_channels),
            _build_activation(activation),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


@dataclass
class UNetDecoderConfig:
    """Configuration for the U-Net style decoder."""

    decoder_channels: Sequence[int] | None = None
    out_channels: int = 1
    norm: str = "bn"
    activation: str = "relu"
    per_stage_heads: bool = True


class UNetDecoder(nn.Module):
    """U-Net decoder that upsamples fused features into manipulation logits."""

    def __init__(self, stage_channels: Sequence[int], config: UNetDecoderConfig | None = None) -> None:
        super().__init__()
        if not stage_channels:
            raise ValueError("stage_channels must contain at least one entry")
        self.stage_channels = tuple(stage_channels)
        self.cfg = config or UNetDecoderConfig()

        if self.cfg.decoder_channels is None:
            decoder_channels = self.stage_channels
        else:
            if len(self.cfg.decoder_channels) != len(self.stage_channels):
                raise ValueError("decoder_channels length must match number of fusion stages")
            decoder_channels = tuple(self.cfg.decoder_channels)
        self.decoder_channels = tuple(decoder_channels)

        self.skip_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, dec_ch, kernel_size=1, bias=False),
                    _build_norm(self.cfg.norm, dec_ch),
                    _build_activation(self.cfg.activation),
                )
                for in_ch, dec_ch in zip(self.stage_channels, self.decoder_channels)
            ]
        )

        self.bottleneck = _ConvBlock(
            self.decoder_channels[-1],
            self.decoder_channels[-1],
            self.cfg.norm,
            self.cfg.activation,
        )

        self.decode_blocks = nn.ModuleList(
            [
                _ConvBlock(
                    self.decoder_channels[idx] + self.decoder_channels[idx + 1],
                    self.decoder_channels[idx],
                    self.cfg.norm,
                    self.cfg.activation,
                )
                for idx in range(len(self.stage_channels) - 1)
            ]
        )

        self.predictors = nn.ModuleList(
            [
                nn.Conv2d(channels, self.cfg.out_channels, kernel_size=1)
                for channels in self.decoder_channels
            ]
        )

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        if len(features) != len(self.stage_channels):
            raise ValueError("Feature list length must match number of decoder stages")

        projected = [proj(feat) for proj, feat in zip(self.skip_projections, features)]
        x = self.bottleneck(projected[-1])

        if self.cfg.per_stage_heads:
            predictions: List[Optional[Tensor]] = [None] * len(projected)
            predictions[-1] = self.predictors[-1](x)
        else:
            predictions = []

        for idx in range(len(projected) - 2, -1, -1):
            skip = projected[idx]
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.decode_blocks[idx](x)
            if self.cfg.per_stage_heads:
                predictions[idx] = self.predictors[idx](x)

        if self.cfg.per_stage_heads:
            return [pred for pred in predictions if pred is not None]

        final = self.predictors[0](x)
        return [final]


__all__ = ["UNetDecoder", "UNetDecoderConfig"]
