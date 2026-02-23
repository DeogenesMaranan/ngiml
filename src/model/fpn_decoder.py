"""Feature Pyramid Network decoder for NGIML segmentation.

Forensic motivation: FPN preserves spatial detail through lateral connections
at every pyramid level, often outperforming U-Net on multi-scale manipulation
detection where tampered regions span multiple resolutions.
"""
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


@dataclass
class FPNDecoderConfig:
    """Configuration for the FPN decoder.

    Forensic motivation: FPN top-down lateral connections preserve multi-scale
    spatial detail critical for precise boundary localisation of tampered
    regions.
    """

    fpn_channels: int = 256
    out_channels: int = 1
    norm: str = "in"
    activation: str = "relu"
    per_stage_heads: bool = True
    use_dropout: bool = False
    dropout_p: float = 0.2


class FPNDecoder(nn.Module):
    """Feature Pyramid Network decoder for segmentation.

    Expects features ordered high → low resolution (same contract as
    ``MultiStageFeatureFusion`` output).
    """

    def __init__(
        self,
        stage_channels: Sequence[int],
        config: FPNDecoderConfig | None = None,
    ) -> None:
        super().__init__()
        if len(stage_channels) < 2:
            raise ValueError("FPNDecoder expects at least two feature stages.")

        self.cfg = config or FPNDecoderConfig()
        self.stage_channels = tuple(stage_channels)
        fpn_ch = self.cfg.fpn_channels

        # 1×1 lateral projections
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(ch, fpn_ch, kernel_size=1, bias=False) for ch in stage_channels]
        )

        # 3×3 output convs after lateral merge
        self.output_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(fpn_ch, fpn_ch, kernel_size=3, padding=1, bias=False),
                    _build_norm(self.cfg.norm, fpn_ch),
                    _build_activation(self.cfg.activation),
                )
                for _ in stage_channels
            ]
        )

        # Final mask head (operates on aggregated highest-res feature)
        self.mask_head = nn.Sequential(
            nn.Conv2d(fpn_ch, fpn_ch, kernel_size=3, padding=1, bias=False),
            _build_norm(self.cfg.norm, fpn_ch),
            _build_activation(self.cfg.activation),
            nn.Conv2d(fpn_ch, self.cfg.out_channels, kernel_size=1),
        )

        # Per-stage prediction heads (deep supervision)
        if self.cfg.per_stage_heads:
            self.stage_predictors = nn.ModuleList(
                [nn.Conv2d(fpn_ch, self.cfg.out_channels, kernel_size=1) for _ in stage_channels]
            )
        else:
            self.stage_predictors = None

        self.use_dropout = self.cfg.use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout2d(self.cfg.dropout_p)

    def forward(self, features: List[Tensor], image: Tensor = None) -> List[Tensor]:
        """Run FPN top-down pathway and return per-stage predictions.

        Args:
            features: Fused pyramid features, ordered high → low resolution.
            image: Original input image (unused by FPN, kept for API parity
                   with UNetDecoder).
        """
        if len(features) != len(self.stage_channels):
            raise ValueError("Feature list length must match number of decoder stages")

        # Build lateral projections
        laterals = [lateral(feat) for lateral, feat in zip(self.lateral_convs, features)]

        # Top-down pathway: propagate from low-res → high-res
        pyramid = laterals.copy()
        for idx in range(len(pyramid) - 1, 0, -1):
            upsampled = F.interpolate(
                pyramid[idx],
                size=pyramid[idx - 1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pyramid[idx - 1] = pyramid[idx - 1] + upsampled

        # Apply output convolutions
        pyramid = [conv(feat) for conv, feat in zip(self.output_convs, pyramid)]

        # Per-stage predictions (deep supervision)
        if self.stage_predictors is not None:
            predictions = [head(feat) for head, feat in zip(self.stage_predictors, pyramid)]
            if self.use_dropout and predictions:
                predictions[-1] = self.dropout(predictions[-1])
            return predictions

        # Aggregate all scales into highest-resolution output
        out = pyramid[0]
        for feat in pyramid[1:]:
            upsampled = F.interpolate(
                feat, size=out.shape[-2:], mode="bilinear", align_corners=False
            )
            out = out + upsampled

        mask = self.mask_head(out)
        if self.use_dropout:
            mask = self.dropout(mask)
        return [mask]


__all__ = ["FPNDecoder", "FPNDecoderConfig"]
