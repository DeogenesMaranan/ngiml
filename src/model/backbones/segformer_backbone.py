"""SegFormer MiT backbone for NGIML – lightweight hierarchical transformer.

Forensic motivation: SegFormer's overlapping patch embeddings capture
fine-grained manipulation boundaries better than shifted-window approaches,
complementing both EfficientNet (local texture) and Swin (global context).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import logging
import torch
import torch.nn.functional as F
from torch import nn

import timm

# Suppress noisy timm builder warnings for adapted pretrained weights.
try:
    logging.getLogger("timm.models._builder").setLevel(logging.ERROR)
except Exception:
    pass


@dataclass
class SegFormerBackboneConfig:
    """Configuration for the SegFormer MiT feature extractor."""

    model_name: str = "segformer_b0"
    pretrained: bool = True
    out_indices: Sequence[int] = (0, 1, 2, 3)
    enforce_input_size: bool = False
    input_size: Union[int, Tuple[int, int], None] = None


class SegFormerBackbone(nn.Module):
    """SegFormer MiT-B0 backbone exposing multi-scale feature maps.

    Forensic motivation: Overlapping patch embeddings preserve sub-pixel
    boundary details critical for manipulation localisation.  The hierarchical
    Mix-Transformer produces four stages at 1/4, 1/8, 1/16 and 1/32 of the
    input resolution, matching the pyramid contract expected by the fusion
    module.
    """

    def __init__(self, config: SegFormerBackboneConfig | None = None) -> None:
        super().__init__()
        cfg = config or SegFormerBackboneConfig()
        self.out_indices: Tuple[int, ...] = tuple(sorted(set(cfg.out_indices)))
        self.enforce_input_size = cfg.enforce_input_size

        extra_args: dict = {}
        if cfg.input_size is not None:
            if isinstance(cfg.input_size, int):
                input_hw = (cfg.input_size, cfg.input_size)
            else:
                input_hw = tuple(cfg.input_size)
            extra_args["img_size"] = input_hw
        else:
            input_hw = None

        self.model = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            features_only=True,
            out_indices=self.out_indices,
            **extra_args,
        )

        self.out_channels: List[int] = list(self.model.feature_info.channels())
        default_cfg_size = self.model.default_cfg.get("input_size", (3, 224, 224))
        self.expected_hw: Tuple[int, int] = (
            input_hw if input_hw is not None else default_cfg_size[1:]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return multi-scale features ordered high → low resolution."""
        if self.enforce_input_size and x.shape[-2:] != self.expected_hw:
            x = F.interpolate(
                x, size=self.expected_hw, mode="bilinear", align_corners=False
            )
        features = self.model(x)
        return list(features)


__all__ = ["SegFormerBackbone", "SegFormerBackboneConfig"]
