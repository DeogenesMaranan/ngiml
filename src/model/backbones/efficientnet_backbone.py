"""EfficientNet backbone for NGIML low-level feature extraction (timm-based).

Forensic motivation: timm EfficientNet provides more stable pretrained weights and better intermediate feature extraction for manipulation localization tasks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, List, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
import timm


@dataclass
class EfficientNetBackboneConfig:
    """Configuration container for EfficientNet backbone."""

    pretrained: bool = True
    out_indices: Sequence[int] = (1, 2, 3, 4, 5)
    enforce_input_size: bool = False
    input_size: Union[int, Tuple[int, int], None] = None



class EfficientNetBackbone(nn.Module):
    """Wrapper that exposes multi-scale EfficientNet feature maps using timm.

    Forensic motivation: Use timm EfficientNet for more stable pretrained weights and better feature extraction for manipulation localization.
    """
    def __init__(self, config: EfficientNetBackboneConfig | None = None) -> None:
        super().__init__()
        cfg = config or EfficientNetBackboneConfig()

        self.out_indices: Tuple[int, ...] = tuple(sorted(set(cfg.out_indices)))
        self.enforce_input_size = cfg.enforce_input_size

        if cfg.input_size is not None:
            if isinstance(cfg.input_size, int):
                self.expected_hw = (cfg.input_size, cfg.input_size)
            else:
                self.expected_hw = tuple(cfg.input_size)
        else:
            self.expected_hw = (224, 224)  # default EfficientNet input

        # Use timm to create EfficientNet backbone
        # Default to 'efficientnet_b0' for backward compatibility
        model_name = getattr(cfg, 'model_name', 'efficientnet_b0')
        self.backbone = timm.create_model(
            model_name,
            pretrained=cfg.pretrained,
            features_only=True,
            out_indices=self.out_indices
        )
        # Cache channel dimensions for downstream heads
        self.out_channels: List[int] = [f['num_chs'] for f in self.backbone.feature_info if f['index'] in self.out_indices]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return multi-scale feature maps."""
        if self.enforce_input_size and x.shape[-2:] != self.expected_hw:
            x = F.interpolate(x, size=self.expected_hw, mode="bilinear", align_corners=False)
        features = self.backbone(x)
        # Ensure returned feature list order matches out_indices
        if isinstance(features, (list, tuple)):
            return list(features)
        return [features]


__all__ = ["EfficientNetBackbone", "EfficientNetBackboneConfig"]
