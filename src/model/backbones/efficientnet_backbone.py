"""EfficientNet-B0 backbone for NGIML low-level feature extraction (torchvision-based)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, List, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


@dataclass
class EfficientNetBackboneConfig:
    """Configuration container for EfficientNet backbone."""

    pretrained: bool = True
    out_indices: Sequence[int] = (1, 2, 3, 4, 5)
    enforce_input_size: bool = False
    input_size: Union[int, Tuple[int, int], None] = None


class EfficientNetBackbone(nn.Module):
    """Wrapper that exposes multi-scale EfficientNet-B0 feature maps."""

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
            self.expected_hw = (224, 224)  # default EfficientNet-B0 input

        weights = EfficientNet_B0_Weights.DEFAULT if cfg.pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.features = backbone.features

        # Cache channel dimensions for downstream heads
        self.out_channels: List[int] = self._infer_out_channels()

    def _infer_out_channels(self) -> List[int]:
        """Infer the channel dimensions of the selected feature maps dynamically."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *self.expected_hw)
            channels: List[int] = []
            x = dummy
            for idx, block in enumerate(self.features):
                x = block(x)
                if idx in self.out_indices:
                    channels.append(x.shape[1])
        return channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return multi-scale feature maps."""
        if self.enforce_input_size and x.shape[-2:] != self.expected_hw:
            x = F.interpolate(x, size=self.expected_hw, mode="bilinear", align_corners=False)

        outputs: List[torch.Tensor] = []
        for idx, block in enumerate(self.features):
            x = block(x)
            if idx in self.out_indices:
                outputs.append(x)
        return outputs


__all__ = ["EfficientNetBackbone", "EfficientNetBackboneConfig"]
