"""EfficientNet-B0 backbone for NGIML low-level feature extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


@dataclass
class EfficientNetBackboneConfig:
    """Configuration container for EfficientNet backbone."""

    pretrained: bool = True
    out_indices: Sequence[int] = (1, 2, 3, 4, 5)


class EfficientNetBackbone(nn.Module):
    """Wrapper that exposes multi-scale EfficientNet-B0 feature maps."""

    def __init__(self, config: EfficientNetBackboneConfig | None = None) -> None:
        super().__init__()
        cfg = config or EfficientNetBackboneConfig()
        self.out_indices: Tuple[int, ...] = tuple(sorted(set(cfg.out_indices)))

        weights = EfficientNet_B0_Weights.DEFAULT if cfg.pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.features = backbone.features

        # Cache channel dimensions for downstream heads.
        self.out_channels: List[int] = self._infer_out_channels()

    def _infer_out_channels(self) -> List[int]:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            channels: List[int] = []
            x = dummy
            for idx, block in enumerate(self.features):
                x = block(x)
                if idx in self.out_indices:
                    channels.append(x.shape[1])
        return channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        for idx, block in enumerate(self.features):
            x = block(x)
            if idx in self.out_indices:
                outputs.append(x)
        return outputs


__all__ = ["EfficientNetBackbone", "EfficientNetBackboneConfig"]
