from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


@dataclass
class ResidualNoiseConfig:
    """Configuration for the SRM residual branch / multi-scale backbone."""
    num_kernels: int = 3  # SRM kernels (fixed)
    base_channels: int = 32  # CNN backbone base channels
    num_stages: int = 4      # Number of pyramid stages


class ConvBlock(nn.Module):
    """Basic 2-layer conv + BN + ReLU block."""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ResidualNoiseModule(nn.Module):
    """
    Combined SRM residual extractor + learnable multi-scale CNN backbone.

    For splicing / copy-move detection:
        Image → SRMResidualBranch → ResidualNoiseBackbone → multi-scale features
    """

    def __init__(self, config: Optional[ResidualNoiseConfig] = None, in_channels: int = 3) -> None:
        super().__init__()
        cfg = config or ResidualNoiseConfig()
        self.cfg = cfg
        self.in_channels = in_channels

        # --- SRM Residual Branch ---
        srm_kernels = torch.tensor([
            [[0, 0, 0, 0, 0],
             [0, -1, 2, -1, 0],
             [0, 2, -4, 2, 0],
             [0, -1, 2, -1, 0],
             [0, 0, 0, 0, 0]],
            [[-1, 2, -2, 2, -1],
             [2, -6, 8, -6, 2],
             [-2, 8, -12, 8, -2],
             [2, -6, 8, -6, 2],
             [-1, 2, -2, 2, -1]],
            [[0, 0, 0, 0, 0],
             [0, 1, -2, 1, 0],
             [0, -2, 4, -2, 0],
             [0, 1, -2, 1, 0],
             [0, 0, 0, 0, 0]],
        ], dtype=torch.float32)
        srm_kernels /= torch.abs(srm_kernels).sum(dim=(1, 2), keepdim=True)
        self.register_buffer("srm_kernels", srm_kernels.view(cfg.num_kernels, 1, 5, 5), persistent=False)
        self.srm_out_channels = in_channels * cfg.num_kernels  # e.g., RGB * 3

        # --- Multi-scale residual backbone ---
        stage_channels = [cfg.base_channels * (2**i) for i in range(cfg.num_stages)]
        self.feature_dims = stage_channels
        self.out_channels = stage_channels

        blocks = []
        downsamplers = []
        current_in = self.srm_out_channels
        for idx, out_channels in enumerate(stage_channels):
            blocks.append(ConvBlock(current_in, out_channels))
            current_in = out_channels
            if idx < cfg.num_stages - 1:
                downsamplers.append(
                    nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False, padding_mode="reflect"),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        self.downsamplers = nn.ModuleList(downsamplers)

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Args:
            x : (B, C, H, W), e.g., RGB image
        Returns:
            List of multi-scale feature tensors from CNN backbone
        """
        # --- SRM residuals ---
        b, c, h, w = x.shape
        kernels = self.srm_kernels.to(x.dtype).repeat(c, 1, 1, 1).contiguous()
        residual_map = F.conv2d(
            x,
            kernels,
            padding=2,
            groups=c,
            padding_mode="reflect",
        )  # shape: (B, C*3, H, W)

        # --- Multi-scale CNN ---
        features = []
        out = residual_map
        for idx, block in enumerate(self.blocks):
            out = block(out)
            features.append(out)
            if idx < len(self.downsamplers):
                out = self.downsamplers[idx](out)
        return features


    # Backward compatibility for existing imports
    ResidualNoiseBranch = ResidualNoiseModule

    __all__ = ["ResidualNoiseModule", "ResidualNoiseBranch", "ResidualNoiseConfig"]
