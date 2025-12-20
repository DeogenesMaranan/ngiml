"""Residual noise branch extracting high-frequency manipulation cues."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class ResidualNoiseConfig:
    """Configuration for the residual noise branch."""

    gaussian_kernel: int = 5
    gaussian_sigma: float = 1.2
    highpass_strength: float = 1.0


class ResidualNoiseBranch(nn.Module):
    """Extracts high-frequency residual signals from RGB inputs."""

    def __init__(self, config: ResidualNoiseConfig | None = None) -> None:
        super().__init__()
        cfg = config or ResidualNoiseConfig()
        if cfg.gaussian_kernel % 2 == 0:
            raise ValueError("gaussian_kernel must be odd to keep spatial alignment")
        self.cfg = cfg

        gaussian = self._build_gaussian_kernel(cfg.gaussian_kernel, cfg.gaussian_sigma)
        laplacian = self._build_highpass_kernel() * cfg.highpass_strength

        self.register_buffer("gaussian_kernel", gaussian, persistent=False)
        self.register_buffer("highpass_kernel", laplacian, persistent=False)

    @staticmethod
    def _build_gaussian_kernel(size: int, sigma: float) -> Tensor:
        radius = size // 2
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()
        return kernel.view(1, 1, size, size)

    @staticmethod
    def _build_highpass_kernel() -> Tensor:
        # Classic Laplacian kernel preserves edges / manipulations.
        kernel = torch.tensor([
            [0.0, -1.0, 0.0],
            [-1.0, 4.0, -1.0],
            [0.0, -1.0, 0.0],
        ], dtype=torch.float32)
        return kernel.view(1, 1, 3, 3)

    def forward(self, x: Tensor) -> Tensor:
        channels = x.shape[1]
        gauss = self.gaussian_kernel.expand(channels, -1, -1, -1)
        blur = F.conv2d(x, gauss, padding=self.cfg.gaussian_kernel // 2, groups=channels)

        highpass_kernel = self.highpass_kernel.expand(channels, -1, -1, -1)
        highpass = F.conv2d(x, highpass_kernel, padding=1, groups=channels)

        residual = (x - blur) + highpass
        return residual


__all__ = ["ResidualNoiseBranch", "ResidualNoiseConfig"]
