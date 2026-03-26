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
    """Configuration for the U-Net style decoder.

    Forensic motivation: Use InstanceNorm by default to improve stability for forensic segmentation. Optionally inject edge-aware refinement for sharper boundaries. Optionally apply Dropout2d to the highest-res decoder output to regularize overfitting to spurious artifacts.
    """

    decoder_channels: Sequence[int] | None = None
    out_channels: int = 1
    norm: str = "in"  # Default to InstanceNorm
    activation: str = "relu"
    per_stage_heads: bool = True
    enable_edge_guidance: bool = True  # Edge-aware decoder refinement (enabled by default)
    use_dropout: bool = True  # Dropout2d in highest-res decoder output enabled by default
    dropout_p: float = 0.2
    enable_boundary_refinement: bool = True  # Sobel-guided residual correction after final logits
    boundary_refine_channels: int = 8  # Lightweight hidden width for post-logit correction
    boundary_refine_scale: float = 1.0  # Multiplicative scale for residual correction
    enable_detail_refinement: bool = True  # Final-stage detail correction for small/low-res regions
    detail_refine_channels: int = 16  # Lightweight hidden width for final detail refinement
    detail_refine_scale: float = 1.0  # Multiplicative scale for final residual correction


class UNetDecoder(nn.Module):
    """U-Net decoder that upsamples fused features into manipulation logits.

    Forensic motivation: Optionally injects Sobel edge map into highest-resolution decoder feature for improved boundary localization.
    """

    def __init__(self, stage_channels: Sequence[int], config: UNetDecoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or UNetDecoderConfig()
        self.use_dropout = getattr(self.cfg, 'use_dropout', False)
        self.dropout_p = getattr(self.cfg, 'dropout_p', 0.2)
        if self.use_dropout:
            self.dropout = nn.Dropout2d(self.dropout_p)
        if not stage_channels:
            raise ValueError("stage_channels must contain at least one entry")
        self.stage_channels = tuple(stage_channels)

        if self.cfg.decoder_channels is None:
            decoder_channels = self.stage_channels
        else:
            if len(self.cfg.decoder_channels) != len(self.stage_channels):
                raise ValueError("decoder_channels length must match number of fusion stages")
            decoder_channels = tuple(self.cfg.decoder_channels)
        self.decoder_channels = tuple(decoder_channels)

        # Edge-aware decoder refinement (optional)
        self.enable_edge_guidance = getattr(self.cfg, 'enable_edge_guidance', False)
        if self.enable_edge_guidance:
            # Project Sobel edge map to decoder feature channels
            self.edge_proj = nn.Sequential(
                nn.Conv2d(1, self.decoder_channels[0], kernel_size=3, padding=1, bias=False),
                _build_norm(self.cfg.norm, self.decoder_channels[0]),
                _build_activation(self.cfg.activation),
            )
            # Sobel kernels
            sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
            self.register_buffer('sobel_x', sobel_x)
            self.register_buffer('sobel_y', sobel_y)

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

        self.enable_detail_refinement = bool(getattr(self.cfg, "enable_detail_refinement", False))
        self.detail_refine_scale = float(getattr(self.cfg, "detail_refine_scale", 1.0))
        if self.enable_detail_refinement:
            refine_channels = int(max(1, getattr(self.cfg, "detail_refine_channels", 16)))
            detail_in_channels = (self.decoder_channels[0] * 2) + self.cfg.out_channels
            self.detail_refine_head = nn.Sequential(
                nn.Conv2d(detail_in_channels, refine_channels, kernel_size=3, padding=1, bias=False),
                _build_norm(self.cfg.norm, refine_channels),
                _build_activation(self.cfg.activation),
                nn.Conv2d(refine_channels, self.cfg.out_channels, kernel_size=1, bias=True),
            )
            # Preserve identity behavior at initialization.
            nn.init.zeros_(self.detail_refine_head[-1].weight)
            if self.detail_refine_head[-1].bias is not None:
                nn.init.zeros_(self.detail_refine_head[-1].bias)

        # Lightweight post-logit boundary refinement (optional).
        self.enable_boundary_refinement = bool(getattr(self.cfg, "enable_boundary_refinement", False))
        self.boundary_refine_scale = float(getattr(self.cfg, "boundary_refine_scale", 1.0))
        if self.enable_boundary_refinement:
            refine_channels = int(max(1, getattr(self.cfg, "boundary_refine_channels", 8)))
            self.boundary_refine_head = nn.Sequential(
                nn.Conv2d(2, refine_channels, kernel_size=3, padding=1, bias=False),
                _build_activation(self.cfg.activation),
                nn.Conv2d(refine_channels, self.cfg.out_channels, kernel_size=1, bias=True),
            )
            # Start from an identity mapping: logits + 0.0 * residual.
            nn.init.zeros_(self.boundary_refine_head[-1].weight)
            if self.boundary_refine_head[-1].bias is not None:
                nn.init.zeros_(self.boundary_refine_head[-1].bias)

            sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
            self.register_buffer("boundary_sobel_x", sobel_x)
            self.register_buffer("boundary_sobel_y", sobel_y)

    def _refine_final_logits(self, logits: Tensor) -> Tensor:
        if not self.enable_boundary_refinement:
            return logits

        if logits.shape[1] != self.cfg.out_channels:
            # Conservative fallback for unexpected channel layouts.
            return logits

        sobel_x = self.boundary_sobel_x.to(dtype=logits.dtype, device=logits.device)
        sobel_y = self.boundary_sobel_y.to(dtype=logits.dtype, device=logits.device)

        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
            grad_x = F.conv2d(probs, sobel_x, padding=1)
            grad_y = F.conv2d(probs, sobel_y, padding=1)
            edge_mag = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)
            refine_in = torch.cat([logits, edge_mag], dim=1)
        else:
            # Grouped Sobel for multi-channel logits (kept for completeness).
            probs = torch.sigmoid(logits)
            groups = int(logits.shape[1])
            sx = sobel_x.repeat(groups, 1, 1, 1)
            sy = sobel_y.repeat(groups, 1, 1, 1)
            grad_x = F.conv2d(probs, sx, padding=1, groups=groups)
            grad_y = F.conv2d(probs, sy, padding=1, groups=groups)
            edge_mag = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)
            edge_mag = edge_mag.mean(dim=1, keepdim=True)
            logit_mean = logits.mean(dim=1, keepdim=True)
            refine_in = torch.cat([logit_mean, edge_mag], dim=1)

        residual = self.boundary_refine_head(refine_in)
        return logits + self.boundary_refine_scale * residual

    def _refine_small_detail_logits(
        self,
        logits: Tensor,
        final_feature: Tensor,
        finest_skip: Tensor,
        coarse_logits: Tensor | None = None,
    ) -> Tensor:
        if not self.enable_detail_refinement:
            return logits

        if coarse_logits is None:
            coarse_logits = logits
        elif coarse_logits.shape[-2:] != logits.shape[-2:]:
            coarse_logits = F.interpolate(
                coarse_logits,
                size=logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        refine_in = torch.cat([final_feature, finest_skip, coarse_logits], dim=1)
        residual = self.detail_refine_head(refine_in)
        return logits + self.detail_refine_scale * residual

    def forward(self, features: List[Tensor], image: Tensor = None, postprocess: Optional[str] = None) -> List[Tensor]:
        if len(features) != len(self.stage_channels):
            raise ValueError("Feature list length must match number of decoder stages")

        projected = [proj(feat) for proj, feat in zip(self.skip_projections, features)]

        # Edge-aware refinement: inject projected Sobel edge map into highest-res decoder feature
        if self.enable_edge_guidance and image is not None:
            # Compute grayscale edge map
            with torch.no_grad():
                if image.shape[1] > 1:
                    gray = image.mean(dim=1, keepdim=True)
                else:
                    gray = image
                grad_x = F.conv2d(gray, self.sobel_x, padding=1)
                grad_y = F.conv2d(gray, self.sobel_y, padding=1)
                edge_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
            edge_proj = self.edge_proj(edge_mag)
            # Ensure edge projection spatial size matches the highest-res
            # decoder feature before addition (avoid mismatched dimensions).
            if edge_proj.shape[-2:] != projected[0].shape[-2:]:
                edge_proj = F.interpolate(
                    edge_proj, size=projected[0].shape[-2:], mode="bilinear", align_corners=False
                )
            projected[0] = projected[0] + edge_proj

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
            # Optionally apply dropout to highest-res output
            out_preds = [pred for pred in predictions if pred is not None]
            if self.use_dropout and out_preds:
                out_preds[0] = self.dropout(out_preds[0])
            if out_preds:
                coarse_logits = out_preds[1] if len(out_preds) > 1 else None
                out_preds[0] = self._refine_small_detail_logits(
                    out_preds[0],
                    x,
                    projected[0],
                    coarse_logits=coarse_logits,
                )
                out_preds[0] = self._refine_final_logits(out_preds[0])
            if postprocess is not None:
                if postprocess.lower() == 'sigmoid':
                    out_preds = [torch.sigmoid(p) for p in out_preds]
                else:
                    raise ValueError(f"Unsupported postprocess: {postprocess}")
            return out_preds

        final = self.predictors[0](x)
        if self.use_dropout:
            final = self.dropout(final)
        final = self._refine_small_detail_logits(final, x, projected[0], coarse_logits=None)
        final = self._refine_final_logits(final)
        if postprocess is not None:
            if postprocess.lower() == 'sigmoid':
                final = torch.sigmoid(final)
            else:
                raise ValueError(f"Unsupported postprocess: {postprocess}")
        return [final]


__all__ = ["UNetDecoder", "UNetDecoderConfig"]
