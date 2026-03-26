"""Adaptive multi-branch feature fusion for NGIML."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _build_norm(norm: str, channels: int) -> nn.Module:
    norm = norm.lower()
    if norm == "bn":
        return nn.BatchNorm2d(channels)
    if norm == "in":
        return nn.InstanceNorm2d(channels, affine=True)
    raise ValueError(f"Unsupported norm type: {norm}")


def _build_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


@dataclass
class FeatureFusionConfig:
    """Config container for the multi-stage fusion module.

    Forensic motivation: Optionally adds a spatial refinement layer after fusion output for each stage, improving spatial detail without large FLOP increase.
    """

    fusion_channels: Sequence[int]
    noise_branch: str = "residual"
    noise_skip_stage: Optional[int] = None
    noise_decay: float = 1.0
    norm: str = "bn"
    activation: str = "relu"
    fusion_refinement: bool = True  # Add Conv3x3+IN+ReLU after fusion output (enabled by default)
    enable_joint_gating: bool = False
    late_residual_boost_start: int = 1
    late_residual_boost: float = 0.0


class _AdaptiveFusionStage(nn.Module):
    """Stage-wise fusion with learned gating and post refinement.

    Forensic motivation: Optionally adds a spatial refinement layer after fusion output for each stage, improving spatial detail without large FLOP increase.
    """
    def __init__(
        self,
        branch_channels: Dict[str, int],
        out_channels: int,
        norm: str,
        activation: str,
        fusion_refinement: bool = False,
        enable_joint_gating: bool = False,
        late_residual_boost: float = 0.0,
    ) -> None:
        super().__init__()
        self.branch_order = tuple(branch_channels.keys())
        # Conv only for projected features before fusion (no norm/activation)
        self.projections = nn.ModuleDict(
            {
                branch: nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False)
                for branch, in_ch in branch_channels.items()
            }
        )
        gate_hidden = max(8, out_channels // 4)
        self.gate_generators = nn.ModuleDict(
            {
                branch: nn.Sequential(
                    nn.Conv2d(out_channels, gate_hidden, kernel_size=1, bias=True),
                    _build_activation(activation),
                    nn.Conv2d(gate_hidden, out_channels, kernel_size=1, bias=True),
                )
                for branch in branch_channels
            }
        )
        self.enable_joint_gating = bool(enable_joint_gating) and len(self.branch_order) > 1
        if self.enable_joint_gating:
            joint_in_channels = out_channels * len(self.branch_order)
            self.joint_gate_generator = nn.Sequential(
                nn.Conv2d(joint_in_channels, gate_hidden, kernel_size=1, bias=True),
                _build_activation(activation),
                nn.Conv2d(gate_hidden, out_channels * len(self.branch_order), kernel_size=1, bias=True),
            )
        else:
            self.joint_gate_generator = None
        self.gate_bias = nn.ParameterDict(
            {
                branch: nn.Parameter(torch.zeros((1, out_channels, 1, 1)))
                for branch in branch_channels
            }
        )
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _build_norm(norm, out_channels),
            _build_activation(activation),
        )
        self.fusion_refinement = fusion_refinement
        self.late_residual_boost = max(late_residual_boost, 0.0)
        if self.fusion_refinement:
            self.refine2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
            )

    def forward(
        self,
        features: Dict[str, Tensor],
        target_size: Optional[Tuple[int, int]],
        noise_branch: Optional[str],
        noise_weight: float,
    ) -> Tensor:
        if not features:
            raise ValueError("Fusion stage received no features to fuse")

        # Determine alignment size: provided target overrides per-stage maximum.
        if target_size is not None:
            align_h, align_w = target_size
        else:
            # Compute per-branch spatial sizes
            sizes = {branch: (tensor.shape[-2], tensor.shape[-1]) for branch, tensor in features.items()}
            # Default to the maximum across branches
            max_h = max(h for h, w in sizes.values())
            max_w = max(w for h, w in sizes.values())

            # If a noise/residual branch is present and is extremely larger than
            # the other branches, avoid upsampling everything to that huge size
            # (which can OOM). Instead prefer the maximum among non-noise
            # branches when the residual spatial size exceeds others by >2x.
            non_noise_sizes = [s for b, s in sizes.items() if b != (noise_branch or "")] if features else []
            if non_noise_sizes:
                non_max_h = max(h for h, w in non_noise_sizes)
                non_max_w = max(w for h, w in non_noise_sizes)
                # If the largest size is more than twice the non-noise max, cap it.
                if max_h >= 2 * non_max_h:
                    max_h = non_max_h
                if max_w >= 2 * non_max_w:
                    max_w = non_max_w

            align_h, align_w = int(max_h), int(max_w)

        fused = None
        weight_sum = None
        eps = 1e-6
        aligned_projections: Dict[str, Tensor] = {}

        for branch, tensor in features.items():
            proj = self.projections[branch](tensor)
            if proj.shape[-2:] != (align_h, align_w):
                proj = F.interpolate(proj, size=(align_h, align_w), mode="bilinear", align_corners=False)
            aligned_projections[branch] = proj

        if len(self.branch_order) > 1 and hasattr(self, "joint_gate_generator") and self.joint_gate_generator is not None:
            joint_gate_tensor = self.joint_gate_generator(
                torch.cat([aligned_projections[branch] for branch in self.branch_order], dim=1)
            )
            joint_gate_chunks = joint_gate_tensor.chunk(len(self.branch_order), dim=1)
            joint_gate_map = {
                branch: gate_chunk
                for branch, gate_chunk in zip(self.branch_order, joint_gate_chunks)
            }
        else:
            joint_gate_map = {
                branch: torch.zeros_like(proj)
                for branch, proj in aligned_projections.items()
            }

        for branch in self.branch_order:
            if branch not in aligned_projections:
                continue
            proj = aligned_projections[branch]
            # Feature-conditioned gating lets each branch adapt by region and channel.
            raw_gate = (
                self.gate_generators[branch](proj)
                + joint_gate_map[branch]
                + self.gate_bias[branch]
            )
            gate = torch.sigmoid(raw_gate) * 0.8 + 0.1
            if noise_branch is not None and branch == noise_branch:
                # weight noise branch by the configured noise weight
                gate = gate * noise_weight

            # Initialize fused/weight_sum tensors on first iteration to avoid
            # creating large intermediate Python floats and to enable in-place
            # accumulation which reduces peak memory.
            if fused is None:
                # Initialize accumulators as regular tensors (not in-place) so
                # autograd can track operations correctly.
                fused = proj * gate
                weight_sum = gate
            else:
                prod = proj * gate
                fused = fused + prod
                weight_sum = weight_sum + gate

        fused = fused / (weight_sum + eps)
        if (
            noise_branch is not None
            and self.late_residual_boost > 0.0
            and noise_branch in aligned_projections
            and noise_weight > 0.0
        ):
            fused = fused + (aligned_projections[noise_branch] * (self.late_residual_boost * noise_weight))
        fused = self.refine(fused)
        if self.fusion_refinement:
            fused = self.refine2(fused)
        return fused


class MultiStageFeatureFusion(nn.Module):
    """Fuses multi-branch features across stages with adaptive gating.

    Forensic motivation: Optionally adds a spatial refinement layer after fusion output for each stage, improving spatial detail without large FLOP increase.
    """
    def __init__(
        self,
        branch_channels: Dict[str, Sequence[int]],
        config: FeatureFusionConfig,
    ) -> None:
        super().__init__()
        self.cfg = config
        self.branches = list(branch_channels.keys())

        num_stages = len(config.fusion_channels)
        self.stages = nn.ModuleList()
        for stage_idx in range(num_stages):
            stage_branch_channels: Dict[str, int] = {}
            for branch, channels in branch_channels.items():
                if stage_idx < len(channels):
                    stage_branch_channels[branch] = channels[stage_idx]
            if not stage_branch_channels:
                raise ValueError(f"No branch provides features for stage {stage_idx}")
            self.stages.append(
                _AdaptiveFusionStage(
                    stage_branch_channels,
                    config.fusion_channels[stage_idx],
                    norm=config.norm,
                    activation=config.activation,
                    fusion_refinement=getattr(config, 'fusion_refinement', False),
                    enable_joint_gating=getattr(config, "enable_joint_gating", False),
                    late_residual_boost=(
                        getattr(config, "late_residual_boost", 0.0)
                        if (
                            config.noise_branch in stage_branch_channels
                            and stage_idx >= getattr(config, "late_residual_boost_start", 1)
                        )
                        else 0.0
                    ),
                )
            )

    def _noise_weight(self, stage_idx: int) -> float:
        skip = self.cfg.noise_skip_stage
        if skip is not None and stage_idx >= skip:
            return 0.0
        decay = max(self.cfg.noise_decay, 0.0)
        return decay ** stage_idx

    def forward(
        self,
        features: Dict[str, List[Tensor]],
        target_size: Optional[Tuple[int, int]] = None,
    ) -> List[Tensor]:
        fused: List[Tensor] = []
        for stage_idx, stage in enumerate(self.stages):
            stage_inputs: Dict[str, Tensor] = {}
            for branch in self.branches:
                branch_feats = features.get(branch, [])
                if stage_idx < len(branch_feats):
                    stage_inputs[branch] = branch_feats[stage_idx]
            if not stage_inputs:
                continue

            fused.append(
                stage(
                    stage_inputs,
                    target_size=target_size,
                    noise_branch=self.cfg.noise_branch,
                    noise_weight=self._noise_weight(stage_idx),
                )
            )
        return fused


__all__ = ["FeatureFusionConfig", "MultiStageFeatureFusion"]
