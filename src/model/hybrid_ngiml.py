"""Hybrid NGIML model that fuses CNN, Transformer, and noise cues."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .backbones.efficientnet_backbone import EfficientNetBackbone, EfficientNetBackboneConfig
from .backbones.residual_noise_branch import ResidualNoiseBranch, ResidualNoiseConfig
from .backbones.segformer_backbone import SegFormerBackbone, SegFormerBackboneConfig
from .backbones.swin_backbone import SwinBackbone, SwinBackboneConfig
from .feature_fusion import FeatureFusionConfig, MultiStageFeatureFusion
from .fpn_decoder import FPNDecoder, FPNDecoderConfig
from .unet_decoder import UNetDecoder, UNetDecoderConfig


@dataclass
class OptimizerGroupConfig:
    """Learning rate / weight decay pair for an optimizer parameter group."""
    lr: float
    weight_decay: float = 1e-5


def _default_efficientnet_optim() -> OptimizerGroupConfig:
    # Forensic motivation: Lower LR for backbone to stabilize early training
    return OptimizerGroupConfig(lr=1e-5, weight_decay=1e-4)


def _default_swin_optim() -> OptimizerGroupConfig:
    return OptimizerGroupConfig(lr=5e-6, weight_decay=5e-5)


def _default_segformer_optim() -> OptimizerGroupConfig:
    return OptimizerGroupConfig(lr=5e-6, weight_decay=5e-5)


def _default_residual_optim() -> OptimizerGroupConfig:
    return OptimizerGroupConfig(lr=3e-4, weight_decay=1e-4)


def _default_fusion_optim() -> OptimizerGroupConfig:
    return OptimizerGroupConfig(lr=1.5e-4, weight_decay=1e-4)


def _default_decoder_optim() -> OptimizerGroupConfig:
    return OptimizerGroupConfig(lr=2e-4, weight_decay=1e-4)


@dataclass
class HybridNGIMLOptimizerConfig:
    """Optimizer hyper-parameters separated per backbone/fusion branch.

    Forensic motivation: Lower backbone LR, higher forensic/fusion/decoder LRs, and support freezing backbone for early epochs.
    """
    efficientnet: OptimizerGroupConfig = field(default_factory=_default_efficientnet_optim)
    swin: OptimizerGroupConfig = field(default_factory=_default_swin_optim)
    segformer: OptimizerGroupConfig = field(default_factory=_default_segformer_optim)
    residual: OptimizerGroupConfig = field(default_factory=_default_residual_optim)
    fusion: OptimizerGroupConfig = field(default_factory=_default_fusion_optim)
    decoder: OptimizerGroupConfig = field(default_factory=_default_decoder_optim)
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    freeze_backbone_epochs: int = 5


@dataclass
class HybridNGIMLConfig:
    """Aggregated configuration for the hybrid NGIML model."""

    efficientnet: EfficientNetBackboneConfig = field(default_factory=EfficientNetBackboneConfig)
    swin: SwinBackboneConfig = field(default_factory=SwinBackboneConfig)
    segformer: SegFormerBackboneConfig = field(default_factory=SegFormerBackboneConfig)
    residual: ResidualNoiseConfig = field(default_factory=ResidualNoiseConfig)
    fusion: FeatureFusionConfig = field(
        default_factory=lambda: FeatureFusionConfig(fusion_channels=(64, 128, 192, 256))
    )
    decoder: UNetDecoderConfig = field(default_factory=UNetDecoderConfig)
    fpn_decoder: FPNDecoderConfig = field(default_factory=FPNDecoderConfig)
    optimizer: HybridNGIMLOptimizerConfig = field(default_factory=HybridNGIMLOptimizerConfig)
    use_low_level: bool = True
    use_context: bool = True
    use_segformer: bool = False  # SegFormer backbone (disabled by default for backward compat)
    use_residual: bool = True
    enable_residual_attention: bool = True  # Residual-guided attention (enabled by default)
    enable_context_attention: bool = True  # Residual-guided attention on context branch too
    decoder_type: str = "unet"  # one of: "unet", "fpn"
    use_boundary_refiner: bool = True  # LIFD-style boundary refinement head
    boundary_refiner_channels: int = 64
    gradient_checkpointing: bool = False  # Trade compute for memory savings


class HybridNGIML(nn.Module):
    """Full NGIML model exposing fused multi-scale features.

    Forensic motivation:
    - Optionally applies residual-guided attention to *both* low-level and
      context features before fusion, improving manipulation localization.
    - Supports SegFormer as a third backbone for fine-grained boundary capture.
    - Supports FPN decoder as alternative to UNet.
    - Includes LIFD-style boundary refiner head (gradient-guided mask
      refinement) for sharper prediction edges.
    - Supports gradient checkpointing for memory-efficient training.
    """

    def __init__(self, config: HybridNGIMLConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or HybridNGIMLConfig()

        # --- Backbones ---
        self.efficientnet = EfficientNetBackbone(self.cfg.efficientnet)
        self.swin = SwinBackbone(self.cfg.swin)
        self.noise = ResidualNoiseBranch(self.cfg.residual)

        layout: Dict[str, List[int]] = {
            "low_level": self.efficientnet.out_channels,
            "context": self.swin.out_channels,
            "residual": self.noise.out_channels,
        }

        # Optional SegFormer backbone
        self.segformer: SegFormerBackbone | None = None
        if self.cfg.use_segformer:
            self.segformer = SegFormerBackbone(self.cfg.segformer)
            layout["segformer"] = self.segformer.out_channels

        self.num_stages = len(self.cfg.fusion.fusion_channels)
        branch_channels: Dict[str, List[int]] = {}

        if self.cfg.use_low_level:
            branch_channels["low_level"] = layout["low_level"]
        if self.cfg.use_context:
            branch_channels["context"] = layout["context"]
        if self.cfg.use_segformer and self.segformer is not None:
            branch_channels["segformer"] = layout["segformer"]
        if self.cfg.use_residual:
            residual_channels = layout.get("residual", [3])
            if len(residual_channels) == 1:
                residual_channels = residual_channels * self.num_stages
            branch_channels["residual"] = residual_channels

        if not branch_channels:
            raise ValueError("At least one backbone branch must be enabled for fusion")

        # --- Fusion ---
        self.fusion = MultiStageFeatureFusion(branch_channels, self.cfg.fusion)

        # --- Decoder ---
        decoder_type = self.cfg.decoder_type.lower()
        if decoder_type == "fpn":
            self.decoder = FPNDecoder(self.cfg.fusion.fusion_channels, self.cfg.fpn_decoder)
        else:
            self.decoder = UNetDecoder(self.cfg.fusion.fusion_channels, self.cfg.decoder)

        # --- Residual-guided attention modules (optional) ---
        self.enable_residual_attention = getattr(self.cfg, 'enable_residual_attention', False)
        self.enable_context_attention = getattr(self.cfg, 'enable_context_attention', False)

        if self.enable_residual_attention and "residual" in branch_channels:
            res_channels = branch_channels["residual"]
            # Low-level attention (stage 0)
            if "low_level" in branch_channels:
                sem_channels = branch_channels["low_level"]
                attn_in_ch = res_channels[0] if res_channels else 0
                attn_out_ch = sem_channels[0] if sem_channels else 0
                self.residual_attention_proj = nn.Conv2d(attn_in_ch, attn_out_ch, kernel_size=1)
                nn.init.zeros_(self.residual_attention_proj.weight)
                if self.residual_attention_proj.bias is not None:
                    nn.init.zeros_(self.residual_attention_proj.bias)
            else:
                self.residual_attention_proj = None

            # Context-branch attention (stage 0) — NEW: extends LIFD's approach
            if self.enable_context_attention and "context" in branch_channels:
                ctx_channels = branch_channels["context"]
                ctx_attn_in = res_channels[0] if res_channels else 0
                ctx_attn_out = ctx_channels[0] if ctx_channels else 0
                self.context_attention_proj = nn.Conv2d(ctx_attn_in, ctx_attn_out, kernel_size=1)
                nn.init.zeros_(self.context_attention_proj.weight)
                if self.context_attention_proj.bias is not None:
                    nn.init.zeros_(self.context_attention_proj.bias)
            else:
                self.context_attention_proj = None
        else:
            self.residual_attention_proj = None
            self.context_attention_proj = None

        # --- Boundary refiner (from LIFD) ---
        self.boundary_refiner: nn.Module | None = None
        if self.cfg.use_boundary_refiner:
            br_ch = self.cfg.boundary_refiner_channels
            self.boundary_refiner = nn.Sequential(
                nn.Conv2d(2, br_ch, kernel_size=3, padding=1),
                nn.GroupNorm(4, br_ch),
                nn.GELU(),
                nn.Conv2d(br_ch, 1, kernel_size=3, padding=1),
            )
            sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
            sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
            self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
            self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False)

    def _image_gradient_magnitude(self, image: Tensor) -> Tensor:
        """Compute per-pixel gradient magnitude via Sobel filters."""
        if self.boundary_refiner is None:
            return torch.zeros_like(image[:, :1])
        kernel_x = self.sobel_x.to(image.device, image.dtype).repeat(image.shape[1], 1, 1, 1)
        kernel_y = self.sobel_y.to(image.device, image.dtype).repeat(image.shape[1], 1, 1, 1)
        grad_x = F.conv2d(image, kernel_x, padding=1, groups=image.shape[1])
        grad_y = F.conv2d(image, kernel_y, padding=1, groups=image.shape[1])
        grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        return grad_mag.sum(dim=1, keepdim=True)

    def _run_backbone(self, backbone: nn.Module, x: Tensor) -> List[Tensor]:
        """Run a backbone with optional gradient checkpointing."""
        if self.cfg.gradient_checkpointing and torch.is_grad_enabled():
            def _fwd(inp, mod=backbone):
                return tuple(mod(inp))
            try:
                features = grad_checkpoint(_fwd, x, use_reentrant=False)
            except TypeError:
                features = grad_checkpoint(_fwd, x)
            return list(features)
        return backbone(x)

    def _extract_features(self, x: Tensor, high_pass: Tensor | None = None) -> Dict[str, List[Tensor]]:
        low_level = self._run_backbone(self.efficientnet, x)
        context = self._run_backbone(self.swin, x)
        residual = self.noise(x, high_pass=high_pass)

        # Residual-guided attention on low-level features (stage 0)
        if (
            self.enable_residual_attention
            and self.residual_attention_proj is not None
            and isinstance(low_level, list)
            and isinstance(residual, list)
        ):
            res_feat = residual[0]
            if res_feat.shape[-2:] != low_level[0].shape[-2:]:
                res_feat = F.interpolate(res_feat, size=low_level[0].shape[-2:], mode="bilinear", align_corners=False)
            attn_map = torch.sigmoid(self.residual_attention_proj(res_feat))
            low_level[0] = low_level[0] * (1.0 + attn_map)

        # Residual-guided attention on context features (stage 0) — NEW
        if (
            self.enable_context_attention
            and self.context_attention_proj is not None
            and isinstance(context, list)
            and isinstance(residual, list)
        ):
            res_feat = residual[0]
            if res_feat.shape[-2:] != context[0].shape[-2:]:
                res_feat = F.interpolate(res_feat, size=context[0].shape[-2:], mode="bilinear", align_corners=False)
            ctx_attn = torch.sigmoid(self.context_attention_proj(res_feat))
            context[0] = context[0] * (1.0 + ctx_attn)

        feats: Dict[str, List[Tensor]] = {
            "low_level": low_level,
            "context": context,
            "residual": residual,
        }

        # SegFormer backbone
        if self.segformer is not None:
            feats["segformer"] = self._run_backbone(self.segformer, x)

        return feats

    def forward_features(
        self,
        x: Tensor,
        target_size: Optional[Tuple[int, int]] = None,
        high_pass: Tensor | None = None,
    ) -> List[Tensor]:
        backbone_feats = self._extract_features(x, high_pass=high_pass)
        fusion_inputs: Dict[str, List[Tensor]] = {}
        if self.cfg.use_low_level:
            fusion_inputs["low_level"] = backbone_feats["low_level"]
        if self.cfg.use_context:
            fusion_inputs["context"] = backbone_feats["context"]
        if self.cfg.use_segformer and "segformer" in backbone_feats:
            fusion_inputs["segformer"] = backbone_feats["segformer"]
        if self.cfg.use_residual:
            fusion_inputs["residual"] = backbone_feats["residual"]
        return self.fusion(fusion_inputs, target_size=None)

    def forward(
        self,
        x: Tensor,
        target_size: Optional[Tuple[int, int]] = None,
        high_pass: Tensor | None = None,
    ) -> List[Tensor]:
        fused = self.forward_features(x, target_size=None, high_pass=high_pass)
        preds = self.decoder(fused)

        # Boundary refiner: refine the final (highest-res) prediction
        if self.boundary_refiner is not None and preds:
            final_pred = preds[-1]
            grad_mag = self._image_gradient_magnitude(x)
            if grad_mag.shape[-2:] != final_pred.shape[-2:]:
                grad_mag = F.interpolate(grad_mag, size=final_pred.shape[-2:], mode="bilinear", align_corners=False)
            ref_input = torch.cat([final_pred, grad_mag], dim=1)
            preds[-1] = final_pred + self.boundary_refiner(ref_input)

        if target_size is None:
            return preds
        return [
            F.interpolate(pred, size=target_size, mode="bilinear", align_corners=False)
            if pred.shape[-2:] != target_size
            else pred
            for pred in preds
        ]

    def optimizer_parameter_groups(self) -> List[Dict[str, object]]:
        """Return AdamW-ready parameter groups with branch-specific LRs/decays."""

        groups: List[Dict[str, object]] = []

        def _append(params, group_cfg: OptimizerGroupConfig) -> None:
            param_list = list(params)
            if not param_list:
                return
            groups.append({
                "params": param_list,
                "lr": group_cfg.lr,
                "weight_decay": group_cfg.weight_decay,
            })

        if self.cfg.use_low_level:
            _append(self.efficientnet.parameters(), self.cfg.optimizer.efficientnet)
        if self.cfg.use_context:
            _append(self.swin.parameters(), self.cfg.optimizer.swin)
        if self.cfg.use_segformer and self.segformer is not None:
            _append(self.segformer.parameters(), self.cfg.optimizer.segformer)
        if self.cfg.use_residual:
            _append(self.noise.parameters(), self.cfg.optimizer.residual)

        _append(self.fusion.parameters(), self.cfg.optimizer.fusion)
        _append(self.decoder.parameters(), self.cfg.optimizer.decoder)

        # Boundary refiner + residual attention projections go with decoder LR
        extra_params = []
        if self.boundary_refiner is not None:
            extra_params.extend(self.boundary_refiner.parameters())
        if self.residual_attention_proj is not None:
            extra_params.extend(self.residual_attention_proj.parameters())
        if self.context_attention_proj is not None:
            extra_params.extend(self.context_attention_proj.parameters())
        if extra_params:
            _append(iter(extra_params), self.cfg.optimizer.decoder)

        if not groups:
            raise ValueError("No parameter groups available for optimization")

        return groups

    def build_optimizer(self) -> AdamW:
        """Instantiate an AdamW optimizer using the configured parameter groups."""

        param_groups = self.optimizer_parameter_groups()
        return AdamW(param_groups, betas=self.cfg.optimizer.betas, eps=self.cfg.optimizer.eps)


__all__ = [
    "HybridNGIML",
    "HybridNGIMLConfig",
    "HybridNGIMLOptimizerConfig",
    "OptimizerGroupConfig",
    "UNetDecoderConfig",
    "FPNDecoderConfig",
    "SegFormerBackboneConfig",
]
