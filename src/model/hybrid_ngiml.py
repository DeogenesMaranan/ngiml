"""Hybrid NGIML model that fuses CNN, Transformer, and noise cues."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW

from .backbones.efficientnet_backbone import EfficientNetBackbone, EfficientNetBackboneConfig
from .backbones.residual_noise_branch import ResidualNoiseBranch, ResidualNoiseConfig
from .backbones.swin_backbone import SwinBackbone, SwinBackboneConfig
from .feature_fusion import FeatureFusionConfig, MultiStageFeatureFusion
from .unet_decoder import UNetDecoder, UNetDecoderConfig


@dataclass
class OptimizerGroupConfig:
    """Learning rate / weight decay pair for an optimizer parameter group."""
    lr: float
    weight_decay: float = 1e-5


def _default_efficientnet_optim() -> OptimizerGroupConfig:
    # Forensic motivation: Lower LR for backbone to stabilize early training
    return OptimizerGroupConfig(lr=1e-5, weight_decay=1.5e-4)


def _default_swin_optim() -> OptimizerGroupConfig:
    return OptimizerGroupConfig(lr=5e-6, weight_decay=1e-4)


def _default_residual_optim() -> OptimizerGroupConfig:
    return OptimizerGroupConfig(lr=2.5e-4, weight_decay=2e-4)


def _default_fusion_optim() -> OptimizerGroupConfig:
    return OptimizerGroupConfig(lr=1.2e-4, weight_decay=2e-4)


def _default_decoder_optim() -> OptimizerGroupConfig:
    return OptimizerGroupConfig(lr=1.8e-4, weight_decay=2e-4)


@dataclass
class HybridNGIMLOptimizerConfig:
    """Optimizer hyper-parameters separated per backbone/fusion branch.

    Forensic motivation: Lower backbone LR, higher forensic/fusion/decoder LRs, and support freezing backbone for early epochs.
    """
    efficientnet: OptimizerGroupConfig = field(default_factory=_default_efficientnet_optim)
    swin: OptimizerGroupConfig = field(default_factory=_default_swin_optim)
    residual: OptimizerGroupConfig = field(default_factory=_default_residual_optim)
    fusion: OptimizerGroupConfig = field(default_factory=_default_fusion_optim)
    decoder: OptimizerGroupConfig = field(default_factory=_default_decoder_optim)
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    freeze_backbone_epochs: int = 3


@dataclass
class HybridNGIMLConfig:
    """Aggregated configuration for the hybrid NGIML model."""

    efficientnet: EfficientNetBackboneConfig = field(default_factory=EfficientNetBackboneConfig)
    swin: SwinBackboneConfig = field(default_factory=SwinBackboneConfig)
    residual: ResidualNoiseConfig = field(default_factory=ResidualNoiseConfig)
    fusion: FeatureFusionConfig = field(
        default_factory=lambda: FeatureFusionConfig(fusion_channels=(64, 128, 192, 256))
    )
    decoder: UNetDecoderConfig = field(default_factory=UNetDecoderConfig)
    optimizer: HybridNGIMLOptimizerConfig = field(default_factory=HybridNGIMLOptimizerConfig)
    use_low_level: bool = True
    use_context: bool = True
    use_residual: bool = True
    enable_residual_attention: bool = True
    enable_low_level_residual_attention: bool = True
    enable_context_residual_attention: bool = False
    residual_attention_init_scale: float = 0.0
    gradient_checkpointing: bool = True  # Enable gradient checkpointing for memory savings
    flash_attention: bool = True  # Enable flash attention by default
    xformers: bool = True  # Enable xformers by default


class HybridNGIML(nn.Module):
    """Full NGIML model exposing fused multi-scale features.

    Forensic motivation: Optionally applies residual-guided attention to semantic features before fusion, improving manipulation localization.
    """

    def __init__(self, config: HybridNGIMLConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or HybridNGIMLConfig()
        self.efficientnet: Optional[EfficientNetBackbone] = None
        self.swin: Optional[SwinBackbone] = None
        self.noise: Optional[ResidualNoiseBranch] = None

        if self.cfg.use_low_level:
            self.efficientnet = EfficientNetBackbone(self.cfg.efficientnet)

        if self.cfg.use_context:
            # Pass flash_attention and xformers flags if present in config
            swin_kwargs = {}
            if hasattr(self.cfg, 'flash_attention'):
                swin_kwargs['flash_attention'] = getattr(self.cfg, 'flash_attention', False)
            if hasattr(self.cfg, 'xformers'):
                swin_kwargs['xformers'] = getattr(self.cfg, 'xformers', False)
            self.swin = SwinBackbone(self.cfg.swin, **swin_kwargs)

        if self.cfg.use_residual:
            self.noise = ResidualNoiseBranch(self.cfg.residual)

        layout = {
            "low_level": self.efficientnet.out_channels if self.efficientnet is not None else [],
            "context": self.swin.out_channels if self.swin is not None else [],
            "residual": self.noise.out_channels if self.noise is not None else [],
        }
        self.num_stages = len(self.cfg.fusion.fusion_channels)
        branch_channels: Dict[str, List[int]] = {}

        if self.cfg.use_low_level:
            branch_channels["low_level"] = layout["low_level"]
        if self.cfg.use_context:
            branch_channels["context"] = layout["context"]
        if self.cfg.use_residual:
            residual_channels = layout.get("residual", [3])
            if len(residual_channels) == 1:
                residual_channels = residual_channels * self.num_stages
            branch_channels["residual"] = residual_channels

        if not branch_channels:
            raise ValueError("At least one backbone branch must be enabled for fusion")

        self.fusion = MultiStageFeatureFusion(branch_channels, self.cfg.fusion)
        self.decoder = UNetDecoder(self.cfg.fusion.fusion_channels, self.cfg.decoder)

        # Residual-guided attention module (optional)
        self.enable_residual_attention = (
            getattr(self.cfg, 'enable_residual_attention', False)
            and self.cfg.use_residual
            and self.noise is not None
            and (self.cfg.use_low_level or self.cfg.use_context)
        )
        if self.enable_residual_attention:
            res_channels = branch_channels.get("residual", [])
            if getattr(self.cfg, "enable_low_level_residual_attention", True):
                self.low_level_residual_attention_proj = self._build_residual_attention_proj(
                    res_channels,
                    branch_channels.get("low_level", []),
                )
                self.low_level_residual_attention_scale = nn.ParameterList(
                    [
                        nn.Parameter(
                            torch.full(
                                (1,),
                                float(getattr(self.cfg, "residual_attention_init_scale", 0.0)),
                            )
                        )
                        for _ in range(len(self.low_level_residual_attention_proj))
                    ]
                )
            if getattr(self.cfg, "enable_context_residual_attention", False):
                self.context_residual_attention_proj = self._build_residual_attention_proj(
                    res_channels,
                    branch_channels.get("context", []),
                )
                self.context_residual_attention_scale = nn.ParameterList(
                    [
                        nn.Parameter(
                            torch.full(
                                (1,),
                                float(getattr(self.cfg, "residual_attention_init_scale", 0.0)),
                            )
                        )
                        for _ in range(len(self.context_residual_attention_proj))
                    ]
                )

    @staticmethod
    def _build_residual_attention_proj(
        residual_channels: List[int],
        target_channels: List[int],
    ) -> nn.ModuleList:
        projections = nn.ModuleList()
        for stage_idx in range(min(len(residual_channels), len(target_channels))):
            proj = nn.Conv2d(residual_channels[stage_idx], target_channels[stage_idx], kernel_size=1)
            nn.init.zeros_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
            projections.append(proj)
        return projections

    @staticmethod
    def _apply_residual_attention(
        target_features: Optional[List[Tensor]],
        residual_features: Optional[List[Tensor]],
        projections: nn.ModuleList,
        scales: Optional[nn.ParameterList] = None,
    ) -> None:
        if not isinstance(target_features, list) or not isinstance(residual_features, list):
            return
        for stage_idx, proj in enumerate(projections):
            if stage_idx >= len(target_features) or stage_idx >= len(residual_features):
                break
            # Start from an identity modulation at zero init so residual guidance
            # does not bias all semantic streams before learning.
            attn_map = (2.0 * torch.sigmoid(proj(residual_features[stage_idx]))) - 1.0
            if scales is not None and stage_idx < len(scales):
                attn_map = attn_map * scales[stage_idx].view(1, 1, 1, 1)
            tgt_h, tgt_w = target_features[stage_idx].shape[-2:]
            if attn_map.shape[-2:] != (tgt_h, tgt_w):
                attn_map = F.interpolate(attn_map, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)
            target_features[stage_idx] = target_features[stage_idx] * (1.0 + attn_map)

    def _extract_features(self, x: Tensor, residual_noise: Tensor | None = None) -> Dict[str, Optional[List[Tensor] | Tensor]]:
        low_level = self.efficientnet(x) if self.efficientnet is not None else None
        context = self.swin(x) if self.swin is not None else None
        residual = self.noise(x, residual_noise=residual_noise) if self.noise is not None else None

        # Residual-guided attention (modulate semantic features before fusion)
        if self.enable_residual_attention and isinstance(residual, list):
            self._apply_residual_attention(
                low_level,
                residual,
                getattr(self, "low_level_residual_attention_proj", nn.ModuleList()),
                getattr(self, "low_level_residual_attention_scale", None),
            )
            self._apply_residual_attention(
                context,
                residual,
                getattr(self, "context_residual_attention_proj", nn.ModuleList()),
                getattr(self, "context_residual_attention_scale", None),
            )

        # Note: previous implementation attempted to checkpoint each child module
        # by calling them directly with the original input `x`. That is incorrect
        # because many backbone children expect the output of prior blocks, not
        # the raw image, which can cause BatchNorm channel mismatches.
        # For now, we avoid per-child checkpointing and rely on the backbone's
        # own forward implementation (which may internally support memory saving).

        return {
            "low_level": low_level,
            "context": context,
            "residual": residual,
        }

    def forward_features(
        self,
        x: Tensor,
        target_size: Optional[Tuple[int, int]] = None,
        residual_noise: Tensor | None = None,
    ) -> List[Tensor]:
        backbone_feats = self._extract_features(x, residual_noise=residual_noise)
        fusion_inputs = {}
        if self.cfg.use_low_level and backbone_feats["low_level"] is not None:
            fusion_inputs["low_level"] = backbone_feats["low_level"]
        if self.cfg.use_context and backbone_feats["context"] is not None:
            fusion_inputs["context"] = backbone_feats["context"]
        if self.cfg.use_residual and backbone_feats["residual"] is not None:
            fusion_inputs["residual"] = backbone_feats["residual"]
        return self.fusion(fusion_inputs, target_size=target_size)

    def forward(
        self,
        x: Tensor,
        target_size: Optional[Tuple[int, int]] = None,
        residual_noise: Tensor | None = None,
    ) -> List[Tensor]:
        fused = self.forward_features(x, target_size=None, residual_noise=residual_noise)
        preds = self.decoder(fused, image=x)
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

        if self.cfg.use_low_level and self.efficientnet is not None:
            _append(self.efficientnet.parameters(), self.cfg.optimizer.efficientnet)
        if self.cfg.use_context and self.swin is not None:
            _append(self.swin.parameters(), self.cfg.optimizer.swin)
        if self.cfg.use_residual and self.noise is not None:
            _append(self.noise.parameters(), self.cfg.optimizer.residual)

        _append(self.fusion.parameters(), self.cfg.optimizer.fusion)
        _append(self.decoder.parameters(), self.cfg.optimizer.decoder)

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
]

