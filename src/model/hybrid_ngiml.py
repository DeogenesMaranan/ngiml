"""Hybrid NGIML model that fuses CNN, Transformer, and noise cues."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
    return OptimizerGroupConfig(lr=5e-5, weight_decay=1e-5)


def _default_swin_optim() -> OptimizerGroupConfig:
    return OptimizerGroupConfig(lr=2e-5, weight_decay=1e-5)


def _default_residual_optim() -> OptimizerGroupConfig:
    return OptimizerGroupConfig(lr=1e-4, weight_decay=0.0)


def _default_fusion_optim() -> OptimizerGroupConfig:
    return OptimizerGroupConfig(lr=1e-4, weight_decay=1e-5)


def _default_decoder_optim() -> OptimizerGroupConfig:
    return OptimizerGroupConfig(lr=1e-4, weight_decay=1e-5)


@dataclass
class HybridNGIMLOptimizerConfig:
    """Optimizer hyper-parameters separated per backbone/fusion branch."""

    efficientnet: OptimizerGroupConfig = field(default_factory=_default_efficientnet_optim)
    swin: OptimizerGroupConfig = field(default_factory=_default_swin_optim)
    residual: OptimizerGroupConfig = field(default_factory=_default_residual_optim)
    fusion: OptimizerGroupConfig = field(default_factory=_default_fusion_optim)
    decoder: OptimizerGroupConfig = field(default_factory=_default_decoder_optim)
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class HybridNGIMLConfig:
    """Aggregated configuration for the hybrid NGIML model."""

    efficientnet: EfficientNetBackboneConfig = field(default_factory=EfficientNetBackboneConfig)
    swin: SwinBackboneConfig = field(default_factory=SwinBackboneConfig)
    residual: ResidualNoiseConfig = field(default_factory=ResidualNoiseConfig)
    fusion: FeatureFusionConfig = field(
        default_factory=lambda: FeatureFusionConfig(fusion_channels=(128, 192, 256, 320))
    )
    decoder: UNetDecoderConfig = field(default_factory=UNetDecoderConfig)
    optimizer: HybridNGIMLOptimizerConfig = field(default_factory=HybridNGIMLOptimizerConfig)
    use_low_level: bool = True
    use_context: bool = True
    use_residual: bool = True


class HybridNGIML(nn.Module):
    """Full NGIML model exposing fused multi-scale features."""

    def __init__(self, config: HybridNGIMLConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or HybridNGIMLConfig()
        self.efficientnet = EfficientNetBackbone(self.cfg.efficientnet)
        self.swin = SwinBackbone(self.cfg.swin)
        self.noise = ResidualNoiseBranch(self.cfg.residual)

        layout = {
            "low_level": self.efficientnet.out_channels,
            "context": self.swin.out_channels,
            "residual": [3],
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

    def _extract_features(self, x: Tensor) -> Dict[str, List[Tensor] | Tensor]:
        low_level = self.efficientnet(x)
        context = self.swin(x)
        residual = self.noise(x)
        return {
            "low_level": low_level,
            "context": context,
            "residual": residual,
        }

    def _broadcast_residual(self, residual: Tensor) -> List[Tensor]:
        return [residual for _ in range(self.num_stages)]

    def forward_features(
        self,
        x: Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> List[Tensor]:
        backbone_feats = self._extract_features(x)
        fusion_inputs = {}
        if self.cfg.use_low_level:
            fusion_inputs["low_level"] = backbone_feats["low_level"]
        if self.cfg.use_context:
            fusion_inputs["context"] = backbone_feats["context"]
        if self.cfg.use_residual:
            fusion_inputs["residual"] = self._broadcast_residual(backbone_feats["residual"])
        return self.fusion(fusion_inputs, target_size=target_size)

    def forward(
        self,
        x: Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> List[Tensor]:
        fused = self.forward_features(x, target_size=target_size)
        return self.decoder(fused)

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
        if self.cfg.use_residual:
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
