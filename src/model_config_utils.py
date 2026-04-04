from __future__ import annotations

from src.model.backbones.efficientnet_backbone import EfficientNetBackboneConfig
from src.model.backbones.residual_noise_branch import ResidualNoiseConfig
from src.model.backbones.swin_backbone import SwinBackboneConfig
from src.model.feature_fusion import FeatureFusionConfig
from src.model.hybrid_ngiml import HybridNGIMLConfig, HybridNGIMLOptimizerConfig, OptimizerGroupConfig
from src.model.losses import MultiStageLossConfig
from src.model.unet_decoder import UNetDecoderConfig


def coerce_model_config(value) -> HybridNGIMLConfig:
    if value is None:
        return HybridNGIMLConfig()
    if isinstance(value, HybridNGIMLConfig):
        return value
    if not isinstance(value, dict):
        raise TypeError("Model config must be HybridNGIMLConfig or dict")

    def _coerce_optimizer_config(opt_value) -> HybridNGIMLOptimizerConfig:
        if opt_value is None:
            return HybridNGIMLOptimizerConfig()
        if isinstance(opt_value, HybridNGIMLOptimizerConfig):
            return opt_value
        if not isinstance(opt_value, dict):
            raise TypeError("Optimizer config must be HybridNGIMLOptimizerConfig or dict")

        default_opt = HybridNGIMLOptimizerConfig()

        def _coerce_group(group_value, default_group: OptimizerGroupConfig) -> OptimizerGroupConfig:
            if isinstance(group_value, OptimizerGroupConfig):
                return group_value
            if group_value is None:
                return default_group
            if isinstance(group_value, dict):
                return OptimizerGroupConfig(**group_value)
            raise TypeError("Optimizer group config must be OptimizerGroupConfig or dict")

        betas_raw = opt_value.get("betas", default_opt.betas)
        if isinstance(betas_raw, list):
            betas = tuple(float(v) for v in betas_raw)
        else:
            betas = tuple(betas_raw)

        return HybridNGIMLOptimizerConfig(
            efficientnet=_coerce_group(opt_value.get("efficientnet"), default_opt.efficientnet),
            swin=_coerce_group(opt_value.get("swin"), default_opt.swin),
            residual=_coerce_group(opt_value.get("residual"), default_opt.residual),
            fusion=_coerce_group(opt_value.get("fusion"), default_opt.fusion),
            decoder=_coerce_group(opt_value.get("decoder"), default_opt.decoder),
            betas=betas,
            eps=float(opt_value.get("eps", default_opt.eps)),
            freeze_backbone_epochs=int(opt_value.get("freeze_backbone_epochs", default_opt.freeze_backbone_epochs)),
        )

    default_model = HybridNGIMLConfig()
    efficientnet = value.get("efficientnet", default_model.efficientnet)
    swin = value.get("swin", default_model.swin)
    residual = value.get("residual", default_model.residual)
    fusion = value.get("fusion", default_model.fusion)
    decoder = value.get("decoder", default_model.decoder)
    optimizer = value.get("optimizer", default_model.optimizer)

    return HybridNGIMLConfig(
        efficientnet=efficientnet if isinstance(efficientnet, EfficientNetBackboneConfig) else EfficientNetBackboneConfig(**efficientnet),
        swin=swin if isinstance(swin, SwinBackboneConfig) else SwinBackboneConfig(**swin),
        residual=residual if isinstance(residual, ResidualNoiseConfig) else ResidualNoiseConfig(**residual),
        fusion=fusion if isinstance(fusion, FeatureFusionConfig) else FeatureFusionConfig(**fusion),
        decoder=decoder if isinstance(decoder, UNetDecoderConfig) else UNetDecoderConfig(**decoder),
        optimizer=_coerce_optimizer_config(optimizer),
        use_low_level=bool(value.get("use_low_level", default_model.use_low_level)),
        use_context=bool(value.get("use_context", default_model.use_context)),
        use_residual=bool(value.get("use_residual", default_model.use_residual)),
        enable_residual_attention=bool(value.get("enable_residual_attention", default_model.enable_residual_attention)),
        gradient_checkpointing=bool(value.get("gradient_checkpointing", default_model.gradient_checkpointing)),
    )


def coerce_loss_config(value) -> MultiStageLossConfig:
    if value is None:
        return MultiStageLossConfig()
    if isinstance(value, MultiStageLossConfig):
        return value
    if isinstance(value, dict):
        return MultiStageLossConfig(**value)
    raise TypeError("Loss config must be MultiStageLossConfig or dict")
