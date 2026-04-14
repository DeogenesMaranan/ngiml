from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from src.data.config import AugmentationConfig
from src.model.backbones.efficientnet_backbone import EfficientNetBackboneConfig
from src.model.backbones.residual_noise_branch import ResidualNoiseConfig
from src.model.backbones.swin_backbone import SwinBackboneConfig
from src.model.feature_fusion import FeatureFusionConfig
from src.model.hybrid_ngiml import HybridNGIMLConfig, HybridNGIMLOptimizerConfig, OptimizerGroupConfig
from src.model.losses import MultiStageLossConfig
from src.model.unet_decoder import UNetDecoderConfig


def build_default_components() -> tuple[HybridNGIMLConfig, MultiStageLossConfig, AugmentationConfig, dict[str, AugmentationConfig]]:
    model_cfg = HybridNGIMLConfig(
        efficientnet=EfficientNetBackboneConfig(pretrained=True),
        swin=SwinBackboneConfig(model_name="swin_tiny_patch4_window7_224", pretrained=True, input_size=448),
        residual=ResidualNoiseConfig(num_kernels=3, base_channels=32, num_stages=4),
        fusion=FeatureFusionConfig(fusion_channels=(64, 128, 192, 256)),
        decoder=UNetDecoderConfig(
            decoder_channels=None,
            out_channels=1,
            per_stage_heads=True,
            enable_edge_guidance=True,
            use_dropout=True,
            dropout_p=0.2,
            enable_boundary_refinement=True,
            enable_detail_refinement=True,
        ),
        optimizer=HybridNGIMLOptimizerConfig(
            efficientnet=OptimizerGroupConfig(lr=1e-5, weight_decay=1.5e-4),
            swin=OptimizerGroupConfig(lr=5e-6, weight_decay=1e-4),
            residual=OptimizerGroupConfig(lr=2.5e-4, weight_decay=2e-4),
            fusion=OptimizerGroupConfig(lr=1.2e-4, weight_decay=2e-4),
            decoder=OptimizerGroupConfig(lr=1.8e-4, weight_decay=2e-4),
            freeze_backbone_epochs=3,
        ),
        use_low_level=True,
        use_context=True,
        use_residual=True,
        enable_residual_attention=True,
        enable_low_level_residual_attention=True,
        enable_context_residual_attention=False,
        residual_attention_init_scale=0.0,
    )

    loss_cfg = MultiStageLossConfig(
        dice_weight=1.0,
        bce_weight=1.0,
        focal_weight=0.0,
        pos_weight=1.0,
        stage_weights=[0.05, 0.1, 0.2, 1.0],
        smooth=1e-6,
        tversky_weight=0.2,
        tversky_alpha=0.3,
        tversky_beta=0.8,
        lovasz_weight=0.05,
        boundary_weight=0.03,
    )

    default_aug = AugmentationConfig(
        enable=True,
        views_per_sample=3,
        enable_flips=True,
        enable_rotations=True,
        max_rotation_degrees=6.0,
        enable_random_crop=True,
        crop_scale_range=(0.75, 1.0),
        object_crop_bias_prob=0.85,
        min_fg_pixels_for_object_crop=8,
        enable_elastic=False,
        elastic_prob=0.0,
        elastic_alpha=8.0,
        elastic_sigma=5.0,
        enable_color_jitter=True,
        brightness_jitter_factors=(0.9, 1.1),
        contrast_jitter_factors=(0.9, 1.1),
        enable_noise=True,
        noise_std_range=(0.0, 0.012),
    )

    per_dataset_aug: dict[str, AugmentationConfig] = {}
    return model_cfg, loss_cfg, default_aug, per_dataset_aug


def _coerce_aug(value) -> AugmentationConfig:
    if isinstance(value, AugmentationConfig):
        return replace(value)
    if isinstance(value, dict):
        allowed_keys = set(AugmentationConfig.__dataclass_fields__.keys())
        filtered = {key: aug_value for key, aug_value in value.items() if key in allowed_keys}
        return AugmentationConfig(**filtered)
    raise TypeError("Augmentation config must be AugmentationConfig or dict")


def build_training_config(
    manifest_path: Path | str,
    output_dir: str,
    model_cfg: HybridNGIMLConfig,
    loss_cfg: MultiStageLossConfig,
    default_aug: AugmentationConfig,
    per_dataset_aug: dict[str, AugmentationConfig],
) -> dict:
    safe_default_aug = replace(_coerce_aug(default_aug), views_per_sample=1)
    safe_per_dataset_aug = {
        name: replace(_coerce_aug(aug), views_per_sample=1)
        for name, aug in per_dataset_aug.items()
    }

    return {
        "manifest": str(manifest_path),
        "output_dir": output_dir,
        "batch_size": 16,
        "num_workers": 0,
        "prefetch_factor": 2,
        "views_per_sample": 1,
        "gpu_aug_batch_chunk_size": 0,
        "resize_max_side": 448,
        "max_rotation_degrees": 6.0,
        "noise_std_max": 0.012,
        "warmup_epochs": 3,
        "resume": None,
        "auto_resume": True,
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 3e-3,
        "early_stopping_monitor": "loss",
        "metric_threshold": 0.5,
        "optimize_threshold": False,
        "foreground_ratio_max_batches": 20,
        "short_side_probe_samples": 0,
        "pos_weight_max": 20.0,
        "dice_weight": float(getattr(loss_cfg, "dice_weight", 1.0)),
        "bce_weight": float(getattr(loss_cfg, "bce_weight", 1.0)),
        "focal_weight": float(getattr(loss_cfg, "focal_weight", 0.0)),
        "focal_gamma": float(getattr(loss_cfg, "focal_gamma", 2.0)),
        "focal_alpha": float(getattr(loss_cfg, "focal_alpha", 0.25)),
        "tversky_weight": float(getattr(loss_cfg, "tversky_weight", 0.2)),
        "tversky_alpha": float(getattr(loss_cfg, "tversky_alpha", 0.3)),
        "tversky_beta": float(getattr(loss_cfg, "tversky_beta", 0.8)),
        "lovasz_weight": float(getattr(loss_cfg, "lovasz_weight", 0.05)),
        "boundary_weight": float(getattr(loss_cfg, "boundary_weight", 0.05)),
        "ema_enabled": False,
        "default_aug": safe_default_aug,
        "per_dataset_aug": safe_per_dataset_aug,
        "model_config": model_cfg,
        "loss_config": loss_cfg,
    }
