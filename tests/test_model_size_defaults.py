from src.model.backbones.efficientnet_backbone import EfficientNetBackboneConfig
from src.model.backbones.residual_noise_branch import ResidualNoiseConfig
from src.model.backbones.swin_backbone import SwinBackboneConfig
from src.model.feature_fusion import FeatureFusionConfig
from src.model.unet_decoder import UNetDecoderConfig
from src.model.hybrid_ngiml import (
    DEFAULT_DECODER_CHANNELS,
    DEFAULT_FUSION_CHANNELS,
    DEFAULT_SWIN_DEPTHS,
    DEFAULT_SWIN_EMBED_DIM,
    DEFAULT_SWIN_NUM_HEADS,
    HybridNGIML,
    HybridNGIMLConfig,
    LEGACY_FUSION_CHANNELS,
)


def _count_params(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _offline_backbones() -> tuple[EfficientNetBackboneConfig, SwinBackboneConfig]:
    return EfficientNetBackboneConfig(pretrained=False), SwinBackboneConfig(pretrained=False)


def _legacy_swin_backbone() -> SwinBackboneConfig:
    return SwinBackboneConfig(
        pretrained=False,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
    )


def test_default_model_is_smaller_than_legacy_head_budget():
    efficientnet_cfg, swin_cfg = _offline_backbones()

    current = HybridNGIML(
        HybridNGIMLConfig(
            efficientnet=efficientnet_cfg,
            swin=swin_cfg,
        )
    )
    legacy = HybridNGIML(
        HybridNGIMLConfig(
            efficientnet=efficientnet_cfg,
            swin=_legacy_swin_backbone(),
            residual=ResidualNoiseConfig(base_channels=32),
            fusion=FeatureFusionConfig(fusion_channels=LEGACY_FUSION_CHANNELS),
            decoder=UNetDecoderConfig(decoder_channels=LEGACY_FUSION_CHANNELS),
        )
    )

    assert DEFAULT_FUSION_CHANNELS == (40, 80, 128, 160)
    assert DEFAULT_DECODER_CHANNELS == (20, 40, 64, 96)
    assert DEFAULT_SWIN_EMBED_DIM == 64
    assert DEFAULT_SWIN_DEPTHS == (2, 2, 4, 2)
    assert DEFAULT_SWIN_NUM_HEADS == (2, 4, 8, 16)
    assert _count_params(current) < _count_params(legacy)
    assert _count_params(current) <= int(_count_params(legacy) * 0.5)
    assert _count_params(current) <= 20_000_000


def test_custom_swin_geometry_changes_context_channels():
    backbone = HybridNGIML(
        HybridNGIMLConfig(
            efficientnet=EfficientNetBackboneConfig(pretrained=False),
            swin=SwinBackboneConfig(
                pretrained=False,
                embed_dim=48,
                depths=(2, 2, 3, 2),
                num_heads=(2, 4, 8, 16),
            ),
        )
    )

    assert backbone.swin.out_channels == [48, 96, 192, 384]


def test_default_decoder_is_narrower_than_fusion_channels():
    model = HybridNGIML(
        HybridNGIMLConfig(
            efficientnet=EfficientNetBackboneConfig(pretrained=False),
            swin=SwinBackboneConfig(pretrained=False),
        )
    )

    assert tuple(model.decoder.decoder_channels) == DEFAULT_DECODER_CHANNELS
    assert tuple(model.cfg.fusion.fusion_channels) == DEFAULT_FUSION_CHANNELS
    assert model.decoder.cfg.depthwise_separable is True


def test_default_model_hits_sub_20m_budget_offline():
    model = HybridNGIML(
        HybridNGIMLConfig(
            efficientnet=EfficientNetBackboneConfig(pretrained=False),
            swin=SwinBackboneConfig(pretrained=False, adapt_pretrained=False),
        )
    )

    assert _count_params(model) <= 20_000_000