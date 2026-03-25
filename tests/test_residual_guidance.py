import torch
from torch import nn

from src.model.feature_fusion import FeatureFusionConfig, MultiStageFeatureFusion
from src.model.hybrid_ngiml import HybridNGIML


class _StaticBackbone(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs

    def forward(self, x, **kwargs):
        return [tensor.clone() for tensor in self.outputs]


def _make_identity_refine(stage: nn.Module) -> None:
    conv = stage.refine[0]
    bn = stage.refine[1]
    conv.weight.data.zero_()
    out_channels, in_channels, _, _ = conv.weight.shape
    for idx in range(min(out_channels, in_channels)):
        conv.weight.data[idx, idx, 1, 1] = 1.0

    if isinstance(bn, nn.BatchNorm2d):
        bn.weight.data.fill_(1.0)
        bn.bias.data.zero_()
        bn.running_mean.zero_()
        bn.running_var.fill_(1.0)


def test_residual_attention_modulates_all_available_low_level_stages():
    model = HybridNGIML.__new__(HybridNGIML)
    nn.Module.__init__(model)

    low_level = [
        torch.ones((1, 2, 16, 16), dtype=torch.float32),
        torch.ones((1, 4, 8, 8), dtype=torch.float32),
        torch.ones((1, 8, 4, 4), dtype=torch.float32),
    ]
    residual = [
        torch.randn((1, 3, 16, 16), dtype=torch.float32),
        torch.randn((1, 5, 8, 8), dtype=torch.float32),
        torch.randn((1, 7, 4, 4), dtype=torch.float32),
    ]

    model.efficientnet = _StaticBackbone(low_level)
    model.swin = None
    model.noise = _StaticBackbone(residual)
    model.enable_residual_attention = True
    model.residual_attention_proj = nn.ModuleList(
        [
            nn.Conv2d(3, 2, kernel_size=1),
            nn.Conv2d(5, 4, kernel_size=1),
            nn.Conv2d(7, 8, kernel_size=1),
        ]
    )
    for proj in model.residual_attention_proj:
        proj.weight.data.zero_()
        proj.bias.data.zero_()

    features = model._extract_features(torch.zeros((1, 3, 16, 16), dtype=torch.float32))

    assert isinstance(features["low_level"], list)
    for stage in features["low_level"]:
        assert torch.allclose(stage, torch.full_like(stage, 1.5))


def test_late_residual_boost_only_changes_later_stages():
    branch_channels = {
        "low_level": (1, 1),
        "context": (1, 1),
        "residual": (1, 1),
    }
    features = {
        "low_level": [
            torch.zeros((1, 1, 8, 8), dtype=torch.float32),
            torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        ],
        "context": [
            torch.zeros((1, 1, 8, 8), dtype=torch.float32),
            torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        ],
        "residual": [
            torch.full((1, 1, 8, 8), 2.0, dtype=torch.float32),
            torch.full((1, 1, 4, 4), 2.0, dtype=torch.float32),
        ],
    }

    torch.manual_seed(7)
    no_boost = MultiStageFeatureFusion(
        branch_channels,
        FeatureFusionConfig(
            fusion_channels=(1, 1),
            fusion_refinement=False,
            late_residual_boost=0.0,
            late_residual_boost_start=1,
        ),
    )
    torch.manual_seed(7)
    boosted = MultiStageFeatureFusion(
        branch_channels,
        FeatureFusionConfig(
            fusion_channels=(1, 1),
            fusion_refinement=False,
            late_residual_boost=0.5,
            late_residual_boost_start=1,
        ),
    )

    for fusion in (no_boost, boosted):
        fusion.eval()
        for stage in fusion.stages:
            _make_identity_refine(stage)
            for projection in stage.projections.values():
                projection.weight.data.fill_(1.0)
            for gate in stage.gate_params.values():
                gate.data.zero_()

    base_out = no_boost(features)
    boosted_out = boosted(features)

    assert torch.allclose(base_out[0], boosted_out[0], atol=1e-6)
    assert boosted_out[1].mean() > base_out[1].mean()
