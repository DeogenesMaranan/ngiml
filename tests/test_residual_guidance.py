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


class _CaptureDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_image = None
        self.last_features = None

    def forward(self, features, image=None, postprocess=None):
        self.last_features = features
        self.last_image = image
        return [torch.zeros((1, 1, 8, 8), dtype=torch.float32)]


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


def test_residual_attention_modulates_all_available_low_level_and_context_stages():
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
    context = [
        torch.ones((1, 6, 16, 16), dtype=torch.float32),
        torch.ones((1, 10, 8, 8), dtype=torch.float32),
        torch.ones((1, 12, 4, 4), dtype=torch.float32),
    ]

    model.efficientnet = _StaticBackbone(low_level)
    model.swin = _StaticBackbone(context)
    model.noise = _StaticBackbone(residual)
    model.enable_residual_attention = True
    model.low_level_residual_attention_proj = nn.ModuleList(
        [
            nn.Conv2d(3, 2, kernel_size=1),
            nn.Conv2d(5, 4, kernel_size=1),
            nn.Conv2d(7, 8, kernel_size=1),
        ]
    )
    model.low_level_residual_attention_scale = nn.ParameterList(
        [nn.Parameter(torch.ones(1)) for _ in range(3)]
    )
    model.context_residual_attention_proj = nn.ModuleList(
        [
            nn.Conv2d(3, 6, kernel_size=1),
            nn.Conv2d(5, 10, kernel_size=1),
            nn.Conv2d(7, 12, kernel_size=1),
        ]
    )
    model.context_residual_attention_scale = nn.ParameterList(
        [nn.Parameter(torch.ones(1)) for _ in range(3)]
    )
    for proj in list(model.low_level_residual_attention_proj) + list(model.context_residual_attention_proj):
        proj.weight.data.zero_()
        proj.bias.data.zero_()

    features = model._extract_features(torch.zeros((1, 3, 16, 16), dtype=torch.float32))

    assert isinstance(features["low_level"], list)
    for stage in features["low_level"]:
        assert torch.allclose(stage, torch.full_like(stage, 1.0))
    assert isinstance(features["context"], list)
    for stage in features["context"]:
        assert torch.allclose(stage, torch.full_like(stage, 1.0))


def test_residual_attention_zero_init_scale_starts_as_identity():
    target = [torch.ones((1, 2, 8, 8), dtype=torch.float32)]
    residual = [torch.randn((1, 3, 8, 8), dtype=torch.float32)]
    projections = nn.ModuleList([nn.Conv2d(3, 2, kernel_size=1)])
    scales = nn.ParameterList([nn.Parameter(torch.zeros(1))])

    HybridNGIML._apply_residual_attention(target, residual, projections, scales)

    assert torch.allclose(target[0], torch.ones_like(target[0]))


def test_residual_attention_default_init_with_zero_scale_keeps_gradients_alive():
    torch.manual_seed(0)
    projections = HybridNGIML._build_residual_attention_proj([3], [2])
    scales = nn.ParameterList([nn.Parameter(torch.zeros(1))])
    target = [torch.randn((2, 2, 8, 8), dtype=torch.float32, requires_grad=True)]
    residual = [torch.randn((2, 3, 8, 8), dtype=torch.float32)]

    HybridNGIML._apply_residual_attention(target, residual, projections, scales)
    loss = target[0].pow(2).mean()
    loss.backward()

    proj_grad = projections[0].weight.grad
    scale_grad = scales[0].grad
    assert proj_grad is not None
    assert scale_grad is not None
    # With zero scale, projection is gated but scale should receive signal.
    assert float(proj_grad.abs().sum().item()) == 0.0
    assert float(scale_grad.abs().sum().item()) > 0.0


def test_hybrid_forward_passes_image_to_decoder():
    model = HybridNGIML.__new__(HybridNGIML)
    nn.Module.__init__(model)
    model.forward_features = lambda x, target_size=None, residual_noise=None: [torch.ones((1, 4, 8, 8))]
    model.decoder = _CaptureDecoder()

    image = torch.randn((1, 3, 32, 32), dtype=torch.float32)
    _ = HybridNGIML.forward(model, image)

    assert model.decoder.last_image is image
    assert len(model.decoder.last_features) == 1


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
            for gate in stage.gate_generators.values():
                for layer in gate:
                    if isinstance(layer, nn.Conv2d):
                        layer.weight.data.zero_()
                        if layer.bias is not None:
                            layer.bias.data.zero_()
            for gate_bias in stage.gate_bias.values():
                gate_bias.data.zero_()

    base_out = no_boost(features)
    boosted_out = boosted(features)

    assert torch.allclose(base_out[0], boosted_out[0], atol=1e-6)
    assert boosted_out[1].mean() > base_out[1].mean()


def test_feature_conditioned_gate_can_vary_spatially():
    fusion = MultiStageFeatureFusion(
        {"low_level": (1,), "residual": (1,)},
        FeatureFusionConfig(
            fusion_channels=(1,),
            fusion_refinement=False,
        ),
    )
    stage = fusion.stages[0]
    gate = stage.gate_generators["low_level"]
    first = gate[0]
    second = gate[2]
    first.weight.data.zero_()
    first.bias.data.zero_()
    first.weight.data[0, 0, 0, 0] = 1.0
    second.weight.data.zero_()
    second.bias.data.zero_()
    second.weight.data[0, 0, 0, 0] = 4.0

    proj = torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]], dtype=torch.float32)
    raw_gate = stage.gate_generators["low_level"](proj) + stage.gate_bias["low_level"]

    assert raw_gate.shape == proj.shape
    assert raw_gate[0, 0, 0, 1] > raw_gate[0, 0, 0, 0]


def test_joint_gate_responds_to_other_branch_features():
    fusion = MultiStageFeatureFusion(
        {"low_level": (1,), "context": (1,)},
        FeatureFusionConfig(
            fusion_channels=(1,),
            fusion_refinement=False,
            enable_joint_gating=True,
        ),
    )
    stage = fusion.stages[0]

    for projection in stage.projections.values():
        projection.weight.data.fill_(1.0)
    for gate in stage.gate_generators.values():
        for layer in gate:
            if isinstance(layer, nn.Conv2d):
                layer.weight.data.zero_()
                if layer.bias is not None:
                    layer.bias.data.zero_()
    for gate_bias in stage.gate_bias.values():
        gate_bias.data.zero_()

    first = stage.joint_gate_generator[0]
    second = stage.joint_gate_generator[2]
    first.weight.data.zero_()
    first.bias.data.zero_()
    second.weight.data.zero_()
    second.bias.data.zero_()

    # Make the low-level branch gate depend only on the context branch signal.
    first.weight.data[0, 1, 0, 0] = 1.0
    second.weight.data[0, 0, 0, 0] = 2.0

    low_level = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    context_weak = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    context_strong = torch.ones((1, 1, 2, 2), dtype=torch.float32)

    weak_joint = stage.joint_gate_generator(torch.cat([low_level, context_weak], dim=1))
    strong_joint = stage.joint_gate_generator(torch.cat([low_level, context_strong], dim=1))

    weak_low_level_gate = weak_joint.chunk(2, dim=1)[0]
    strong_low_level_gate = strong_joint.chunk(2, dim=1)[0]

    assert strong_low_level_gate.mean() > weak_low_level_gate.mean()
