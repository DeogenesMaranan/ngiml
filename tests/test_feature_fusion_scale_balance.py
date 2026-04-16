import torch

from src.model.feature_fusion import FeatureFusionConfig, MultiStageFeatureFusion


def _make_equal_gate_fusion(balance_branch_scales: bool) -> MultiStageFeatureFusion:
    fusion = MultiStageFeatureFusion(
        {"a": (1,), "b": (1,)},
        FeatureFusionConfig(
            fusion_channels=(1,),
            fusion_refinement=False,
            enable_joint_gating=False,
            balance_branch_scales=balance_branch_scales,
        ),
    )
    stage = fusion.stages[0]

    for projection in stage.projections.values():
        projection.weight.data.fill_(1.0)
    for gate in stage.gate_generators.values():
        for layer in gate:
            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.data.zero_()
                if layer.bias is not None:
                    layer.bias.data.zero_()
    for gate_bias in stage.gate_bias.values():
        gate_bias.data.zero_()
    for layer in stage.refine:
        if isinstance(layer, torch.nn.Conv2d):
            layer.weight.data.zero_()
            layer.weight.data[0, 0, 1, 1] = 1.0
    return fusion.eval()


def test_scale_balancing_reduces_branch_magnitude_domination():
    unbalanced = _make_equal_gate_fusion(balance_branch_scales=False)
    balanced = _make_equal_gate_fusion(balance_branch_scales=True)

    features = {
        "a": [torch.ones((1, 1, 8, 8), dtype=torch.float32)],
        "b": [torch.full((1, 1, 8, 8), 100.0, dtype=torch.float32)],
    }

    with torch.no_grad():
        out_unbalanced = unbalanced(features)[0]
        out_balanced = balanced(features)[0]

    mean_unbalanced = float(out_unbalanced.mean().item())
    mean_balanced = float(out_balanced.mean().item())

    assert mean_unbalanced > 10.0
    assert mean_balanced < mean_unbalanced / 5.0
