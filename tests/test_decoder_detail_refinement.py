import torch

from src.model.unet_decoder import UNetDecoder, UNetDecoderConfig


def test_detail_refinement_starts_as_identity():
    decoder = UNetDecoder(
        stage_channels=(4, 8),
        config=UNetDecoderConfig(
            decoder_channels=(4, 8),
            per_stage_heads=True,
            enable_edge_guidance=False,
            enable_boundary_refinement=False,
            enable_detail_refinement=True,
            use_dropout=False,
        ),
    )
    decoder.eval()

    features = [
        torch.randn((1, 4, 16, 16), dtype=torch.float32),
        torch.randn((1, 8, 8, 8), dtype=torch.float32),
    ]

    with torch.no_grad():
        refined = decoder(features)
        decoder.enable_detail_refinement = False
        baseline = decoder(features)

    assert torch.allclose(refined[0], baseline[0], atol=1e-6)


def test_detail_refinement_can_use_coarse_logits():
    decoder = UNetDecoder(
        stage_channels=(1, 1),
        config=UNetDecoderConfig(
            decoder_channels=(1, 1),
            norm="bn",
            activation="relu",
            per_stage_heads=True,
            enable_edge_guidance=False,
            enable_boundary_refinement=False,
            enable_detail_refinement=True,
            detail_refine_channels=1,
            use_dropout=False,
        ),
    )
    decoder.eval()

    # Make the detail-refinement head respond only to the coarse-logit input.
    first = decoder.detail_refine_head[0]
    bn = decoder.detail_refine_head[1]
    second = decoder.detail_refine_head[3]
    first.weight.data.zero_()
    first.weight.data[0, 2, 1, 1] = 1.0
    if isinstance(bn, torch.nn.BatchNorm2d):
        bn.weight.data.fill_(1.0)
        bn.bias.data.zero_()
        bn.running_mean.zero_()
        bn.running_var.fill_(1.0)
    second.weight.data.zero_()
    second.weight.data[0, 0, 0, 0] = 1.0
    if second.bias is not None:
        second.bias.data.zero_()

    # Force deterministic logits from the coarse stage only.
    for predictor in decoder.predictors:
        predictor.weight.data.zero_()
        if predictor.bias is not None:
            predictor.bias.data.zero_()
    decoder.predictors[1].bias.data.fill_(2.0)

    features = [
        torch.zeros((1, 1, 8, 8), dtype=torch.float32),
        torch.zeros((1, 1, 4, 4), dtype=torch.float32),
    ]

    with torch.no_grad():
        refined = decoder(features)
        decoder.enable_detail_refinement = False
        baseline = decoder(features)

    assert refined[0].mean() > baseline[0].mean()
