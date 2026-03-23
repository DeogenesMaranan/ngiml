import torch

from src.model.unet_decoder import UNetDecoder, UNetDecoderConfig


def test_boundary_refinement_is_identity_at_init():
    decoder = UNetDecoder(
        stage_channels=(16, 32, 64, 128),
        config=UNetDecoderConfig(
            out_channels=1,
            use_dropout=False,
            enable_boundary_refinement=True,
            boundary_refine_channels=8,
            boundary_refine_scale=1.0,
        ),
    )

    logits = torch.randn(2, 1, 32, 32)
    refined = decoder._refine_final_logits(logits)

    assert refined.shape == logits.shape
    assert torch.allclose(refined, logits, atol=1e-7)


def test_boundary_refinement_can_be_disabled():
    decoder = UNetDecoder(
        stage_channels=(16, 32, 64, 128),
        config=UNetDecoderConfig(
            out_channels=1,
            use_dropout=False,
            enable_boundary_refinement=False,
        ),
    )

    logits = torch.randn(1, 1, 16, 16)
    refined = decoder._refine_final_logits(logits)

    assert refined.shape == logits.shape
    assert torch.equal(refined, logits)


def test_per_stage_refinement_targets_highest_resolution_head():
    decoder = UNetDecoder(
        stage_channels=(8, 16, 24, 32),
        config=UNetDecoderConfig(
            out_channels=1,
            per_stage_heads=True,
            use_dropout=False,
            enable_boundary_refinement=True,
        ),
    )

    features = [
        torch.randn(1, 8, 64, 64),
        torch.randn(1, 16, 32, 32),
        torch.randn(1, 24, 16, 16),
        torch.randn(1, 32, 8, 8),
    ]

    baseline = decoder(features)

    def _plus_one(logits: torch.Tensor) -> torch.Tensor:
        return logits + 1.0

    decoder._refine_final_logits = _plus_one  # type: ignore[method-assign]
    shifted = decoder(features)

    assert torch.allclose(shifted[0], baseline[0] + 1.0, atol=1e-6)
    for idx in range(1, len(shifted)):
        assert torch.allclose(shifted[idx], baseline[idx], atol=1e-6)
