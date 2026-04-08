import pytest
import torch

from src.model.unet_decoder import UNetDecoder, UNetDecoderConfig


@pytest.mark.parametrize("block_type", ["conv", "mbconv"])
def test_decoder_supports_configurable_block_type(block_type: str):
    decoder = UNetDecoder(
        stage_channels=(8, 16, 24, 32),
        config=UNetDecoderConfig(
            decoder_channels=(8, 16, 24, 32),
            decoder_block_type=block_type,
            activation="relu",
            use_dropout=False,
            enable_edge_guidance=False,
            enable_boundary_refinement=False,
            enable_detail_refinement=False,
            per_stage_heads=False,
        ),
    )
    decoder.eval()

    features = [
        torch.randn(1, 8, 64, 64),
        torch.randn(1, 16, 32, 32),
        torch.randn(1, 24, 16, 16),
        torch.randn(1, 32, 8, 8),
    ]

    with torch.no_grad():
        out = decoder(features)

    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0].shape == (1, 1, 64, 64)


def test_decoder_rejects_invalid_block_type():
    with pytest.raises(ValueError, match="Unsupported decoder block type"):
        UNetDecoder(
            stage_channels=(8, 16),
            config=UNetDecoderConfig(
                decoder_channels=(8, 16),
                decoder_block_type="invalid_block",
                use_dropout=False,
                enable_edge_guidance=False,
                enable_boundary_refinement=False,
                enable_detail_refinement=False,
            ),
        )
