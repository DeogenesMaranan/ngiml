import pytest
import torch

from src.model.hybrid_ngiml import HybridNGIML, HybridNGIMLConfig
from src.model.unet_decoder import UNetDecoderConfig


class _FakeSwinBackbone(torch.nn.Module):
    def __init__(self, _config=None):
        super().__init__()
        self.out_channels = [16, 32, 64, 128]

    def forward(self, x: torch.Tensor):
        n, _, h, w = x.shape
        return [
            torch.randn(n, 16, h, w, device=x.device, dtype=x.dtype),
            torch.randn(n, 32, h // 2, w // 2, device=x.device, dtype=x.dtype),
            torch.randn(n, 64, h // 4, w // 4, device=x.device, dtype=x.dtype),
            torch.randn(n, 128, h // 8, w // 8, device=x.device, dtype=x.dtype),
        ]


def test_swin_only_no_fusion_forward(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("src.model.hybrid_ngiml.SwinBackbone", _FakeSwinBackbone)
    fusion_cfg = HybridNGIMLConfig().fusion

    cfg = HybridNGIMLConfig(
        use_low_level=False,
        use_context=True,
        use_residual=False,
        use_fusion=False,
        fusion=fusion_cfg,
        decoder=UNetDecoderConfig(
            decoder_channels=tuple(fusion_cfg.fusion_channels),
            per_stage_heads=False,
            use_dropout=False,
            enable_edge_guidance=False,
            enable_boundary_refinement=False,
            enable_detail_refinement=False,
        ),
    )
    model = HybridNGIML(cfg)
    x = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        preds = model(x)

    assert isinstance(preds, list)
    assert len(preds) == 1
    assert preds[0].shape == (2, 1, 64, 64)


def test_no_fusion_rejects_multi_branch():
    cfg = HybridNGIMLConfig(
        use_low_level=True,
        use_context=True,
        use_residual=False,
        use_fusion=False,
    )
    with pytest.raises(ValueError, match="requires exactly one enabled branch"):
        _ = HybridNGIML(cfg)
