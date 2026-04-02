import torch

import src.infer_helpers as infer_helpers
from src.model.hybrid_ngiml import HybridNGIML
from src.training_defaults import build_default_components


def test_get_model_complexity_stats_prefers_thop():
    model_cfg, _, _, _ = build_default_components()
    model = HybridNGIML(model_cfg).cpu().eval()

    stats = infer_helpers.get_model_complexity_stats(model, input_size=(1, 3, 448, 448))

    assert stats["flops_source"] == "thop"
    assert float(stats["macs"]) > 3.0e10
    assert float(stats["flops"]) > 6.0e10

