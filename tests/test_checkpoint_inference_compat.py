import types
from pathlib import Path
import sys

import torch
from torch import nn

from src.model.hybrid_ngiml import HybridNGIML

if "matplotlib" not in sys.modules:
    matplotlib_stub = types.ModuleType("matplotlib")
    pyplot_stub = types.ModuleType("matplotlib.pyplot")
    matplotlib_stub.pyplot = pyplot_stub
    sys.modules["matplotlib"] = matplotlib_stub
    sys.modules["matplotlib.pyplot"] = pyplot_stub

from tools import infer_helpers


def test_build_model_config_from_checkpoint_restores_new_fusion_fields():
    checkpoint = {
        "train_config": {
            "model_config": {
                "swin": {
                    "model_name": "swin_tiny_patch4_window7_224",
                    "pretrained": False,
                    "out_indices": [0, 1, 2, 3],
                    "input_size": 512,
                    "allow_variable_input": True,
                },
                "fusion": {
                    "fusion_channels": [32, 64, 96, 160],
                    "noise_branch": "residual",
                    "noise_skip_stage": 3,
                    "noise_decay": 0.8,
                    "norm": "in",
                    "activation": "silu",
                    "fusion_refinement": False,
                    "late_residual_boost_start": 2,
                    "late_residual_boost": 0.75,
                },
                "decoder": {
                    "enable_edge_guidance": False,
                },
                "use_low_level": True,
                "use_context": True,
                "use_residual": True,
                "enable_residual_attention": True,
            }
        }
    }

    model_cfg, source = infer_helpers._build_model_config_from_checkpoint(checkpoint)

    assert source == "train_config.model_config"
    assert tuple(model_cfg.fusion.fusion_channels) == (32, 64, 96, 160)
    assert model_cfg.fusion.late_residual_boost_start == 2
    assert model_cfg.fusion.late_residual_boost == 0.75
    assert model_cfg.swin.input_size == 512
    assert model_cfg.decoder.enable_edge_guidance is False


def test_load_model_from_checkpoint_uses_cpu_map_location(monkeypatch):
    captured = {"map_location": None}

    class _FakeModel:
        def __init__(self, cfg):
            self.cfg = cfg
            self.loaded = None
            self.eval_called = False
            self.device = None

        def load_state_dict(self, state_dict, strict=False):
            self.loaded = state_dict
            return [], []

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.eval_called = True
            return self

    def _fake_torch_load(path, map_location="cpu"):
        captured["map_location"] = map_location
        return {
            "epoch": 7,
            "model_state": {"any": torch.tensor(1.0)},
            "train_config": {"precision": "fp32"},
        }

    monkeypatch.setattr(infer_helpers.torch, "load", _fake_torch_load)
    monkeypatch.setattr(infer_helpers, "HybridNGIML", _FakeModel)
    monkeypatch.setattr(infer_helpers, "resolve_threshold_for_checkpoint", lambda *args, **kwargs: (0.5, "fallback"))

    model, device, info = infer_helpers.load_model_from_checkpoint(Path("dummy.pt"), device=torch.device("cpu"))

    assert captured["map_location"] == "cpu"
    assert isinstance(model, _FakeModel)
    assert device.type == "cpu"
    assert info["epoch"] == 7


def test_forward_features_passes_target_size_to_fusion():
    model = HybridNGIML.__new__(HybridNGIML)
    nn.Module.__init__(model)
    model.cfg = types.SimpleNamespace(use_low_level=True, use_context=False, use_residual=False)
    model._extract_features = lambda x, residual_noise=None: {
        "low_level": [torch.ones((1, 4, 8, 8), dtype=torch.float32)],
        "context": None,
        "residual": None,
    }

    captured = {"target_size": None}

    class _CaptureFusion:
        def __call__(self, features, target_size=None):
            captured["target_size"] = target_size
            return features["low_level"]

    model.fusion = _CaptureFusion()

    _ = HybridNGIML.forward_features(model, torch.randn((1, 3, 16, 16)), target_size=(32, 32))

    assert captured["target_size"] == (32, 32)
