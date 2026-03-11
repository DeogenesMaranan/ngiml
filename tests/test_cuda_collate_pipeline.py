from pathlib import Path

import torch

from src.data.config import AugmentationConfig
from src.data.dataloaders import _collate_impl
from tools.colab_train_helpers import build_default_components, build_training_config


def test_collate_defers_cuda_transforms_but_keeps_view_expansion():
    aug_cfg = AugmentationConfig(
        enable=True,
        views_per_sample=2,
        enable_flips=False,
        enable_rotations=False,
        enable_random_crop=False,
        enable_elastic=False,
        enable_color_jitter=False,
        enable_noise=False,
    )
    sample = {
        "image": torch.full((3, 8, 8), 0.5, dtype=torch.float32),
        "mask": torch.zeros((1, 8, 8), dtype=torch.float32),
        "label": torch.tensor(0, dtype=torch.long),
        "dataset": "CASIA2",
        "high_pass": None,
        "edge_mask": None,
        "has_edge_mask": False,
    }

    batch = _collate_impl(
        {"CASIA2": aug_cfg},
        "imagenet",
        True,
        None,
        True,
        True,
        [sample],
    )

    assert batch["images"].shape[0] == 2
    assert torch.allclose(batch["images"], torch.full_like(batch["images"], 0.5))


def test_training_config_uses_compact_defaults():
    model_cfg, loss_cfg, default_aug, per_dataset_aug = build_default_components()
    config = build_training_config(
        manifest_path=Path("prepared/manifest_local.json"),
        output_dir="runs/test",
        model_cfg=model_cfg,
        loss_cfg=loss_cfg,
        default_aug=default_aug,
        per_dataset_aug=per_dataset_aug,
    )

    assert config["batch_size"] == 12
    assert config["views_per_sample"] == 2
    assert config["max_short_side"] == 384