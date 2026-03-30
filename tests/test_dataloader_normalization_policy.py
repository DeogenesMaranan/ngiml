import torch

from src.data.config import AugmentationConfig
from src.data.dataloaders import _collate_impl, _normalize


def test_imagenet_normalization_applies_to_rgb_only():
    rgb = torch.ones((3, 4, 4), dtype=torch.float32)
    normalized = _normalize(rgb, mode="imagenet")

    expected = torch.tensor(
        [
            (1.0 - 0.485) / 0.229,
            (1.0 - 0.456) / 0.224,
            (1.0 - 0.406) / 0.225,
        ],
        dtype=torch.float32,
    ).view(3, 1, 1)

    assert torch.allclose(normalized, expected.expand_as(normalized), atol=1e-6)


def test_imagenet_normalization_skips_non_rgb_tensors():
    non_rgb = torch.full((1, 4, 4), 0.25, dtype=torch.float32)
    out = _normalize(non_rgb, mode="imagenet")
    assert torch.allclose(out, non_rgb)


def test_collate_keeps_noise_branch_separate_from_rgb_normalization():
    image = torch.full((3, 4, 4), 1.0, dtype=torch.float32)
    noise = torch.full((3, 4, 4), 0.25, dtype=torch.float32)
    mask = torch.zeros((1, 4, 4), dtype=torch.float32)

    batch = [
        {
            "image": image,
            "mask": mask,
            "label": torch.tensor(1, dtype=torch.long),
            "dataset": "CASIA2",
            "residual_noise": noise,
        }
    ]

    out = _collate_impl(
        per_dataset_aug={"CASIA2": AugmentationConfig(enable=False)},
        normalization_mode="imagenet",
        training=False,
        aug_seed=None,
        batch=batch,
    )

    out_image = out["images"][0]
    out_noise = out["residual_noise"][0]

    # RGB tensor should be ImageNet-normalized.
    assert not torch.allclose(out_image, image)
    # Noise branch tensor should remain in its original scale.
    assert torch.allclose(out_noise, noise)


def test_collate_pads_variable_size_masks_after_batch_resize():
    batch = [
        {
            "image": torch.ones((3, 448, 448), dtype=torch.float32),
            "mask": torch.ones((1, 448, 448), dtype=torch.float32),
            "label": torch.tensor(1, dtype=torch.long),
            "dataset": "CASIA2",
            "residual_noise": torch.zeros((3, 448, 448), dtype=torch.float32),
        },
        {
            "image": torch.ones((3, 448, 320), dtype=torch.float32),
            "mask": torch.ones((1, 448, 320), dtype=torch.float32),
            "label": torch.tensor(0, dtype=torch.long),
            "dataset": "CASIA2",
            "residual_noise": torch.zeros((3, 448, 320), dtype=torch.float32),
        },
    ]

    out = _collate_impl(
        per_dataset_aug={"CASIA2": AugmentationConfig(enable=False)},
        normalization_mode="imagenet",
        training=False,
        aug_seed=None,
        batch=batch,
    )

    assert out["images"].shape == (2, 3, 448, 448)
    assert out["masks"].shape == (2, 1, 448, 448)
    assert out["residual_noise"].shape == (2, 3, 448, 448)

