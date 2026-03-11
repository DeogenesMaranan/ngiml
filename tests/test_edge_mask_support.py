import io
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from src.data.config import Manifest, SampleRecord
from src.model.losses import MultiStageLossConfig, MultiStageManipulationLoss
from tools.prepare_datasets import _build_npz_bytes, _load_rgb_image_for_npz


def test_manifest_roundtrip_preserves_edge_mask_path():
    manifest = Manifest(
        samples=[
            SampleRecord(
                dataset="CASIA2",
                split="train",
                image_path="prepared/CASIA2/train/fake/sample_01.npz",
                mask_path=None,
                label=1,
                edge_mask_path="prepared/CASIA2/train/fake/sample_01_edge.png",
            )
        ],
        normalization_mode="imagenet",
    )

    recovered = Manifest.from_dataframe(manifest.to_dataframe())

    assert recovered.samples[0].edge_mask_path == "prepared/CASIA2/train/fake/sample_01_edge.png"


def test_boundary_loss_uses_explicit_edge_target_when_present():
    loss_fn = MultiStageManipulationLoss(
        MultiStageLossConfig(
            dice_weight=0.0,
            bce_weight=0.0,
            use_boundary_loss=True,
            boundary_weight=1.0,
        )
    )
    preds = [torch.zeros((2, 1, 8, 8), dtype=torch.float32)]
    target = torch.zeros((2, 1, 8, 8), dtype=torch.float32)
    edge_target = torch.zeros((2, 1, 8, 8), dtype=torch.float32)
    edge_target[0, :, 3:5, 3:5] = 1.0
    edge_present = torch.tensor([True, False])

    explicit_loss = loss_fn(preds, target, edge_target=edge_target, edge_target_present=edge_present)
    fallback_loss = loss_fn(preds, target)

    assert explicit_loss.item() > fallback_loss.item()


def test_prepare_dataset_computes_train_edge_mask_from_main_mask(tmp_path):
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[..., 0] = 255
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255

    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)

    npz_bytes = _build_npz_bytes(
        image_path=image_path,
        mask_path=mask_path,
        edge_mask_path=None,
        target_size=8,
        include_high_pass=False,
        compute_edge_mask=True,
    )

    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as data:
        assert "edge_mask" in data
        edge_mask = data["edge_mask"]

    assert edge_mask.shape == (8, 8)
    assert edge_mask.dtype == np.uint8
    assert edge_mask.max() == 255
    assert edge_mask[2, 2] == 255
    assert edge_mask[3, 3] == 0


def test_prepare_dataset_skips_computed_edge_mask_outside_train(tmp_path):
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255

    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)

    npz_bytes = _build_npz_bytes(
        image_path=image_path,
        mask_path=mask_path,
        edge_mask_path=None,
        target_size=8,
        include_high_pass=False,
        compute_edge_mask=False,
    )

    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as data:
        assert "edge_mask" not in data


def test_non_jpeg_sources_are_standardized_through_jpeg_before_npz(tmp_path):
    image_path = tmp_path / "image.png"
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    Image.fromarray(image).save(image_path)

    npz_bytes = _build_npz_bytes(
        image_path=image_path,
        mask_path=None,
        edge_mask_path=None,
        target_size=0,
        include_high_pass=False,
        compute_edge_mask=False,
    )

    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as data:
        stored = data["image"]

    assert stored.shape == image.shape
    assert stored.dtype == np.uint8
    assert not np.array_equal(stored, image)


def test_jpeg_sources_are_not_reencoded_again(tmp_path):
    image_path = tmp_path / "image.jpg"
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    Image.fromarray(image).save(image_path, format="JPEG", quality=95)

    expected = np.asarray(_load_rgb_image_for_npz(Path(image_path)), dtype=np.uint8)
    npz_bytes = _build_npz_bytes(
        image_path=image_path,
        mask_path=None,
        edge_mask_path=None,
        target_size=0,
        include_high_pass=False,
        compute_edge_mask=False,
    )

    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as data:
        stored = data["image"]

    assert np.array_equal(stored, expected)