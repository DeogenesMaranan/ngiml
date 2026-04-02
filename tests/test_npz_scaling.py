import numpy as np
import torch

from src.data.config import SampleRecord
from src.data.dataloaders import _compute_residual_noise, _load_from_npz
from src.infer_helpers import load_image_mask_from_record


def test_npz_uint8_scaled_to_unit_float32(tmp_path):
    npz_path = tmp_path / "sample_uint8.npz"
    image = np.full((8, 8, 3), 255, dtype=np.uint8)
    mask = np.ones((8, 8), dtype=np.uint8)
    residual_noise = np.full((8, 8, 3), 128, dtype=np.uint8)
    np.savez(npz_path, image=image, mask=mask, residual_noise=residual_noise)

    out_image, out_mask, out_residual_noise = _load_from_npz(str(npz_path))

    assert out_image.dtype == torch.float32
    assert out_residual_noise is not None
    assert out_residual_noise.dtype == torch.float32
    assert out_mask is not None
    assert out_mask.dtype == torch.float32
    assert torch.isclose(out_image.max(), torch.tensor(1.0), atol=1e-6)
    expected_residual = _compute_residual_noise(out_image)
    assert torch.allclose(out_residual_noise, expected_residual, atol=1e-6)


def test_npz_float_unit_range_not_double_scaled(tmp_path):
    npz_path = tmp_path / "sample_float.npz"
    image = np.full((8, 8, 3), 0.5, dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.float32)
    residual_noise = np.full((8, 8, 3), 0.4, dtype=np.float32)
    np.savez(npz_path, image=image, mask=mask, residual_noise=residual_noise)

    out_image, out_mask, out_residual_noise = _load_from_npz(str(npz_path))

    assert out_image.dtype == torch.float32
    assert out_residual_noise is not None
    assert out_residual_noise.dtype == torch.float32
    assert out_mask is not None
    assert out_mask.dtype == torch.float32
    assert torch.isclose(out_image.mean(), torch.tensor(0.5), atol=1e-6)
    expected_residual = _compute_residual_noise(out_image)
    assert torch.allclose(out_residual_noise, expected_residual, atol=1e-6)


def test_infer_helper_handles_npz_records(tmp_path):
    npz_path = tmp_path / "sample_record.npz"
    image = np.full((8, 8, 3), 255, dtype=np.uint8)
    mask = np.ones((8, 8), dtype=np.uint8)
    residual_noise = np.full((8, 8, 3), 64, dtype=np.uint8)
    np.savez(npz_path, image=image, mask=mask, residual_noise=residual_noise)

    record = SampleRecord(
        dataset="CASIA2",
        split="test",
        image_path=str(npz_path),
        mask_path=None,
        label=1,
    )

    out_image, out_mask, out_residual_noise = load_image_mask_from_record(record)

    assert out_image.shape == (3, 8, 8)
    assert out_mask.shape == (1, 8, 8)
    assert out_residual_noise is not None
    assert out_residual_noise.shape == (3, 8, 8)
    assert torch.isclose(out_mask.max(), torch.tensor(1.0), atol=1e-6)


def test_npz_residual_noise_key_is_loaded(tmp_path):
    npz_path = tmp_path / "sample_residual.npz"
    image = np.full((8, 8, 3), 200, dtype=np.uint8)
    mask = np.ones((8, 8), dtype=np.uint8)
    residual_noise = np.full((8, 8, 3), -0.1, dtype=np.float32)
    np.savez(npz_path, image=image, mask=mask, residual_noise=residual_noise)

    out_image, out_mask, out_residual_noise = _load_from_npz(str(npz_path))

    assert out_image.dtype == torch.float32
    assert out_mask is not None
    assert out_residual_noise is not None
    assert out_residual_noise.dtype == torch.float32
    expected_residual = _compute_residual_noise(out_image)
    assert torch.allclose(out_residual_noise, expected_residual, atol=1e-6)


def test_npz_mask_is_resized_to_match_image_shape(tmp_path):
    npz_path = tmp_path / "sample_mismatch.npz"
    image = np.full((448, 448, 3), 255, dtype=np.uint8)
    mask = np.ones((448, 320), dtype=np.uint8)
    np.savez(npz_path, image=image, mask=mask)

    out_image, out_mask, _out_residual_noise = _load_from_npz(str(npz_path))

    assert out_image.shape == (3, 448, 448)
    assert out_mask is not None
    assert out_mask.shape == (1, 448, 448)


