from io import BytesIO

import numpy as np

from src.data.dataloaders import _load_from_npz


def _build_npz_buffer(image: np.ndarray, mask: np.ndarray) -> BytesIO:
    buffer = BytesIO()
    np.savez(buffer, image=image, mask=mask)
    buffer.seek(0)
    return buffer


def test_load_from_npz_skips_residual_noise_when_disabled():
    image = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, size=(64, 64), dtype=np.uint8)
    sample = _build_npz_buffer(image, mask)

    loaded_image, loaded_mask, residual_noise = _load_from_npz(
        sample,
        include_residual_noise=False,
    )

    assert loaded_image is not None
    assert loaded_mask is not None
    assert residual_noise is None


def test_load_from_npz_computes_residual_noise_when_enabled():
    image = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, size=(64, 64), dtype=np.uint8)
    sample = _build_npz_buffer(image, mask)

    _, _, residual_noise = _load_from_npz(
        sample,
        include_residual_noise=True,
    )

    assert residual_noise is not None
