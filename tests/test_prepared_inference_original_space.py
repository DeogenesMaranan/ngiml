import numpy as np

from src.infer_helpers import _original_space_ground_truth, _original_space_probability


def test_original_space_restore_for_padded_sample():
    processed_mask = np.zeros((6, 6), dtype=np.uint8)
    processed_mask[1:5, 2:4] = 1
    processed_prob = processed_mask.astype(np.float32)
    meta = {
        "preproc_mode": "pad",
        "original_size_hw": [4, 2],
        "original_mask_path": None,
    }

    restored_gt = _original_space_ground_truth(processed_mask, meta)
    restored_prob = _original_space_probability(processed_prob, meta)

    assert restored_gt.shape == (4, 2)
    assert restored_prob.shape == (4, 2)
    assert np.array_equal(restored_gt, np.ones((4, 2), dtype=np.uint8))
    assert np.allclose(restored_prob, np.ones((4, 2), dtype=np.float32))


def test_original_space_restore_for_resize_then_pad_sample():
    processed_mask = np.zeros((6, 6), dtype=np.uint8)
    processed_mask[:, 2:4] = 1
    processed_prob = processed_mask.astype(np.float32)
    meta = {
        "preproc_mode": "resize_then_pad",
        "original_size_hw": [6, 2],
        "original_mask_path": None,
    }

    restored_gt = _original_space_ground_truth(processed_mask, meta)
    restored_prob = _original_space_probability(processed_prob, meta)

    assert restored_gt.shape == (6, 2)
    assert restored_prob.shape == (6, 2)
    assert np.array_equal(restored_gt, np.ones((6, 2), dtype=np.uint8))
    assert np.allclose(restored_prob, np.ones((6, 2), dtype=np.float32))

