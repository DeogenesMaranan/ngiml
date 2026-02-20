import pytest

from tools.local_train_helpers import _infer_label_from_path


def test_path_based_label_inference_fake_and_real():
    assert _infer_label_from_path("prepared/CASIA2/train/fake/sample_01.npz") == 1
    assert _infer_label_from_path("prepared/CASIA2/train/real/sample_01.npz") == 0


def test_path_based_label_inference_raises_on_ambiguous_or_missing():
    with pytest.raises(ValueError):
        _infer_label_from_path("prepared/CASIA2/train/fake/real/sample_01.npz")
    with pytest.raises(ValueError):
        _infer_label_from_path("prepared/CASIA2/train/unknown/sample_01.npz")
