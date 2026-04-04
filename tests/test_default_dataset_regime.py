from src.training_defaults import build_default_components
from tools.prepare_datasets import build_default_configs


def test_prepare_default_configs_use_train_val_only_datasets_by_default():
    datasets, per_dataset_splits, prep_cfg = build_default_configs()

    assert [dataset.dataset_name for dataset in datasets] == [
        "CASIA2",
        "TampCOCO",
        "NIST",
        "IMD2020",
    ]
    assert prep_cfg.target_sizes == (448,)
    assert prep_cfg.resize_max_side == 896

    casia2_split = per_dataset_splits["CASIA2"]
    assert casia2_split.train == 0.8
    assert casia2_split.val == 0.2
    assert casia2_split.test == 0.0

    tampcoco_split = per_dataset_splits["TampCOCO"]
    assert tampcoco_split.train == 0.8
    assert tampcoco_split.val == 0.2
    assert tampcoco_split.test == 0.0

    nist_split = per_dataset_splits["NIST"]
    assert nist_split.train == 0.8
    assert nist_split.val == 0.2
    assert nist_split.test == 0.0

    imd_split = per_dataset_splits["IMD2020"]
    assert imd_split.train == 0.8
    assert imd_split.val == 0.2
    assert imd_split.test == 0.0


def test_default_components_use_shared_augmentation_defaults():
    _model_cfg, _loss_cfg, default_aug, per_dataset_aug = build_default_components()

    assert default_aug.enable is True
    assert default_aug.views_per_sample == 3
    assert default_aug.max_rotation_degrees == 6.0
    assert default_aug.crop_scale_range == (0.75, 1.0)
    assert default_aug.noise_std_range == (0.0, 0.012)
    assert per_dataset_aug == {}
