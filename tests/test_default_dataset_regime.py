from tools.colab_train_helpers import build_default_components
from tools.prepare_datasets import build_default_configs


def test_prepare_default_configs_use_casia2_for_training_and_coverage_columbia_for_test():
    datasets, per_dataset_splits, prep_cfg = build_default_configs()

    assert [dataset.dataset_name for dataset in datasets] == ["CASIA2", "COVERAGE", "Columbia"]
    assert prep_cfg.target_sizes == (384,)

    casia2_split = per_dataset_splits["CASIA2"]
    assert casia2_split.train == 0.8
    assert casia2_split.val == 0.2
    assert casia2_split.test == 0.0

    coverage_split = per_dataset_splits["COVERAGE"]
    assert coverage_split.train == 0.0
    assert coverage_split.val == 0.0
    assert coverage_split.test == 1.0

    columbia_split = per_dataset_splits["Columbia"]
    assert columbia_split.train == 0.0
    assert columbia_split.val == 0.0
    assert columbia_split.test == 1.0


def test_default_components_use_shared_augmentation_defaults():
    model_cfg, _loss_cfg, default_aug, per_dataset_aug = build_default_components()

    assert default_aug.enable is True
    assert default_aug.views_per_sample == 2
    assert default_aug.max_rotation_degrees == 6.0
    assert default_aug.crop_scale_range == (0.75, 1.0)
    assert default_aug.noise_std_range == (0.0, 0.012)
    assert model_cfg.swin.pretrained is False
    assert model_cfg.swin.embed_dim == 64
    assert model_cfg.swin.depths == (2, 2, 4, 2)
    assert model_cfg.swin.num_heads == (2, 4, 8, 16)
    assert model_cfg.residual.base_channels == 24
    assert model_cfg.fusion.fusion_channels == (48, 96, 144, 192)
    assert per_dataset_aug == {}