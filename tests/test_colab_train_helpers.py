from src.colab_train_helpers import apply_colab_runtime_settings
from src.training_defaults import build_default_components, build_training_config
from src.model.hybrid_ngiml import HybridNGIMLConfig
from src.model.feature_fusion import FeatureFusionConfig


def test_apply_colab_runtime_settings_preserves_explicit_compile_choice():
    cfg = {
        "compile_model": False,
        "compile_mode": "default",
        "channels_last": False,
        "balance_sampling": False,
    }

    updated = apply_colab_runtime_settings(
        cfg,
        balance_sampling=False,
        local_cache_dir="/content/cache",
    )

    assert updated["compile_model"] is False
    assert updated["channels_last"] is False
    assert updated["local_cache_dir"] == "/content/cache"


def test_apply_colab_runtime_settings_caps_excessive_num_workers():
    cfg = {
        "num_workers": 6,
    }
    import src.colab_train_helpers as helpers

    original_cpu_count = helpers.os.cpu_count
    try:
        helpers.os.cpu_count = lambda: 2
        updated = apply_colab_runtime_settings(
            cfg,
            balance_sampling=False,
            local_cache_dir="/content/cache",
        )
    finally:
        helpers.os.cpu_count = original_cpu_count

    assert updated["num_workers"] == 2


def test_apply_colab_runtime_settings_sets_spawn_context_on_py312_plus():
    cfg = {}
    updated = apply_colab_runtime_settings(
        cfg,
        balance_sampling=False,
        local_cache_dir="/content/cache",
    )

    import sys

    if sys.version_info >= (3, 12):
        assert updated["multiprocessing_context"] == "spawn"
    else:
        assert updated["multiprocessing_context"] is None


def test_build_training_config_uses_fixed_threshold_iou_early_stopping_defaults():
    model_cfg, loss_cfg, default_aug, per_dataset_aug = build_default_components()
    cfg = build_training_config(
        manifest_path="dummy.parquet",
        output_dir="runs/test",
        model_cfg=model_cfg,
        loss_cfg=loss_cfg,
        default_aug=default_aug,
        per_dataset_aug=per_dataset_aug,
    )

    assert cfg["early_stopping_patience"] == 5
    assert cfg["early_stopping_min_delta"] == 3e-3
    assert cfg["early_stopping_monitor"] == "loss"
    assert cfg["metric_threshold"] == 0.5
    assert cfg["optimize_threshold"] is False


def test_further_lite_defaults_enable_low_level_only_residual_attention_and_disable_joint_gating():
    model_cfg = HybridNGIMLConfig()
    fusion_cfg = FeatureFusionConfig(fusion_channels=(64, 128, 192, 256))

    assert model_cfg.enable_residual_attention is True
    assert model_cfg.enable_low_level_residual_attention is True
    assert model_cfg.enable_context_residual_attention is False
    assert model_cfg.residual_attention_init_scale == 0.0
    assert fusion_cfg.enable_joint_gating is False

