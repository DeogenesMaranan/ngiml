from tools.colab_train_helpers import apply_colab_runtime_settings
from tools.train_ngiml import build_default_components, build_training_config
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
    assert cfg["early_stopping_monitor"] == "iou"
    assert cfg["metric_threshold"] == 0.5
    assert cfg["optimize_threshold"] is False


def test_further_lite_defaults_disable_residual_attention_and_joint_gating():
    model_cfg = HybridNGIMLConfig()
    fusion_cfg = FeatureFusionConfig(fusion_channels=(64, 128, 192, 256))

    assert model_cfg.enable_residual_attention is False
    assert fusion_cfg.enable_joint_gating is False
