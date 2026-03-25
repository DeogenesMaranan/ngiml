from tools.colab_train_helpers import apply_colab_runtime_settings


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
