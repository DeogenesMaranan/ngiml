import os
import shutil
from pathlib import Path

from src.data.config import AugmentationConfig
from src.model.hybrid_ngiml import HybridNGIMLConfig
from src.model.losses import MultiStageLossConfig
from tools.train_ngiml import (
    build_default_components as _build_default_components_top_level,
    build_training_config as _build_training_config_top_level,
)


def _cfg_update(config, values: dict, *, overwrite: bool = True) -> None:
    if isinstance(config, dict):
        if overwrite:
            config.update(values)
        else:
            for key, value in values.items():
                config.setdefault(key, value)
        return
    for key, value in values.items():
        if overwrite or not hasattr(config, key):
            setattr(config, key, value)


def _cfg_as_dict(config) -> dict:
    if isinstance(config, dict):
        return config
    return dict(vars(config))


def apply_colab_runtime_settings(
    training_config,
    balance_sampling: bool = False,
    local_cache_dir: str | None = None,
) -> dict:
    recommended_workers = max(2, min(6, (os.cpu_count() or 4)))
    cache_dir = local_cache_dir or "/content/cache"

    updates = {
        "num_workers": recommended_workers,
        "persistent_workers": False,
        "pin_memory": True,
        "auto_local_cache": True,
        "local_cache_dir": cache_dir,
        "reuse_local_cache_manifest": True,
        "compile_model": True,
        "compile_mode": "default",
        "channels_last": True,
        "use_tf32": True,
        "balance_sampling": bool(balance_sampling),
    }

    _cfg_update(training_config, updates, overwrite=False)
    return _cfg_as_dict(training_config)


def stage_persistent_cache_to_runtime(
    persistent_cache_dir: str | Path,
    runtime_cache_dir: str | Path = "/content/cache",
    force: bool = False,
) -> dict[str, object]:
    persistent = Path(persistent_cache_dir)
    runtime = Path(runtime_cache_dir)
    runtime.mkdir(parents=True, exist_ok=True)

    if not persistent.exists():
        return {
            "staged": False,
            "reason": f"Persistent cache not found: {persistent}",
            "persistent_cache_dir": str(persistent),
            "runtime_cache_dir": str(runtime),
        }

    runtime_has_content = any(runtime.iterdir())
    if runtime_has_content and not force:
        return {
            "staged": False,
            "reason": "Runtime cache already populated; skipping copy",
            "persistent_cache_dir": str(persistent),
            "runtime_cache_dir": str(runtime),
        }

    copied_entries = 0
    for src in persistent.iterdir():
        dst = runtime / src.name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            copied_entries += 1
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied_entries += 1

    return {
        "staged": True,
        "copied_entries": copied_entries,
        "persistent_cache_dir": str(persistent),
        "runtime_cache_dir": str(runtime),
    }


def build_default_components():
    return _build_default_components_top_level()


def build_training_config(
    manifest_path: Path,
    output_dir: str,
    model_cfg: HybridNGIMLConfig,
    loss_cfg: MultiStageLossConfig,
    default_aug: AugmentationConfig,
    per_dataset_aug: dict[str, AugmentationConfig],
) -> dict:
    return _build_training_config_top_level(
        manifest_path=manifest_path,
        output_dir=output_dir,
        model_cfg=model_cfg,
        loss_cfg=loss_cfg,
        default_aug=default_aug,
        per_dataset_aug=per_dataset_aug,
    )
