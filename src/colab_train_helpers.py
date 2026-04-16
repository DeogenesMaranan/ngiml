import os
import shutil
import json
import sys
from pathlib import Path
from typing import Any

from src.data.dataloaders import load_manifest
from src.manifest_utils import build_tar_index, resolve_path, sample_files_exist

__all__ = [
    "apply_colab_runtime_settings",
    "find_or_resolve_manifest",
    "stage_persistent_cache_to_runtime",
]


def _cfg_update(config: Any, values: dict[str, Any], *, overwrite: bool = True) -> None:
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


def _cfg_as_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return config
    return dict(vars(config))


def apply_colab_runtime_settings(
    training_config: Any,
    balance_sampling: bool = False,
    local_cache_dir: str | None = None,
) -> dict[str, Any]:
    recommended_workers = max(2, min(6, (os.cpu_count() or 4)))
    cache_dir = local_cache_dir or "/content/cache"

    updates = {
        "num_workers": recommended_workers,
        "persistent_workers": False,
        "multiprocessing_context": "spawn" if sys.version_info >= (3, 12) else None,
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
    cfg_out = _cfg_as_dict(training_config)
    current_workers = int(cfg_out.get("num_workers", recommended_workers) or 0)
    if current_workers > recommended_workers:
        cfg_out["num_workers"] = recommended_workers
    return cfg_out


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


def find_or_resolve_manifest(data_root: Path, manifest_names: tuple[str, ...] = ("manifest.parquet", "manifest.json")) -> Path:
    data_root = Path(data_root)
    resolved_manifest_path = data_root / "manifest_resolved.json"
    manifest_candidates = [
        resolved_manifest_path,
        data_root / "manifest.parquet",
        data_root / "manifest.json",
        data_root / "prepared" / "manifest.parquet",
        data_root / "prepared" / "manifest.json",
        data_root / "ngiml" / "manifest.parquet",
        data_root / "ngiml" / "manifest.json",
    ]
    manifest_path = next((p for p in manifest_candidates if p.exists()), None)
    if manifest_path is None:
        discovered = sorted(
            p
            for p in data_root.rglob("manifest.*")
            if p.name in manifest_names or p.name == "manifest_resolved.json"
        )
        if discovered:
            manifest_path = discovered[0]
        else:
            raise FileNotFoundError(
                f"No manifest.parquet or manifest.json found under {data_root}. "
                "Check dataset download path, or set DATA_DIR to the folder containing the manifest file."
            )
    if resolved_manifest_path.exists() and resolved_manifest_path.stat().st_size > 0:
        print(f"Using cached resolved manifest: {resolved_manifest_path}")
        return resolved_manifest_path
    print("Using manifest:", manifest_path)
    tar_files, tar_by_name = build_tar_index(data_root)
    print(f"Indexed tar files under {data_root}: {len(tar_files)}")
    manifest_obj = load_manifest(manifest_path)
    rewritten = 0
    for sample in manifest_obj.samples:
        image_new = resolve_path(sample.image_path, manifest_path, data_root, tar_by_name)
        mask_new = resolve_path(sample.mask_path, manifest_path, data_root, tar_by_name) if sample.mask_path else None
        if image_new != sample.image_path:
            sample.image_path = image_new
            rewritten += 1
        if mask_new != sample.mask_path:
            sample.mask_path = mask_new
            rewritten += 1
        if sample.residual_noise_path is not None:
            sample.residual_noise_path = None
            rewritten += 1
    original_count = len(manifest_obj.samples)
    manifest_obj.samples = [s for s in manifest_obj.samples if sample_files_exist(s)]
    filtered_out = original_count - len(manifest_obj.samples)
    if not manifest_obj.samples:
        raise FileNotFoundError(
            "No valid samples remain after path resolution. "
            f"Indexed tar files: {len(tar_files)} under {data_root}. "
            "Likely the downloaded dataset does not contain prepared shards referenced by the manifest."
        )
    with open(resolved_manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_obj.to_dict(), handle)
    print(
        f"Wrote resolved manifest to {resolved_manifest_path} "
        f"(updated fields: {rewritten}, removed missing samples: {filtered_out})"
    )
    return resolved_manifest_path
