import os
import shutil
import json
from pathlib import Path
from typing import Tuple

from src.data.config import AugmentationConfig
from src.data.dataloaders import load_manifest
from src.model.hybrid_ngiml import HybridNGIMLConfig
from src.model.losses import MultiStageLossConfig
from src.training_defaults import (
    build_default_components as _build_default_components_top_level,
    build_training_config as _build_training_config_top_level
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


def _norm(value: str) -> str:
    return str(value).replace("\\", "/")


def _suffix_score(a_parts, b_parts) -> int:
    score = 0
    for ax, bx in zip(reversed(a_parts), reversed(b_parts)):
        if ax != bx:
            break
        score += 1
    return score


def _candidate_paths(value: str, manifest_path: Path, data_root: Path):
    normalized = _norm(value)
    path_value = Path(normalized)
    candidates = []
    if path_value.is_absolute():
        candidates.append(path_value)
    else:
        candidates.extend([
            manifest_path.parent / path_value,
            data_root / path_value,
            data_root / "ngiml" / path_value,
            Path("/content") / path_value,
            Path("/content/data") / path_value,
            Path("/content/ngiml") / path_value,
        ])
    if "prepared/" in normalized:
        suffix = normalized.split("prepared/", 1)[1]
        candidates.extend([
            data_root / "prepared" / suffix,
            data_root / "ngiml" / "prepared" / suffix,
            Path("/content") / "prepared" / suffix,
            Path("/content/ngiml") / "prepared" / suffix,
        ])
    if "datasets/" in normalized:
        suffix = normalized.split("datasets/", 1)[1]
        candidates.extend([
            data_root / "datasets" / suffix,
            data_root / "ngiml" / "datasets" / suffix,
            Path("/content") / "datasets" / suffix,
            Path("/content/ngiml") / "datasets" / suffix,
        ])
    seen = set()
    unique = []
    for candidate in candidates:
        key = candidate.as_posix()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _build_tar_index(data_root: Path):
    tar_files = []
    for pattern in ("*.tar", "*.tar.gz", "*.tgz"):
        tar_files.extend(data_root.rglob(pattern))
    tar_by_name = {}
    for tar_path in tar_files:
        tar_by_name.setdefault(tar_path.name, []).append(tar_path)
    return tar_files, tar_by_name


def _match_tar_by_basename(value: str, tar_by_name: dict[str, list[Path]]):
    name = Path(_norm(value)).name
    matches = tar_by_name.get(name, [])
    if not matches:
        return None
    hint_parts = Path(_norm(value)).parts
    return max(matches, key=lambda path: _suffix_score(path.parts, hint_parts))


def _resolve_file(value: str, manifest_path: Path, data_root: Path, tar_by_name: dict[str, list[Path]]) -> Path:
    candidates = _candidate_paths(value, manifest_path, data_root)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if str(value).endswith((".tar", ".tar.gz", ".tgz")):
        tar_match = _match_tar_by_basename(value, tar_by_name)
        if tar_match is not None:
            return tar_match
    return candidates[0] if candidates else Path(_norm(value))


def _resolve_path(path_str: str | None, manifest_path: Path, data_root: Path, tar_by_name: dict[str, list[Path]]) -> str | None:
    if path_str is None:
        return None
    normalized = _norm(path_str)
    if "::" in normalized:
        archive, member = normalized.split("::", 1)
        archive_path = _resolve_file(archive, manifest_path, data_root, tar_by_name).as_posix()
        member_path = _norm(member)
        return f"{archive_path}::{member_path}"
    return _resolve_file(normalized, manifest_path, data_root, tar_by_name).as_posix()


def _sample_files_exist(sample) -> bool:
    image_path = str(sample.image_path)
    if "::" in image_path:
        archive_path, _ = image_path.split("::", 1)
        if not Path(archive_path).exists():
            return False
    else:
        if not Path(image_path).exists():
            return False
    if sample.mask_path is not None and not Path(sample.mask_path).exists():
        return False
    return True


def find_or_resolve_manifest(data_root: Path, manifest_names: Tuple[str, ...] = ("manifest.parquet", "manifest.json")) -> Path:
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
    tar_files, tar_by_name = _build_tar_index(data_root)
    print(f"Indexed tar files under {data_root}: {len(tar_files)}")
    manifest_obj = load_manifest(manifest_path)
    rewritten = 0
    for sample in manifest_obj.samples:
        image_new = _resolve_path(sample.image_path, manifest_path, data_root, tar_by_name)
        mask_new = _resolve_path(sample.mask_path, manifest_path, data_root, tar_by_name) if sample.mask_path else None
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
    manifest_obj.samples = [s for s in manifest_obj.samples if _sample_files_exist(s)]
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
