from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

from src.data.config import Manifest, SampleRecord
from tools.colab_train_helpers import (
    apply_colab_runtime_settings,
    build_default_components,
    build_training_config,
)


def ensure_repo_on_syspath(repo_root: Path) -> None:
    repo_root = Path(repo_root).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _infer_label_from_path(path_value: str) -> int:
    tokens = [token.strip().lower() for token in path_value.replace("\\", "/").split("/") if token.strip()]
    has_fake = "fake" in tokens
    has_real = "real" in tokens

    if has_fake and not has_real:
        return 1
    if has_real and not has_fake:
        return 0
    raise ValueError(f"Unable to infer label from path (expected folder 'fake' or 'real'): {path_value}")


def build_manifest_from_prepared(prepared_root: Path, manifest_out: Path | None = None) -> Path:
    prepared_root = Path(prepared_root)
    if not prepared_root.exists():
        raise FileNotFoundError(f"Prepared root not found: {prepared_root}")

    if manifest_out is None:
        manifest_out = prepared_root / "manifest_local.json"

    records: list[SampleRecord] = []
    for dataset_dir in sorted(p for p in prepared_root.iterdir() if p.is_dir()):
        dataset_name = dataset_dir.name
        for split_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            split = split_dir.name
            if split not in {"train", "val", "test"}:
                continue

            npz_files = sorted(split_dir.rglob("*.npz"))
            for npz_path in npz_files:
                label = _infer_label_from_path(npz_path.as_posix())
                records.append(
                    SampleRecord(
                        dataset=dataset_name,
                        split=split,
                        image_path=npz_path.as_posix(),
                        mask_path=None,
                        label=label,
                        high_pass_path=None,
                    )
                )

            tar_files = []
            for pattern in ("*.tar", "*.tar.gz", "*.tgz"):
                tar_files.extend(sorted(split_dir.rglob(pattern)))

            for tar_path in tar_files:
                with tarfile.open(tar_path, mode="r:*") as tf:
                    for member in tf.getmembers():
                        if not member.isfile() or not member.name.endswith(".npz"):
                            continue
                        label = _infer_label_from_path(member.name)
                        records.append(
                            SampleRecord(
                                dataset=dataset_name,
                                split=split,
                                image_path=f"{tar_path.as_posix()}::{member.name}",
                                mask_path=None,
                                label=label,
                                high_pass_path=None,
                            )
                        )

    if not records:
        raise RuntimeError(
            f"No prepared samples found under {prepared_root}. Expected NPZ files or TAR shards inside dataset/split folders."
        )

    manifest = Manifest(samples=records, normalization_mode="imagenet")
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_out, "w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle)

    return manifest_out


def build_local_training_config(
    manifest_path: Path,
    output_dir: Path,
    balance_sampling: bool = False,
    vram_profile: str = "6gb",
) -> dict:
    model_cfg, loss_cfg, default_aug, per_dataset_aug = build_default_components()
    training_config = build_training_config(
        manifest_path=manifest_path,
        output_dir=str(output_dir),
        model_cfg=model_cfg,
        loss_cfg=loss_cfg,
        default_aug=default_aug,
        per_dataset_aug=per_dataset_aug,
    )

    training_config = apply_colab_runtime_settings(
        training_config,
        balance_sampling=balance_sampling,
    )

    base_batch_size = int(training_config.get("batch_size", 8))
    profile = vram_profile.lower().strip()
    if profile in {"6gb", "rtx4050", "low"}:
        training_config["batch_size"] = 1
        training_config["grad_accum_steps"] = max(1, base_batch_size)
    else:
        training_config["grad_accum_steps"] = 1

    output_dir = Path(output_dir)
    local_cache = output_dir / "local_cache"
    training_config["local_cache_dir"] = str(local_cache)
    try:
        import torch

        training_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        training_config["device"] = "cpu"
    return training_config
