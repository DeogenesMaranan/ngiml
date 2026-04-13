"""Prepare datasets into a common manifest."""
from __future__ import annotations

import argparse
import hashlib
import io
import random
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(
        iterable: Iterable | None = None,
        total: int | None = None,
        desc: str | None = None,
        **_: object,
    ) -> Iterable | list[object]:
        return iterable if iterable is not None else []

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.config import DatasetStructureConfig, Manifest, PreparationConfig, SampleRecord, SplitConfig
from src.prepare_shared import (
    IMAGE_EXTENSIONS,
    TarShardWriter,
    build_mask_index,
    discover_images,
    pad_to_size,
    resolve_mask_from_candidates,
)


def _clean_optional_token(text: str | None) -> str:
    if text is None:
        return ""
    return text.lower().strip()


def _build_grouping_rules(cfg: DatasetStructureConfig) -> tuple[set[str], tuple[str, ...]]:
    dir_tokens = {
        _clean_optional_token(cfg.real_subdir),
        _clean_optional_token(cfg.fake_subdir),
        _clean_optional_token(cfg.mask_subdir),
    }
    dir_tokens = {token for token in dir_tokens if token}

    stem_suffixes: list[str] = []
    suffix = cfg.mask_suffix.lower().strip()
    if suffix:
        stem_suffixes.append(suffix)

    return dir_tokens, tuple(stem_suffixes)


def _normalize_source_stem(stem: str, stem_suffixes: Sequence[str]) -> str:
    text = stem.lower()
    for suffix in stem_suffixes:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    text = re.sub(r"(?:[_\-.]?)(?:copy|clone|tampered|forged|splice|spliced|edited)$", "", text)
    text = re.sub(r"(?:[_\-.]?)(?:v\d+|ver\d+)$", "", text)
    text = re.sub(r"[\s\-]+", "_", text)
    return text.strip("_") or stem.lower()


def _source_group_key(
    path: Path,
    root: Path,
    dir_tokens: set[str],
    stem_suffixes: Sequence[str],
) -> str:
    rel = path.relative_to(root)
    parts = [p.lower() for p in rel.parts[:-1] if p.lower() not in dir_tokens]
    stem = _normalize_source_stem(path.stem, stem_suffixes)
    prefix = "/".join(parts[-2:]) if parts else ""
    return f"{prefix}/{stem}" if prefix else stem


def _assign_grouped_items(
    grouped_items: Sequence[list[SampleRecord]],
    ratios: Sequence[float],
    rng: random.Random,
) -> dict[str, list[SampleRecord]]:
    split_names = ["train", "val", "test"]
    total = sum(len(group) for group in grouped_items)
    target = [float(total) * r for r in ratios]
    assigned_counts = [0, 0, 0]
    splits: dict[str, list[SampleRecord]] = {"train": [], "val": [], "test": []}

    shuffled = [list(group) for group in grouped_items]
    rng.shuffle(shuffled)
    shuffled.sort(key=len, reverse=True)

    for group in shuffled:
        valid_indices = [idx for idx, ratio in enumerate(ratios) if ratio > 0]
        if not valid_indices:
            valid_indices = [0]
        chosen_idx = max(valid_indices, key=lambda idx: (target[idx] - assigned_counts[idx], ratios[idx]))
        split_name = split_names[chosen_idx]
        splits[split_name].extend(group)
        assigned_counts[chosen_idx] += len(group)

    return splits


def _split_records(
    records: Sequence[SampleRecord],
    split_cfg: SplitConfig,
    dataset_root: Path,
    dir_tokens: set[str],
    stem_suffixes: Sequence[str],
) -> dict[str, list[SampleRecord]]:
    split_cfg.validate()
    rng = random.Random(split_cfg.seed)
    per_label_groups: dict[int, dict[str, list[SampleRecord]]] = {0: {}, 1: {}}
    for rec in records:
        key = _source_group_key(Path(rec.image_path), dataset_root, dir_tokens, stem_suffixes)
        label_groups = per_label_groups.setdefault(rec.label, {})
        label_groups.setdefault(key, []).append(rec)

    splits = {"train": [], "val": [], "test": []}
    for label_groups in per_label_groups.values():
        groups = list(label_groups.values())
        n = sum(len(group) for group in groups)
        if n == 0:
            continue
        ratios = [split_cfg.train, split_cfg.val, split_cfg.test]
        label_splits = _assign_grouped_items(groups, ratios, rng)
        splits["train"].extend(label_splits["train"])
        splits["val"].extend(label_splits["val"])
        splits["test"].extend(label_splits["test"])
    return splits


def _build_npz_bytes(
    image_path: Path,
    mask_path: Path | None,
    split_name: str,
    crop_size: int,
    resize_max_side: int,
    rng: random.Random,
) -> bytes:
    image = Image.open(image_path).convert("RGB")
    mask_img = Image.open(mask_path).convert("L") if mask_path is not None else None

    if resize_max_side > 0:
        w, h = image.size
        long_side = max(w, h)
        if long_side > resize_max_side:
            scale = float(resize_max_side) / float(long_side)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            image = image.resize((new_w, new_h), Image.BILINEAR)
            if mask_img is not None:
                mask_img = mask_img.resize((new_w, new_h), Image.NEAREST)

    if crop_size > 0 and split_name in {"train", "val"}:
        image_np_full = np.asarray(image, dtype=np.uint8)
        mask_np_full = np.asarray(mask_img, dtype=np.uint8) if mask_img is not None else None

        h_full, w_full = image_np_full.shape[:2]
        pad_h = max(0, crop_size - h_full)
        pad_w = max(0, crop_size - w_full)
        if pad_h > 0 or pad_w > 0:
            image_np_full = pad_to_size(image_np_full, crop_size, mode="symmetric")
            if mask_np_full is not None:
                mask_np_full = pad_to_size(mask_np_full, crop_size, mode="constant", constant=0)

        h_full, w_full = image_np_full.shape[:2]
        max_top = max(0, h_full - crop_size)
        max_left = max(0, w_full - crop_size)

        top = 0
        left = 0
        if mask_np_full is not None:
            mask_bin = mask_np_full > 127
            fg_coords = np.argwhere(mask_bin)
            if fg_coords.shape[0] > 0:
                up = np.pad(mask_bin[:-1, :], ((1, 0), (0, 0)), constant_values=False)
                down = np.pad(mask_bin[1:, :], ((0, 1), (0, 0)), constant_values=False)
                left_n = np.pad(mask_bin[:, :-1], ((0, 0), (1, 0)), constant_values=False)
                right_n = np.pad(mask_bin[:, 1:], ((0, 0), (0, 1)), constant_values=False)
                boundary = mask_bin & (~(up & down & left_n & right_n))
                boundary_coords = np.argwhere(boundary)

                use_boundary = boundary_coords.shape[0] > 0 and rng.random() < 0.7
                coords = boundary_coords if use_boundary else fg_coords
                center_y, center_x = coords[rng.randrange(coords.shape[0])]
                jitter_y = rng.randint(-crop_size // 6, crop_size // 6)
                jitter_x = rng.randint(-crop_size // 6, crop_size // 6)
                top = int(center_y) - crop_size // 2 + jitter_y
                left = int(center_x) - crop_size // 2 + jitter_x
                top = max(0, min(top, max_top))
                left = max(0, min(left, max_left))

                crop_mask = mask_bin[top : top + crop_size, left : left + crop_size]
                if crop_mask.sum() == 0:
                    for _ in range(12):
                        center_y, center_x = fg_coords[rng.randrange(fg_coords.shape[0])]
                        top = max(0, min(int(center_y) - crop_size // 2, max_top))
                        left = max(0, min(int(center_x) - crop_size // 2, max_left))
                        crop_mask = mask_bin[top : top + crop_size, left : left + crop_size]
                        if crop_mask.sum() > 0:
                            break
            else:
                top = rng.randint(0, max_top) if max_top > 0 else 0
                left = rng.randint(0, max_left) if max_left > 0 else 0
        else:
            top = rng.randint(0, max_top) if max_top > 0 else 0
            left = rng.randint(0, max_left) if max_left > 0 else 0

        image_np = image_np_full[top : top + crop_size, left : left + crop_size]
        if mask_np_full is not None:
            mask_np = mask_np_full[top : top + crop_size, left : left + crop_size]
        else:
            mask_np = None
    else:
        image_np = np.asarray(image, dtype=np.uint8)
        mask_np = np.asarray(mask_img, dtype=np.uint8) if mask_img is not None else None

    payload = {"image": image_np}
    if mask_np is not None:
        payload["mask"] = (mask_np > 127).astype(np.uint8)

    buf = io.BytesIO()
    np.savez(buf, **payload)
    return buf.getvalue()


def prepare_single_dataset(
    cfg: DatasetStructureConfig,
    split_cfg: SplitConfig,
    prep_cfg: PreparationConfig,
    sample_limit: int = 0,
) -> list[SampleRecord]:
    root = cfg.root()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root missing: {root}")

    real_dir = (root / cfg.real_subdir) if cfg.real_subdir else None
    fake_dir = (root / cfg.fake_subdir) if cfg.fake_subdir else None
    mask_dir = (root / cfg.mask_subdir) if cfg.mask_subdir else None

    real_images = discover_images(real_dir, IMAGE_EXTENSIONS) if real_dir is not None and real_dir.exists() else []
    fake_images = discover_images(fake_dir, IMAGE_EXTENSIONS) if fake_dir is not None and fake_dir.exists() else []
    mask_index = build_mask_index(mask_dir, IMAGE_EXTENSIONS) if mask_dir is not None and mask_dir.exists() else {}
    dir_tokens, stem_suffixes = _build_grouping_rules(cfg)
    target_sizes = sorted(prep_cfg.target_size_set())
    if len(target_sizes) != 1:
        raise ValueError(
            f"prepare_datasets expects exactly one target size, got {target_sizes}"
        )
    crop_size = target_sizes[0]

    records: list[SampleRecord] = []
    skipped_fake_missing_mask = 0

    for real_img in tqdm(real_images, desc=f"{cfg.dataset_name} real", leave=False):
        records.append(
            SampleRecord(
                dataset=cfg.dataset_name,
                split="train",
                image_path=str(real_img),
                mask_path=None,
                label=0,
            )
        )

    for fake_img in tqdm(fake_images, desc=f"{cfg.dataset_name} fake", leave=False):
        stem = fake_img.stem
        candidates = (f"{stem}{cfg.mask_suffix}", stem)
        mask_path = resolve_mask_from_candidates(stem, mask_index, candidates)
        if mask_path is None:
            skipped_fake_missing_mask += 1
            print(f"Skipping fake image without mask: {fake_img}", file=sys.stderr)
            continue
        records.append(
            SampleRecord(
                dataset=cfg.dataset_name,
                split="train",
                image_path=str(fake_img),
                mask_path=str(mask_path),
                label=1,
            )
        )

    if sample_limit > 0 and len(records) > sample_limit:
        sample_seed_text = f"{split_cfg.seed}|{cfg.dataset_name}|sample_limit"
        sample_seed = int.from_bytes(hashlib.blake2b(sample_seed_text.encode("utf-8"), digest_size=8).digest(), "big")
        rng = random.Random(sample_seed)
        records = rng.sample(records, sample_limit)

    splits = _split_records(records, split_cfg, root, dir_tokens, stem_suffixes)

    prepared_records: list[SampleRecord] = []
    for split_name, split_records in splits.items():
        if not split_records:
            continue
        tar_writer: TarShardWriter | None = None
        if prep_cfg.tar_shard_size > 0:
            tar_root = cfg.prepared_dir() / split_name
            tar_writer = TarShardWriter(tar_root, prep_cfg.tar_shard_size)

        for idx, rec in enumerate(tqdm(split_records, desc=f"{cfg.dataset_name} {split_name}", leave=False)):
            image_path = Path(rec.image_path)
            mask_path = Path(rec.mask_path) if rec.mask_path is not None else None
            seed_text = f"{split_cfg.seed}|{cfg.dataset_name}|{split_name}|{image_path.as_posix()}"
            sample_seed = int.from_bytes(hashlib.blake2b(seed_text.encode("utf-8"), digest_size=8).digest(), "big")
            sample_rng = random.Random(sample_seed)
            npz_bytes = _build_npz_bytes(
                image_path=image_path,
                mask_path=mask_path,
                split_name=split_name,
                crop_size=crop_size,
                resize_max_side=prep_cfg.resize_max_side,
                rng=sample_rng,
            )

            stem = f"{cfg.dataset_name}_{split_name}_{'fake' if rec.label else 'real'}_{idx:06d}"
            if tar_writer is not None:
                tar_path, member_name = tar_writer.add(npz_bytes, member_name=stem + ".npz")
                sample_path = f"{tar_path}::{member_name}"
            else:
                out_root = cfg.prepared_dir() / split_name
                out_npz = out_root / (stem + ".npz")
                out_npz.parent.mkdir(parents=True, exist_ok=True)
                out_npz.write_bytes(npz_bytes)
                sample_path = str(out_npz)

            prepared_records.append(
                SampleRecord(
                    dataset=rec.dataset,
                    split=split_name,
                    image_path=sample_path,
                    mask_path=None,
                    label=rec.label,
                    residual_noise_path=None,
                    original_image_path=rec.image_path,
                    original_mask_path=rec.mask_path,
                    metadata={
                        "dataset": rec.dataset,
                        "path": sample_path,
                        "original_image_path": rec.image_path,
                        "original_mask_path": rec.mask_path,
                        "storage": "npz",
                        "residual_noise_mode": "on_the_fly",
                    },
                )
            )

        if tar_writer is not None:
            tar_writer.close()

    if skipped_fake_missing_mask > 0:
        print(
            f"[{cfg.dataset_name}] Skipped fake images without masks: {skipped_fake_missing_mask}",
            file=sys.stderr,
        )
    return prepared_records


def prepare_all(
    datasets: Sequence[DatasetStructureConfig],
    per_dataset_splits: dict[str, SplitConfig],
    prep_cfg: PreparationConfig,
    manifest_out: Path,
    sample_limit: int = 0,
) -> Manifest:
    all_records: list[SampleRecord] = []
    for cfg in tqdm(datasets, desc="datasets"):
        split_cfg = per_dataset_splits.get(cfg.dataset_name)
        if split_cfg is None:
            raise ValueError(f"Missing split config for dataset {cfg.dataset_name}")
        records = prepare_single_dataset(cfg, split_cfg, prep_cfg, sample_limit=sample_limit)
        all_records.extend(records)

    manifest = Manifest(samples=all_records, normalization_mode=prep_cfg.normalization_mode)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    df = manifest.to_dataframe()
    df.to_parquet(manifest_out, index=False)
    return manifest


def build_default_configs() -> tuple[list[DatasetStructureConfig], dict[str, SplitConfig], PreparationConfig]:
    shared_seed = 42
    datasets = [
        DatasetStructureConfig(
            dataset_root="./datasets",
            dataset_name="CASIA1",
            real_subdir="Au",
            fake_subdir="Tp",
            mask_subdir="Gt",
            mask_suffix="_gt",
            prepared_root="./prepared",
        ),
        DatasetStructureConfig(
            dataset_root="./datasets",
            dataset_name="CASIA2",
            real_subdir="Au",
            fake_subdir="Tp",
            mask_subdir="Gt",
            mask_suffix="_gt",
            prepared_root="./prepared",
        ),
        DatasetStructureConfig(
            dataset_root="./datasets",
            dataset_name="TampCOCO",
            real_subdir=None,
            fake_subdir="tp",
            mask_subdir="mask",
            mask_suffix="",
            prepared_root="./prepared",
        ),
        DatasetStructureConfig(
            dataset_root="./datasets",
            dataset_name="Columbia",
            real_subdir=None,
            fake_subdir="fake",
            mask_subdir="mask",
            mask_suffix="_edgemask",
            prepared_root="./prepared",
        ),
        DatasetStructureConfig(
            dataset_root="./datasets",
            dataset_name="COVERAGE",
            real_subdir="real",
            fake_subdir="fake",
            mask_subdir="mask",
            mask_suffix="forged",
            prepared_root="./prepared",
        )
    ]

    per_dataset_splits = {
        "CASIA1": SplitConfig(train=0.8, val=0.2, test=0.0, seed=shared_seed),
        "CASIA2": SplitConfig(train=0.8, val=0.2, test=0.0, seed=shared_seed),
        "TampCOCO": SplitConfig(train=0.8, val=0.2, test=0.0, seed=shared_seed),
        "Columbia": SplitConfig(train=0.8, val=0.2, test=0.0, seed=shared_seed),
        "COVERAGE": SplitConfig(train=0.8, val=0.2, test=0.0, seed=shared_seed),
    }

    prep_cfg = PreparationConfig(
        target_sizes=(448,),
        normalization_mode="imagenet",
        tar_shard_size=500,
        resize_max_side=896,
    )

    return datasets, per_dataset_splits, prep_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare datasets and emit manifest.parquet")
    parser.add_argument("--manifest", type=str, default=None, help="Output manifest path. Defaults to <prepared_root>/manifest.parquet")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Name of a single dataset to process (e.g., CASIA2, IMD2020, NIST, TampCOCO). If omitted, processes all.")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=0,
        help="Maximum number of samples to use per dataset (0 = use all). Applies before splitting. Uses split seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    """Run dataset preparation."""
    args = parse_args()
    datasets, per_dataset_splits, prep_cfg = build_default_configs()

    if args.dataset:
        selected = args.dataset.strip().lower()
        datasets = [d for d in datasets if d.dataset_name.lower() == selected]
        if not datasets:
            raise ValueError(f"Dataset '{args.dataset}' not found in config. Available: {[d.dataset_name for d in build_default_configs()[0]]}")
        print(f"Processing only dataset: {datasets[0].dataset_name}")

    prepared_root = Path(datasets[0].prepared_root)
    if args.manifest:
        manifest_out = Path(args.manifest)
    elif len(datasets) == 1:
        manifest_out = prepared_root / datasets[0].dataset_name / "manifest.parquet"
    else:
        manifest_out = prepared_root / "manifest.parquet"

    manifest = prepare_all(
        datasets,
        per_dataset_splits,
        prep_cfg,
        manifest_out,
        sample_limit=int(args.sample_limit),
    )
    print(f"Wrote manifest with {len(manifest.samples)} samples to {manifest_out}")


if __name__ == "__main__":
    main()


