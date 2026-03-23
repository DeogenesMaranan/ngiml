"""Prepare datasets into a common manifest with optional resizing."""
from __future__ import annotations

import argparse
import hashlib
import io
import random
import re
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

try:  # tqdm makes progress clearer; fall back to no-op if missing
    from tqdm import tqdm
except ImportError:  # pragma: no cover - lightweight fallback
    def tqdm(iterable: Iterable | None = None, total: int | None = None, desc: str | None = None, **_: object):
        return iterable if iterable is not None else []

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # allow running as a script without installing the package

from src.data.config import DatasetStructureConfig, Manifest, PreparationConfig, SampleRecord, SplitConfig

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class TarShardWriter:
    """Utility to write NPZ payloads into sequential tar shards."""

    def __init__(self, out_root: Path, shard_size: int) -> None:
        self.out_root = out_root
        self.shard_size = max(1, shard_size)
        self.shard_idx = 0
        self.current: tarfile.TarFile | None = None
        self.current_path: Path | None = None
        self.count_in_shard = 0

    def _start_new_shard(self) -> None:
        self.out_root.mkdir(parents=True, exist_ok=True)
        tar_path = self.out_root / f"shard_{self.shard_idx:05d}.tar"
        self.shard_idx += 1
        self.count_in_shard = 0
        if self.current is not None:
            self.current.close()
        self.current = tarfile.open(tar_path, mode="w")
        self.current_path = tar_path

    def add(self, payload_bytes: bytes, member_name: str) -> tuple[str, str]:
        if self.current is None or self.count_in_shard >= self.shard_size:
            self._start_new_shard()
        assert self.current is not None and self.current_path is not None
        info = tarfile.TarInfo(name=member_name)
        info.size = len(payload_bytes)
        info.mtime = time.time()
        self.current.addfile(info, io.BytesIO(payload_bytes))
        self.count_in_shard += 1
        return str(self.current_path), member_name

    def close(self) -> None:
        if self.current is not None:
            self.current.close()
            self.current = None
            self.current_path = None


def _discover_images(directory: Path) -> List[Path]:
    return sorted(
        [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    )


def _find_mask(fake_path: Path, mask_dir: Path, mask_suffix: str) -> Path | None:
    candidates = []
    stem = fake_path.stem
    for ext in IMAGE_EXTENSIONS:
        candidates.append(mask_dir / f"{stem}{mask_suffix}{ext}")
        candidates.append(mask_dir / f"{stem}{ext}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _build_grouping_rules(cfg: DatasetStructureConfig) -> tuple[set[str], tuple[str, ...]]:
    dir_tokens = {
        cfg.real_subdir.lower().strip(),
        cfg.fake_subdir.lower().strip(),
        cfg.mask_subdir.lower().strip(),
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
    grouped_items: Sequence[List[SampleRecord]],
    ratios: Sequence[float],
    rng: random.Random,
) -> Dict[str, List[SampleRecord]]:
    split_names = ["train", "val", "test"]
    total = sum(len(group) for group in grouped_items)
    target = [float(total) * r for r in ratios]
    assigned_counts = [0, 0, 0]
    splits: Dict[str, List[SampleRecord]] = {"train": [], "val": [], "test": []}

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
) -> Dict[str, List[SampleRecord]]:
    split_cfg.validate()
    rng = random.Random(split_cfg.seed)
    per_label_groups: Dict[int, Dict[str, List[SampleRecord]]] = {0: {}, 1: {}}
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
    # Tiny-image filtering is handled upstream in prepare_single_dataset.
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
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            image_np_full = np.pad(
                image_np_full,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="symmetric",
            )
            if mask_np_full is not None:
                mask_np_full = np.pad(
                    mask_np_full,
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode="constant",
                    constant_values=0,
                )

        h_full, w_full = image_np_full.shape[:2]
        max_top = max(0, h_full - crop_size)
        max_left = max(0, w_full - crop_size)

        top = 0
        left = 0
        if mask_np_full is not None:
            mask_bin = mask_np_full > 127
            fg_coords = np.argwhere(mask_bin)
            if fg_coords.shape[0] > 0:
                # Bias fake crops to boundaries when possible.
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
    # np.savez (not compressed) to avoid CPU overhead from compression.
    np.savez(buf, **payload)
    return buf.getvalue()


def prepare_single_dataset(
    cfg: DatasetStructureConfig,
    split_cfg: SplitConfig,
    prep_cfg: PreparationConfig,
    sample_limit: int = 0,
) -> List[SampleRecord]:
    root = cfg.root()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root missing: {root}")

    real_dir = root / cfg.real_subdir
    fake_dir = root / cfg.fake_subdir
    mask_dir = root / cfg.mask_subdir

    real_images = _discover_images(real_dir) if real_dir.exists() else []
    fake_images = _discover_images(fake_dir) if fake_dir.exists() else []
    dir_tokens, stem_suffixes = _build_grouping_rules(cfg)
    target_sizes = sorted(prep_cfg.target_size_set())
    if len(target_sizes) != 1:
        raise ValueError(
            f"prepare_datasets expects exactly one target size, got {target_sizes}"
        )
    crop_size = target_sizes[0]
    tiny_threshold = max(1, crop_size // 2)

    records: List[SampleRecord] = []
    skipped_fake_missing_mask = 0
    skipped_tiny_images = 0

    for real_img in tqdm(real_images, desc=f"{cfg.dataset_name} real", leave=False):
        with Image.open(real_img) as real_pil:
            w, h = real_pil.size
        if min(w, h) < tiny_threshold:
            skipped_tiny_images += 1
            continue
        records.append(
            SampleRecord(
                dataset=cfg.dataset_name,
                split="train",  # placeholder, real split decided later
                image_path=str(real_img),
                mask_path=None,
                label=0,
            )
        )

    for fake_img in tqdm(fake_images, desc=f"{cfg.dataset_name} fake", leave=False):
        with Image.open(fake_img) as fake_pil:
            w, h = fake_pil.size
        if min(w, h) < tiny_threshold:
            skipped_tiny_images += 1
            continue
        mask_path = _find_mask(fake_img, mask_dir, cfg.mask_suffix)
        if mask_path is None:
            skipped_fake_missing_mask += 1
            print(f"Skipping fake image without mask: {fake_img}", file=sys.stderr)
            continue
        records.append(
            SampleRecord(
                dataset=cfg.dataset_name,
                split="train",  # placeholder
                image_path=str(fake_img),
                mask_path=str(mask_path),
                label=1,
            )
        )

    # Apply sampling limit if set
    if sample_limit > 0 and len(records) > sample_limit:
        sample_seed_text = f"{split_cfg.seed}|{cfg.dataset_name}|sample_limit"
        sample_seed = int.from_bytes(hashlib.blake2b(sample_seed_text.encode("utf-8"), digest_size=8).digest(), "big")
        rng = random.Random(sample_seed)
        records = rng.sample(records, sample_limit)

    splits = _split_records(records, split_cfg, root, dir_tokens, stem_suffixes)

    prepared_records: List[SampleRecord] = []
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
    if skipped_tiny_images > 0:
        print(
            f"[{cfg.dataset_name}] Filtered tiny images (short side < {tiny_threshold}px): {skipped_tiny_images}",
            file=sys.stderr,
        )
    return prepared_records


def prepare_all(
    datasets: Sequence[DatasetStructureConfig],
    per_dataset_splits: Dict[str, SplitConfig],
    prep_cfg: PreparationConfig,
    manifest_out: Path,
    sample_limit: int = 0,
) -> Manifest:
    all_records: List[SampleRecord] = []
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


def build_default_configs() -> Tuple[List[DatasetStructureConfig], Dict[str, SplitConfig], PreparationConfig]:
    shared_seed = 42
    datasets = [
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
            real_subdir="au",
            fake_subdir="tp",
            mask_subdir="mask",
            mask_suffix="",
            prepared_root="./prepared",
        ),
        DatasetStructureConfig(
            dataset_root="./datasets",
            dataset_name="NIST",
            real_subdir="au",
            fake_subdir="tp",
            mask_subdir="mask",
            mask_suffix="",
            prepared_root="./prepared",
        ),
        DatasetStructureConfig(
            dataset_root="./datasets",
            dataset_name="IMD2020",
            real_subdir="au",
            fake_subdir="fake",
            mask_subdir="mask",
            mask_suffix="_mask",
            prepared_root="./prepared",
        ),
    ]

    per_dataset_splits = {
        "CASIA2": SplitConfig(train=0.8, val=0.2, test=0.0, seed=shared_seed),
        "TampCOCO": SplitConfig(train=0.8, val=0.2, test=0.0, seed=shared_seed),
        "NIST": SplitConfig(train=0.8, val=0.2, test=0.0, seed=shared_seed),
        "IMD2020": SplitConfig(train=0.8, val=0.2, test=0.0, seed=shared_seed),
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
    args = parse_args()
    datasets, per_dataset_splits, prep_cfg = build_default_configs()

    # Filter datasets if --dataset is specified
    if args.dataset:
        selected = args.dataset.strip().lower()
        datasets = [d for d in datasets if d.dataset_name.lower() == selected]
        if not datasets:
            raise ValueError(f"Dataset '{args.dataset}' not found in config. Available: {[d.dataset_name for d in build_default_configs()[0]]}")
        print(f"Processing only dataset: {datasets[0].dataset_name}")

    prepared_root = Path(datasets[0].prepared_root)
    # If only one dataset, default manifest name includes dataset name
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

