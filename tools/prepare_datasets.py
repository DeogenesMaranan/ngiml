"""Prepare datasets into a common manifest with optional resizing and high-pass exports."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from src.data.config import DatasetStructureConfig, Manifest, PreparationConfig, SampleRecord, SplitConfig

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


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


def _split_records(records: Sequence[SampleRecord], split_cfg: SplitConfig) -> Dict[str, List[SampleRecord]]:
    split_cfg.validate()
    rng = random.Random(split_cfg.seed)
    per_label: Dict[int, List[SampleRecord]] = {0: [], 1: []}
    for rec in records:
        per_label.setdefault(rec.label, []).append(rec)

    splits = {"train": [], "val": [], "test": []}
    for label, items in per_label.items():
        items = items.copy()
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * split_cfg.train)
        n_val = int(n * split_cfg.val)
        train_items = items[:n_train]
        val_items = items[n_train : n_train + n_val]
        test_items = items[n_train + n_val :]
        splits["train"].extend(train_items)
        splits["val"].extend(val_items)
        splits["test"].extend(test_items)
    return splits


def _save_npz_sample(
    image_path: Path,
    mask_path: Path | None,
    target_size: int,
    out_npz: Path,
    compute_high_pass: bool,
) -> None:
    image = Image.open(image_path).convert("RGB")
    mask_img = Image.open(mask_path).convert("L") if mask_path is not None else None

    if target_size > 0:
        image = image.resize((target_size, target_size), Image.BILINEAR)
        if mask_img is not None:
            mask_img = mask_img.resize((target_size, target_size), Image.NEAREST)

    if mask_img is None:
        mask_img = Image.new("L", image.size, color=0)

    image_np = np.array(image, dtype=np.uint8)
    mask_np = np.array(mask_img, dtype=np.uint8)

    payload = {"image": image_np, "mask": mask_np}

    if compute_high_pass:
        hp_img = ImageOps.autocontrast(image.filter(ImageFilter.FIND_EDGES))
        hp_np = np.array(hp_img, dtype=np.uint8)
        payload["high_pass"] = hp_np

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    # np.savez (not compressed) to avoid CPU overhead from compression.
    np.savez(out_npz, **payload)


def prepare_single_dataset(
    cfg: DatasetStructureConfig,
    split_cfg: SplitConfig,
    prep_cfg: PreparationConfig,
) -> List[SampleRecord]:
    root = cfg.root()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root missing: {root}")

    real_dir = root / cfg.real_subdir
    fake_dir = root / cfg.fake_subdir
    mask_dir = root / cfg.mask_subdir

    real_images = _discover_images(real_dir) if real_dir.exists() else []
    fake_images = _discover_images(fake_dir) if fake_dir.exists() else []

    records: List[SampleRecord] = []

    for real_img in real_images:
        records.append(
            SampleRecord(
                dataset=cfg.dataset_name,
                split="train",  # placeholder, real split decided later
                image_path=str(real_img),
                mask_path=None,
                label=0,
            )
        )

    for fake_img in fake_images:
        mask_path = _find_mask(fake_img, mask_dir, cfg.mask_suffix)
        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for fake image {fake_img}")
        records.append(
            SampleRecord(
                dataset=cfg.dataset_name,
                split="train",  # placeholder
                image_path=str(fake_img),
                mask_path=str(mask_path),
                label=1,
            )
        )

    splits = _split_records(records, split_cfg)

    prepared_records: List[SampleRecord] = []
    target_size = sorted(prep_cfg.target_size_set())[0]
    for split_name, split_records in splits.items():
        for idx, rec in enumerate(split_records):
            image_path = Path(rec.image_path)
            mask_path = Path(rec.mask_path) if rec.mask_path is not None else None
            out_root = cfg.prepared_dir() / f"size_{target_size}" / split_name / "npz"

            stem = f"{cfg.dataset_name}_{split_name}_{'fake' if rec.label else 'real'}_{idx:06d}"
            out_npz = out_root / (stem + ".npz")

            _save_npz_sample(
                image_path=image_path,
                mask_path=mask_path,
                target_size=target_size,
                out_npz=out_npz,
                compute_high_pass=prep_cfg.compute_high_pass,
            )

            prepared_records.append(
                SampleRecord(
                    dataset=rec.dataset,
                    split=split_name,
                    image_path=str(out_npz),
                    mask_path=None,
                    label=rec.label,
                    high_pass_path=None,
                )
            )
    return prepared_records


def prepare_all(
    datasets: Sequence[DatasetStructureConfig],
    per_dataset_splits: Dict[str, SplitConfig],
    prep_cfg: PreparationConfig,
    manifest_out: Path,
) -> Manifest:
    all_records: List[SampleRecord] = []
    for cfg in datasets:
        split_cfg = per_dataset_splits.get(cfg.dataset_name)
        if split_cfg is None:
            raise ValueError(f"Missing split config for dataset {cfg.dataset_name}")
        records = prepare_single_dataset(cfg, split_cfg, prep_cfg)
        all_records.extend(records)

    manifest = Manifest(samples=all_records, normalization_mode=prep_cfg.normalization_mode)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_out, "w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2)
    return manifest


def build_default_configs() -> Tuple[List[DatasetStructureConfig], Dict[str, SplitConfig], PreparationConfig]:
    datasets = [
        DatasetStructureConfig(
            dataset_root="./datasets",
            dataset_name="IMD2020",
            real_subdir="real",
            fake_subdir="fake",
            mask_subdir="mask",
            mask_suffix="_mask",
            prepared_root="prepared",
        ),
        DatasetStructureConfig(
            dataset_root="./datasets",
            dataset_name="CASIA2",
            real_subdir="real",
            fake_subdir="fake",
            mask_subdir="mask",
            mask_suffix="_gt",
            prepared_root="prepared",
        ),
        DatasetStructureConfig(
            dataset_root="./datasets",
            dataset_name="COVERAGE",
            real_subdir="real",
            fake_subdir="fake",
            mask_subdir="mask",
            mask_suffix="forged",
            prepared_root="prepared",
        ),
    ]

    per_dataset_splits = {
        "CASIA2": SplitConfig(train=0.8, val=0.2, test=0.0, seed=6),
        "IMD2020": SplitConfig(train=0.8, val=0.2, test=0.0, seed=20),
        "COVERAGE": SplitConfig(train=0.0, val=0.0, test=1.0, seed=4),
    }

    prep_cfg = PreparationConfig(target_sizes=(320,), normalization_mode="zero_one", compute_high_pass=True)

    return datasets, per_dataset_splits, prep_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare datasets and emit manifest.json")
    parser.add_argument("--manifest", type=str, default=None, help="Output manifest path. Defaults to <prepared_root>/manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets, per_dataset_splits, prep_cfg = build_default_configs()

    prepared_root = Path(datasets[0].prepared_root)
    manifest_out = Path(args.manifest) if args.manifest else prepared_root / "manifest.json"

    manifest = prepare_all(datasets, per_dataset_splits, prep_cfg, manifest_out)
    print(f"Wrote manifest with {len(manifest.samples)} samples to {manifest_out}")


if __name__ == "__main__":
    main()
