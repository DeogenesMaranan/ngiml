from pathlib import Path

import numpy as np
from PIL import Image

from src.data.config import DatasetStructureConfig, PreparationConfig, SplitConfig
from tools.prepare_datasets import prepare_single_dataset


def _write_rgb(path: Path, height: int, width: int) -> None:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[..., 0] = 128
    Image.fromarray(arr, mode="RGB").save(path)


def _write_mask(path: Path, height: int, width: int) -> None:
    arr = np.zeros((height, width), dtype=np.uint8)
    arr[height // 4 : height // 2, width // 4 : width // 2] = 255
    Image.fromarray(arr, mode="L").save(path)


def test_prepare_single_dataset_keeps_small_images(tmp_path):
    dataset_root = tmp_path / "datasets"
    toy_root = dataset_root / "Toy"
    real_dir = toy_root / "Au"
    fake_dir = toy_root / "Tp"
    mask_dir = toy_root / "Gt"
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    _write_rgb(real_dir / "real_small.png", 96, 120)
    _write_rgb(fake_dir / "fake_small.png", 96, 120)
    _write_mask(mask_dir / "fake_small_gt.png", 96, 120)

    cfg = DatasetStructureConfig(
        dataset_root=str(dataset_root),
        dataset_name="Toy",
        real_subdir="Au",
        fake_subdir="Tp",
        mask_subdir="Gt",
        mask_suffix="_gt",
        prepared_root=str(tmp_path / "prepared"),
    )
    split_cfg = SplitConfig(train=0.8, val=0.2, test=0.0, seed=42)
    prep_cfg = PreparationConfig(
        target_sizes=(448,),
        normalization_mode="imagenet",
        tar_shard_size=0,
        resize_max_side=896,
    )

    records = prepare_single_dataset(cfg, split_cfg, prep_cfg)

    assert len(records) == 2
    assert sorted(record.label for record in records) == [0, 1]
