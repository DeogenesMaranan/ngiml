from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image

_fake_pandas = types.ModuleType("pandas")


class _FakeDataFrame:
    def __init__(self, *_args, **_kwargs):
        pass

    def to_parquet(self, *_args, **_kwargs):
        return None


_fake_pandas.DataFrame = _FakeDataFrame
sys.modules.setdefault("pandas", _fake_pandas)

from tools.prepare_test_datasets import prepare_test_datasets


def _write_rgb(path: Path, size: tuple[int, int], value: int) -> None:
    image = np.full((size[0], size[1], 3), value, dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(path)


def _write_mask(path: Path, size: tuple[int, int], value: int) -> None:
    mask = np.full(size, value, dtype=np.uint8)
    Image.fromarray(mask, mode="L").save(path)


def test_prepare_test_datasets_defaults_to_fake_only(tmp_path: Path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"

    dataset_root = input_root / "CASIA1"
    (dataset_root / "Au").mkdir(parents=True)
    (dataset_root / "Tp").mkdir(parents=True)
    (dataset_root / "Gt").mkdir(parents=True)

    _write_rgb(dataset_root / "Au" / "real_01.png", (64, 64), 32)
    _write_rgb(dataset_root / "Tp" / "fake_01.png", (64, 64), 192)
    _write_mask(dataset_root / "Gt" / "fake_01_gt.png", (64, 64), 255)

    manifest = prepare_test_datasets(
        input_root=input_root,
        output_root=output_root,
        manifest_path=output_root / "manifest.parquet",
        row_log_path=output_root / "manifest_rows.jsonl",
        size=64,
        split_name="test",
        dataset_name="CASIA1",
        fake_only=True,
        max_samples=0,
        clean_output=True,
        fail_on_missing_mask=True,
        tar_shard_size=0,
        remove_unsharded_after_shard=False,
    )

    assert len(manifest.samples) == 1
    assert [sample.label for sample in manifest.samples] == [1]
    assert all(sample.metadata["kind"] == "fake" for sample in manifest.samples)


def test_prepare_test_datasets_can_include_real_images(tmp_path: Path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"

    dataset_root = input_root / "CASIA1"
    (dataset_root / "Au").mkdir(parents=True)
    (dataset_root / "Tp").mkdir(parents=True)
    (dataset_root / "Gt").mkdir(parents=True)

    _write_rgb(dataset_root / "Au" / "real_01.png", (64, 64), 32)
    _write_rgb(dataset_root / "Tp" / "fake_01.png", (64, 64), 192)
    _write_mask(dataset_root / "Gt" / "fake_01_gt.png", (64, 64), 255)

    manifest = prepare_test_datasets(
        input_root=input_root,
        output_root=output_root,
        manifest_path=output_root / "manifest.parquet",
        row_log_path=output_root / "manifest_rows.jsonl",
        size=64,
        split_name="test",
        dataset_name="CASIA1",
        fake_only=False,
        max_samples=0,
        clean_output=True,
        fail_on_missing_mask=True,
        tar_shard_size=0,
        remove_unsharded_after_shard=False,
    )

    assert len(manifest.samples) == 2
    assert sorted(sample.label for sample in manifest.samples) == [0, 1]
    assert {sample.metadata["kind"] for sample in manifest.samples} == {"fake", "real"}
