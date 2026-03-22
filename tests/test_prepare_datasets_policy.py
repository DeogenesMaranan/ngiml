import io
import random
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.config import Manifest, SampleRecord, SplitConfig
from tools.prepare_datasets import _build_npz_bytes, _source_group_key, _split_records


def test_split_records_keeps_source_groups_in_single_split(tmp_path: Path):
    root = tmp_path / "MockSet"
    (root / "Au").mkdir(parents=True)
    (root / "Tp").mkdir(parents=True)

    records = [
        SampleRecord(dataset="MockSet", split="train", image_path=str(root / "Au" / "scene01_v1.jpg"), mask_path=None, label=0),
        SampleRecord(dataset="MockSet", split="train", image_path=str(root / "Au" / "scene01_copy.jpg"), mask_path=None, label=0),
        SampleRecord(dataset="MockSet", split="train", image_path=str(root / "Tp" / "scene07_mask.png"), mask_path=str(root / "mask" / "scene07.png"), label=1),
        SampleRecord(dataset="MockSet", split="train", image_path=str(root / "Tp" / "scene07_clone.png"), mask_path=str(root / "mask" / "scene07_b.png"), label=1),
    ]

    split_cfg = SplitConfig(train=0.5, val=0.5, test=0.0, seed=123)
    dir_tokens = {"au", "tp", "mask"}
    stem_suffixes = ("_mask",)
    splits = _split_records(records, split_cfg, root, dir_tokens, stem_suffixes)

    group_to_split: dict[str, str] = {}
    for split_name, split_items in splits.items():
        for rec in split_items:
            key = _source_group_key(Path(rec.image_path), root, dir_tokens, stem_suffixes)
            previous = group_to_split.get(key)
            if previous is None:
                group_to_split[key] = split_name
            else:
                assert previous == split_name


def test_build_npz_bytes_uses_mask_aware_crop_and_binary_mask(tmp_path: Path):
    image_path = tmp_path / "fake.png"
    mask_path = tmp_path / "fake_mask.png"

    yy, xx = np.meshgrid(np.arange(900), np.arange(1400), indexing="ij")
    image = np.zeros((900, 1400, 3), dtype=np.uint8)
    image[..., 0] = ((xx * 3 + yy * 5) % 256).astype(np.uint8)
    image[..., 1] = ((xx * 7 + 40) % 256).astype(np.uint8)
    image[..., 2] = ((yy * 11 + 80) % 256).astype(np.uint8)

    mask = np.zeros((900, 1400), dtype=np.uint8)
    mask[420:520, 640:760] = 255

    Image.fromarray(image, mode="RGB").save(image_path)
    Image.fromarray(mask, mode="L").save(mask_path)

    payload = _build_npz_bytes(
        image_path=image_path,
        mask_path=mask_path,
        split_name="train",
        crop_size=384,
        resize_max_side=1024,
        rng=random.Random(7),
    )

    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        out_image = data["image"]
        out_mask = data["mask"]
        payload_keys = set(data.files)

    assert out_image.shape == (384, 384, 3)
    assert out_mask.shape == (384, 384)
    assert "residual_noise" not in payload_keys
    assert int(out_mask.sum()) > 0
    assert set(np.unique(out_mask).tolist()).issubset({0, 1})


def test_manifest_dataframe_includes_path_and_metadata():
    manifest = Manifest(
        samples=[
            SampleRecord(
                dataset="CASIA2",
                split="train",
                image_path="prepared/CASIA2/train/sample_000001.npz",
                mask_path=None,
                label=1,
                metadata={"source": "unit-test", "residual_noise": True},
            )
        ],
        normalization_mode="imagenet",
    )

    df = manifest.to_dataframe()
    assert "path" in df.columns
    assert "metadata" in df.columns
    assert df.loc[0, "path"] == "prepared/CASIA2/train/sample_000001.npz"
    assert df.loc[0, "metadata"]["source"] == "unit-test"
