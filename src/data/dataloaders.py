from __future__ import annotations

import io
import json
import tarfile
import atexit
from collections import OrderedDict
import pandas as pd
from bisect import bisect_right
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode

from .config import AugmentationConfig, Manifest, SampleRecord

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".npz"}

# Per-process LRU cache of open tar archives to avoid reopening on every sample.
_TAR_CACHE_LIMIT = 8
_TAR_CACHE: "OrderedDict[str, tarfile.TarFile]" = OrderedDict()


def _close_all_tars() -> None:
    while _TAR_CACHE:
        _, tar = _TAR_CACHE.popitem(last=False)
        try:
            tar.close()
        except Exception:
            pass


atexit.register(_close_all_tars)


def _get_tarfile(archive_path: str) -> tarfile.TarFile:
    tar = _TAR_CACHE.pop(archive_path, None)
    if tar is None or tar.closed:
        tar = tarfile.open(archive_path, "r:*")
    _TAR_CACHE[archive_path] = tar
    if len(_TAR_CACHE) > _TAR_CACHE_LIMIT:
        _, old_tar = _TAR_CACHE.popitem(last=False)
        try:
            old_tar.close()
        except Exception:
            pass
    return tar


def _load_image(path: str) -> torch.Tensor:
    image = torchvision_load_image(path).float() / 255.0
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    return image


def _load_mask(mask_path: str | None, target_hw: Sequence[int]) -> torch.Tensor:
    if mask_path is None:
        return torch.zeros((1, target_hw[0], target_hw[1]), dtype=torch.float32)
    mask = torchvision_load_image(mask_path, as_mask=True).float()
    if mask.shape[0] > 1:
        mask = mask[:1]
    if mask.max() > 1.0:
        mask = mask / 255.0
    if mask.shape[-2:] != tuple(target_hw):
        mask = F.resize(mask, target_hw, interpolation=InterpolationMode.NEAREST)
    return mask


def _load_from_npz(path: str | io.BytesIO) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    data = np.load(path)
    image_np = data["image"]
    image = torch.from_numpy(image_np)
    if image.ndim == 3:
        if image.shape[0] in (1, 3):
            pass
        elif image.shape[-1] in (1, 3):
            image = image.permute(2, 0, 1)
        else:
            raise ValueError(f"Unexpected image shape in NPZ: {image.shape}")
    else:
        raise ValueError(f"Image array must be 3D, got shape {image.shape}")

    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    image = image.float() / 255.0

    mask = None
    if "mask" in data:
        mask_np = data["mask"]
        mask = torch.from_numpy(mask_np)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.permute(2, 0, 1)
        elif mask.ndim == 3 and mask.shape[0] == 1:
            pass
        else:
            raise ValueError(f"Unexpected mask shape in NPZ: {mask.shape}")
        if mask.max() > 1.0:
            mask = mask / 255.0

    high_pass = None
    if "high_pass" in data:
        hp_np = data["high_pass"]
        if hp_np.size > 0:
            high_pass = torch.from_numpy(hp_np)
            if high_pass.ndim == 2:
                high_pass = high_pass.unsqueeze(0)
            elif high_pass.ndim == 3 and high_pass.shape[-1] in (1, 3):
                high_pass = high_pass.permute(2, 0, 1)
            elif high_pass.ndim == 3 and high_pass.shape[0] in (1, 3):
                pass
            else:
                raise ValueError(f"Unexpected high_pass shape in NPZ: {high_pass.shape}")
            if high_pass.shape[0] == 1:
                high_pass = high_pass.repeat(3, 1, 1)
            high_pass = high_pass.float() / 255.0

    return image, mask, high_pass


def _load_from_tar_npz(tar_spec: str) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if "::" not in tar_spec:
        raise ValueError(f"Invalid tar npz spec: {tar_spec}")
    archive_path, member_name = tar_spec.split("::", 1)
    tar = _get_tarfile(archive_path)
    member = tar.extractfile(member_name)
    if member is None:
        raise FileNotFoundError(f"Missing member {member_name} in {archive_path}")
    npz_bytes = member.read()
    return _load_from_npz(io.BytesIO(npz_bytes))


def torchvision_load_image(path: str, as_mask: bool = False) -> torch.Tensor:
    # Lazy import to avoid hard dependency at module import time.
    from torchvision.io import read_image

    image = read_image(path)
    if as_mask and image.shape[0] > 1:
        image = image[:1]
    return image


class PerDatasetDataset(Dataset):
    def __init__(self, samples: Sequence[SampleRecord], aug_cfg: AugmentationConfig, training: bool) -> None:
        self.samples = list(samples)
        self.aug_cfg = aug_cfg
        self.training = training

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:  # type: ignore[override]
        record = self.samples[index]

        if "::" in record.image_path and record.image_path.endswith(".npz"):
            image, mask, high_pass = _load_from_tar_npz(record.image_path)
        elif record.image_path.endswith(".npz"):
            image, mask, high_pass = _load_from_npz(record.image_path)
            if mask is None:
                mask = torch.zeros((1, image.shape[-2], image.shape[-1]), dtype=torch.float32)
        else:
            image = _load_image(record.image_path)
            mask = _load_mask(record.mask_path, image.shape[-2:])
            high_pass = None

        label = torch.tensor(record.label, dtype=torch.long)

        return {
            "image": image,
            "mask": mask,
            "label": label,
            "dataset": record.dataset,
            "high_pass": high_pass,
        }


class CombinedDataset(Dataset):
    def __init__(self, datasets: Sequence[Dataset]) -> None:
        self.datasets = list(datasets)
        self.offsets: List[int] = []
        total = 0
        for ds in self.datasets:
            self.offsets.append(total)
            total += len(ds)
        self.total_len = total

    def __len__(self) -> int:  # type: ignore[override]
        return self.total_len

    def __getitem__(self, index: int) -> dict[str, object]:  # type: ignore[override]
        ds_idx = bisect_right(self.offsets, index) - 1
        local_index = index - self.offsets[ds_idx]
        return self.datasets[ds_idx][local_index]


class RoundRobinSampler(Sampler[int]):
    def __init__(
        self,
        datasets: Sequence[Dataset],
        shuffle: bool = True,
        seed: int | None = None,
        balance: bool = True,
    ) -> None:
        self.datasets = list(datasets)
        self.shuffle = shuffle
        self.seed = seed
        self.balance = balance
        self.lengths = [len(ds) for ds in self.datasets]
        self.offsets: List[int] = []
        total = 0
        for length in self.lengths:
            self.offsets.append(total)
            total += length

    def __iter__(self):  # type: ignore[override]
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed)
        per_dataset_indices: List[List[int]] = []
        for length in self.lengths:
            indices = list(range(length))
            if self.shuffle:
                perm = torch.randperm(length, generator=generator).tolist()
                indices = [indices[i] for i in perm]
            per_dataset_indices.append(indices)

        max_len = max(self.lengths)
        total_rounds = max_len
        for offset in range(total_rounds):
            for ds_idx, indices in enumerate(per_dataset_indices):
                if not indices:
                    continue
                local_len = len(indices)
                if not self.balance and offset >= local_len:
                    continue
                local_idx = indices[offset % local_len] if self.balance else indices[offset]
                yield self.offsets[ds_idx] + local_idx

    def __len__(self) -> int:  # type: ignore[override]
        if self.balance:
            return max(self.lengths) * len(self.datasets)
        return sum(self.lengths)


def _normalize(image: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "zero_one":
        return image
    if mode == "imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
        return (image - mean) / std
    return image


def _apply_gpu_augmentations(
    image: torch.Tensor,
    mask: torch.Tensor,
    cfg: AugmentationConfig,
    high_pass: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    def _rand_scalar() -> torch.Tensor:
        return torch.rand((), device=image.device, generator=generator)

    if cfg.enable_flips:
        if _rand_scalar() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])
            if high_pass is not None:
                high_pass = torch.flip(high_pass, dims=[2])
        if _rand_scalar() < 0.2:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])
            if high_pass is not None:
                high_pass = torch.flip(high_pass, dims=[1])

    if cfg.enable_rotations and cfg.max_rotation_degrees > 0:
        angle = float((_rand_scalar() * 2 - 1) * cfg.max_rotation_degrees)
        image = F.rotate(image, angle=angle, interpolation=InterpolationMode.BILINEAR)
        mask = F.rotate(mask, angle=angle, interpolation=InterpolationMode.NEAREST)
        if high_pass is not None:
            high_pass = F.rotate(high_pass, angle=angle, interpolation=InterpolationMode.BILINEAR)

    if cfg.enable_random_crop:
        scale = float(
            cfg.crop_scale_range[0]
            + _rand_scalar() * (cfg.crop_scale_range[1] - cfg.crop_scale_range[0])
        )
        _, h, w = image.shape
        crop_h = max(1, int(h * scale))
        crop_w = max(1, int(w * scale))
        top = int(_rand_scalar() * (h - crop_h + 1))
        left = int(_rand_scalar() * (w - crop_w + 1))
        image = F.resized_crop(image, top, left, crop_h, crop_w, size=[h, w], interpolation=InterpolationMode.BILINEAR)
        mask = F.resized_crop(mask, top, left, crop_h, crop_w, size=[h, w], interpolation=InterpolationMode.NEAREST)
        if high_pass is not None:
            high_pass = F.resized_crop(high_pass, top, left, crop_h, crop_w, size=[h, w], interpolation=InterpolationMode.BILINEAR)

    if cfg.enable_color_jitter:
        factor = float(
            cfg.color_jitter_factors[0]
            + _rand_scalar() * (cfg.color_jitter_factors[1] - cfg.color_jitter_factors[0])
        )
        image = torch.clamp(image * factor, 0.0, 1.0)

    if cfg.enable_noise and cfg.noise_std_range[1] > 0:
        std = float(cfg.noise_std_range[0] + _rand_scalar() * (cfg.noise_std_range[1] - cfg.noise_std_range[0]))
        if std > 0:
            noise = torch.randn_like(image) * std
            image = torch.clamp(image + noise, 0.0, 1.0)

    return image, mask, high_pass


def _collate_builder(
    device: torch.device,
    per_dataset_aug: Dict[str, AugmentationConfig],
    normalization_mode: str,
    training: bool,
    aug_seed: int | None = None,
):
    def _collate(batch: List[dict[str, object]]) -> dict[str, object]:
        aug_generator: torch.Generator | None = None
        if aug_seed is not None:
            aug_generator = torch.Generator(device=device)
            aug_generator.manual_seed(aug_seed)

        images: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        datasets: List[str] = []
        high_passes: List[torch.Tensor] = []
        collect_high_pass = True

        for sample in batch:
            image = sample["image"].to(device, non_blocking=True)
            mask = sample["mask"].to(device, non_blocking=True)
            label = sample["label"].to(device, non_blocking=True)
            dataset_name = str(sample["dataset"])
            high_pass = sample.get("high_pass")
            if high_pass is not None:
                high_pass = high_pass.to(device, non_blocking=True)
            else:
                collect_high_pass = False
            aug_cfg = per_dataset_aug.get(dataset_name, AugmentationConfig(enable=False))
            views = aug_cfg.views_per_sample if aug_cfg.enable else 1
            views = max(1, views)

            base_image = image
            base_mask = mask
            base_high_pass = high_pass

            for _ in range(views):
                view_image = base_image
                view_mask = base_mask
                view_high_pass = base_high_pass

                if training and aug_cfg.enable:
                    view_image, view_mask, view_high_pass = _apply_gpu_augmentations(
                        view_image,
                        view_mask,
                        aug_cfg,
                        high_pass=view_high_pass,
                        generator=aug_generator,
                    )

                view_image = _normalize(view_image, normalization_mode)

                images.append(view_image)
                masks.append(view_mask)
                labels.append(label)
                datasets.append(dataset_name)
                if collect_high_pass and view_high_pass is not None:
                    high_passes.append(view_high_pass)

        batch_dict = {
            "images": torch.stack(images, dim=0),
            "masks": torch.stack(masks, dim=0),
            "labels": torch.stack(labels, dim=0),
            "datasets": datasets,
        }

        if collect_high_pass and high_passes:
            batch_dict["high_pass"] = torch.stack(high_passes, dim=0)

        return batch_dict

    return _collate


def _group_by(split: str, samples: Iterable[SampleRecord]) -> Dict[str, list[SampleRecord]]:
    grouped: Dict[str, list[SampleRecord]] = {}
    for record in samples:
        if record.split != split:
            continue
        grouped.setdefault(record.dataset, []).append(record)
    return grouped


def load_manifest(path: str | Path) -> Manifest:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        return Manifest.from_dataframe(df)
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return Manifest.from_dict(data)


def create_dataloaders(
    manifest_path: str | Path,
    per_dataset_augmentations: Dict[str, AugmentationConfig],
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
    pin_memory: bool = True,
    round_robin_seed: int | None = 0,
    drop_last: bool = True,
    aug_seed: int | None = None,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
) -> Dict[str, DataLoader]:
    manifest = load_manifest(manifest_path)
    normalization_mode = manifest.normalization_mode

    splits = {
        split: _group_by(split, manifest.samples)
        for split in ("train", "val", "test")
    }

    loaders: Dict[str, DataLoader] = {}

    for split_name, per_dataset_records in splits.items():
        training = split_name == "train"
        datasets: List[PerDatasetDataset] = []
        for dataset_name, records in per_dataset_records.items():
            aug_cfg = per_dataset_augmentations.get(dataset_name, AugmentationConfig(enable=False))
            datasets.append(PerDatasetDataset(records, aug_cfg=aug_cfg, training=training))

        if not datasets:
            continue

        combined = CombinedDataset(datasets)

        if training:
            sampler = RoundRobinSampler(datasets, shuffle=True, seed=round_robin_seed, balance=True)
        else:
            sampler = None

        collate_fn = _collate_builder(
            device,
            per_dataset_augmentations,
            normalization_mode,
            training=training,
            aug_seed=aug_seed,
        )

        pf = prefetch_factor if num_workers > 0 else None
        persistent = persistent_workers if num_workers > 0 else False

        loader = DataLoader(
            combined,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            drop_last=drop_last if training else False,
            prefetch_factor=pf,
            persistent_workers=persistent,
        )
        loaders[split_name] = loader

    return loaders
