from __future__ import annotations

import io
import tarfile
import time
from pathlib import Path

import numpy as np


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


def discover_images(root: Path, image_extensions: set[str], *, return_empty_if_missing: bool = False) -> list[Path]:
    if return_empty_if_missing and not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    )


def build_mask_index(mask_dir: Path, image_extensions: set[str] | None = None) -> dict[str, Path]:
    exts = image_extensions or IMAGE_EXTENSIONS
    index: dict[str, Path] = {}
    for path in discover_images(mask_dir, exts, return_empty_if_missing=True):
        key = path.stem.lower()
        if key not in index:
            index[key] = path
    return index


def resolve_mask_from_candidates(
    image_stem: str,
    mask_index: dict[str, Path],
    candidates: list[str] | tuple[str, ...],
) -> Path | None:
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in mask_index:
            return mask_index[key]
    return None


def pad_to_size(arr: np.ndarray, size: int, mode: str, constant: int = 0) -> np.ndarray:
    """Pad a HW or HWC array to size x size."""
    h, w = arr.shape[:2]
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    if pad_h == 0 and pad_w == 0:
        return arr

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if arr.ndim == 3:
        padding = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    else:
        padding = ((pad_top, pad_bottom), (pad_left, pad_right))

    if mode == "constant":
        return np.pad(arr, padding, mode="constant", constant_values=constant)
    return np.pad(arr, padding, mode=mode)


__all__ = [
    "IMAGE_EXTENSIONS",
    "TarShardWriter",
    "build_mask_index",
    "discover_images",
    "pad_to_size",
    "resolve_mask_from_candidates",
]