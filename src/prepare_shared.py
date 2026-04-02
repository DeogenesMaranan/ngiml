from __future__ import annotations

import io
import tarfile
import time
from pathlib import Path


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