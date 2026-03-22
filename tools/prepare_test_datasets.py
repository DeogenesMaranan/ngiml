from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable | None = None, **_: object):
        return iterable if iterable is not None else []

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.config import Manifest, SampleRecord

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    real_dir: str | None
    fake_dir: str | None
    mask_dir: str | None
    mask_stem_candidates: Callable[[str], Sequence[str]]


def _casia1_mask_candidates(fake_stem: str) -> Sequence[str]:
    return (f"{fake_stem}_gt", fake_stem)


def _coverage_mask_candidates(fake_stem: str) -> Sequence[str]:
    return (f"{fake_stem}forged", fake_stem)


def _columbia_mask_candidates(fake_stem: str) -> Sequence[str]:
    return (f"{fake_stem}_edgemask", fake_stem)


def default_specs() -> list[DatasetSpec]:
    return [
        DatasetSpec(
            name="CASIA1",
            real_dir="Au",
            fake_dir="Tp",
            mask_dir="Gt",
            mask_stem_candidates=_casia1_mask_candidates,
        ),
        DatasetSpec(
            name="Columbia",
            real_dir=None,
            fake_dir="fake",
            mask_dir="mask",
            mask_stem_candidates=_columbia_mask_candidates,
        ),
        DatasetSpec(
            name="COVERAGE",
            real_dir="real",
            fake_dir="fake",
            mask_dir="mask",
            mask_stem_candidates=_coverage_mask_candidates,
        ),
    ]


def _discover_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _safe_name(value: str) -> str:
    text = value.replace("\\", "_").replace("/", "_").replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in text)


def _build_mask_index(mask_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in _discover_images(mask_dir):
        key = path.stem.lower()
        if key not in index:
            index[key] = path
    return index


def _resolve_mask(fake_path: Path, mask_index: dict[str, Path], candidates_fn: Callable[[str], Sequence[str]]) -> Path | None:
    for candidate in candidates_fn(fake_path.stem):
        key = candidate.lower()
        if key in mask_index:
            return mask_index[key]
    return None


def _resize_image_rgb(image_path: Path, size: int) -> tuple[np.ndarray, list[int]]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    image = image.resize((size, size), resample=Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8), [height, width]


def _resize_mask(mask_path: Path, size: int) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((size, size), resample=Image.NEAREST)
    mask_np = np.asarray(mask, dtype=np.uint8)
    return (mask_np > 127).astype(np.uint8)


def _black_mask(size: int) -> np.ndarray:
    return np.zeros((size, size), dtype=np.uint8)


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
        # mode="w" writes plain uncompressed tar archives.
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


def _save_npz(npz_path: Path, image_np: np.ndarray, mask_np: np.ndarray, metadata: dict[str, object]) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(metadata, ensure_ascii=True)
    # np.savez keeps arrays uncompressed (ZIP_STORED); do not switch to savez_compressed.
    np.savez(npz_path, image=image_np, mask=mask_np, metadata_json=np.asarray(metadata_json))


def _save_metadata(metadata_path: Path, metadata: dict[str, object]) -> None:
    # Saved as .npy (object array wrapping the metadata dict) for consistency
    # with the numpy-native pipeline; use np.load(..., allow_pickle=True).item()
    # to reload.
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(metadata_path, np.asarray(metadata, dtype=object))


def _append_manifest_row(jsonl_path: Path, row: dict[str, object]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True))
        handle.write("\n")


def _iter_selected_specs(all_specs: Sequence[DatasetSpec], requested: str | None) -> list[DatasetSpec]:
    if not requested:
        return list(all_specs)
    requested_lower = requested.strip().lower()
    selected = [s for s in all_specs if s.name.lower() == requested_lower]
    if not selected:
        known = ", ".join(s.name for s in all_specs)
        raise ValueError(f"Unknown dataset '{requested}'. Available: {known}")
    return selected


def _shard_records(
    records: Sequence[SampleRecord],
    output_root: Path,
    shard_size: int,
    remove_unsharded_after_shard: bool,
) -> tuple[list[SampleRecord], int]:
    if shard_size <= 0 or not records:
        return list(records), 0

    split_names = sorted({rec.split for rec in records})
    by_split: dict[str, list[SampleRecord]] = {split: [] for split in split_names}
    for rec in records:
        by_split[rec.split].append(rec)

    sharded_records: list[SampleRecord] = []
    tar_count = 0

    for split_name in split_names:
        split_records = by_split[split_name]
        shard_root = output_root
        writer = TarShardWriter(shard_root, shard_size)
        try:
            for idx, rec in enumerate(tqdm(split_records, desc=f"Sharding {split_name}", leave=False)):
                npz_path = Path(rec.image_path)
                payload = npz_path.read_bytes()
                # Store directly under dataset/ with no split subdirectory.
                member_name = f"{rec.dataset}/{idx:07d}_{npz_path.name}"
                tar_path, member_name = writer.add(payload, member_name=member_name)
                tar_spec = f"{tar_path}::{member_name}"

                if rec.metadata is None:
                    rec.metadata = {}
                rec.metadata["sharded_sample_path"] = tar_spec
                rec.metadata["processed_sample_path"] = tar_spec
                rec.metadata["storage"] = "tar_npz"

                rec.image_path = tar_spec
                rec.mask_path = None
                sharded_records.append(rec)

                if remove_unsharded_after_shard and npz_path.exists():
                    npz_path.unlink()
        finally:
            writer.close()
            tar_count += writer.shard_idx

    if remove_unsharded_after_shard:
        # Remove entire per-dataset subdirectories that held the unsharded NPZ
        # files.  Only directories that are not shard tar files are removed so
        # the freshly written shards are left untouched.
        for child in sorted(output_root.iterdir()):
            if child.is_dir() and not child.name.startswith("shard_"):
                shutil.rmtree(child, ignore_errors=True)
        # Safety sweep for any stray NPZ files that survived the directory
        # removal (e.g. from an interrupted prior run).
        for leftover_npz in output_root.rglob("*.npz"):
            leftover_npz.unlink(missing_ok=True)

    return sharded_records, tar_count


def prepare_test_datasets(
    input_root: Path,
    output_root: Path,
    manifest_path: Path,
    row_log_path: Path,
    size: int,
    split_name: str,
    dataset_name: str | None,
    max_samples: int,
    clean_output: bool,
    fail_on_missing_mask: bool,
    tar_shard_size: int,
    remove_unsharded_after_shard: bool,
) -> Manifest:
    specs = _iter_selected_specs(default_specs(), dataset_name)

    if clean_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if clean_output and row_log_path.exists():
        row_log_path.unlink()

    records: list[SampleRecord] = []
    skipped_missing_mask = 0

    for spec in specs:
        dataset_root = input_root / spec.name
        if not dataset_root.exists():
            print(f"[WARN] Missing dataset folder: {dataset_root}")
            continue

        mask_index: dict[str, Path] = {}
        if spec.mask_dir:
            mask_index = _build_mask_index(dataset_root / spec.mask_dir)

        entries: list[tuple[str, Path, Path | None, int]] = []

        if spec.real_dir:
            for image_path in _discover_images(dataset_root / spec.real_dir):
                entries.append(("real", image_path, None, 0))

        if spec.fake_dir:
            for image_path in _discover_images(dataset_root / spec.fake_dir):
                mask_path = _resolve_mask(image_path, mask_index, spec.mask_stem_candidates)
                if mask_path is None:
                    skipped_missing_mask += 1
                    msg = f"[WARN] Missing mask for fake image: {image_path}"
                    if fail_on_missing_mask:
                        raise FileNotFoundError(msg)
                    print(msg)
                    continue
                entries.append(("fake", image_path, mask_path, 1))

        if max_samples > 0:
            entries = entries[:max_samples]

        for idx, (kind, image_path, mask_path, label) in enumerate(
            tqdm(entries, desc=f"{spec.name} {split_name}", leave=False)
        ):
            rel_source = image_path.relative_to(dataset_root).as_posix()
            stem = _safe_name(Path(rel_source).with_suffix("").as_posix())
            sample_id = f"{spec.name.lower()}_{kind}_{idx:06d}_{stem}"

            npz_path = output_root / spec.name / split_name / kind / f"{sample_id}.npz"
            # Metadata stored as .npy (pickled object array) instead of JSON.
            metadata_path = output_root / spec.name / split_name / kind / "metadata" / f"{sample_id}.npy"

            image_np, original_size = _resize_image_rgb(image_path=image_path, size=size)
            if mask_path is not None:
                mask_np = _resize_mask(mask_path=mask_path, size=size)
            else:
                mask_np = _black_mask(size=size)

            metadata = {
                "dataset": spec.name,
                "split": split_name,
                "kind": kind,
                "label": label,
                "original_image_path": str(image_path),
                "original_mask_path": str(mask_path) if mask_path is not None else None,
                "original_size": original_size,
                "processed_sample_path": str(npz_path),
                "processed_size": [size, size],
                "mask_is_generated_black": bool(mask_path is None),
                "mask_foreground_pixels": int(mask_np.sum()),
                "storage": "npz",
            }

            _save_npz(npz_path=npz_path, image_np=image_np, mask_np=mask_np, metadata=metadata)
            _save_metadata(metadata_path=metadata_path, metadata=metadata)

            record = SampleRecord(
                dataset=spec.name,
                split=split_name,
                image_path=str(npz_path),
                mask_path=None,
                label=label,
                residual_noise_path=None,
                original_image_path=str(image_path),
                original_mask_path=str(mask_path) if mask_path is not None else None,
                metadata=metadata,
            )
            records.append(record)

    records, tar_count = _shard_records(
        records=records,
        output_root=output_root,
        shard_size=tar_shard_size,
        remove_unsharded_after_shard=remove_unsharded_after_shard,
    )

    if row_log_path.exists():
        row_log_path.unlink()
    for rec in records:
        _append_manifest_row(row_log_path, rec.to_dict())

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([rec.to_dict() for rec in records]).to_parquet(manifest_path, index=False)

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote row log: {row_log_path}")
    print(f"Total samples: {len(records)}")
    if tar_shard_size > 0:
        print(f"Tar shards written: {tar_count}")
    if skipped_missing_mask:
        print(f"Skipped fake samples without masks: {skipped_missing_mask}")

    return Manifest(samples=list(records))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare standardized benchmark test datasets")
    parser.add_argument("--input-root", type=str, default="./test_datasets", help="Input test dataset root")
    parser.add_argument(
        "--output-root",
        type=str,
        default="./prepared_test_datasets",
        help="Output root for standardized samples",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Manifest output path (default: <output-root>/manifest.parquet)",
    )
    parser.add_argument(
        "--row-log",
        type=str,
        default=None,
        help="JSONL row log path (default: <output-root>/manifest_rows.jsonl)",
    )
    parser.add_argument("--size", type=int, default=384, help="Square resize size")
    parser.add_argument("--split", type=str, default="test", help="Split name to assign in manifest")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Single dataset to process: CASIA1, Columbia, COVERAGE (default: all)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Cap number of samples per dataset after discovery (0 = all)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove output root before writing new samples",
    )
    parser.add_argument(
        "--fail-on-missing-mask",
        action="store_true",
        help="Fail immediately when a fake sample has no matching mask",
    )
    parser.add_argument(
        "--tar-shard-size",
        type=int,
        default=1024,
        help="Samples per tar shard (0 disables sharding)",
    )
    parser.add_argument(
        "--remove-unsharded-after-shard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete local NPZ files after they are packed into tar shards (default: true)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    manifest_path = Path(args.manifest) if args.manifest else output_root / "manifest.parquet"
    row_log_path = Path(args.row_log) if args.row_log else output_root / "manifest_rows.jsonl"

    prepare_test_datasets(
        input_root=input_root,
        output_root=output_root,
        manifest_path=manifest_path,
        row_log_path=row_log_path,
        size=int(args.size),
        split_name=str(args.split),
        dataset_name=args.dataset,
        max_samples=int(args.max_samples),
        clean_output=bool(args.clean),
        fail_on_missing_mask=bool(args.fail_on_missing_mask),
        tar_shard_size=int(args.tar_shard_size),
        remove_unsharded_after_shard=bool(args.remove_unsharded_after_shard),
    )


if __name__ == "__main__":
    main()