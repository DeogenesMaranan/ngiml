from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable: Iterable | None = None, **_: object) -> Iterable | list[object]:
        return iterable if iterable is not None else []

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.config import Manifest, SampleRecord
from src.prepare_shared import (
    IMAGE_EXTENSIONS,
    TarShardWriter,
    build_mask_index,
    discover_images,
    pad_to_size,
    resolve_mask_from_candidates,
)


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

def _tampcoco_mask_candidates(fake_stem: str) -> Sequence[str]:
    return (f"{fake_stem}", fake_stem)


def default_specs() -> list[DatasetSpec]:
    return [
        # DatasetSpec(
        #     name="CASIA1",
        #     real_dir="Au",
        #     fake_dir="Tp",
        #     mask_dir="Gt",
        #     mask_stem_candidates=_casia1_mask_candidates,
        # ),
        # DatasetSpec(
        #     name="Columbia",
        #     real_dir=None,  # fakes + masks only
        #     fake_dir="fake",
        #     mask_dir="mask",
        #     mask_stem_candidates=_columbia_mask_candidates,
        # ),
        # DatasetSpec(
        #     name="COVERAGE",
        #     real_dir="real",
        #     fake_dir="fake",
        #     mask_dir="mask",
        #     mask_stem_candidates=_coverage_mask_candidates,
        # ),
        DatasetSpec(
            name="CASIA2",
            real_dir="Au",
            fake_dir="Tp",
            mask_dir="Gt",
            mask_stem_candidates=_casia1_mask_candidates,
        ),
    ]


def _safe_name(value: str) -> str:
    text = value.replace("\\", "_").replace("/", "_").replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in text)


def _process_image(image_path: Path, size: int) -> tuple[np.ndarray, tuple[int, int], str]:
    """Load and bring image to size x size using resize/pad policy."""
    image = Image.open(image_path).convert("RGB")
    original_hw = (image.height, image.width)
    h, w = original_hw

    if h >= size and w >= size:
        if image.size != (size, size):
            image = image.resize((size, size), resample=Image.BILINEAR)
        return np.asarray(image, dtype=np.uint8), original_hw, "resize"

    if h <= size and w <= size:
        arr = np.asarray(image, dtype=np.uint8)
        arr = pad_to_size(arr, size, mode="symmetric")
        return arr, original_hw, "pad"

    long_side = max(h, w)
    scale = size / long_side
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    image = image.resize((new_w, new_h), resample=Image.BILINEAR)
    arr = np.asarray(image, dtype=np.uint8)
    arr = pad_to_size(arr, size, mode="symmetric")
    return arr, original_hw, "resize_then_pad"


def _process_mask(mask_path: Path, size: int, original_hw: tuple[int, int], preproc_mode: str) -> np.ndarray:
    """Load and bring mask to size x size using the image preprocessing path."""
    mask = Image.open(mask_path).convert("L")
    h, w = original_hw

    if preproc_mode == "resize":
        if mask.size != (size, size):
            mask = mask.resize((size, size), resample=Image.NEAREST)
        return (np.asarray(mask, dtype=np.uint8) > 127).astype(np.uint8)

    if preproc_mode == "pad":
        arr = np.asarray(mask, dtype=np.uint8)
        arr = pad_to_size(arr, size, mode="constant", constant=0)
        return (arr > 127).astype(np.uint8)

    long_side = max(h, w)
    scale = size / long_side
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    mask = mask.resize((new_w, new_h), resample=Image.NEAREST)
    arr = np.asarray(mask, dtype=np.uint8)
    arr = pad_to_size(arr, size, mode="constant", constant=0)
    return (arr > 127).astype(np.uint8)


def _save_npz(
    npz_path: Path,
    image_np: np.ndarray,
    mask_np: np.ndarray | None,
    metadata: dict[str, object],
) -> None:
    """Save image and optional mask to uncompressed NPZ."""
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(metadata, ensure_ascii=True)
    arrays: dict[str, np.ndarray] = {
        "image": image_np,
        "metadata_json": np.asarray(metadata_json),
    }
    if mask_np is not None:
        arrays["mask"] = mask_np
    np.savez(npz_path, **arrays)


def _append_manifest_row(jsonl_path: Path, row: dict[str, object]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True))
        handle.write("\n")


def _shard_records(
    records: list[SampleRecord],
    output_root: Path,
    shard_size: int,
    remove_unsharded: bool,
) -> tuple[list[SampleRecord], int]:
    if shard_size <= 0 or not records:
        return records, 0

    writer = TarShardWriter(output_root, shard_size)
    sharded: list[SampleRecord] = []
    try:
        for idx, rec in enumerate(tqdm(records, desc="Sharding", leave=False)):
            npz_path = Path(rec.image_path)
            payload = npz_path.read_bytes()
            member_name = f"{rec.dataset}/{idx:07d}_{npz_path.name}"
            tar_path, member_name = writer.add(payload, member_name=member_name)
            tar_spec = f"{tar_path}::{member_name}"

            if rec.metadata is None:
                rec.metadata = {}
            rec.metadata["processed_sample_path"] = tar_spec
            rec.metadata["storage"] = "tar_npz"
            rec.image_path = tar_spec
            rec.mask_path = None
            sharded.append(rec)

            if remove_unsharded and npz_path.exists():
                npz_path.unlink()
    finally:
        writer.close()

    if remove_unsharded:
        for child in sorted(output_root.iterdir()):
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
        for leftover in output_root.rglob("*.npz"):
            leftover.unlink(missing_ok=True)

    return sharded, writer.shard_idx


def _iter_selected_specs(
    all_specs: Sequence[DatasetSpec],
    requested: str | None,
) -> list[DatasetSpec]:
    if not requested:
        return list(all_specs)
    requested_lower = requested.strip().lower()
    selected = [s for s in all_specs if s.name.lower() == requested_lower]
    if not selected:
        known = ", ".join(s.name for s in all_specs)
        raise ValueError(f"Unknown dataset '{requested}'. Available: {known}")
    return selected


def prepare_test_datasets(
    input_root: Path,
    output_root: Path,
    manifest_path: Path,
    row_log_path: Path,
    size: int,
    split_name: str,
    dataset_name: str | None,
    fake_only: bool,
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

    mode_counts: dict[str, int] = {"resize": 0, "pad": 0, "resize_then_pad": 0}

    for spec in specs:
        dataset_root = input_root / spec.name
        if not dataset_root.exists():
            print(f"[WARN] Missing dataset folder: {dataset_root}", file=sys.stderr)
            continue

        mask_index: dict[str, Path] = {}
        if spec.mask_dir:
            mask_index = build_mask_index(dataset_root / spec.mask_dir, IMAGE_EXTENSIONS)

        entries: list[tuple[str, Path, Path | None, int]] = []

        if spec.real_dir:
            if fake_only:
                print(f"[{spec.name}] Skipping real images (fake-only mode)")
            else:
                for image_path in discover_images(
                    dataset_root / spec.real_dir,
                    IMAGE_EXTENSIONS,
                    return_empty_if_missing=True,
                ):
                    entries.append(("real", image_path, None, 0))

        if spec.fake_dir:
            for image_path in discover_images(dataset_root / spec.fake_dir, IMAGE_EXTENSIONS, return_empty_if_missing=True):
                mask_path = resolve_mask_from_candidates(
                    image_stem=image_path.stem,
                    mask_index=mask_index,
                    candidates=tuple(spec.mask_stem_candidates(image_path.stem)),
                )
                if mask_path is None:
                    skipped_missing_mask += 1
                    msg = f"[WARN] Missing mask for fake image: {image_path}"
                    if fail_on_missing_mask:
                        raise FileNotFoundError(msg)
                    print(msg, file=sys.stderr)
                    continue
                entries.append(("fake", image_path, mask_path, 1))

        if max_samples > 0:
            entries = entries[:max_samples]

        dataset_records: list[SampleRecord] = []

        for idx, (kind, image_path, mask_path, label) in enumerate(
            tqdm(entries, desc=f"{spec.name}", leave=False)
        ):
            rel_source = image_path.relative_to(dataset_root).as_posix()
            stem = _safe_name(Path(rel_source).with_suffix("").as_posix())
            sample_id = f"{spec.name.lower()}_{kind}_{idx:06d}_{stem}"
            npz_path = output_root / spec.name / f"{sample_id}.npz"

            image_np, original_hw, preproc_mode = _process_image(image_path, size)
            mode_counts[preproc_mode] += 1

            mask_np: np.ndarray | None = None
            if mask_path is not None:
                mask_np = _process_mask(mask_path, size, original_hw, preproc_mode)

            metadata: dict[str, object] = {
                "dataset": spec.name,
                "split": split_name,
                "kind": kind,
                "label": label,
                "original_image_path": str(image_path),
                "original_mask_path": str(mask_path) if mask_path is not None else None,
                "original_size_hw": list(original_hw),
                "processed_size_hw": [size, size],
                "processed_sample_path": str(npz_path),
                "preproc_mode": preproc_mode,
                "mask_is_all_zero": mask_np is None,
                "mask_foreground_pixels": int(mask_np.sum()) if mask_np is not None else 0,
                "storage": "npz",
            }

            _save_npz(
                npz_path=npz_path,
                image_np=image_np,
                mask_np=mask_np,
                metadata=metadata,
            )

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
            dataset_records.append(record)

        real_count = sum(1 for r in dataset_records if r.label == 0)
        fake_count = sum(1 for r in dataset_records if r.label == 1)
        suffix = "  (no real_dir)" if not spec.real_dir else ""
        print(
            f"[{spec.name}] real={real_count}  fake={fake_count}"
            f"  total={len(dataset_records)}{suffix}"
        )
        records.extend(dataset_records)

    records, tar_count = _shard_records(
        records=records,
        output_root=output_root,
        shard_size=tar_shard_size,
        remove_unsharded=remove_unsharded_after_shard,
    )

    if row_log_path.exists():
        row_log_path.unlink()
    for rec in records:
        _append_manifest_row(row_log_path, rec.to_dict())

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([rec.to_dict() for rec in records]).to_parquet(manifest_path, index=False)

    print(f"\nWrote manifest:    {manifest_path}")
    print(f"Wrote row log:     {row_log_path}")
    print(f"Total samples:     {len(records)}")
    print(
        f"Preproc modes:     resize={mode_counts['resize']}"
        f"  pad={mode_counts['pad']}"
        f"  resize_then_pad={mode_counts['resize_then_pad']}"
    )
    if tar_shard_size > 0:
        print(f"Tar shards:        {tar_count}")
    if skipped_missing_mask:
        print(f"Skipped (no mask): {skipped_missing_mask}", file=sys.stderr)

    return Manifest(samples=records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare benchmark test datasets")
    parser.add_argument(
        "--input-root",
        type=str,
        default="./test_datasets",
        help="Root folder containing one subfolder per dataset (CASIA1/, Columbia/, COVERAGE/)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="./prepared_test_datasets",
        help="Output root for prepared NPZ samples and tar shards",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Manifest parquet path (default: <output-root>/manifest.parquet)",
    )
    parser.add_argument(
        "--row-log",
        type=str,
        default=None,
        help="JSONL row log path (default: <output-root>/manifest_rows.jsonl)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=448,
        help="Target square size - images resized or padded to size x size (default: 448)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split name written into the manifest (default: test)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Single dataset to process: CASIA1, Columbia, COVERAGE (default: all)",
    )
    parser.add_argument(
        "--fake-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Process fake images only and skip real images (default: false)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Cap samples per dataset after discovery, 0 = all",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output root before writing (clean re-run from scratch)",
    )
    parser.add_argument(
        "--fail-on-missing-mask",
        action="store_true",
        help="Raise immediately when a fake image has no matching mask (default: warn and skip)",
    )
    parser.add_argument(
        "--tar-shard-size",
        type=int,
        default=1024,
        help="Samples per tar shard, 0 disables sharding (default: 1024)",
    )
    parser.add_argument(
        "--remove-unsharded-after-shard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete local NPZ files after packing into tar shards (default: true)",
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
        fake_only=bool(args.fake_only),
        max_samples=int(args.max_samples),
        clean_output=bool(args.clean),
        fail_on_missing_mask=bool(args.fail_on_missing_mask),
        tar_shard_size=int(args.tar_shard_size),
        remove_unsharded_after_shard=bool(args.remove_unsharded_after_shard),
    )


if __name__ == "__main__":
    main()
