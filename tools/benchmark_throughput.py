"""Lightweight dataloader and transfer benchmark for NGIML.

This script measures DataLoader throughput, CPU->device transfer time, and
prints simple recommendations. It gracefully falls back to CPU-only machines.

Usage examples:
  python tools/benchmark_throughput.py --manifest prepared/CASIA2/manifest.json --batch-size 8 --num-workers 8 --batches 50

"""
from __future__ import annotations

import argparse
import time
import json
import sys
from pathlib import Path
import subprocess

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataloaders import create_dataloaders


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark dataloader throughput and transfer times")
    p.add_argument("--manifest", required=True, help="Path to prepared manifest (json/parquet)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=max(4, (torch.get_num_threads() or 4)))
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true", default=True)
    p.add_argument("--batches", type=int, default=50, help="Number of timed batches to sample")
    p.add_argument("--warmup", type=int, default=10, help="Number of warmup batches before timing")
    p.add_argument("--device", type=str, default=None, help="Device override (e.g., cuda:0)")
    return p.parse_args()


def sample_nvidia_smi(duration_s: float = 1.0, interval: float = 0.2) -> dict:
    # Try to sample GPU utilization via nvidia-smi if available
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory", "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL)
        lines = out.decode("utf-8").strip().splitlines()
        vals = [tuple(int(x.strip()) for x in l.split(",")) for l in lines if l.strip()]
        # return list of per-gpu (gpu_util, mem_util)
        return {"per_gpu": vals}
    except Exception:
        return {"per_gpu": None}


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Benchmark device: {device}")

    print("Creating dataloaders (this may take a moment)...")
    loaders = create_dataloaders(
        Path(args.manifest),
        per_dataset_augmentations={},
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True,
    )

    if "train" not in loaders:
        print("Manifest has no 'train' split; aborting.")
        return 2

    dl = loaders["train"]

    # Warmup
    print(f"Warming up {args.warmup} batches...")
    it = iter(dl)
    for i in range(min(args.warmup, args.batches)):
        try:
            batch = next(it)
        except StopIteration:
            break

    # Timed sampling
    print(f"Timing {args.batches} batches (moving to {device})...")
    it = iter(dl)
    timings = []
    transfer_times = []
    batch_sizes = []

    for i in range(args.batches):
        try:
            t0 = time.time()
            batch = next(it)
            t1 = time.time()
        except StopIteration:
            break

        # measure move to device
        t_move0 = time.time()
        if device.type == "cuda":
            for k in ("images", "masks", "high_pass"):
                v = batch.get(k)
                if isinstance(v, torch.Tensor):
                    v = v.to(device, non_blocking=True)
        else:
            # no-op for CPU
            pass
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_move1 = time.time()

        timings.append(t1 - t0)
        transfer_times.append(t_move1 - t_move0)
        batch_sizes.append(getattr(batch.get("images"), "shape", (None,))[0] if batch.get("images") is not None else 0)

    n = len(timings)
    if n == 0:
        print("No batches measured; check manifest and dataset size.")
        return 3

    avg_load = sum(timings) / n
    avg_transfer = sum(transfer_times) / n
    avg_batch = sum(batch_sizes) / max(1, sum(1 for s in batch_sizes if s))

    results = {
        "device": str(device),
        "batches_sampled": n,
        "avg_collate_load_sec": avg_load,
        "avg_transfer_sec": avg_transfer,
        "batches_per_sec_including_transfer": 1.0 / (avg_load + avg_transfer) if (avg_load + avg_transfer) > 0 else None,
        "suggestions": [],
    }

    # Simple suggestions
    if device.type == "cuda":
        results["suggestions"].append("Use bf16 on A100 ( --precision bf16 ) if supported")
        results["suggestions"].append("Try increasing --num-workers and --prefetch-factor until CPU keeps up")
        results["suggestions"].append("Enable --pin-memory and use non_blocking transfers (already used by this project)")
        results["suggestions"].append("Consider --compile (torch.compile) for faster kernels if using PyTorch 2.x")
    else:
        results["suggestions"].append("No CUDA detected: benchmark measured CPU-only pipeline. Increase --num-workers for throughput or run on GPU for faster training.")

    # Try quick nvidia-smi sample
    if device.type == "cuda":
        results["nvidia_smi"] = sample_nvidia_smi()

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
