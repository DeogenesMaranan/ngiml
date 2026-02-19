"""End-to-end NGIML training loop with checkpointing.

Run example (Colab-ready):
    python tools/train_ngiml.py --manifest /content/data/manifest.json --output-dir /content/runs

The script expects a prepared manifest (see src/data/config.py) and will
save checkpoints plus a copy of the training arguments inside the output dir.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataloaders import AugmentationConfig, create_dataloaders, load_manifest
from src.model.hybrid_ngiml import HybridNGIML, HybridNGIMLConfig
from src.model.losses import MultiStageLossConfig, MultiStageManipulationLoss


@dataclass
class TrainConfig:
    manifest: str
    output_dir: str = "runs/ngiml"
    batch_size: int = 4
    epochs: int = 10
    num_workers: int = 2
    amp: bool = True
    grad_clip: float = 1.0
    val_every: int = 1
    checkpoint_every: int = 1
    resume: Optional[str] = None
    round_robin_seed: Optional[int] = 0
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False
    drop_last: bool = True
    views_per_sample: int = 1
    max_rotation_degrees: float = 5.0
    noise_std_max: float = 0.02
    disable_aug: bool = False
    device: Optional[str] = None
    aug_seed: Optional[int] = None
    default_aug: Optional[AugmentationConfig] = None
    per_dataset_aug: Optional[Dict[str, AugmentationConfig]] = None
    model_config: Optional[HybridNGIMLConfig] = None
    loss_config: Optional[MultiStageLossConfig] = None


@dataclass
class Checkpoint:
    epoch: int
    global_step: int
    model_state: dict
    optimizer_state: dict
    scaler_state: Optional[dict]
    train_config: dict


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train NGIML manipulation localization")
    parser.add_argument("--manifest", required=True, help="Path to prepared manifest JSON")
    parser.add_argument("--output-dir", default="runs/ngiml", help="Directory to write checkpoints/logs")
    parser.add_argument("--batch-size", type=int, default=4, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm; <=0 disables")
    parser.add_argument("--val-every", type=int, default=1, help="Validate every N epochs")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Write checkpoint every N epochs (includes last epoch)",
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--round-robin-seed", type=int, default=0, help="Seed for round-robin sampler")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="DataLoader prefetch factor")
    parser.add_argument("--persistent-workers", action="store_true", help="Enable persistent workers")
    parser.add_argument("--drop-last", action="store_true", help="Drop last incomplete batch in training")
    parser.add_argument("--views-per-sample", type=int, default=1, help="Number of augmented views per sample (on-the-fly)")
    parser.add_argument("--max-rotation-degrees", type=float, default=5.0, help="Random rotation range (+/-)")
    parser.add_argument("--noise-std-max", type=float, default=0.02, help="Max Gaussian noise std")
    parser.add_argument("--disable-aug", action="store_true", help="Disable GPU augmentations")
    parser.add_argument("--device", type=str, default=None, help="Override device (e.g., cuda:0 or cpu)")
    args = parser.parse_args()
    return TrainConfig(
        manifest=args.manifest,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        grad_clip=args.grad_clip,
        val_every=args.val_every,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        round_robin_seed=args.round_robin_seed,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        drop_last=args.drop_last,
        views_per_sample=args.views_per_sample,
        max_rotation_degrees=args.max_rotation_degrees,
        noise_std_max=args.noise_std_max,
        disable_aug=args.disable_aug,
        device=args.device,
    )


def _collect_dataset_names(manifest_path: Path) -> Sequence[str]:
    manifest = load_manifest(manifest_path)
    names = sorted({sample.dataset for sample in manifest.samples})
    if not names:
        raise ValueError("Manifest contains no samples")
    return names


def _coerce_aug(value) -> AugmentationConfig:
    if isinstance(value, AugmentationConfig):
        return replace(value)
    if isinstance(value, dict):
        return AugmentationConfig(**value)
    raise TypeError("Augmentation config must be AugmentationConfig or dict")


def _build_aug_map(names: Sequence[str], cfg: TrainConfig) -> Dict[str, AugmentationConfig]:
    base_aug = cfg.default_aug or AugmentationConfig(
        enable=not cfg.disable_aug,
        views_per_sample=cfg.views_per_sample,
        enable_flips=True,
        enable_rotations=cfg.max_rotation_degrees > 0,
        max_rotation_degrees=cfg.max_rotation_degrees,
        enable_random_crop=True,
        enable_color_jitter=True,
        enable_noise=cfg.noise_std_max > 0,
        noise_std_range=(0.0, max(0.0, cfg.noise_std_max)),
    )

    aug_map: Dict[str, AugmentationConfig] = {name: _coerce_aug(base_aug) for name in names}

    if cfg.per_dataset_aug:
        for name, aug in cfg.per_dataset_aug.items():
            aug_map[name] = _coerce_aug(aug)

    return aug_map


def _prepare_dataloaders(cfg: TrainConfig, device: torch.device):
    manifest_path = Path(cfg.manifest)
    dataset_names = _collect_dataset_names(manifest_path)
    per_dataset_aug = _build_aug_map(dataset_names, cfg)
    return create_dataloaders(
        manifest_path,
        per_dataset_aug,
        batch_size=cfg.batch_size,
        device=device,
        num_workers=cfg.num_workers,
        round_robin_seed=cfg.round_robin_seed,
        drop_last=cfg.drop_last,
        aug_seed=cfg.aug_seed,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=cfg.persistent_workers,
    )


def _dice_coefficient(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    target = target.float()
    dims = (1, 2, 3)
    intersection = torch.sum(probs * target, dim=dims)
    denom = torch.sum(probs, dim=dims) + torch.sum(target, dim=dims)
    return ((2 * intersection + eps) / (denom + eps)).mean()


def save_checkpoint(path: Path, model: HybridNGIML, optimizer: torch.optim.Optimizer, scaler: GradScaler, epoch: int, global_step: int, cfg: TrainConfig) -> None:
    ckpt = Checkpoint(
        epoch=epoch,
        global_step=global_step,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scaler_state=scaler.state_dict() if scaler.is_enabled() else None,
        train_config=asdict(cfg),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt.__dict__, path)


def append_checkpoint_log(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def load_checkpoint(path: Path, model: HybridNGIML, optimizer: torch.optim.Optimizer, scaler: GradScaler, device: torch.device) -> Tuple[int, int]:
    data = torch.load(path, map_location=device)
    model.load_state_dict(data["model_state"])
    optimizer.load_state_dict(data["optimizer_state"])
    if data.get("scaler_state") and scaler.is_enabled():
        scaler.load_state_dict(data["scaler_state"])
    start_epoch = int(data.get("epoch", 0))
    global_step = int(data.get("global_step", 0))
    return start_epoch, global_step


def train_one_epoch(model: HybridNGIML, loader, optimizer, scaler: GradScaler, loss_fn, device: torch.device, cfg: TrainConfig, epoch: int, global_step: int):
    model.train()
    running_loss = 0.0
    num_batches = 0
    progress = tqdm(loader, desc=f"Epoch {epoch:03d}", leave=False, dynamic_ncols=True)
    for step, batch in enumerate(progress):
        images = batch["images"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        use_amp = cfg.amp and device.type == "cuda"
        with autocast(device_type=device.type, enabled=use_amp):
            preds = model(images, target_size=masks.shape[-2:])
            loss = loss_fn(preds, masks)
        scaler.scale(loss).backward()

        if cfg.grad_clip and cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        num_batches += 1
        global_step += 1

        avg_loss = running_loss / max(1, num_batches)
        progress.set_postfix(loss=f"{avg_loss:.4f}", step=f"{step:05d}")

    return running_loss / max(1, num_batches), global_step


@torch.no_grad()
def evaluate(model: HybridNGIML, loader, loss_fn, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    batches = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        preds = model(images, target_size=masks.shape[-2:])
        loss = loss_fn(preds, masks)
        dice = _dice_coefficient(preds[-1], masks)
        total_loss += loss.item()
        total_dice += dice.item()
        batches += 1

    normalizer = max(1, batches)
    return {"loss": total_loss / normalizer, "dice": total_dice / normalizer}


def run_training(cfg: TrainConfig) -> None:
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    loaders = _prepare_dataloaders(cfg, device)
    if "train" not in loaders:
        raise ValueError("Train split missing in manifest; cannot start training")

    model_cfg = cfg.model_config or HybridNGIMLConfig()
    model = HybridNGIML(model_cfg).to(device)
    optimizer = model.build_optimizer()
    scaler = GradScaler(device.type, enabled=(cfg.amp and device.type == "cuda"))
    loss_cfg = cfg.loss_config or MultiStageLossConfig()
    loss_fn = MultiStageManipulationLoss(loss_cfg)

    start_epoch = 0
    global_step = 0
    if cfg.resume:
        resume_path = Path(cfg.resume)
        if resume_path.is_file():
            start_epoch, global_step = load_checkpoint(resume_path, model, optimizer, scaler, device)
            print(f"Resumed from {resume_path} at epoch {start_epoch} step {global_step}")
        else:
            print(f"Resume path {resume_path} not found; starting fresh")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_log_path = checkpoint_dir / "checkpoint_metrics.jsonl"
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2)

    for epoch in range(start_epoch, cfg.epochs):
        start_time = time.time()
        train_loss, global_step = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            scaler,
            loss_fn,
            device,
            cfg,
            epoch,
            global_step,
        )

        elapsed = time.time() - start_time
        print(f"Epoch {epoch:03d} done | loss {train_loss:.4f} | time {elapsed:.1f}s")

        val_loss = None
        val_dice = None
        if "val" in loaders and (epoch + 1) % cfg.val_every == 0:
            metrics = evaluate(model, loaders["val"], loss_fn, device)
            val_loss = float(metrics["loss"])
            val_dice = float(metrics["dice"])
            print(f"Val | loss {val_loss:.4f} | dice {val_dice:.4f}")

        should_checkpoint = ((epoch + 1) % cfg.checkpoint_every == 0) or (epoch + 1 == cfg.epochs)
        if should_checkpoint:
            ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
            save_checkpoint(ckpt_path, model, optimizer, scaler, epoch + 1, global_step, cfg)
            append_checkpoint_log(
                checkpoint_log_path,
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "train_loss": float(train_loss),
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "epoch_seconds": float(elapsed),
                    "checkpoint_path": str(ckpt_path),
                },
            )
            print(f"Saved checkpoint to {ckpt_path}")

    print("Training complete")


if __name__ == "__main__":
    configuration = parse_args()
    run_training(configuration)
