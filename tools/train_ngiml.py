"""End-to-end NGIML training loop with checkpointing.

Run example (Colab-ready):
    python tools/train_ngiml.py --manifest /content/data/manifest.json --output-dir /content/runs

The script expects a prepared manifest (see src/data/config.py) and will
save checkpoints plus a copy of the training arguments inside the output dir.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple
import re
import math

import numpy as np
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
    batch_size: int = 8
    epochs: int = 50
    num_workers: int = max(2, (os.cpu_count() or 4) // 2)
    amp: bool = True
    pin_memory: bool = True
    channels_last: bool = True
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"
    deterministic: bool = False
    use_tf32: bool = True
    lr_schedule: bool = True
    warmup_epochs: int = 2
    min_lr_scale: float = 0.1
    grad_clip: float = 1.0
    val_every: int = 1
    checkpoint_every: int = 1
    resume: Optional[str] = None
    auto_resume: bool = False
    round_robin_seed: Optional[int] = 42
    balance_sampling: bool = False
    prefetch_factor: Optional[int] = 2
    persistent_workers: bool = True
    drop_last: bool = True
    views_per_sample: int = 1
    max_rotation_degrees: float = 5.0
    noise_std_max: float = 0.02
    disable_aug: bool = False
    device: Optional[str] = None
    aug_seed: Optional[int] = None
    seed: int = 42
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-4
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
    scheduler_state: Optional[dict]
    scaler_state: Optional[dict]
    train_config: dict


def parse_args() -> TrainConfig:
    default_workers = max(2, (os.cpu_count() or 4) // 2)
    parser = argparse.ArgumentParser(description="Train NGIML manipulation localization")
    parser.add_argument("--manifest", required=True, help="Path to prepared manifest JSON")
    parser.add_argument("--output-dir", default="runs/ngiml", help="Directory to write checkpoints/logs")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--num-workers", type=int, default=default_workers, help="DataLoader workers")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable DataLoader pinned memory")
    parser.add_argument("--no-channels-last", action="store_true", help="Disable channels-last memory format on CUDA")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile on model")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead", help="torch.compile mode")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic kernels (slower)")
    parser.add_argument("--no-tf32", action="store_true", help="Disable TF32 matrix math on CUDA")
    parser.add_argument("--no-lr-schedule", action="store_true", help="Disable warmup+cosine LR schedule")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Number of warmup epochs")
    parser.add_argument("--min-lr-scale", type=float, default=0.1, help="Final LR scale for cosine schedule")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm; <=0 disables")
    parser.add_argument("--val-every", type=int, default=1, help="Validate every N epochs")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Write checkpoint every N epochs (includes last epoch)",
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Automatically resume from latest checkpoint in output_dir/checkpoints when available",
    )
    parser.add_argument("--round-robin-seed", type=int, default=42, help="Seed for round-robin sampler")
    parser.add_argument(
        "--balance-sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Balance per-dataset sampling by oversampling smaller datasets",
    )
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor")
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable persistent DataLoader workers",
    )
    parser.add_argument(
        "--drop-last",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop last incomplete batch in training",
    )
    parser.add_argument("--views-per-sample", type=int, default=1, help="Number of augmented views per sample (on-the-fly)")
    parser.add_argument("--max-rotation-degrees", type=float, default=5.0, help="Random rotation range (+/-)")
    parser.add_argument("--noise-std-max", type=float, default=0.02, help="Max Gaussian noise std")
    parser.add_argument("--disable-aug", action="store_true", help="Disable GPU augmentations")
    parser.add_argument("--device", type=str, default=None, help="Override device (e.g., cuda:0 or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for reproducibility")
    parser.add_argument("--early-stopping-patience", type=int, default=8, help="Stop after N validations without improvement; <=0 disables")
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4, help="Minimum Dice improvement to reset early stopping")
    args = parser.parse_args()
    return TrainConfig(
        manifest=args.manifest,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        pin_memory=not args.no_pin_memory,
        channels_last=not args.no_channels_last,
        compile_model=args.compile,
        compile_mode=args.compile_mode,
        deterministic=args.deterministic,
        use_tf32=not args.no_tf32,
        lr_schedule=not args.no_lr_schedule,
        warmup_epochs=args.warmup_epochs,
        min_lr_scale=args.min_lr_scale,
        grad_clip=args.grad_clip,
        val_every=args.val_every,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        auto_resume=args.auto_resume,
        round_robin_seed=args.round_robin_seed,
        balance_sampling=args.balance_sampling,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        drop_last=args.drop_last,
        views_per_sample=args.views_per_sample,
        max_rotation_degrees=args.max_rotation_degrees,
        noise_std_max=args.noise_std_max,
        disable_aug=args.disable_aug,
        device=args.device,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


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
        pin_memory=cfg.pin_memory,
        num_workers=cfg.num_workers,
        round_robin_seed=cfg.round_robin_seed,
        balance_sampling=cfg.balance_sampling,
        drop_last=cfg.drop_last,
        aug_seed=cfg.aug_seed if cfg.aug_seed is not None else cfg.seed,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=cfg.persistent_workers,
    )


def _build_lr_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig):
    if not cfg.lr_schedule or cfg.epochs <= 1:
        return None

    warmup_epochs = max(0, min(cfg.warmup_epochs, max(cfg.epochs - 1, 0)))
    min_lr_scale = float(max(0.0, min(cfg.min_lr_scale, 1.0)))

    def _lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return max(1e-6, float(epoch + 1) / float(warmup_epochs))

        cosine_total = max(cfg.epochs - warmup_epochs, 1)
        cosine_epoch = min(max(epoch - warmup_epochs, 0), cosine_total)
        cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_epoch / cosine_total))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


def _dice_coefficient(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    target = target.float()
    dims = (1, 2, 3)
    intersection = torch.sum(probs * target, dim=dims)
    denom = torch.sum(probs, dim=dims) + torch.sum(target, dim=dims)
    return ((2 * intersection + eps) / (denom + eps)).mean()


def _segmentation_metrics(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = (probs >= 0.5).float()
    target = target.float()

    dims = (1, 2, 3)
    tp = torch.sum(pred * target, dim=dims)
    tn = torch.sum((1.0 - pred) * (1.0 - target), dim=dims)
    fp = torch.sum(pred * (1.0 - target), dim=dims)
    fn = torch.sum((1.0 - pred) * target, dim=dims)

    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return {
        "iou": float(iou.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
        "accuracy": float(accuracy.mean().item()),
    }


def save_checkpoint(
    path: Path,
    model: HybridNGIML,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    cfg: TrainConfig,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> None:
    ckpt = Checkpoint(
        epoch=epoch,
        global_step=global_step,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict() if scheduler is not None else None,
        scaler_state=scaler.state_dict() if scaler.is_enabled() else None,
        train_config=asdict(cfg),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt.__dict__, path)


def append_checkpoint_log(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def load_checkpoint(
    path: Path,
    model: HybridNGIML,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Tuple[int, int]:
    data = torch.load(path, map_location=device)
    model.load_state_dict(data["model_state"])
    optimizer.load_state_dict(data["optimizer_state"])
    if scheduler is not None and data.get("scheduler_state") is not None:
        scheduler.load_state_dict(data["scheduler_state"])
    if data.get("scaler_state") and scaler.is_enabled():
        scaler.load_state_dict(data["scaler_state"])
    start_epoch = int(data.get("epoch", 0))
    global_step = int(data.get("global_step", 0))
    return start_epoch, global_step


def _checkpoint_epoch(path: Path) -> int:
    match = re.search(r"checkpoint_epoch_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    checkpoint_dir = output_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"), key=_checkpoint_epoch)
    return candidates[-1] if candidates else None


def train_one_epoch(model: HybridNGIML, loader, optimizer, scaler: GradScaler, loss_fn, device: torch.device, cfg: TrainConfig, epoch: int, global_step: int):
    model.train()
    running_loss = 0.0
    num_batches = 0
    progress = tqdm(loader, desc=f"Epoch {epoch:03d}", leave=False, dynamic_ncols=True)
    for step, batch in enumerate(progress):
        images = batch["images"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        if cfg.channels_last and device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)

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


@torch.inference_mode()
def evaluate(model: HybridNGIML, loader, loss_fn, device: torch.device, cfg: TrainConfig) -> dict:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_accuracy = 0.0
    batches = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        if cfg.channels_last and device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        use_amp = cfg.amp and device.type == "cuda"
        with autocast(device_type=device.type, enabled=use_amp):
            preds = model(images, target_size=masks.shape[-2:])
            loss = loss_fn(preds, masks)
        logits = preds[-1]
        dice = _dice_coefficient(logits, masks)
        extra_metrics = _segmentation_metrics(logits, masks)
        total_loss += loss.item()
        total_dice += dice.item()
        total_iou += extra_metrics["iou"]
        total_precision += extra_metrics["precision"]
        total_recall += extra_metrics["recall"]
        total_accuracy += extra_metrics["accuracy"]
        batches += 1

    normalizer = max(1, batches)
    return {
        "loss": total_loss / normalizer,
        "dice": total_dice / normalizer,
        "iou": total_iou / normalizer,
        "precision": total_precision / normalizer,
        "recall": total_recall / normalizer,
        "accuracy": total_accuracy / normalizer,
    }


def run_training(cfg: TrainConfig) -> None:
    set_global_seed(cfg.seed, deterministic=cfg.deterministic)
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = cfg.use_tf32
        torch.backends.cudnn.allow_tf32 = cfg.use_tf32
        torch.set_float32_matmul_precision("high" if cfg.use_tf32 else "highest")

    loaders = _prepare_dataloaders(cfg, device)
    if "train" not in loaders:
        raise ValueError("Train split missing in manifest; cannot start training")

    model_cfg = cfg.model_config or HybridNGIMLConfig()
    model = HybridNGIML(model_cfg).to(device)
    if cfg.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    optimizer = model.build_optimizer()
    scheduler = _build_lr_scheduler(optimizer, cfg)
    scaler = GradScaler(device.type, enabled=(cfg.amp and device.type == "cuda"))
    loss_cfg = cfg.loss_config or MultiStageLossConfig()
    loss_fn = MultiStageManipulationLoss(loss_cfg)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_log_path = checkpoint_dir / "checkpoint_metrics.jsonl"

    start_epoch = 0
    global_step = 0
    resume_path: Optional[Path] = None
    if cfg.resume:
        resume_path = Path(cfg.resume)
    elif cfg.auto_resume:
        resume_path = find_latest_checkpoint(out_dir)
        if resume_path is not None:
            print(f"Auto-resume selected latest checkpoint: {resume_path}")

    if resume_path:
        if resume_path.is_file():
            start_epoch, global_step = load_checkpoint(resume_path, model, optimizer, scaler, device, scheduler=scheduler)
            print(f"Resumed from {resume_path} at epoch {start_epoch} step {global_step}")
        else:
            print(f"Resume path {resume_path} not found; starting fresh")
    elif cfg.auto_resume:
        print("Auto-resume enabled but no checkpoint found; starting fresh")

    if cfg.compile_model:
        if hasattr(torch, "compile"):
            model = torch.compile(model, mode=cfg.compile_mode)
            print(f"torch.compile enabled with mode={cfg.compile_mode}")
        else:
            print("torch.compile requested but not available in this torch build")

    with open(out_dir / "train_config.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2)

    best_val_dice = float("-inf")
    no_improve_epochs = 0
    early_stopping_enabled = "val" in loaders and cfg.early_stopping_patience > 0

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
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:03d} done | loss {train_loss:.4f} | lr {current_lr:.6e} | time {elapsed:.1f}s")
        else:
            print(f"Epoch {epoch:03d} done | loss {train_loss:.4f} | time {elapsed:.1f}s")

        val_loss = None
        val_dice = None
        val_iou = None
        val_precision = None
        val_recall = None
        val_accuracy = None
        if "val" in loaders and (epoch + 1) % cfg.val_every == 0:
            metrics = evaluate(model, loaders["val"], loss_fn, device, cfg)
            val_loss = float(metrics["loss"])
            val_dice = float(metrics["dice"])
            val_iou = float(metrics["iou"])
            val_precision = float(metrics["precision"])
            val_recall = float(metrics["recall"])
            val_accuracy = float(metrics["accuracy"])
            print(
                f"Val | loss {val_loss:.4f} | dice {val_dice:.4f} | iou {val_iou:.4f} "
                f"| precision {val_precision:.4f} | recall {val_recall:.4f} | accuracy {val_accuracy:.4f}"
            )

            improved = val_dice > (best_val_dice + cfg.early_stopping_min_delta)
            if improved:
                best_val_dice = val_dice
                no_improve_epochs = 0
                best_path = checkpoint_dir / "best_checkpoint.pt"
                save_checkpoint(best_path, model, optimizer, scaler, epoch + 1, global_step, cfg, scheduler=scheduler)
                print(f"New best val dice {best_val_dice:.4f}; saved to {best_path}")
            elif early_stopping_enabled:
                no_improve_epochs += 1
                print(
                    f"Early stopping patience: {no_improve_epochs}/{cfg.early_stopping_patience} "
                    f"without val dice improvement"
                )

        should_checkpoint = ((epoch + 1) % cfg.checkpoint_every == 0) or (epoch + 1 == cfg.epochs)
        if should_checkpoint:
            ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
            save_checkpoint(ckpt_path, model, optimizer, scaler, epoch + 1, global_step, cfg, scheduler=scheduler)
            append_checkpoint_log(
                checkpoint_log_path,
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "train_loss": float(train_loss),
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_accuracy": val_accuracy,
                    "epoch_seconds": float(elapsed),
                    "checkpoint_path": str(ckpt_path),
                },
            )
            print(f"Saved checkpoint to {ckpt_path}")

        if early_stopping_enabled and "val" in loaders and (epoch + 1) % cfg.val_every == 0:
            if no_improve_epochs >= cfg.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    print("Training complete")


if __name__ == "__main__":
    configuration = parse_args()
    run_training(configuration)
