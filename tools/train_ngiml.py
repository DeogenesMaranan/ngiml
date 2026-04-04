"""End-to-end NGIML training loop with checkpointing."""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Optional


import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataloaders import (
    AugmentationConfig,
    _apply_gpu_augmentations_batch,
    _normalize,
)
from src.checkpoint_utils import (
    append_checkpoint_log,
    disable_pretrained_backbones_for_checkpoint_load,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
    select_highest_resolution_head,
)
from src.model.hybrid_ngiml import HybridNGIML, HybridNGIMLConfig
from src.model.losses import MultiStageManipulationLoss
from src.training_cli import parse_args as _parse_args
from src.training_types import TrainConfig
from src.training_loop_helpers import (
    PrefetchLoader,
    _build_lr_scheduler,
    _build_threshold_grid,
    _coerce_loss_config,
    _coerce_model_config,
    _empty_bin_stats,
    _finalize_bin_stats,
    _initial_best_for_monitor,
    _metric_for_monitor,
    _metrics_from_counts,
    _monitor_improved,
    _parity_check,
    _prepare_dataloaders,
    _print_and_validate_train_dataset_integrity,
    _print_resolved_config_summary,
    _resolve_cuda_runtime_stability,
    _resolve_manifest_for_training,
    _segmentation_counts,
    _select_threshold_with_precision_guard,
    _set_backbone_trainability_for_epoch,
    _should_disable_compile_for_device,
    _size_bin_name,
    _validate_startup_config,
    _write_best_threshold_metadata,
    format_status_flags,
    infer_loader_total_batches,
    resolve_gpu_aug_chunk_size,
    should_chunk_gpu_aug,
    to_float_label_ratio,
)

def parse_args() -> TrainConfig:
    return _parse_args()


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


@torch.inference_mode()
def compute_foreground_pixel_ratio(loader, max_batches: int | None = 200) -> float:
    """Compute foreground pixel ratio with optional batch sampling and progress prints.

    This avoids iterating the entire dataset in slow or memory-constrained environments.
    """
    foreground = 0.0
    total = 0.0
    for i, batch in enumerate(loader):
        masks = batch["masks"]
        masks = (masks > 0.5).float()
        foreground += float(masks.sum().item())
        total += float(masks.numel())
        if (i + 1) % 10 == 0:
            print(f"Foreground sampling: processed {i+1} batches")
        if max_batches is not None and (i + 1) >= int(max_batches):
            print(f"Foreground sampling: reached max_batches={max_batches}, stopping early")
            break
    if total <= 0:
        return 0.0
    return foreground / total


def _init_ema_model(model: HybridNGIML, model_cfg: HybridNGIMLConfig, enabled: bool) -> Optional[HybridNGIML]:
    if not enabled:
        return None
    ema_model = HybridNGIML(model_cfg)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ema_model


@torch.no_grad()
def _update_ema_model(ema_model: Optional[HybridNGIML], model: HybridNGIML, decay: float) -> None:
    if ema_model is None:
        return
    decay = float(min(max(decay, 0.0), 0.999999))
    msd = model.state_dict()
    for key, value in ema_model.state_dict().items():
        model_value = msd[key].detach()
        if not torch.is_floating_point(value):
            value.copy_(model_value)
        else:
            value.mul_(decay).add_(model_value, alpha=1.0 - decay)


def train_one_epoch(
    model: HybridNGIML,
    loader,
    optimizer,
    scaler: GradScaler,
    loss_fn,
    device: torch.device,
    cfg: TrainConfig,
    epoch: int,
    global_step: int,
    ema_model: Optional[HybridNGIML] = None,
    per_dataset_aug: Optional[Dict[str, AugmentationConfig]] = None,
    normalization_mode: Optional[str] = None,
):
    """Run one training epoch with GPU-side augmentations and EMA updates."""
    model.train()
    running_loss = 0.0
    num_batches = 0
    sampled_pos = 0.0
    sampled_total = 0.0
    accum_steps = max(1, int(cfg.grad_accum_steps))
    if device.type == "cuda":
        loader = PrefetchLoader(loader, device)

    total = infer_loader_total_batches(loader)

    progress = tqdm(loader, desc=f"Epoch {epoch:03d}", leave=False, dynamic_ncols=True, total=total)
    imagenet_mean = None
    imagenet_std = None
    if device.type == "cuda" and normalization_mode == "imagenet":
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(progress):
        batch_start = time.perf_counter()
        images = batch["images"]
        masks = batch["masks"]
        if images.device != device:
            images = images.to(device, non_blocking=True)
        if masks.device != device:
            masks = masks.to(device, non_blocking=True)
        residual_noise = batch.get("residual_noise")
        if isinstance(residual_noise, torch.Tensor):
            if residual_noise.device != device:
                residual_noise = residual_noise.to(device, non_blocking=True)
        else:
            residual_noise = None
        aug_start = None
        forward_start = None
        backward_end = None
        opt_end = None
        if device.type == "cuda" and per_dataset_aug is not None and normalization_mode is not None:
            aug_start = time.perf_counter()
            try:
                gen = torch.Generator(device=device)
            except TypeError:
                gen = torch.Generator()
            seed_base = int(cfg.aug_seed) if cfg.aug_seed is not None else int(cfg.seed)
            try:
                gen.manual_seed(seed_base + int(global_step))
            except Exception:
                gen.manual_seed(seed_base)

            datasets_list = batch.get("datasets", None)
            if datasets_list is not None:
                bsz = images.shape[0]
                idxs_by_ds: dict[str, list[int]] = {}
                for i in range(bsz):
                    ds_name = str(datasets_list[i])
                    idxs_by_ds.setdefault(ds_name, []).append(i)

                for ds_name, idxs in idxs_by_ds.items():
                    aug_cfg = per_dataset_aug.get(ds_name, None)
                    if aug_cfg is None or not getattr(aug_cfg, "enable", False):
                        if normalization_mode == "imagenet":
                            mean = imagenet_mean.view(1, 3, 1, 1)
                            std = imagenet_std.view(1, 3, 1, 1)
                            images[idxs] = (images[idxs] - mean) / std
                        continue

                    aug_chunk = resolve_gpu_aug_chunk_size(
                        len(idxs),
                        int(getattr(cfg, "gpu_aug_batch_chunk_size", 0)),
                    )
                    use_chunking = should_chunk_gpu_aug(len(idxs), aug_chunk)

                    if not use_chunking:
                        img_slice = images[idxs]
                        mask_slice = masks[idxs]
                        hp_slice = None
                        if residual_noise is not None:
                            hp_slice = residual_noise[idxs]

                        img_out, mask_out, hp_out = _apply_gpu_augmentations_batch(
                            img_slice,
                            mask_slice,
                            aug_cfg,
                            residual_noise=hp_slice,
                            generator=gen,
                        )

                        if normalization_mode == "imagenet":
                            mean = imagenet_mean.view(1, 3, 1, 1)
                            std = imagenet_std.view(1, 3, 1, 1)
                            img_out = (img_out - mean) / std

                        images[idxs] = img_out
                        masks[idxs] = mask_out
                        if hp_out is not None and residual_noise is not None:
                            residual_noise[idxs] = hp_out
                    else:
                        for chunk_start in range(0, len(idxs), aug_chunk):
                            chunk_idxs = idxs[chunk_start : chunk_start + aug_chunk]
                            img_slice = images[chunk_idxs]
                            mask_slice = masks[chunk_idxs]
                            hp_slice = None
                            if residual_noise is not None:
                                hp_slice = residual_noise[chunk_idxs]

                            img_out, mask_out, hp_out = _apply_gpu_augmentations_batch(
                                img_slice,
                                mask_slice,
                                aug_cfg,
                                residual_noise=hp_slice,
                                generator=gen,
                            )

                            if normalization_mode == "imagenet":
                                mean = imagenet_mean.view(1, 3, 1, 1)
                                std = imagenet_std.view(1, 3, 1, 1)
                                img_out = (img_out - mean) / std

                            images[chunk_idxs] = img_out
                            masks[chunk_idxs] = mask_out
                            if hp_out is not None and residual_noise is not None:
                                residual_noise[chunk_idxs] = hp_out
            aug_end = time.perf_counter()
        else:
            aug_end = None
        labels = batch["labels"]
        pos_count, total_count = to_float_label_ratio(labels)
        sampled_pos += pos_count
        sampled_total += total_count
        if cfg.channels_last and device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
            if residual_noise is not None:
                residual_noise = residual_noise.contiguous(memory_format=torch.channels_last)

        precision_name = (getattr(cfg, "precision", "fp32") or "fp32").lower()
        amp_dtype = torch.bfloat16 if precision_name == "bf16" else (torch.float16 if precision_name == "fp16" else None)
        use_amp = cfg.amp and device.type == "cuda" and (amp_dtype is not None)
        if amp_dtype is not None:
            forward_start = time.perf_counter()
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                preds = model(images, target_size=masks.shape[-2:], residual_noise=residual_noise)
                loss = loss_fn(preds, masks)
            forward_end = time.perf_counter()
        else:
            forward_start = time.perf_counter()
            preds = model(images, target_size=masks.shape[-2:], residual_noise=residual_noise)
            loss = loss_fn(preds, masks)
            forward_end = time.perf_counter()

        if cfg.hard_mining_enabled and epoch >= int(max(0, cfg.hard_mining_start_epoch)):
            final_logits = select_highest_resolution_head(preds)
            if final_logits.shape[-2:] != masks.shape[-2:]:
                final_logits = torch.nn.functional.interpolate(
                    final_logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            bce_per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
                final_logits,
                masks,
                reduction="none",
            ).mean(dim=(1, 2, 3))

            with torch.no_grad():
                pred_bin = (torch.sigmoid(final_logits) >= 0.5).float()
                tp = (pred_bin * masks).sum(dim=(1, 2, 3))
                fp = (pred_bin * (1.0 - masks)).sum(dim=(1, 2, 3))
                fn = ((1.0 - pred_bin) * masks).sum(dim=(1, 2, 3))
                iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
                difficulty = (1.0 - iou).clamp(0.0, 1.0)
                hard_weights = 1.0 + float(max(0.0, cfg.hard_mining_gamma)) * difficulty
                hard_weights = hard_weights / hard_weights.mean().clamp_min(1e-6)

            hard_loss = (hard_weights * bce_per_sample).mean()
            loss = loss + float(max(0.0, cfg.hard_mining_weight)) * hard_loss

        scaled_loss = loss / accum_steps
        backward_start = time.perf_counter()
        scaler.scale(scaled_loss).backward()
        backward_end = time.perf_counter()

        do_step = ((step + 1) % accum_steps == 0) or ((step + 1) == len(loader))

        if do_step:
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            opt_end = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            _update_ema_model(ema_model, model, cfg.ema_decay)

        running_loss += loss.item()
        num_batches += 1
        global_step += 1

        if cfg.debug_timing:
            batch_end = time.perf_counter()
            batch_time = batch_end - batch_start
            aug_time = (aug_end - aug_start) if (aug_start is not None and aug_end is not None) else 0.0
            forward_time = (forward_end - forward_start) if (forward_start is not None and forward_end is not None) else 0.0
            backward_time = (backward_end - backward_start) if (backward_end is not None and backward_start is not None) else 0.0
            opt_time = (opt_end - backward_end) if (opt_end is not None and backward_end is not None) else 0.0
            if num_batches % 50 == 0 or num_batches == 1:
                print(
                    f"[timing] batch={num_batches} total={batch_time:.3f}s aug={aug_time:.3f}s forward={forward_time:.3f}s backward={backward_time:.3f}s opt={opt_time:.3f}s"
                )

        avg_loss = running_loss / max(1, num_batches)
        progress.set_postfix(loss=f"{avg_loss:.4f}", step=f"{step:05d}", accum=f"{accum_steps}")

    sampled_positive_ratio = sampled_pos / max(sampled_total, 1.0)
    return running_loss / max(1, num_batches), global_step, sampled_positive_ratio


@torch.inference_mode()
def evaluate(model: HybridNGIML, loader, loss_fn, device: torch.device, cfg: TrainConfig, normalization_mode: Optional[str] = None) -> dict:
    """Evaluate model loss and segmentation metrics on a validation loader."""
    model.eval()
    total_loss = 0.0
    batches = 0
    thresholds = _build_threshold_grid(cfg) if cfg.optimize_threshold else [float(cfg.metric_threshold)]
    threshold_stats = {
        float(th): {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
        for th in thresholds
    }
    threshold_bin_stats = {
        float(th): _empty_bin_stats()
        for th in thresholds
    }
    imagenet_mean = None
    imagenet_std = None
    if device.type == "cuda" and normalization_mode == "imagenet":
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    progress = tqdm(loader, desc="Validation", leave=False, dynamic_ncols=True)
    for batch in progress:
        images = batch["images"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        residual_noise = batch.get("residual_noise")
        if isinstance(residual_noise, torch.Tensor):
            residual_noise = residual_noise.to(device, non_blocking=True)
        else:
            residual_noise = None
        if device.type == "cuda" and normalization_mode is not None:
            bsz = images.shape[0]
            for i in range(bsz):
                images[i] = _normalize(images[i], normalization_mode, imagenet_mean=imagenet_mean, imagenet_std=imagenet_std)
        if cfg.channels_last and device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
            if residual_noise is not None:
                residual_noise = residual_noise.contiguous(memory_format=torch.channels_last)
        precision_name = (getattr(cfg, "precision", "fp32") or "fp32").lower()
        amp_dtype = torch.bfloat16 if precision_name == "bf16" else (torch.float16 if precision_name == "fp16" else None)
        use_amp = cfg.amp and device.type == "cuda" and (amp_dtype is not None)
        if amp_dtype is not None:
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                preds = model(images, target_size=masks.shape[-2:], residual_noise=residual_noise)
                loss = loss_fn(preds, masks)
        else:
            preds = model(images, target_size=masks.shape[-2:], residual_noise=residual_noise)
            loss = loss_fn(preds, masks)
        logits = select_highest_resolution_head(preds)

        with torch.no_grad():
            fg_ratio = masks.float().mean(dim=(1, 2, 3))
            size_bin_idx = _size_bin_name(fg_ratio, cfg)

        for threshold in thresholds:
            counts = _segmentation_counts(logits, masks, threshold=threshold)
            threshold_stats[float(threshold)]["tp"] += counts["tp"]
            threshold_stats[float(threshold)]["tn"] += counts["tn"]
            threshold_stats[float(threshold)]["fp"] += counts["fp"]
            threshold_stats[float(threshold)]["fn"] += counts["fn"]

            pred = (torch.sigmoid(logits) >= float(threshold)).float()
            target = masks.float()
            tp_b = (pred * target).sum(dim=(1, 2, 3))
            tn_b = ((1.0 - pred) * (1.0 - target)).sum(dim=(1, 2, 3))
            fp_b = (pred * (1.0 - target)).sum(dim=(1, 2, 3))
            fn_b = ((1.0 - pred) * target).sum(dim=(1, 2, 3))

            bin_names = ["small", "medium", "large"]
            for bin_id, bin_name in enumerate(bin_names):
                mask_sel = size_bin_idx == bin_id
                if not torch.any(mask_sel):
                    continue
                stats = threshold_bin_stats[float(threshold)][bin_name]
                stats["tp"] += float(tp_b[mask_sel].sum().item())
                stats["tn"] += float(tn_b[mask_sel].sum().item())
                stats["fp"] += float(fp_b[mask_sel].sum().item())
                stats["fn"] += float(fn_b[mask_sel].sum().item())
                stats["count"] += float(mask_sel.sum().item())

        total_loss += loss.item()
        batches += 1
        progress.set_postfix(loss=f"{(total_loss / max(1, batches)):.4f}", step=f"{batches:05d}")

    optimize_key = cfg.threshold_metric.lower()
    if optimize_key not in {"iou", "f1"}:
        optimize_key = "f1"

    scored_thresholds: list[tuple[float, dict]] = []
    for threshold in thresholds:
        stats = threshold_stats[float(threshold)]
        metrics = _metrics_from_counts(
            stats["tp"],
            stats["tn"],
            stats["fp"],
            stats["fn"],
        )
        scored_thresholds.append((float(threshold), metrics))

    if cfg.optimize_threshold:
        best_threshold, best_metrics = _select_threshold_with_precision_guard(scored_thresholds, optimize_key=optimize_key)
    else:
        fixed_threshold = float(cfg.metric_threshold)
        nearest_threshold, best_metrics = min(scored_thresholds, key=lambda item: abs(item[0] - fixed_threshold))
        best_threshold = float(nearest_threshold)

    best_bin_metrics = _finalize_bin_stats(threshold_bin_stats[float(best_threshold)])

    normalizer = max(1, batches)
    return {
        "loss": total_loss / normalizer,
        "iou": float(best_metrics["iou"]),
        "precision": float(best_metrics["precision"]),
        "recall": float(best_metrics["recall"]),
        "f1": float(best_metrics["f1"]),
        "accuracy": float(best_metrics["accuracy"]),
        "threshold": float(best_threshold),
        "threshold_metric": optimize_key,
        "size_bins": best_bin_metrics,
    }


def run_training(cfg: TrainConfig) -> None:
    """Run full NGIML training."""
    set_global_seed(cfg.seed, deterministic=cfg.deterministic)
    startup_t0 = time.time()

    if cfg.cuda_expandable_segments and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if torch.cuda.is_available():
        cfg = replace(cfg, device="cuda")
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    cfg = _resolve_cuda_runtime_stability(cfg, device)
    if _should_disable_compile_for_device(cfg, device):
        try:
            total_gib = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        except Exception:
            total_gib = 0.0
        cfg = replace(cfg, compile_model=False)
        print(
            "torch.compile disabled on this CUDA device to avoid long warmup and high host RAM usage | "
            f"detected_vram={total_gib:.1f} GiB"
        )

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high" if cfg.use_tf32 else "highest")

        cudnn_backend = getattr(torch.backends, "cudnn", None)
        cuda_backend = getattr(torch.backends, "cuda", None)
        cudnn_conv = getattr(cudnn_backend, "conv", None)
        cuda_matmul = getattr(cuda_backend, "matmul", None)
        if cudnn_conv is not None and hasattr(cudnn_conv, "fp32_precision"):
            cudnn_conv.fp32_precision = "tf32" if cfg.use_tf32 else "ieee"
        elif cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
            cudnn_backend.allow_tf32 = cfg.use_tf32

        if cuda_matmul is not None and hasattr(cuda_matmul, "fp32_precision"):
            cuda_matmul.fp32_precision = "tf32" if cfg.use_tf32 else "ieee"
        elif cuda_matmul is not None and hasattr(cuda_matmul, "allow_tf32"):
            cuda_matmul.allow_tf32 = cfg.use_tf32

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_manifest = _resolve_manifest_for_training(cfg, out_dir)
    t_after_manifest = time.time()
    if resolved_manifest != Path(cfg.manifest):
        cfg = replace(cfg, manifest=str(resolved_manifest))

    _, normalization_mode_checked = _validate_startup_config(cfg, Path(cfg.manifest), device)
    _print_resolved_config_summary(cfg, normalization_mode_checked)
    _parity_check(cfg, Path(cfg.manifest), normalization_mode_checked)

    _print_and_validate_train_dataset_integrity(Path(cfg.manifest))

    loaders, per_dataset_aug, normalization_mode = _prepare_dataloaders(cfg, device)
    t_after_dataloaders = time.time()
    if "train" not in loaders:
        raise ValueError("Train split missing in manifest; cannot start training")
    if cfg.balance_real_fake:
        print(
            "Train sampler: round-robin + real/fake balanced | "
            f"target_positive_ratio={cfg.balanced_positive_ratio:.3f} | "
            f"num_samples={cfg.balanced_sampler_num_samples or 'dataset_len'}"
        )

    foreground_ratio = None
    if cfg.compute_foreground_ratio:
        sampled_batches = cfg.foreground_ratio_max_batches if cfg.foreground_ratio_max_batches > 0 else None
        if sampled_batches is None:
            print("Computing foreground pixel ratio (sampling full train loader)...")
        else:
            print(f"Computing foreground pixel ratio (sampling up to {sampled_batches} batches)...")
        foreground_ratio = compute_foreground_pixel_ratio(loaders["train"], max_batches=sampled_batches)
        print(f"Foreground pixel ratio (train): {foreground_ratio:.6f}")

    start_epoch = 0
    global_step = 0
    resume_path: Optional[Path] = None
    if cfg.resume:
        resume_path = Path(cfg.resume)
    elif cfg.auto_resume:
        resume_path = find_latest_checkpoint(out_dir)
        if resume_path is not None:
            print(f"Auto-resume selected latest checkpoint: {resume_path}")

    model_cfg = _coerce_model_config(cfg.model_config)
    if resume_path is not None and resume_path.is_file():
        model_cfg = disable_pretrained_backbones_for_checkpoint_load(model_cfg)
    base_loss_cfg = _coerce_loss_config(cfg.loss_config)
    cfg = replace(cfg, model_config=model_cfg, loss_config=base_loss_cfg)

    try:
        from dataclasses import replace as _dc_replace

        model_cfg = _dc_replace(model_cfg, gradient_checkpointing=cfg.gradient_checkpointing)
    except Exception:
        model_cfg.gradient_checkpointing = cfg.gradient_checkpointing

    model = HybridNGIML(model_cfg).to(device)
    if cfg.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    optimizer = model.build_optimizer()
    scheduler = _build_lr_scheduler(optimizer, cfg)
    requested_precision = str(cfg.precision).lower()
    if requested_precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        print("[Warning] bf16 precision requested but not supported on this device. Falling back to fp16 with AMP enabled.")
        cfg = replace(cfg, precision="fp16", amp=True)
    scaler = GradScaler(enabled=(str(cfg.precision).lower() == "fp16" and cfg.amp and device.type == "cuda"))
    ema_model = _init_ema_model(model, model_cfg, cfg.ema_enabled)
    if ema_model is not None:
        ema_model = ema_model.to(device)
        if cfg.channels_last and device.type == "cuda":
            ema_model = ema_model.to(memory_format=torch.channels_last)
    loss_cfg = replace(
        base_loss_cfg,
        hybrid_mode=cfg.loss_hybrid_mode,
        dice_weight=cfg.dice_weight,
        bce_weight=cfg.bce_weight,
        focal_gamma=cfg.focal_gamma,
        focal_alpha=cfg.focal_alpha,
        tversky_weight=cfg.tversky_weight,
        tversky_alpha=cfg.tversky_alpha,
        tversky_beta=cfg.tversky_beta,
        lovasz_weight=cfg.lovasz_weight,
        use_boundary_loss=cfg.use_boundary_loss,
        boundary_weight=cfg.boundary_weight,
    )
    if cfg.auto_pos_weight and foreground_ratio is not None:
        ratio = max(1e-6, min(1.0 - 1e-6, foreground_ratio))
        pos_weight = (1.0 - ratio) / ratio
        pos_weight = float(min(max(pos_weight, cfg.pos_weight_min), cfg.pos_weight_max))
        if cfg.balance_real_fake:
            cap = float(getattr(cfg, "balanced_pos_weight_cap", 0.0))
            if cap > 0:
                capped = min(pos_weight, cap)
                if capped < pos_weight:
                    print(
                        "Balanced class sampling is enabled; capping auto pos_weight "
                        f"from {pos_weight:.4f} to {capped:.4f}"
                    )
                pos_weight = capped
        loss_cfg = replace(loss_cfg, pos_weight=pos_weight)
        print(f"Auto pos_weight from foreground ratio: {pos_weight:.4f}")
    else:
        fixed_pos_weight = float(getattr(loss_cfg, "pos_weight", 1.0))
        print(f"Using fixed pos_weight from loss config: {fixed_pos_weight:.4f}")
    loss_fn = MultiStageManipulationLoss(loss_cfg)
    t_after_model = time.time()

    print(
        "Startup timings | "
        f"manifest/cache {t_after_manifest - startup_t0:.1f}s | "
        f"dataloaders {t_after_dataloaders - t_after_manifest:.1f}s | "
        f"model+optim {t_after_model - t_after_dataloaders:.1f}s | "
        f"total {t_after_model - startup_t0:.1f}s"
    )

    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_log_path = checkpoint_dir / "checkpoint_metrics.json"

    restored_training_state: dict = {}
    if resume_path:
        if resume_path.is_file():
            start_epoch, global_step, restored_training_state = load_checkpoint(
                resume_path,
                model,
                optimizer,
                scaler,
                device,
                scheduler=scheduler,
                ema_model=ema_model,
            )
            print(f"Resumed from {resume_path} at epoch {start_epoch} step {global_step}")
        else:
            print(f"Resume path {resume_path} not found; starting fresh")
    elif cfg.auto_resume:
        print("Auto-resume enabled but no checkpoint found; starting fresh")

    if cfg.compile_model:
        if ema_model is not None:
            print("torch.compile skipped because EMA is enabled (keeps EMA/state_dict keys consistent)")
        elif hasattr(torch, "compile"):
            if device.type == "cuda" and cfg.compile_mode == "reduce-overhead":
                try:
                    import torch._inductor.config as inductor_config

                    if hasattr(inductor_config, "triton") and hasattr(inductor_config.triton, "cudagraphs"):
                        inductor_config.triton.cudagraphs = False
                        print("Disabled Triton CUDA graphs for reduce-overhead compile mode to reduce memory pressure")
                except Exception:
                    pass
            model = torch.compile(model, mode=cfg.compile_mode)
            print(f"torch.compile enabled with mode={cfg.compile_mode}")
        else:
            print("torch.compile requested but not available in this torch build")

    with open(out_dir / "train_config.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2)

    best_monitor_value = float(
        restored_training_state.get(
            "best_monitor_value",
            _initial_best_for_monitor(cfg.early_stopping_monitor),
        )
    )
    best_val_iou = float(restored_training_state.get("best_val_iou", float("-inf")))
    best_val_f1 = float(restored_training_state.get("best_val_f1", float("-inf")))
    no_improve_epochs = int(restored_training_state.get("no_improve_epochs", 0))
    early_stopping_enabled = "val" in loaders and cfg.early_stopping_patience > 0
    best_threshold_path = checkpoint_dir / "best_threshold.json"

    freeze_backbone_epochs = int(max(0, getattr(model_cfg.optimizer, "freeze_backbone_epochs", 0)))
    backbone_phase: Optional[str] = None

    for epoch in range(start_epoch, cfg.epochs):
        current_backbone_phase = _set_backbone_trainability_for_epoch(
            model,
            epoch=epoch,
            freeze_backbone_epochs=freeze_backbone_epochs,
        )
        if current_backbone_phase != backbone_phase:
            if current_backbone_phase == "frozen":
                print(
                    "Backbone freeze enabled: freezing EfficientNet/Swin "
                    f"for first {freeze_backbone_epochs} epochs"
                )
            elif current_backbone_phase == "all-trainable":
                print("Backbone freeze finished: fully unfreezing EfficientNet/Swin")
            else:
                print(
                    "Backbone staged unfreeze: "
                    f"{current_backbone_phase} backbone groups trainable"
                )
            backbone_phase = current_backbone_phase

        start_time = time.time()
        train_loss, global_step, train_positive_ratio = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            scaler,
            loss_fn,
            device,
            cfg,
            epoch,
            global_step,
            ema_model=ema_model,
            per_dataset_aug=per_dataset_aug,
            normalization_mode=normalization_mode,
        )

        elapsed = time.time() - start_time
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1:03d}/{cfg.epochs:03d} Train | "
                f"loss {train_loss:.4f} | pos_ratio {train_positive_ratio:.3f} | "
                f"lr {current_lr:.6e} | time {elapsed:.1f}s"
            )
        else:
            print(
                f"Epoch {epoch + 1:03d}/{cfg.epochs:03d} Train | "
                f"loss {train_loss:.4f} | pos_ratio {train_positive_ratio:.3f} | time {elapsed:.1f}s"
            )

        val_loss = None
        val_iou = None
        val_f1 = None
        val_precision = None
        val_recall = None
        val_accuracy = None
        val_threshold = None
        val_size_bins = None
        epoch_status_flags: list[str] = []
        if "val" in loaders and (epoch + 1) % cfg.val_every == 0:
            eval_model = ema_model if ema_model is not None else model
            metrics = evaluate(eval_model, loaders["val"], loss_fn, device, cfg, normalization_mode=normalization_mode)
            val_loss = float(metrics["loss"])
            val_iou = float(metrics["iou"])
            val_f1 = float(metrics["f1"])
            val_precision = float(metrics["precision"])
            val_recall = float(metrics["recall"])
            val_accuracy = float(metrics["accuracy"])
            val_threshold = float(metrics["threshold"])
            val_size_bins = metrics.get("size_bins")
            val_summary = (
                f"Epoch {epoch + 1:03d}/{cfg.epochs:03d} Val   | "
                f"loss {val_loss:.4f} | iou {val_iou:.4f} | f1 {val_f1:.4f} | "
                f"prec {val_precision:.4f} | rec {val_recall:.4f} | acc {val_accuracy:.4f} | "
                f"thr {val_threshold:.2f}"
            )
            print(
                val_summary
            )
            if isinstance(val_size_bins, dict):
                small_iou = float(val_size_bins.get("small", {}).get("iou", 0.0))
                medium_iou = float(val_size_bins.get("medium", {}).get("iou", 0.0))
                large_iou = float(val_size_bins.get("large", {}).get("iou", 0.0))
                print(
                    "               bins | "
                    f"small {small_iou:.4f} | medium {medium_iou:.4f} | large {large_iou:.4f}"
                )

            iou_improved = val_iou > (best_val_iou + cfg.early_stopping_min_delta)
            f1_improved = val_f1 > (best_val_f1 + cfg.early_stopping_min_delta)
            if iou_improved:
                best_val_iou = val_iou
            if f1_improved:
                best_val_f1 = val_f1

            overlap_improved = iou_improved or f1_improved
            if overlap_improved:
                best_f1_iou_path = checkpoint_dir / "best_f1_iou_checkpoint.pt"
                save_checkpoint(
                    best_f1_iou_path,
                    model,
                    optimizer,
                    scaler,
                    epoch + 1,
                    global_step,
                    cfg,
                    scheduler=scheduler,
                    ema_model=ema_model,
                    use_ema_for_model_state=(ema_model is not None),
                    training_state={
                        "best_monitor_value": best_monitor_value,
                        "best_val_iou": best_val_iou,
                        "best_val_f1": best_val_f1,
                        "no_improve_epochs": no_improve_epochs,
                    },
                )
                epoch_status_flags.append(f"best-overlap -> {best_f1_iou_path.name}")

            monitor_value = _metric_for_monitor(metrics, cfg.early_stopping_monitor)
            monitor_improved = _monitor_improved(
                cfg.early_stopping_monitor,
                monitor_value,
                best_monitor_value,
                cfg.early_stopping_min_delta,
            )

            if monitor_improved:
                monitor_for_metadata = str(getattr(cfg, "early_stopping_monitor", "loss")).strip().lower()
                monitor_value_for_metadata = _metric_for_monitor(metrics, monitor_for_metadata)
                best_alias_path = checkpoint_dir / "best_checkpoint.pt"
                save_checkpoint(
                    best_alias_path,
                    model,
                    optimizer,
                    scaler,
                    epoch + 1,
                    global_step,
                    cfg,
                    scheduler=scheduler,
                    ema_model=ema_model,
                    use_ema_for_model_state=(ema_model is not None),
                    training_state={
                        "best_monitor_value": best_monitor_value,
                        "best_val_iou": best_val_iou,
                        "best_val_f1": best_val_f1,
                        "no_improve_epochs": no_improve_epochs,
                    },
                )
                if cfg.optimize_threshold:
                    _ = _write_best_threshold_metadata(
                        best_threshold_path,
                        epoch=epoch + 1,
                        threshold=val_threshold,
                        threshold_metric=str(metrics.get("threshold_metric", cfg.threshold_metric)),
                        monitor=monitor_for_metadata,
                        monitor_value=monitor_value_for_metadata,
                        metrics=metrics,
                        checkpoint_path=best_alias_path,
                    )
                epoch_status_flags.append(
                    f"best-{monitor_for_metadata} {monitor_value_for_metadata:.4f} -> {best_alias_path.name}"
                )

            if monitor_improved:
                best_monitor_value = monitor_value
                no_improve_epochs = 0
                epoch_status_flags.append(f"monitor improved ({cfg.early_stopping_monitor}={monitor_value:.4f})")
            else:
                if early_stopping_enabled:
                    no_improve_epochs += 1
                    epoch_status_flags.append(
                        f"patience {no_improve_epochs}/{cfg.early_stopping_patience}"
                    )

        should_checkpoint = ((epoch + 1) % cfg.checkpoint_every == 0) or (epoch + 1 == cfg.epochs)
        if should_checkpoint:
            ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                scaler,
                epoch + 1,
                global_step,
                cfg,
                scheduler=scheduler,
                ema_model=ema_model,
                use_ema_for_model_state=False,
                training_state={
                    "best_monitor_value": best_monitor_value,
                    "best_val_iou": best_val_iou,
                    "best_val_f1": best_val_f1,
                    "no_improve_epochs": no_improve_epochs,
                },
            )
            append_checkpoint_log(
                checkpoint_log_path,
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "train_loss": float(train_loss),
                    "train_positive_ratio": float(train_positive_ratio),
                    "val_loss": val_loss,
                    "val_iou": val_iou,
                    "val_f1": val_f1,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_accuracy": val_accuracy,
                    "val_threshold": val_threshold,
                    "val_size_bins": val_size_bins,
                    "epoch_seconds": float(elapsed),
                    "checkpoint_path": str(ckpt_path),
                },
            )
            epoch_status_flags.append(f"checkpoint {ckpt_path.name}")

        if epoch_status_flags:
            print(f"               status | {format_status_flags(epoch_status_flags)}")

        if early_stopping_enabled and "val" in loaders and (epoch + 1) % cfg.val_every == 0:
            if no_improve_epochs >= cfg.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    print("Training complete")


if __name__ == "__main__":
    configuration = parse_args()
    run_training(configuration)