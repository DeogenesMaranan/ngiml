from __future__ import annotations

import hashlib
import io
import json
import math
import re
import shutil
import tarfile
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.data.dataloaders import AugmentationConfig, create_dataloaders, load_manifest
from src.model.hybrid_ngiml import HybridNGIML, HybridNGIMLConfig
from src.model.losses import MultiStageLossConfig
from src.training_defaults import _coerce_aug as _shared_coerce_aug, build_default_components as _build_default_components
from src.training_types import TrainConfig

class PrefetchLoader:
    """Simple async prefetcher that moves next batch to CUDA in a background stream."""

    def __init__(self, loader, device: torch.device):
        self._loader = loader
        self._device = device
        self._stream = None

    def __iter__(self):
        if self._device.type != "cuda":
            return iter(self._loader)
        self._iter = iter(self._loader)
        self._stream = torch.cuda.Stream()
        self._next_batch = None
        self._preload()
        return self

    def __next__(self):
        if self._device.type != "cuda":
            return next(self._iter)
        if self._next_batch is None:
            raise StopIteration
        torch.cuda.current_stream().wait_stream(self._stream)
        batch = self._next_batch
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                value.record_stream(torch.cuda.current_stream())
        self._preload()
        return batch

    def _preload(self):
        try:
            nxt = next(self._iter)
        except StopIteration:
            self._next_batch = None
            return

        with torch.cuda.stream(self._stream):
            for k, v in list(nxt.items()):
                if isinstance(v, torch.Tensor):
                    try:
                        nxt[k] = v.to(self._device, non_blocking=True)
                    except Exception:
                        nxt[k] = v.to(self._device)
        self._next_batch = nxt

    def __len__(self):
        try:
            return len(self._loader)
        except Exception:
            raise TypeError("wrapped loader has no __len__")

    def __getattr__(self, name: str):
        return getattr(self._loader, name)


def format_status_flags(flags: Sequence[str]) -> str:
    compact = [str(flag).strip() for flag in flags if str(flag).strip()]
    return " | ".join(compact) if compact else "none"


def should_chunk_gpu_aug(group_size: int, chunk_size: int) -> bool:
    if int(chunk_size) <= 0:
        return False
    return int(chunk_size) < int(group_size)


def resolve_gpu_aug_chunk_size(group_size: int, chunk_size: int) -> int:
    if int(chunk_size) <= 0:
        return int(group_size)
    return max(1, int(chunk_size))


def to_float_label_ratio(labels: torch.Tensor) -> tuple[float, float]:
    labels_f = labels.float()
    positives = float((labels_f >= 0.5).sum().item())
    total = float(labels_f.numel())
    return positives, total


def _safe_len(obj) -> Optional[int]:
    if obj is None:
        return None
    try:
        value = len(obj)
    except Exception:
        return None
    try:
        value_int = int(value)
    except Exception:
        return None
    return value_int if value_int >= 0 else None


def infer_loader_total_batches(loader) -> Optional[int]:
    total = _safe_len(loader)
    base_loader = getattr(loader, "_loader", loader)

    if total is None:
        total = _safe_len(base_loader)

    if total is None:
        batch_sampler = getattr(base_loader, "batch_sampler", None)
        total = _safe_len(batch_sampler)
        if total is None and batch_sampler is not None:
            base_sampler = getattr(batch_sampler, "base_sampler", None)
            batch_size = int(getattr(batch_sampler, "batch_size", 0) or 0)
            if base_sampler is not None and batch_size > 0:
                sample_count = _safe_len(base_sampler)
                if sample_count is not None:
                    drop_last = bool(getattr(batch_sampler, "drop_last", False))
                    total = (sample_count // batch_size) if drop_last else ((sample_count + batch_size - 1) // batch_size)

    if total is None:
        sampler = getattr(base_loader, "sampler", None)
        batch_size = int(getattr(base_loader, "batch_size", 0) or 0)
        if sampler is not None and batch_size > 0:
            sample_count = _safe_len(sampler)
            if sample_count is not None:
                drop_last = bool(getattr(base_loader, "drop_last", False))
                total = (sample_count // batch_size) if drop_last else ((sample_count + batch_size - 1) // batch_size)

    return total


def _segmentation_counts(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = (probs >= float(threshold)).float()
    target = target.float()

    tp = torch.sum(pred * target).item()
    tn = torch.sum((1.0 - pred) * (1.0 - target)).item()
    fp = torch.sum(pred * (1.0 - target)).item()
    fn = torch.sum((1.0 - pred) * target).item()
    return {"tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn)}


def _select_pred_head(preds: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return the highest-resolution decoder prediction tensor."""
    if not preds:
        raise ValueError("Model returned empty predictions list")
    return preds[0]


def _metrics_from_counts(tp: float, tn: float, fp: float, fn: float, eps: float = 1e-6) -> Dict[str, float]:
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def _build_threshold_grid(cfg: TrainConfig) -> Sequence[float]:
    start = float(min(max(cfg.threshold_start, 0.0), 1.0))
    end = float(min(max(cfg.threshold_end, 0.0), 1.0))
    step = float(max(cfg.threshold_step, 1e-6))
    if end < start:
        start, end = end, start

    values = []
    t = start
    while t <= (end + 1e-9):
        values.append(round(t, 4))
        t += step

    if not values:
        values = [0.5]

    if 0.5 not in values:
        values.append(0.5)
    values = sorted(set(values))
    return values


def _select_threshold_with_precision_guard(
    scored_thresholds: Sequence[tuple[float, dict]],
    optimize_key: str,
    min_precision: float = 0.1,
    min_recall: float = 0.05,
    metric_tolerance: float = 0.98,
    cold_start_metric_floor: float = 1e-4,
) -> tuple[float, dict]:
    if not scored_thresholds:
        raise ValueError("No scored thresholds provided")

    metric_key = optimize_key if optimize_key in {"iou", "f1"} else "f1"
    baseline_threshold, baseline_metrics = max(scored_thresholds, key=lambda item: item[1][metric_key])
    baseline_metric = float(baseline_metrics[metric_key])

    if baseline_metric <= float(cold_start_metric_floor):
        neutral_threshold, neutral_metrics = min(scored_thresholds, key=lambda item: abs(float(item[0]) - 0.5))
        return float(neutral_threshold), neutral_metrics

    eligible = [
        (threshold, metrics)
        for threshold, metrics in scored_thresholds
        if (
            float(metrics.get("precision", 0.0)) >= float(min_precision)
            and float(metrics.get("recall", 0.0)) >= float(min_recall)
        )
    ]
    if not eligible:
        return float(baseline_threshold), baseline_metrics

    metric_floor = baseline_metric * float(metric_tolerance)
    close = [
        (threshold, metrics)
        for threshold, metrics in eligible
        if float(metrics[metric_key]) >= metric_floor
    ]
    candidate_pool = close if close else eligible

    selected_threshold, selected_metrics = max(
        candidate_pool,
        key=lambda item: (float(item[1][metric_key]), float(item[1].get("precision", 0.0))),
    )
    return float(selected_threshold), selected_metrics


def _metric_for_monitor(metrics: dict, monitor: str) -> float:
    key = str(monitor).strip().lower()
    if key not in metrics:
        raise KeyError(f"Unsupported monitor metric: {monitor}")
    return float(metrics[key])


def _monitor_improved(monitor: str, current: float, best: float, min_delta: float) -> bool:
    key = str(monitor).strip().lower()
    delta = float(min_delta)
    if key == "loss":
        return float(current) < (float(best) - delta)
    return float(current) > (float(best) + delta)


def _initial_best_for_monitor(monitor: str) -> float:
    key = str(monitor).strip().lower()
    if key == "loss":
        return float("inf")
    return float("-inf")


def _size_bin_name(fg_ratio: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    small_max = float(max(0.0, cfg.small_mask_ratio_max))
    medium_max = float(max(small_max, cfg.medium_mask_ratio_max))
    bins = torch.full_like(fg_ratio, 2, dtype=torch.long)
    bins = torch.where(fg_ratio <= small_max, torch.zeros_like(bins), bins)
    bins = torch.where((fg_ratio > small_max) & (fg_ratio <= medium_max), torch.ones_like(bins), bins)
    return bins


def _empty_bin_stats() -> Dict[str, Dict[str, float]]:
    return {
        "small": {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0, "count": 0.0},
        "medium": {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0, "count": 0.0},
        "large": {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0, "count": 0.0},
    }


def _finalize_bin_stats(bin_stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name, stats in bin_stats.items():
        metrics = _metrics_from_counts(stats["tp"], stats["tn"], stats["fp"], stats["fn"])
        out[name] = {
            "count": float(stats["count"]),
            "iou": float(metrics["iou"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "accuracy": float(metrics["accuracy"]),
        }
    return out


def _write_best_threshold_metadata(
    path: Path,
    *,
    epoch: int,
    threshold: float | None,
    threshold_metric: str,
    monitor: str,
    monitor_value: float,
    metrics: dict,
    checkpoint_path: Path,
) -> dict:
    payload = {
        "epoch": int(epoch),
        "checkpoint_path": str(checkpoint_path),
        "threshold": float(threshold) if threshold is not None else None,
        "threshold_metric": str(threshold_metric),
        "monitor": str(monitor),
        "monitor_value": float(monitor_value),
        "val_iou": float(metrics.get("iou")) if metrics.get("iou") is not None else None,
        "val_f1": float(metrics.get("f1")) if metrics.get("f1") is not None else None,
        "val_precision": float(metrics.get("precision")) if metrics.get("precision") is not None else None,
        "val_recall": float(metrics.get("recall")) if metrics.get("recall") is not None else None,
        "val_accuracy": float(metrics.get("accuracy")) if metrics.get("accuracy") is not None else None,
        "val_size_bins": metrics.get("size_bins"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def _cuda_supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            pass
    try:
        major, _minor = torch.cuda.get_device_capability()
        return int(major) >= 8
    except Exception:
        return False


def _resolve_cuda_runtime_stability(cfg: TrainConfig, device: torch.device) -> TrainConfig:
    if device.type != "cuda":
        return cfg

    updates: dict[str, object] = {}
    precision = (getattr(cfg, "precision", "fp32") or "fp32").lower()
    if precision == "bf16" and not _cuda_supports_bf16():
        updates["precision"] = "fp16"

    if updates:
        resolved = replace(cfg, **updates)
        print(
            "Adjusted CUDA runtime config for stability | "
            f"precision: {cfg.precision} -> {resolved.precision}"
        )
        return resolved
    return cfg


def _disable_pretrained_backbones_for_checkpoint_load(model_cfg: HybridNGIMLConfig) -> HybridNGIMLConfig:
    """Return a config that avoids external backbone downloads during checkpoint restore."""
    cfg_out = model_cfg
    try:
        if hasattr(cfg_out.efficientnet, "pretrained"):
            cfg_out = replace(cfg_out, efficientnet=replace(cfg_out.efficientnet, pretrained=False))
    except Exception:
        try:
            cfg_out.efficientnet.pretrained = False
        except Exception:
            pass

    try:
        if hasattr(cfg_out.swin, "pretrained"):
            cfg_out = replace(cfg_out, swin=replace(cfg_out.swin, pretrained=False))
    except Exception:
        try:
            cfg_out.swin.pretrained = False
        except Exception:
            pass

    return cfg_out


def _should_disable_compile_for_device(cfg: TrainConfig, device: torch.device) -> bool:
    if not bool(getattr(cfg, "compile_model", False)):
        return False
    if device.type != "cuda":
        return False
    try:
        total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    except Exception:
        return False
    return total_memory <= (16 * 1024**3)


def _build_lr_scheduler(optimizer, cfg):
    """Builds a learning rate scheduler with optional warmup and cosine/step decay."""
    if not cfg.lr_schedule or cfg.epochs <= 1:
        return None

    warmup_epochs = max(0, min(cfg.warmup_epochs, max(cfg.epochs - 1, 0)))
    min_lr_scale = float(max(0.0, min(cfg.min_lr_scale, 1.0)))

    if getattr(cfg, "scheduler_type", "cosine") == "step":
        step_size = getattr(cfg, "step_size", 10)
        gamma = getattr(cfg, "gamma", 0.5)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    def _lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return min_lr_scale + (1.0 - min_lr_scale) * (float(epoch + 1) / float(warmup_epochs))
        cosine_total = max(cfg.epochs - warmup_epochs, 1)
        cosine_epoch = min(max(epoch - warmup_epochs, 0), cosine_total)
        cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_epoch / cosine_total))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


def _collect_dataset_names(manifest_path: Path) -> Sequence[str]:
    manifest = load_manifest(manifest_path)
    names = sorted({sample.dataset for sample in manifest.samples})
    if not names:
        raise ValueError("Manifest contains no samples")
    return names


def _coerce_model_config(value) -> HybridNGIMLConfig:
    if value is None:
        return HybridNGIMLConfig()
    if isinstance(value, HybridNGIMLConfig):
        return value
    if not isinstance(value, dict):
        raise TypeError("Model config must be HybridNGIMLConfig or dict")

    from src.model.backbones.efficientnet_backbone import EfficientNetBackboneConfig
    from src.model.backbones.residual_noise_branch import ResidualNoiseConfig
    from src.model.backbones.swin_backbone import SwinBackboneConfig
    from src.model.feature_fusion import FeatureFusionConfig
    from src.model.hybrid_ngiml import HybridNGIMLOptimizerConfig, OptimizerGroupConfig
    from src.model.unet_decoder import UNetDecoderConfig

    def _coerce_optimizer_config(opt_value) -> HybridNGIMLOptimizerConfig:
        if opt_value is None:
            return HybridNGIMLOptimizerConfig()
        if isinstance(opt_value, HybridNGIMLOptimizerConfig):
            return opt_value
        if not isinstance(opt_value, dict):
            raise TypeError("Optimizer config must be HybridNGIMLOptimizerConfig or dict")

        default_opt = HybridNGIMLOptimizerConfig()

        def _coerce_group(group_value, default_group: OptimizerGroupConfig) -> OptimizerGroupConfig:
            if isinstance(group_value, OptimizerGroupConfig):
                return group_value
            if group_value is None:
                return default_group
            if isinstance(group_value, dict):
                return OptimizerGroupConfig(**group_value)
            raise TypeError("Optimizer group config must be OptimizerGroupConfig or dict")

        betas_raw = opt_value.get("betas", default_opt.betas)
        if isinstance(betas_raw, list):
            betas = tuple(float(v) for v in betas_raw)
        else:
            betas = tuple(betas_raw)

        return HybridNGIMLOptimizerConfig(
            efficientnet=_coerce_group(opt_value.get("efficientnet"), default_opt.efficientnet),
            swin=_coerce_group(opt_value.get("swin"), default_opt.swin),
            residual=_coerce_group(opt_value.get("residual"), default_opt.residual),
            fusion=_coerce_group(opt_value.get("fusion"), default_opt.fusion),
            decoder=_coerce_group(opt_value.get("decoder"), default_opt.decoder),
            betas=betas,
            eps=float(opt_value.get("eps", default_opt.eps)),
            freeze_backbone_epochs=int(opt_value.get("freeze_backbone_epochs", default_opt.freeze_backbone_epochs)),
        )

    default_model = HybridNGIMLConfig()
    efficientnet = value.get("efficientnet", default_model.efficientnet)
    swin = value.get("swin", default_model.swin)
    residual = value.get("residual", default_model.residual)
    fusion = value.get("fusion", default_model.fusion)
    decoder = value.get("decoder", default_model.decoder)
    optimizer = value.get("optimizer", default_model.optimizer)

    return HybridNGIMLConfig(
        efficientnet=efficientnet if isinstance(efficientnet, EfficientNetBackboneConfig) else EfficientNetBackboneConfig(**efficientnet),
        swin=swin if isinstance(swin, SwinBackboneConfig) else SwinBackboneConfig(**swin),
        residual=residual if isinstance(residual, ResidualNoiseConfig) else ResidualNoiseConfig(**residual),
        fusion=fusion if isinstance(fusion, FeatureFusionConfig) else FeatureFusionConfig(**fusion),
        decoder=decoder if isinstance(decoder, UNetDecoderConfig) else UNetDecoderConfig(**decoder),
        optimizer=_coerce_optimizer_config(optimizer),
        use_low_level=bool(value.get("use_low_level", default_model.use_low_level)),
        use_context=bool(value.get("use_context", default_model.use_context)),
        use_residual=bool(value.get("use_residual", default_model.use_residual)),
        enable_residual_attention=bool(value.get("enable_residual_attention", default_model.enable_residual_attention)),
        gradient_checkpointing=bool(value.get("gradient_checkpointing", default_model.gradient_checkpointing)),
    )


def _coerce_loss_config(value) -> MultiStageLossConfig:
    if value is None:
        return MultiStageLossConfig()
    if isinstance(value, MultiStageLossConfig):
        return value
    if isinstance(value, dict):
        return MultiStageLossConfig(**value)
    raise TypeError("Loss config must be MultiStageLossConfig or dict")


def _build_aug_map(names: Sequence[str], cfg: TrainConfig) -> Dict[str, AugmentationConfig]:
    if cfg.default_aug is not None:
        base_aug = cfg.default_aug
    else:
        _, _, default_aug, _ = _build_default_components()
        base_aug = default_aug

    aug_map: Dict[str, AugmentationConfig] = {name: _shared_coerce_aug(base_aug) for name in names}

    if cfg.per_dataset_aug:
        for name, aug in cfg.per_dataset_aug.items():
            aug_map[name] = _shared_coerce_aug(aug)

    return aug_map


def _prepare_dataloaders(cfg: TrainConfig, device: torch.device):
    """Create train/val/test loaders and return runtime augmentation metadata."""
    manifest_path = Path(cfg.manifest)
    dataset_names = _collect_dataset_names(manifest_path)
    per_dataset_aug = _build_aug_map(dataset_names, cfg)
    manifest = load_manifest(manifest_path)
    normalization_mode = manifest.normalization_mode
    collate_aug_map = per_dataset_aug
    collate_norm_mode = normalization_mode
    if device.type == "cuda":
        from dataclasses import replace as _dc_replace

        collate_aug_map = {name: _dc_replace(aug, enable=False) for name, aug in per_dataset_aug.items()}
        collate_norm_mode = "zero_one"

    loaders = create_dataloaders(
        manifest_path,
        collate_aug_map,
        batch_size=cfg.batch_size,
        device=device,
        pin_memory=cfg.pin_memory,
        num_workers=cfg.num_workers,
        round_robin_seed=cfg.round_robin_seed,
        balance_sampling=cfg.balance_sampling,
        balance_real_fake=cfg.balance_real_fake,
        balanced_positive_ratio=cfg.balanced_positive_ratio,
        balanced_sampler_seed=cfg.balanced_sampler_seed,
        balanced_sampler_num_samples=cfg.balanced_sampler_num_samples,
        drop_last=cfg.drop_last,
        aug_seed=cfg.aug_seed if cfg.aug_seed is not None else cfg.seed,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=cfg.persistent_workers,
        resize_max_side=int(cfg.resize_max_side),
        short_side_probe_samples=cfg.short_side_probe_samples,
        normalization_mode_override=collate_norm_mode,
    )
    return loaders, per_dataset_aug, normalization_mode


def _safe_cache_name(spec: str) -> str:
    digest = hashlib.sha1(spec.encode("utf-8")).hexdigest()
    return f"{digest}.npz"


def _materialize_tar_npz_manifest(manifest_path: Path, cache_root: Path) -> Path:
    """Resolve tar::npz references into local files and write a cache manifest."""
    manifest = load_manifest(manifest_path)
    cache_root.mkdir(parents=True, exist_ok=True)
    samples_dir = cache_root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    tar_handles: Dict[str, tarfile.TarFile] = {}

    def _extract_if_needed(spec: str) -> str:
        if "::" not in spec or not spec.endswith(".npz"):
            return spec
        archive_path, member_name = spec.split("::", 1)
        out_path = samples_dir / _safe_cache_name(spec)
        if out_path.exists() and out_path.stat().st_size > 0:
            return str(out_path)

        tar = tar_handles.get(archive_path)
        if tar is None or tar.closed:
            tar = tarfile.open(archive_path, mode="r:*")
            tar_handles[archive_path] = tar

        member = tar.extractfile(member_name)
        if member is None:
            raise FileNotFoundError(f"Missing tar member {member_name} in {archive_path}")

        with open(out_path, "wb") as out_f:
            shutil.copyfileobj(member, out_f)
        return str(out_path)

    try:
        changed = False
        for sample in manifest.samples:
            new_image_path = _extract_if_needed(sample.image_path)
            if new_image_path != sample.image_path:
                sample.image_path = new_image_path
                changed = True

            if sample.mask_path is not None:
                new_mask_path = _extract_if_needed(sample.mask_path)
                if new_mask_path != sample.mask_path:
                    sample.mask_path = new_mask_path
                    changed = True

        resolved_manifest = cache_root / "manifest_local_cache.parquet"
        if changed or not resolved_manifest.exists():
            manifest.to_dataframe().to_parquet(resolved_manifest, index=False)
        return resolved_manifest
    finally:
        for tar in tar_handles.values():
            try:
                tar.close()
            except Exception:
                pass


def _resolve_manifest_for_training(cfg: TrainConfig, out_dir: Path) -> Path:
    manifest_path = Path(cfg.manifest)
    if not cfg.auto_local_cache:
        return manifest_path

    manifest = load_manifest(manifest_path)
    has_tar_npz = any("::" in s.image_path and s.image_path.endswith(".npz") for s in manifest.samples)
    if not has_tar_npz:
        return manifest_path

    cache_root = Path(cfg.local_cache_dir) if cfg.local_cache_dir else (out_dir / "local_cache")
    cache_root = cache_root / manifest_path.stem
    resolved_manifest = cache_root / "manifest_local_cache.parquet"
    if cfg.reuse_local_cache_manifest and resolved_manifest.exists():
        print(f"Reusing pre-materialized local cache manifest: {resolved_manifest}")
        return resolved_manifest

    print(f"Materializing tar::npz samples to local cache: {cache_root}")
    return _materialize_tar_npz_manifest(manifest_path, cache_root)


def _checkpoint_epoch(path: Path) -> int:
    match = re.search(r"checkpoint_epoch_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def _set_backbone_trainable(model: HybridNGIML, trainable: bool) -> None:
    for module_name in ("efficientnet", "swin"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = bool(trainable)


def _backbone_trainable_groups(module: nn.Module) -> list[nn.Module]:
    """Return progressively-unfreezable groups for a backbone wrapper."""
    if hasattr(module, "backbone"):
        backbone = getattr(module, "backbone")
        blocks = getattr(backbone, "blocks", None)
        if isinstance(blocks, (list, nn.ModuleList)) and len(blocks) > 0:
            groups: list[nn.Module] = []
            stem_modules = [
                child
                for name, child in backbone.named_children()
                if name != "blocks"
            ]
            if stem_modules:
                groups.append(nn.ModuleList(stem_modules))
            groups.extend(list(blocks))
            return groups
    stages = getattr(module, "stages", None)
    if isinstance(stages, (list, nn.ModuleList)) and len(stages) > 0:
        groups = []
        patch_embed = getattr(module, "patch_embed", None)
        if isinstance(patch_embed, nn.Module):
            groups.append(patch_embed)
        groups.extend(list(stages))
        return groups
    return [module]


def _set_backbone_trainability_for_epoch(
    model: HybridNGIML,
    epoch: int,
    freeze_backbone_epochs: int,
    progressive_unfreeze_epochs: int = 3,
) -> str:
    """Apply a smoother staged backbone unfreeze schedule and describe it."""
    progressive_unfreeze_epochs = max(1, int(progressive_unfreeze_epochs))
    if epoch < freeze_backbone_epochs:
        _set_backbone_trainable(model, trainable=False)
        return "frozen"

    groups_by_backbone: list[tuple[str, list[nn.Module]]] = []
    total_groups = 0
    for module_name in ("efficientnet", "swin"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        groups = _backbone_trainable_groups(module)
        groups_by_backbone.append((module_name, groups))
        total_groups += len(groups)

    if total_groups == 0:
        return "all-trainable"

    for _, groups in groups_by_backbone:
        for group in groups:
            for param in group.parameters():
                param.requires_grad = False

    unfreeze_progress = min(epoch - freeze_backbone_epochs + 1, progressive_unfreeze_epochs)
    progress_ratio = float(unfreeze_progress) / float(progressive_unfreeze_epochs)

    trainable_groups = 0
    for _, groups in groups_by_backbone:
        groups_to_enable = max(1, math.ceil(len(groups) * progress_ratio))
        for group in groups[-groups_to_enable:]:
            for param in group.parameters():
                param.requires_grad = True
        trainable_groups += groups_to_enable

    if trainable_groups >= total_groups:
        return "all-trainable"
    return f"progressive {trainable_groups}/{total_groups}"


def _sample_has_mask(record) -> bool:
    has_mask = bool(record.mask_path)
    image_path = str(record.image_path)
    if not image_path.endswith(".npz"):
        return has_mask

    try:
        if "::" in image_path:
            archive_path, member_name = image_path.split("::", 1)
            with tarfile.open(archive_path, "r:*") as tf:
                member = tf.extractfile(member_name)
                if member is None:
                    raise FileNotFoundError(f"Missing member {member_name} in {archive_path}")
                with np.load(io.BytesIO(member.read()), allow_pickle=False) as npz_data:
                    has_mask = has_mask or ("mask" in npz_data and npz_data["mask"].size > 0)
        else:
            with np.load(image_path, allow_pickle=False) as npz_data:
                has_mask = has_mask or ("mask" in npz_data and npz_data["mask"].size > 0)
    except Exception as exc:
        raise ValueError(f"Failed to inspect NPZ sample for mask field: {image_path}") from exc

    return has_mask


def _print_and_validate_train_dataset_integrity(manifest_path: Path) -> None:
    manifest = load_manifest(manifest_path)
    train_samples = [sample for sample in manifest.samples if sample.split == "train"]
    if not train_samples:
        raise ValueError("Train split has no samples; cannot start training")

    per_dataset_counts: Dict[str, int] = {}
    real_count = 0
    fake_count = 0
    mask_count = 0

    for sample in train_samples:
        per_dataset_counts[sample.dataset] = per_dataset_counts.get(sample.dataset, 0) + 1
        label = int(sample.label)
        if label == 0:
            real_count += 1
        elif label == 1:
            fake_count += 1
        else:
            raise ValueError(f"Unexpected train label {label} for sample: {sample.image_path}")

        has_mask = _sample_has_mask(sample)
        if has_mask:
            mask_count += 1

    total = len(train_samples)
    print("Train dataset integrity summary")
    print("  Per-dataset sample counts:")
    for dataset_name in sorted(per_dataset_counts):
        print(f"    {dataset_name}: {per_dataset_counts[dataset_name]}")

    fake_ratio = fake_count / max(total, 1)
    real_ratio = real_count / max(total, 1)
    print(
        "  Class ratio (real/fake): "
        f"{real_count}/{fake_count} "
        f"(real={real_ratio:.3f}, fake={fake_ratio:.3f})"
    )
    print(
        "  Coverage: "
        f"masks={100.0 * (mask_count / max(total, 1)):.1f}%"
    )

    if fake_count <= 0:
        raise ValueError(
            "Train split has no positive (fake) samples. "
            "Expected at least one sample with label=1."
        )

    minority_ratio = min(real_count, fake_count) / max(total, 1)
    if minority_ratio < 0.01:
        raise ValueError(
            "Train split class ratio is extreme "
            f"(real={real_count}, fake={fake_count}, total={total}). "
            "Please rebalance data before training."
        )


def _manifest_split_counts(manifest_path: Path) -> dict[str, int]:
    manifest = load_manifest(manifest_path)
    counts = {"train": 0, "val": 0, "test": 0}
    unknown_splits: set[str] = set()
    for sample in manifest.samples:
        split_name = str(sample.split).strip().lower()
        if split_name in counts:
            counts[split_name] += 1
        else:
            unknown_splits.add(split_name)
    if unknown_splits:
        unknown = ", ".join(sorted(unknown_splits))
        raise ValueError(
            "Manifest contains unsupported split names: "
            f"{unknown}. Expected only train/val/test."
        )
    return counts


def _validate_startup_config(cfg: TrainConfig, manifest_path: Path, device: torch.device) -> tuple[dict[str, int], str]:
    manifest = load_manifest(manifest_path)
    normalization_mode = str(manifest.normalization_mode).strip().lower()
    if normalization_mode not in {"zero_one", "imagenet"}:
        raise ValueError(
            "Manifest normalization_mode is incompatible with runtime expectations: "
            f"{manifest.normalization_mode!r}. Supported values: 'zero_one' or 'imagenet'."
        )

    if device.type == "cuda" and normalization_mode not in {"zero_one", "imagenet"}:
        raise ValueError(
            "CUDA runtime normalization path only supports 'zero_one' and 'imagenet'. "
            f"Got: {manifest.normalization_mode!r}."
        )

    if cfg.optimize_threshold:
        if cfg.threshold_step <= 0:
            raise ValueError(
                "Invalid threshold search range: threshold_step must be > 0 when optimize_threshold is enabled. "
                f"Got {cfg.threshold_step}."
            )
        if not (0.0 <= float(cfg.threshold_start) <= 1.0 and 0.0 <= float(cfg.threshold_end) <= 1.0):
            raise ValueError(
                "Invalid threshold search range: threshold_start/threshold_end must be within [0, 1]. "
                f"Got start={cfg.threshold_start}, end={cfg.threshold_end}."
            )
        if float(cfg.threshold_end) < float(cfg.threshold_start):
            raise ValueError(
                "Invalid threshold search range: threshold_end must be >= threshold_start. "
                f"Got start={cfg.threshold_start}, end={cfg.threshold_end}."
            )
    else:
        if not (0.0 <= float(cfg.metric_threshold) <= 1.0):
            raise ValueError(
                "Invalid fixed threshold: metric_threshold must be within [0, 1] when optimize_threshold is disabled. "
                f"Got {cfg.metric_threshold}."
            )

    split_counts = _manifest_split_counts(manifest_path)
    train_count = int(split_counts.get("train", 0))
    val_count = int(split_counts.get("val", 0))

    if train_count <= 0:
        raise ValueError("Manifest train split has no samples; cannot start training.")

    if cfg.val_every > 0 and val_count <= 0:
        raise ValueError(
            "Validation is enabled (val_every > 0) but manifest has no val split samples. "
            "Provide a val split or set val_every <= 0."
        )

    if cfg.early_stopping_patience > 0 and val_count <= 0:
        raise ValueError(
            "Early stopping requires validation data, but manifest has no val split samples."
        )

    if cfg.optimize_threshold and val_count <= 0:
        raise ValueError(
            "Threshold optimization requires validation data, but manifest has no val split samples."
        )

    return split_counts, normalization_mode


def _parity_check(cfg: TrainConfig, manifest_path: Path, normalization_mode: str) -> None:
    manifest = load_manifest(manifest_path)
    train_labels = [int(sample.label) for sample in manifest.samples if str(sample.split).strip().lower() == "train"]
    total = len(train_labels)
    positives = sum(1 for label in train_labels if label == 1)
    negatives = sum(1 for label in train_labels if label == 0)
    fake_ratio = (float(positives) / float(total)) if total > 0 else 0.0
    threshold_policy = "optimized" if cfg.optimize_threshold else f"fixed@{float(cfg.metric_threshold):.3f}"

    print(
        "Parity check | "
        f"normalization={normalization_mode} | "
        f"train_class_ratio(real/fake)={negatives}/{positives} (fake={fake_ratio:.3f}) | "
        f"balanced_sampler_active={bool(cfg.balance_real_fake)} | "
        f"eval_threshold_policy={threshold_policy}"
    )


def _print_resolved_config_summary(cfg: TrainConfig, normalization_mode: str) -> None:
    threshold_mode = (
        f"optimized[{cfg.threshold_metric}:{cfg.threshold_start:.2f}-{cfg.threshold_end:.2f}@{cfg.threshold_step:.3f}]"
        if cfg.optimize_threshold
        else f"fixed[{cfg.metric_threshold:.2f}]"
    )
    sampler_mode = "round_robin_balanced" if cfg.balance_real_fake else "round_robin"
    print(
        "Resolved config | "
        f"normalization={normalization_mode} | "
        f"balance_sampling={bool(cfg.balance_sampling)} | "
        f"balance_real_fake={bool(cfg.balance_real_fake)} | "
        f"sampler_mode={sampler_mode} | "
        f"threshold_mode={threshold_mode}"
    )

__all__ = [
    "PrefetchLoader",
    "format_status_flags",
    "infer_loader_total_batches",
    "resolve_gpu_aug_chunk_size",
    "should_chunk_gpu_aug",
    "to_float_label_ratio",
    "_build_lr_scheduler",
    "_build_threshold_grid",
    "_checkpoint_epoch",
    "_coerce_loss_config",
    "_coerce_model_config",
    "_disable_pretrained_backbones_for_checkpoint_load",
    "_empty_bin_stats",
    "_finalize_bin_stats",
    "_initial_best_for_monitor",
    "_metric_for_monitor",
    "_metrics_from_counts",
    "_monitor_improved",
    "_parity_check",
    "_prepare_dataloaders",
    "_print_and_validate_train_dataset_integrity",
    "_print_resolved_config_summary",
    "_resolve_cuda_runtime_stability",
    "_resolve_manifest_for_training",
    "_segmentation_counts",
    "_select_pred_head",
    "_select_threshold_with_precision_guard",
    "_set_backbone_trainability_for_epoch",
    "_should_disable_compile_for_device",
    "_size_bin_name",
    "_validate_startup_config",
    "_write_best_threshold_metadata",
]