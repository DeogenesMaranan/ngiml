from __future__ import annotations

import argparse
import os

from src.training_types import TrainConfig


def parse_args() -> TrainConfig:
    default_workers = max(4, (os.cpu_count() or 4))
    parser = argparse.ArgumentParser(description="Train NGIML manipulation localization")
    parser.add_argument("--scheduler-type", type=str, default="cosine", choices=["cosine", "step"], help="LR scheduler type (cosine or step)")
    parser.add_argument("--manifest", required=True, help="Path to prepared manifest JSON")
    parser.add_argument("--output-dir", default="runs/ngiml", help="Directory to write checkpoints/logs")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--num-workers", type=int, default=default_workers, help="DataLoader workers")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable DataLoader pinned memory")
    parser.add_argument("--no-channels-last", action="store_true", help="Disable channels-last memory format on CUDA")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile on model")
    parser.add_argument("--compile-mode", type=str, default="default", help="torch.compile mode")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic kernels (slower)")
    parser.add_argument("--no-tf32", action="store_true", help="Disable TF32 matrix math on CUDA")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Numerical precision for training")
    parser.add_argument("--debug-timing", action="store_true", help="Enable lightweight per-stage timing prints during training")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True, help="Enable gradient checkpointing for memory savings")
    parser.add_argument(
        "--cuda-expandable-segments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True when CUDA is used",
    )
    parser.add_argument("--no-lr-schedule", action="store_true", help="Disable warmup+cosine LR schedule")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Number of warmup epochs (linear, default=5)")
    parser.add_argument("--min-lr-scale", type=float, default=0.1, help="Initial LR scale for warmup (default=0.1)")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm; <=0 disables")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
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
    parser.add_argument(
        "--balance-real-fake",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable weighted sampling to match a target fake-positive ratio in train batches",
    )
    parser.add_argument(
        "--balanced-positive-ratio",
        type=float,
        default=0.6,
        help="Target fake-positive sampling ratio when --balance-real-fake is enabled",
    )
    parser.add_argument(
        "--balanced-sampler-seed",
        type=int,
        default=42,
        help="Random seed used by the real/fake balanced sampler",
    )
    parser.add_argument(
        "--balanced-sampler-num-samples",
        type=int,
        default=None,
        help="Optional number of sampled training items per epoch for real/fake balancing",
    )
    parser.add_argument("--prefetch-factor", type=int, default=1, help="DataLoader prefetch factor")
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable persistent DataLoader workers",
    )
    parser.add_argument(
        "--multiprocessing-context",
        type=str,
        default=None,
        choices=["fork", "spawn", "forkserver"],
        help="Optional DataLoader multiprocessing context (recommended: spawn on notebook runtimes)",
    )
    parser.add_argument(
        "--drop-last",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop last incomplete batch in training",
    )
    parser.add_argument(
        "--auto-local-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically materialize tar::npz samples to local cache before training",
    )
    parser.add_argument(
        "--local-cache-dir",
        type=str,
        default=None,
        help="Directory for local materialized samples (defaults to output_dir/local_cache)",
    )
    parser.add_argument(
        "--reuse-local-cache-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing local cached manifest when available to shorten startup",
    )
    parser.add_argument("--views-per-sample", type=int, default=2, help="Number of augmented views per sample (on-the-fly)")
    parser.add_argument("--gpu-aug-batch-chunk-size", type=int, default=1, help="Chunk size for GPU-side batched augmentations; smaller uses less memory")
    parser.add_argument("--resize-max-side", type=int, default=448, help="Cap image short side before batching (lower is faster)")
    parser.add_argument("--max-rotation-degrees", type=float, default=6.0, help="Random rotation range (+/-)")
    parser.add_argument("--noise-std-max", type=float, default=0.01, help="Max Gaussian noise std")
    parser.add_argument("--disable-aug", action="store_true", help="Disable GPU augmentations")
    parser.add_argument("--device", type=str, default=None, help="Override device (e.g., cuda:0 or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for reproducibility")
    parser.add_argument(
        "--aug-seed",
        type=int,
        default=None,
        help="Augmentation RNG seed override (defaults to --seed when omitted)",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=5, help="Stop after N validations without improvement; <=0 disables")
    parser.add_argument("--early-stopping-min-delta", type=float, default=3e-3, help="Minimum monitored-metric improvement to reset early stopping")
    parser.add_argument(
        "--early-stopping-monitor",
        type=str,
        default="loss",
        choices=["loss", "iou", "f1", "recall", "precision", "accuracy"],
        help="Validation metric used for early stopping and best checkpoint",
    )
    parser.add_argument(
        "--monitor-source-policy",
        type=str,
        default="best",
        choices=["best", "ema", "raw"],
        help="Source for early-stopping monitor and best_checkpoint selection when EMA is enabled",
    )
    parser.add_argument(
        "--overlap-source-policy",
        type=str,
        default="best",
        choices=["best", "ema", "raw"],
        help="Source for best_f1_iou_checkpoint selection when EMA is enabled",
    )
    parser.add_argument("--metric-threshold", type=float, default=0.5, help="Fixed threshold for sigmoid outputs when threshold optimization is disabled")
    parser.add_argument("--optimize-threshold", action=argparse.BooleanOptionalAction, default=False, help="Search validation thresholds and use the best for metric reporting")
    parser.add_argument("--threshold-metric", type=str, default="f1", choices=["iou", "f1"], help="Metric used to select best threshold")
    parser.add_argument("--threshold-start", type=float, default=0.2, help="Threshold search range start")
    parser.add_argument("--threshold-end", type=float, default=0.8, help="Threshold search range end")
    parser.add_argument("--threshold-step", type=float, default=0.02, help="Threshold search step size")
    parser.add_argument("--small-mask-ratio-max", type=float, default=0.01, help="Upper foreground-ratio bound for small-mask validation bin")
    parser.add_argument("--medium-mask-ratio-max", type=float, default=0.05, help="Upper foreground-ratio bound for medium-mask validation bin")
    parser.add_argument("--compute-foreground-ratio", action=argparse.BooleanOptionalAction, default=True, help="Compute foreground pixel ratio from train loader")
    parser.add_argument(
        "--foreground-ratio-max-batches",
        type=int,
        default=40,
        help="Max train batches sampled when computing foreground pixel ratio",
    )
    parser.add_argument(
        "--short-side-probe-samples",
        type=int,
        default=128,
        help="Max samples per split to probe on disk for size bucketing (0 disables probing)",
    )
    parser.add_argument("--auto-pos-weight", action=argparse.BooleanOptionalAction, default=True, help="Auto-compute BCE pos_weight from foreground ratio")
    parser.add_argument("--pos-weight-min", type=float, default=0.5, help="Lower clamp for auto pos_weight")
    parser.add_argument("--pos-weight-max", type=float, default=10.0, help="Upper clamp for auto pos_weight")
    parser.add_argument(
        "--balanced-pos-weight-cap",
        type=float,
        default=0.0,
        help="When --balance-real-fake is enabled, cap auto pos_weight to this value (<=0 disables cap)",
    )
    parser.add_argument("--dice-weight", type=float, default=1.0, help="Dice loss weight")
    parser.add_argument("--bce-weight", type=float, default=1.0, help="BCE term weight (0 disables)")
    parser.add_argument("--focal-weight", type=float, default=0.0, help="Focal term weight (0 disables)")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument("--focal-alpha", type=float, default=0.25, help="Focal loss alpha")
    parser.add_argument("--tversky-weight", type=float, default=0.2, help="Optional Tversky loss weight to improve recall")
    parser.add_argument("--tversky-alpha", type=float, default=0.3, help="Tversky alpha (FP penalty)")
    parser.add_argument("--tversky-beta", type=float, default=0.8, help="Tversky beta (FN penalty)")
    parser.add_argument("--lovasz-weight", type=float, default=0.05, help="Lovasz Hinge Loss weight for IoU optimization")
    parser.add_argument("--boundary-weight", type=float, default=0.03, help="Boundary loss weight (0 disables)")
    parser.add_argument("--ema-enabled", action=argparse.BooleanOptionalAction, default=True, help="Use EMA weights for validation and best checkpoints")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay factor")
    parser.add_argument("--hard-mining-enabled", action=argparse.BooleanOptionalAction, default=False, help="Enable low-IoU hard-example weighting")
    parser.add_argument("--hard-mining-start-epoch", type=int, default=5, help="Epoch to start hard-example weighting")
    parser.add_argument("--hard-mining-weight", type=float, default=0.03, help="Weight of hard-example auxiliary loss")
    parser.add_argument("--hard-mining-gamma", type=float, default=2.0, help="Scale for low-IoU hard-example weights")
    parser.add_argument(
        "--decoder-block-type",
        type=str,
        default="conv",
        choices=["conv", "mbconv"],
        help="UNet decoder block family used in bottleneck and upsampling decode blocks",
    )
    parser.add_argument(
        "--mbconv-expand-ratio",
        type=int,
        default=4,
        help="Expansion ratio used when --decoder-block-type=mbconv",
    )
    parser.add_argument(
        "--mbconv-use-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable residual shortcut inside MBConv blocks when shapes match",
    )
    args = parser.parse_args()
    resolved_resize_max_side = max(64, int(args.resize_max_side))

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
        cuda_expandable_segments=args.cuda_expandable_segments,
        lr_schedule=not args.no_lr_schedule,
        warmup_epochs=args.warmup_epochs,
        min_lr_scale=args.min_lr_scale,
        grad_clip=args.grad_clip,
        grad_accum_steps=max(1, int(args.grad_accum_steps)),
        val_every=args.val_every,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        auto_resume=args.auto_resume,
        round_robin_seed=args.round_robin_seed,
        balance_sampling=args.balance_sampling,
        balance_real_fake=args.balance_real_fake,
        balanced_positive_ratio=args.balanced_positive_ratio,
        balanced_sampler_seed=args.balanced_sampler_seed,
        balanced_sampler_num_samples=args.balanced_sampler_num_samples,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        multiprocessing_context=args.multiprocessing_context,
        drop_last=args.drop_last,
        auto_local_cache=args.auto_local_cache,
        local_cache_dir=args.local_cache_dir,
        reuse_local_cache_manifest=args.reuse_local_cache_manifest,
        views_per_sample=args.views_per_sample,
        gpu_aug_batch_chunk_size=int(args.gpu_aug_batch_chunk_size),
        resize_max_side=resolved_resize_max_side,
        max_rotation_degrees=args.max_rotation_degrees,
        noise_std_max=args.noise_std_max,
        disable_aug=args.disable_aug,
        device=args.device,
        aug_seed=args.aug_seed,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_monitor=args.early_stopping_monitor,
        monitor_source_policy=args.monitor_source_policy,
        overlap_source_policy=args.overlap_source_policy,
        metric_threshold=args.metric_threshold,
        optimize_threshold=args.optimize_threshold,
        threshold_metric=args.threshold_metric,
        threshold_start=args.threshold_start,
        threshold_end=args.threshold_end,
        threshold_step=args.threshold_step,
        small_mask_ratio_max=args.small_mask_ratio_max,
        medium_mask_ratio_max=args.medium_mask_ratio_max,
        compute_foreground_ratio=args.compute_foreground_ratio,
        foreground_ratio_max_batches=max(0, int(args.foreground_ratio_max_batches)),
        short_side_probe_samples=max(0, int(args.short_side_probe_samples)),
        auto_pos_weight=args.auto_pos_weight,
        pos_weight_min=args.pos_weight_min,
        pos_weight_max=args.pos_weight_max,
        balanced_pos_weight_cap=args.balanced_pos_weight_cap,
        dice_weight=args.dice_weight,
        bce_weight=args.bce_weight,
        focal_weight=args.focal_weight,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        tversky_weight=args.tversky_weight,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        lovasz_weight=args.lovasz_weight,
        boundary_weight=args.boundary_weight,
        ema_enabled=args.ema_enabled,
        ema_decay=args.ema_decay,
        hard_mining_enabled=args.hard_mining_enabled,
        hard_mining_start_epoch=args.hard_mining_start_epoch,
        hard_mining_weight=args.hard_mining_weight,
        hard_mining_gamma=args.hard_mining_gamma,
        decoder_block_type=args.decoder_block_type,
        mbconv_expand_ratio=max(1, int(args.mbconv_expand_ratio)),
        mbconv_use_residual=args.mbconv_use_residual,
        scheduler_type=args.scheduler_type,
        precision=args.precision,
        gradient_checkpointing=args.gradient_checkpointing,
    )
