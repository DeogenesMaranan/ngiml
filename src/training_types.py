from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from src.data.config import AugmentationConfig
from src.model.hybrid_ngiml import HybridNGIMLConfig
from src.model.losses import MultiStageLossConfig


@dataclass
class TrainConfig:
    manifest: str
    scheduler_type: str = "cosine"
    output_dir: str = "runs/ngiml"
    batch_size: int = 16
    epochs: int = 50
    num_workers: int = 6
    amp: bool = True
    pin_memory: bool = True
    channels_last: bool = True
    compile_model: bool = True
    compile_mode: str = "default"
    deterministic: bool = False
    use_tf32: bool = True
    precision: str = "bf16"
    gradient_checkpointing: bool = True
    cuda_expandable_segments: bool = True
    lr_schedule: bool = True
    warmup_epochs: int = 3
    min_lr_scale: float = 0.1
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    val_every: int = 1
    checkpoint_every: int = 1
    resume: Optional[str] = None
    auto_resume: bool = True
    round_robin_seed: Optional[int] = 42
    balance_sampling: bool = False
    balance_real_fake: bool = True
    balanced_positive_ratio: float = 0.6
    balanced_sampler_seed: int = 42
    balanced_sampler_num_samples: Optional[int] = None
    prefetch_factor: Optional[int] = 2
    persistent_workers: bool = False
    drop_last: bool = True
    auto_local_cache: bool = True
    local_cache_dir: Optional[str] = "/cache"
    reuse_local_cache_manifest: bool = True
    views_per_sample: int = 2
    gpu_aug_batch_chunk_size: int = 0
    resize_max_side: int = 448
    max_rotation_degrees: float = 6.0
    noise_std_max: float = 0.012
    disable_aug: bool = False
    device: Optional[str] = "cuda"
    aug_seed: Optional[int] = None
    seed: int = 42
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 3e-3
    early_stopping_monitor: str = "loss"
    metric_threshold: float = 0.5
    optimize_threshold: bool = False
    threshold_metric: str = "f1"
    threshold_start: float = 0.2
    threshold_end: float = 0.8
    threshold_step: float = 0.02
    small_mask_ratio_max: float = 0.01
    medium_mask_ratio_max: float = 0.05
    compute_foreground_ratio: bool = True
    foreground_ratio_max_batches: int = 20
    short_side_probe_samples: int = 0
    auto_pos_weight: bool = True
    pos_weight_min: float = 0.5
    pos_weight_max: float = 10.0
    balanced_pos_weight_cap: float = 0.0
    dice_weight: float = 1.0
    bce_weight: float = 1.0
    focal_weight: float = 0.0
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    tversky_weight: float = 0.2
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.8
    lovasz_weight: float = 0.05
    boundary_weight: float = 0.03
    ema_enabled: bool = True
    ema_decay: float = 0.999
    hard_mining_enabled: bool = False
    hard_mining_start_epoch: int = 5
    hard_mining_weight: float = 0.03
    hard_mining_gamma: float = 2.0
    decoder_block_type: str = "conv"
    mbconv_expand_ratio: int = 4
    mbconv_use_residual: bool = True
    default_aug: Optional[AugmentationConfig] = None
    per_dataset_aug: Optional[Dict[str, AugmentationConfig]] = None
    model_config: Optional[HybridNGIMLConfig] = None
    loss_config: Optional[MultiStageLossConfig] = None
    debug_timing: bool = False


@dataclass
class Checkpoint:
    epoch: int
    global_step: int
    model_state: dict
    raw_model_state: Optional[dict]
    ema_state: Optional[dict]
    optimizer_state: dict
    scheduler_state: Optional[dict]
    scaler_state: Optional[dict]
    train_config: dict
    training_state: Optional[dict] = None
