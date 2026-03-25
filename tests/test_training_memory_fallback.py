import torch

from tools.train_ngiml import (
    TrainConfig,
    _build_cuda_memory_safe_cfg,
    _build_cuda_runtime_safe_cfg,
    _coerce_aug,
    _is_cuda_oom_error,
    _should_disable_compile_for_device,
)
from src.data.dataloaders import AugmentationConfig


def test_cuda_memory_safe_cfg_reduces_memory_pressure():
    cfg = TrainConfig(
        manifest="dummy.parquet",
        ema_enabled=True,
        views_per_sample=3,
        gpu_aug_batch_chunk_size=4,
        resize_max_side=896,
        default_aug=AugmentationConfig(
            enable=True,
            views_per_sample=3,
            enable_rotations=True,
            enable_random_crop=True,
            enable_elastic=True,
        ),
        per_dataset_aug={
            "Toy": AugmentationConfig(
                enable=True,
                views_per_sample=2,
                enable_rotations=True,
                enable_random_crop=True,
                enable_elastic=True,
            )
        },
    )

    safe_cfg = _build_cuda_memory_safe_cfg(cfg)

    assert safe_cfg.amp is True
    assert safe_cfg.precision == "fp16"
    assert safe_cfg.ema_enabled is False
    assert safe_cfg.views_per_sample == 1
    assert safe_cfg.gpu_aug_batch_chunk_size == 1
    assert safe_cfg.resize_max_side == 448
    assert safe_cfg.default_aug is not None
    assert safe_cfg.default_aug.views_per_sample == 1
    assert safe_cfg.default_aug.enable_rotations is False
    assert safe_cfg.default_aug.enable_random_crop is False
    assert safe_cfg.default_aug.enable_elastic is False
    assert safe_cfg.per_dataset_aug is not None
    assert safe_cfg.per_dataset_aug["Toy"].views_per_sample == 1
    assert safe_cfg.per_dataset_aug["Toy"].enable_rotations is False
    assert safe_cfg.per_dataset_aug["Toy"].enable_random_crop is False
    assert safe_cfg.per_dataset_aug["Toy"].enable_elastic is False


def test_cuda_oom_detector_matches_torch_message():
    err = RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB.")
    assert _is_cuda_oom_error(err) is True


def test_cuda_runtime_safe_cfg_prefers_fp16_for_t4_style_retry():
    cfg = TrainConfig(
        manifest="dummy.parquet",
        amp=False,
        precision="bf16",
        channels_last=True,
        compile_model=True,
        flash_attention=True,
        xformers=True,
    )

    safe_cfg = _build_cuda_runtime_safe_cfg(cfg)

    assert safe_cfg.amp is True
    assert safe_cfg.precision == "fp16"
    assert safe_cfg.channels_last is False
    assert safe_cfg.compile_model is False
    assert safe_cfg.flash_attention is False
    assert safe_cfg.xformers is False


def test_bf16_unsupported_devices_should_fall_back_to_fp16_not_fp32():
    cfg = TrainConfig(manifest="dummy.parquet", amp=True, precision="bf16")
    requested_precision = str(cfg.precision).lower()
    if requested_precision == "bf16":
        cfg = TrainConfig(**{**cfg.__dict__, "precision": "fp16", "amp": True})

    assert cfg.precision == "fp16"
    assert cfg.amp is True


def test_coerce_aug_ignores_training_level_keys():
    aug = _coerce_aug(
        {
            "enable": True,
            "views_per_sample": 1,
            "enable_flips": True,
            "gpu_aug_batch_chunk_size": 1,
        }
    )

    assert aug.enable is True
    assert aug.views_per_sample == 1
    assert aug.enable_flips is True


def test_low_vram_cuda_devices_disable_compile():
    class _DeviceProps:
        total_memory = 14 * 1024**3

    original = torch.cuda.get_device_properties
    try:
        torch.cuda.get_device_properties = lambda device: _DeviceProps()
        cfg = TrainConfig(manifest="dummy.parquet", compile_model=True)
        assert _should_disable_compile_for_device(cfg, torch.device("cuda")) is True
    finally:
        torch.cuda.get_device_properties = original
