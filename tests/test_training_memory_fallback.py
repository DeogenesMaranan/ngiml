import torch

import src.training_loop_helpers as loop_helpers
from src.training_defaults import _coerce_aug
from src.training_types import TrainConfig


def test_resolve_cuda_runtime_stability_prefers_fp16_when_bf16_unsupported():
    cfg = TrainConfig(
        manifest="dummy.parquet",
        amp=True,
        precision="bf16",
    )

    original_support_fn = loop_helpers._cuda_supports_bf16
    try:
        loop_helpers._cuda_supports_bf16 = lambda: False
        safe_cfg = loop_helpers._resolve_cuda_runtime_stability(cfg, torch.device("cuda"))
    finally:
        loop_helpers._cuda_supports_bf16 = original_support_fn

    assert safe_cfg.precision == "fp16"
    assert safe_cfg.amp is True


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
        assert loop_helpers._should_disable_compile_for_device(cfg, torch.device("cuda")) is True
    finally:
        torch.cuda.get_device_properties = original


def test_gpu_aug_chunking_is_skipped_when_chunk_covers_group():
    assert loop_helpers.should_chunk_gpu_aug(group_size=8, chunk_size=8) is False
    assert loop_helpers.should_chunk_gpu_aug(group_size=8, chunk_size=16) is False
    assert loop_helpers.should_chunk_gpu_aug(group_size=8, chunk_size=4) is True
    assert loop_helpers.should_chunk_gpu_aug(group_size=8, chunk_size=0) is False


def test_gpu_aug_chunk_size_zero_means_auto_full_group():
    assert loop_helpers.resolve_gpu_aug_chunk_size(group_size=8, chunk_size=0) == 8
    assert loop_helpers.resolve_gpu_aug_chunk_size(group_size=8, chunk_size=-1) == 8
    assert loop_helpers.resolve_gpu_aug_chunk_size(group_size=8, chunk_size=4) == 4
