from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TVF
from torchvision.transforms.functional import InterpolationMode

from src.data.dataloaders import (
    _compute_residual_noise_fallback,
    _load_from_npz,
    _load_from_tar_npz,
    _load_image,
    _normalize,
    load_manifest,
)
from src.data.config import SampleRecord
from src.model.hybrid_ngiml import HybridNGIML
from tools.train_ngiml import build_default_components


def _zero_flop_jit(_inputs, _outputs) -> Counter[str]:
    return Counter()


def _build_flop_analysis(model: torch.nn.Module, sample: torch.Tensor):
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn.jit_handles import elementwise_flop_counter, generic_activation_jit

    elementwise = elementwise_flop_counter(1, 0)
    analysis = FlopCountAnalysis(model, sample).unsupported_ops_warnings(False)
    analysis = analysis.set_op_handle(
        "aten::add",
        elementwise,
        "aten::sub",
        elementwise,
        "aten::rsub",
        elementwise,
        "aten::mul",
        elementwise,
        "aten::div",
        elementwise,
        "aten::mean",
        elementwise,
        "aten::ne",
        elementwise,
        "aten::sigmoid",
        generic_activation_jit("sigmoid"),
        "aten::gelu",
        generic_activation_jit("gelu"),
        "aten::silu_",
        generic_activation_jit("silu"),
        "aten::softmax",
        generic_activation_jit("softmax"),
        "aten::pad",
        _zero_flop_jit,
        "aten::fill_",
        _zero_flop_jit,
        "aten::repeat",
        _zero_flop_jit,
        "aten::expand_as",
        _zero_flop_jit,
        "aten::feature_dropout",
        _zero_flop_jit,
    )
    return analysis


def find_latest_checkpoint(runs_root: Path) -> Path:
    runs_root = Path(runs_root)
    candidates = sorted(runs_root.rglob("checkpoints/checkpoint_epoch_*.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {runs_root}/**/checkpoints/checkpoint_epoch_*.pt")
    return candidates[-1]

def load_default_threshold(checkpoint_path: Path, fallback: float = 0.5) -> float:
    checkpoint_path = Path(checkpoint_path)
    candidate_files = [
        checkpoint_path.parent / "best_threshold.json",
        checkpoint_path.parent.parent / "best_threshold.json",
    ]
    for candidate in candidate_files:
        if not candidate.exists():
            continue
        try:
            import json

            with open(candidate, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            threshold = payload.get("threshold", fallback)
            return float(threshold)
        except Exception:
            continue
    return float(fallback)


def resolve_threshold_for_checkpoint(
    checkpoint_path: Path,
    checkpoint_epoch: int | None = None,
    fallback: float = 0.5,
) -> tuple[float, str]:
    checkpoint_path = Path(checkpoint_path)

    # First prefer explicit threshold metadata when it belongs to this checkpoint.
    candidate_files = [
        checkpoint_path.parent / "best_threshold.json",
        checkpoint_path.parent.parent / "best_threshold.json",
    ]
    for candidate in candidate_files:
        if not candidate.exists():
            continue
        try:
            import json

            with open(candidate, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            payload_ckpt = str(payload.get("checkpoint_path", ""))
            if payload_ckpt and Path(payload_ckpt).name == checkpoint_path.name:
                return float(payload.get("threshold", fallback)), f"{candidate.name}:matching_checkpoint"
            if checkpoint_epoch is not None and int(payload.get("epoch", -1)) == int(checkpoint_epoch):
                return float(payload.get("threshold", fallback)), f"{candidate.name}:matching_epoch"
        except Exception:
            continue

    # Fallback to per-epoch checkpoint metrics when available.
    metrics_candidates = [
        checkpoint_path.parent / "checkpoint_metrics.json",
        checkpoint_path.parent.parent / "checkpoint_metrics.json",
    ]
    for metrics_path in metrics_candidates:
        if not metrics_path.exists():
            continue
        try:
            import json

            with open(metrics_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, list):
                continue

            by_path = next(
                (
                    record for record in payload
                    if isinstance(record, dict)
                    and str(record.get("checkpoint_path", "")).endswith(checkpoint_path.name)
                    and record.get("val_threshold") is not None
                ),
                None,
            )
            if by_path is not None:
                return float(by_path["val_threshold"]), f"{metrics_path.name}:matching_path"

            if checkpoint_epoch is not None:
                by_epoch = next(
                    (
                        record for record in reversed(payload)
                        if isinstance(record, dict)
                        and int(record.get("epoch", -1)) == int(checkpoint_epoch)
                        and record.get("val_threshold") is not None
                    ),
                    None,
                )
                if by_epoch is not None:
                    return float(by_epoch["val_threshold"]), f"{metrics_path.name}:matching_epoch"
        except Exception:
            continue

    return float(load_default_threshold(checkpoint_path, fallback=fallback)), "fallback"


def _infer_fusion_channels_from_state_dict(model_state: dict) -> tuple[int, ...] | None:
    stage_channels: dict[int, int] = {}
    pattern = re.compile(r"^fusion\.stages\.(\d+)\.projections\.[^.]+\.weight$")
    for key, tensor in model_state.items():
        match = pattern.match(key)
        if not match or not isinstance(tensor, torch.Tensor):
            continue
        stage_idx = int(match.group(1))
        out_channels = int(tensor.shape[0])
        stage_channels[stage_idx] = out_channels

    if not stage_channels:
        return None

    ordered = [stage_channels[idx] for idx in sorted(stage_channels)]
    return tuple(int(value) for value in ordered)


def _build_model_config_from_checkpoint(checkpoint: dict) -> tuple[object, str]:
    model_cfg, _, _, _ = build_default_components()

    train_config = checkpoint.get("train_config") if isinstance(checkpoint, dict) else None
    model_config = train_config.get("model_config") if isinstance(train_config, dict) else None

    if isinstance(model_config, dict):
        fusion_cfg = model_config.get("fusion")
        if isinstance(fusion_cfg, dict):
            fusion_channels = fusion_cfg.get("fusion_channels")
            if isinstance(fusion_channels, (list, tuple)) and fusion_channels:
                model_cfg.fusion.fusion_channels = tuple(int(value) for value in fusion_channels)
            for attr in ("noise_branch", "noise_skip_stage", "noise_decay", "norm", "activation", "fusion_refinement"):
                if attr in fusion_cfg and hasattr(model_cfg.fusion, attr):
                    setattr(model_cfg.fusion, attr, fusion_cfg[attr])

        decoder_cfg = model_config.get("decoder")
        if isinstance(decoder_cfg, dict):
            for attr in (
                "decoder_channels",
                "out_channels",
                "norm",
                "activation",
                "per_stage_heads",
                "enable_edge_guidance",
                "use_dropout",
                "dropout_p",
                "enable_boundary_refinement",
                "boundary_refine_channels",
                "boundary_refine_scale",
            ):
                if attr in decoder_cfg and hasattr(model_cfg.decoder, attr):
                    setattr(model_cfg.decoder, attr, decoder_cfg[attr])

        for attr in (
            "use_low_level",
            "use_context",
            "use_residual",
            "enable_residual_attention",
            "gradient_checkpointing",
            "flash_attention",
            "xformers",
        ):
            if attr in model_config and hasattr(model_cfg, attr):
                setattr(model_cfg, attr, model_config[attr])

        return model_cfg, "train_config.model_config"

    inferred_channels = _infer_fusion_channels_from_state_dict(checkpoint.get("model_state", {}))
    if inferred_channels:
        model_cfg.fusion.fusion_channels = inferred_channels
        return model_cfg, "state_dict.inferred_fusion_channels"

    return model_cfg, "defaults"


def _select_output_head(outputs: Sequence[torch.Tensor]) -> torch.Tensor:
    if not outputs:
        raise ValueError("Model returned empty predictions list")
    # Highest-resolution decoder output is index 0 by contract.
    return outputs[0]


def _dtype_name(value: torch.dtype | None) -> str:
    return str(value).replace("torch.", "") if isinstance(value, torch.dtype) else "none"


def _resolve_checkpoint_autocast_dtype(train_config: dict, device: torch.device) -> tuple[torch.dtype | None, str]:
    precision_raw = str(train_config.get("precision", "") or "").strip().lower()
    amp_enabled = bool(train_config.get("amp", False))

    preferred: torch.dtype | None = None
    source = "checkpoint_precision"

    if precision_raw in {"bf16", "bfloat16"}:
        preferred = torch.bfloat16
    elif precision_raw in {"fp16", "float16", "half"}:
        preferred = torch.float16
    elif precision_raw in {"fp32", "float32", "32", "full", "none", "off", "disabled"}:
        preferred = None
    elif amp_enabled:
        # Older checkpoints may have amp=True but unset precision; choose a safe CUDA autocast dtype.
        preferred = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        source = "checkpoint_amp_fallback"

    if device.type != "cuda":
        return None, f"{source}:cpu"

    if preferred is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return torch.float16, f"{source}:bf16_unsupported_fallback_fp16"

    return preferred, source


def get_inference_autocast_dtype(model: HybridNGIML, device: torch.device) -> torch.dtype | None:
    dtype = getattr(model, "default_autocast_dtype", None)
    if not isinstance(dtype, torch.dtype):
        return None
    if device.type != "cuda":
        return None
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return torch.float16
    if dtype in {torch.float16, torch.bfloat16}:
        return dtype
    return None


def _load_state_dict_with_fallback(model: HybridNGIML, model_state: dict) -> tuple[list[str], list[str], int]:
    try:
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        return list(missing), list(unexpected), 0
    except RuntimeError:
        current_state = model.state_dict()
        compatible_state = {
            key: value
            for key, value in model_state.items()
            if key in current_state and hasattr(value, "shape") and current_state[key].shape == value.shape
        }
        skipped = int(len(model_state) - len(compatible_state))
        missing, unexpected = model.load_state_dict(compatible_state, strict=False)
        return list(missing), list(unexpected), skipped


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device | None = None) -> tuple[HybridNGIML, torch.device, dict]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    model_cfg, config_source = _build_model_config_from_checkpoint(checkpoint)
    model = HybridNGIML(model_cfg).to(device)

    missing, unexpected, skipped_mismatched = _load_state_dict_with_fallback(model, checkpoint["model_state"])
    model.eval()
    resolved_threshold, threshold_source = resolve_threshold_for_checkpoint(
        Path(checkpoint_path),
        checkpoint_epoch=checkpoint_epoch,
        fallback=0.5,
    )

    train_config = checkpoint.get("train_config") or {}

    has_train_max_short = "max_short_side" in train_config
    autocast_dtype, autocast_source = _resolve_checkpoint_autocast_dtype(train_config, device)
    precision_raw = str(train_config.get("precision", "") or "").strip().lower() or "unset"
    info = {
        "epoch": checkpoint_epoch,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "skipped_mismatched_keys": int(skipped_mismatched),
        "config_source": str(config_source),
        "fusion_channels": tuple(int(value) for value in model.cfg.fusion.fusion_channels),
        "default_threshold": float(resolved_threshold),
        "threshold_source": str(threshold_source),
        "max_short_side": int(train_config.get("max_short_side", 0) or 0),
        "max_short_side_source": "train_config" if has_train_max_short else "default",
        "runtime_precision": precision_raw,
        "inference_autocast_dtype": _dtype_name(autocast_dtype),
        "inference_autocast_source": autocast_source,
    }
    setattr(model, "default_threshold", float(info["default_threshold"]))
    setattr(model, "default_runtime_precision", precision_raw)
    setattr(model, "default_autocast_dtype", autocast_dtype)
    return model, device, info


def select_manifest_sample(
    manifest_path: Path,
    split_priority: Sequence[str] = ("test", "val", "train"),
    fake_only: bool = True,
) -> SampleRecord:
    manifest = load_manifest(manifest_path)
    samples = manifest.samples

    if fake_only:
        fake_samples = [s for s in samples if int(getattr(s, "label", 0)) == 1 or s.mask_path is not None]
    else:
        fake_samples = samples

    for split_name in split_priority:
        split_samples = [s for s in fake_samples if s.split == split_name]
        if split_samples:
            return split_samples[0]

    if fake_samples:
        return fake_samples[0]

    raise RuntimeError(f"No samples available in manifest: {manifest_path}")


def _resolve_possible_local_path(path_str: str) -> str:
    path = Path(path_str)
    return path.as_posix()


def resize_for_inference(
    image: torch.Tensor,
    mask: torch.Tensor | None = None,
    residual_noise: torch.Tensor | None = None,
    max_short_side: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    cap = int(max_short_side or 0)
    if cap <= 0:
        return image, mask, residual_noise

    h, w = image.shape[-2:]
    short_side = min(h, w)
    if short_side <= 0 or short_side <= cap:
        return image, mask, residual_noise

    scale = float(cap) / float(short_side)
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    image = TVF.resize(image, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)
    if mask is not None:
        mask = TVF.resize(mask, [new_h, new_w], interpolation=InterpolationMode.NEAREST)
    if residual_noise is not None:
        residual_noise = TVF.resize(residual_noise, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)
    return image, mask, residual_noise


def should_use_residual_noise_for_records(records: Sequence[SampleRecord]) -> bool:
    """Mirror dataloader behavior: if any non-NPZ sample lacks high-pass, disable high-pass for the whole split."""
    for record in records:
        image_path = str(record.image_path)
        is_npz_like = image_path.endswith(".npz")
        if is_npz_like:
            # NPZ/tar::NPZ samples synthesize high-pass fallback at load time.
            continue
        if record.residual_noise_path is None:
            return False
    return True


def collate_eval_batch_like_training(
    records: Sequence[SampleRecord],
    max_short_side: int | None = None,
    use_residual_noise: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, list[str]]:
    """Load and collate a batch using the same median-short-side resize + padding policy as training eval."""
    images: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    residual_noisees: list[torch.Tensor] = []
    datasets: list[str] = []

    for record in records:
        image, mask, residual_noise = load_image_mask_from_record(record, max_short_side=max_short_side)
        images.append(image)
        masks.append(mask)
        datasets.append(str(record.dataset))
        if use_residual_noise and residual_noise is not None:
            residual_noisees.append(residual_noise)

    shorts = [min(int(img.shape[-2]), int(img.shape[-1])) for img in images]
    target_short = int(round(float(torch.tensor(shorts, dtype=torch.float32).median().item()))) if shorts else 0

    if target_short > 0:
        for idx, image in enumerate(images):
            h, w = image.shape[-2:]
            short_side = min(h, w)
            if short_side > 0 and short_side != target_short:
                scale = float(target_short) / float(short_side)
                new_h = max(1, int(round(h * scale)))
                new_w = max(1, int(round(w * scale)))
                images[idx] = TVF.resize(images[idx], [new_h, new_w], interpolation=InterpolationMode.BILINEAR)
                masks[idx] = TVF.resize(masks[idx], [new_h, new_w], interpolation=InterpolationMode.NEAREST)
                if use_residual_noise and idx < len(residual_noisees):
                    residual_noisees[idx] = TVF.resize(residual_noisees[idx], [new_h, new_w], interpolation=InterpolationMode.BILINEAR)

    max_h = max(int(img.shape[-2]) for img in images)
    max_w = max(int(img.shape[-1]) for img in images)

    padded_images: list[torch.Tensor] = []
    padded_masks: list[torch.Tensor] = []
    padded_residual_noisees: list[torch.Tensor] = []

    for idx, (image, mask) in enumerate(zip(images, masks)):
        h, w = image.shape[-2:]
        pad_h = max_h - h
        pad_w = max_w - w
        if pad_h > 0 or pad_w > 0:
            image = F.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h), mode="constant", value=0.0).squeeze(0)
            mask = F.pad(mask.unsqueeze(0), (0, pad_w, 0, pad_h), mode="constant", value=0.0).squeeze(0)
        padded_images.append(image)
        padded_masks.append(mask)

        if use_residual_noise and idx < len(residual_noisees):
            hp = residual_noisees[idx]
            hh, hw = hp.shape[-2:]
            hp_pad_h = max_h - hh
            hp_pad_w = max_w - hw
            if hp_pad_h > 0 or hp_pad_w > 0:
                hp = F.pad(hp.unsqueeze(0), (0, hp_pad_w, 0, hp_pad_h), mode="constant", value=0.0).squeeze(0)
            padded_residual_noisees.append(hp)

    image_batch = torch.stack(padded_images, dim=0)
    mask_batch = torch.stack(padded_masks, dim=0)

    residual_noise_batch = None
    if use_residual_noise and len(padded_residual_noisees) == len(padded_images):
        residual_noise_batch = torch.stack(padded_residual_noisees, dim=0)

    return image_batch, mask_batch, residual_noise_batch, datasets


def load_image_mask_from_record(
    record: SampleRecord,
    max_short_side: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    image_path = str(record.image_path)
    if "::" in image_path and image_path.endswith(".npz"):
        image, mask, residual_noise = _load_from_tar_npz(image_path)
    elif image_path.endswith(".npz"):
        image, mask, residual_noise = _load_from_npz(_resolve_possible_local_path(image_path))
    else:
        image = _load_image(_resolve_possible_local_path(image_path))
        residual_noise = None
        mask = None
        if record.mask_path is not None:
            loaded = _load_image(_resolve_possible_local_path(record.mask_path))
            mask = loaded[:1] if loaded.shape[0] > 1 else loaded
        if record.residual_noise_path is not None:
            loaded_high = _load_image(_resolve_possible_local_path(record.residual_noise_path))
            residual_noise = loaded_high if loaded_high.shape[0] in (1, 3) else loaded_high[:3]

    image = image.float()
    if image.max() > 1.0:
        image = image / 255.0

    if mask is None:
        mask = torch.zeros((1, image.shape[-2], image.shape[-1]), dtype=torch.float32)
    else:
        mask = mask.float()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[0] > 1:
            mask = mask[:1]
        if mask.max() > 1.0:
            mask = mask / 255.0
        if tuple(mask.shape[-2:]) != tuple(image.shape[-2:]):
            mask = F.interpolate(mask.unsqueeze(0), size=image.shape[-2:], mode="nearest").squeeze(0)

    if residual_noise is not None:
        residual_noise = residual_noise.float()
        if residual_noise.ndim == 2:
            residual_noise = residual_noise.unsqueeze(0)
        if residual_noise.shape[0] == 1:
            residual_noise = residual_noise.repeat(3, 1, 1)
        elif residual_noise.shape[0] > 3:
            residual_noise = residual_noise[:3]
        if residual_noise.max() > 1.0:
            residual_noise = residual_noise / 255.0
        if tuple(residual_noise.shape[-2:]) != tuple(image.shape[-2:]):
            residual_noise = F.interpolate(residual_noise.unsqueeze(0), size=image.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
    else:
        residual_noise = _compute_residual_noise_fallback(image)

    image, mask, residual_noise = resize_for_inference(image, mask=mask, residual_noise=residual_noise, max_short_side=max_short_side)
    return image, mask, residual_noise


def normalize_image_for_inference(image: torch.Tensor, normalization_mode: str = "zero_one") -> torch.Tensor:
    image = image.float()
    if image.max() > 1.0:
        image = image / 255.0
    return _normalize(image, str(normalization_mode).strip().lower())


def predict_probability_map(
    model: HybridNGIML,
    image: torch.Tensor,
    device: torch.device,
    normalization_mode: str = "zero_one",
    residual_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    normalized = normalize_image_for_inference(image, normalization_mode=normalization_mode)
    x = normalized.unsqueeze(0).to(device)
    hp = None
    if residual_noise is not None:
        hp = residual_noise.float()
        if hp.max() > 1.0:
            hp = hp / 255.0
        hp = hp.unsqueeze(0).to(device)
    autocast_dtype = get_inference_autocast_dtype(model, device)
    use_amp = device.type == "cuda" and autocast_dtype is not None
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=autocast_dtype or torch.float16, enabled=use_amp):
            outputs = model(x, target_size=image.shape[-2:], residual_noise=hp)
            logits = _select_output_head(outputs)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu()
    return prob


def predict_binary_map(
    model: HybridNGIML,
    image: torch.Tensor,
    device: torch.device,
    threshold: float | None = None,
    normalization_mode: str = "zero_one",
    residual_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    prob = predict_probability_map(
        model,
        image,
        device,
        normalization_mode=normalization_mode,
        residual_noise=residual_noise,
    )
    if threshold is None:
        threshold = float(getattr(model, "default_threshold", 0.5))
    return (prob >= float(threshold)).float()


def resolve_normalization_mode_for_inference(
    manual_mode: str | None = None,
    manifest_path: str | Path | None = None,
    training_config: dict | None = None,
    checkpoint_path: str | Path | None = None,
    default_mode: str = "imagenet",
) -> str:
    """Resolve inference normalization with explicit override first, then manifest sources."""
    if isinstance(manual_mode, str) and manual_mode.strip():
        mode = manual_mode.strip().lower()
        if mode in {"imagenet", "zero_one"}:
            return mode
        raise ValueError(
            f"Unsupported normalization mode: {manual_mode!r}. "
            "Use 'imagenet', 'zero_one', or None."
        )

    candidates: list[Path] = []

    if manifest_path:
        candidates.append(Path(manifest_path))

    if isinstance(training_config, dict):
        train_manifest = training_config.get("manifest")
        if train_manifest:
            candidates.append(Path(train_manifest))

    if checkpoint_path:
        try:
            checkpoint_blob = torch.load(Path(checkpoint_path), map_location="cpu")
            train_cfg = checkpoint_blob.get("train_config") if isinstance(checkpoint_blob, dict) else None
            checkpoint_manifest = train_cfg.get("manifest") if isinstance(train_cfg, dict) else None
            if checkpoint_manifest:
                candidates.append(Path(checkpoint_manifest))
        except Exception:
            pass

    for candidate in candidates:
        try:
            manifest_obj = load_manifest(candidate)
            mode = str(manifest_obj.normalization_mode).strip().lower()
            if mode in {"imagenet", "zero_one"}:
                return mode
        except Exception:
            continue

    fallback = str(default_mode).strip().lower()
    if fallback not in {"imagenet", "zero_one"}:
        fallback = "imagenet"
    return fallback


def _tile_starts(full_size: int, tile_size: int, stride: int) -> list[int]:
    if full_size <= tile_size:
        return [0]
    starts = list(range(0, full_size - tile_size + 1, stride))
    tail = full_size - tile_size
    if starts[-1] != tail:
        starts.append(tail)
    return starts


def _hann_weight_2d(h: int, w: int) -> torch.Tensor:
    wy = torch.hann_window(h, periodic=False).float().clamp_min(1e-3)
    wx = torch.hann_window(w, periodic=False).float().clamp_min(1e-3)
    weight = wy[:, None] * wx[None, :]
    return weight / weight.max().clamp_min(1e-6)


def predict_probability_map_sliding_window(
    model: HybridNGIML,
    image: torch.Tensor,
    device: torch.device,
    normalization_mode: str = "zero_one",
    residual_noise: torch.Tensor | None = None,
    tile_size: int = 768,
    overlap: float = 0.25,
    tile_batch_size: int = 4,
) -> torch.Tensor:
    """Run overlap-weighted tiled inference and return full-resolution probability map."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected RGB CHW image tensor, got shape={tuple(image.shape)}")

    image = image.float()
    if image.max() > 1.0:
        image = image / 255.0
    image = image.clamp(0.0, 1.0)

    if residual_noise is None:
        residual_noise = _compute_residual_noise_fallback(image)
    else:
        residual_noise = residual_noise.float()
        if residual_noise.ndim == 2:
            residual_noise = residual_noise.unsqueeze(0)
        if residual_noise.shape[0] == 1:
            residual_noise = residual_noise.repeat(3, 1, 1)
        elif residual_noise.shape[0] > 3:
            residual_noise = residual_noise[:3]
        if residual_noise.max() > 1.0:
            residual_noise = residual_noise / 255.0
        if tuple(residual_noise.shape[-2:]) != tuple(image.shape[-2:]):
            residual_noise = F.interpolate(
                residual_noise.unsqueeze(0),
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

    _, h, w = image.shape
    tile = max(64, int(tile_size))
    stride = max(1, int(round(tile * (1.0 - float(overlap)))))
    tile_batch = max(1, int(tile_batch_size))

    pad_h = max(0, tile - h)
    pad_w = max(0, tile - w)
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect").squeeze(0)
        residual_noise = F.pad(residual_noise.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect").squeeze(0)

    _, hp, wp = image.shape
    ys = _tile_starts(hp, tile, stride)
    xs = _tile_starts(wp, tile, stride)

    weight = _hann_weight_2d(tile, tile)
    accum = torch.zeros((hp, wp), dtype=torch.float32)
    accum_w = torch.zeros((hp, wp), dtype=torch.float32)

    def _flush(
        rgb_tiles: list[torch.Tensor],
        residual_tiles: list[torch.Tensor],
        coords: list[tuple[int, int]],
    ) -> None:
        if not rgb_tiles:
            return
        xb = torch.stack(rgb_tiles, dim=0).to(device, non_blocking=True)
        hb = torch.stack(residual_tiles, dim=0).to(device, non_blocking=True)

        autocast_dtype = get_inference_autocast_dtype(model, device)
        use_amp = device.type == "cuda" and autocast_dtype is not None
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=autocast_dtype or torch.float16, enabled=use_amp):
                outputs = model(xb, target_size=xb.shape[-2:], residual_noise=hb)
                logits = _select_output_head(outputs) if isinstance(outputs, (list, tuple)) else outputs
                probs = torch.sigmoid(logits[:, 0]).float().cpu()

        for idx, (y0, x0) in enumerate(coords):
            accum[y0 : y0 + tile, x0 : x0 + tile] += probs[idx] * weight
            accum_w[y0 : y0 + tile, x0 : x0 + tile] += weight

    tile_images: list[torch.Tensor] = []
    tile_residuals: list[torch.Tensor] = []
    tile_coords: list[tuple[int, int]] = []

    for y0 in ys:
        for x0 in xs:
            rgb_tile = image[:, y0 : y0 + tile, x0 : x0 + tile]
            hp_tile = residual_noise[:, y0 : y0 + tile, x0 : x0 + tile]

            tile_images.append(normalize_image_for_inference(rgb_tile, normalization_mode=normalization_mode))
            tile_residuals.append(hp_tile.float())
            tile_coords.append((y0, x0))

            if len(tile_images) >= tile_batch:
                _flush(tile_images, tile_residuals, tile_coords)
                tile_images.clear()
                tile_residuals.clear()
                tile_coords.clear()

    _flush(tile_images, tile_residuals, tile_coords)
    prob = accum / accum_w.clamp_min(1e-6)
    return prob[:h, :w]


def _center_square_crop(tensor: torch.Tensor) -> torch.Tensor:
    h, w = tensor.shape[-2:]
    if h == w:
        return tensor
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    return tensor[..., top : top + side, left : left + side]


def infer_from_image_path(
    model: HybridNGIML,
    image_path: Path,
    device: torch.device,
    normalization_mode: str = "zero_one",
    max_short_side: int | None = None,
    crop_square: bool = False,
    prep_target_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    orig_image = _load_image(str(Path(image_path).as_posix())).float()
    if orig_image.max() > 1.0:
        orig_image = orig_image / 255.0

    image = orig_image
    if crop_square:
        image = _center_square_crop(image)

    residual_noise = _compute_residual_noise_fallback(image)
    if prep_target_size is not None and prep_target_size > 0:
        image = TVF.resize(image, [prep_target_size, prep_target_size], interpolation=InterpolationMode.BILINEAR)
        residual_noise = TVF.resize(residual_noise, [prep_target_size, prep_target_size], interpolation=InterpolationMode.BILINEAR)
    else:
        image, _, residual_noise = resize_for_inference(image, mask=None, residual_noise=residual_noise, max_short_side=max_short_side)
    pred = predict_probability_map(model, image, device, normalization_mode=normalization_mode, residual_noise=residual_noise)
    return image, pred, orig_image


def multiscale_infer_from_image_path(
    model: HybridNGIML,
    image_path: Path,
    device: torch.device,
    normalization_mode: str = "zero_one",
    max_short_side: int | None = None,
    scales: Sequence[float] = (1.0,),
    merge_mode: str = "mean",
    crop_square: bool = False,
    prep_target_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, list[tuple[float, torch.Tensor]], torch.Tensor]:
    orig_image = _load_image(str(Path(image_path).as_posix())).float()
    if orig_image.max() > 1.0:
        orig_image = orig_image / 255.0

    image = orig_image
    if crop_square:
        image = _center_square_crop(image)
    residual_noise = _compute_residual_noise_fallback(image)
    if prep_target_size is not None and prep_target_size > 0:
        image = TVF.resize(image, [prep_target_size, prep_target_size], interpolation=InterpolationMode.BILINEAR)
        residual_noise = TVF.resize(residual_noise, [prep_target_size, prep_target_size], interpolation=InterpolationMode.BILINEAR)
    else:
        image, _, residual_noise = resize_for_inference(image, mask=None, residual_noise=residual_noise, max_short_side=max_short_side)

    base_h, base_w = image.shape[-2:]
    merge = None if merge_mode is None else ("max" if str(merge_mode).lower() == "max" else "mean")

    cleaned_scales: list[float] = []
    for scale in scales or (1.0,):
        value = float(scale)
        if value <= 0:
            continue
        cleaned_scales.append(value)
    if not cleaned_scales:
        cleaned_scales = [1.0]

    merged: torch.Tensor | None = None
    count = 0
    scale_outputs: list[tuple[float, torch.Tensor]] = []

    for scale in cleaned_scales:
        if abs(scale - 1.0) < 1e-6:
            scaled_img = image
            scaled_hp = residual_noise
        else:
            new_h = max(1, int(round(base_h * scale)))
            new_w = max(1, int(round(base_w * scale)))
            scaled_img = TVF.resize(image, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)
            scaled_hp = None if residual_noise is None else TVF.resize(residual_noise, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)

        prob = predict_probability_map(
            model,
            scaled_img,
            device,
            normalization_mode=normalization_mode,
            residual_noise=scaled_hp,
        )

        if prob.shape[-2:] != (base_h, base_w):
            prob = F.interpolate(prob.unsqueeze(0).unsqueeze(0), size=(base_h, base_w), mode="bilinear", align_corners=False)[0, 0]

        scale_outputs.append((scale, prob))

        # Accumulate merged map only when caller wants a merge (len(cleaned_scales) > 1 handled upstream)
        if merge is not None:
            if merge == "max":
                merged = prob if merged is None else torch.maximum(merged, prob)
            else:
                merged = prob if merged is None else merged + prob
                count += 1

    if merge is None:
        merged = None
    else:
        if merged is None:
            merged = torch.empty((base_h, base_w))
        elif merge != "max":
            merged = merged / float(count or len(cleaned_scales))

    return image, merged, scale_outputs, orig_image


def get_model_complexity_stats(
    model: HybridNGIML,
    input_size: tuple[int, int, int, int] = (1, 3, 384, 384),
) -> dict[str, object]:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen_params = total_params - trainable_params

    stats: dict[str, object] = {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "frozen_params": int(frozen_params),
        "input_size": tuple(int(v) for v in input_size),
    }

    sample_device = next(model.parameters()).device
    sample = torch.randn(*input_size, device=sample_device)

    class _ProfileWrapper(torch.nn.Module):
        def __init__(self, base_model: HybridNGIML):
            super().__init__()
            self.base_model = base_model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.base_model(x, target_size=x.shape[-2:], residual_noise=None)
            if isinstance(out, (list, tuple)):
                return _select_output_head(out)
            return out

    profile_model = _ProfileWrapper(model).to(sample_device)

    was_training = model.training
    model.eval()
    profile_model.eval()
    try:
        try:
            with torch.no_grad():
                analysis = _build_flop_analysis(profile_model, sample)
                total_flops = float(analysis.total())
                unsupported_ops = {str(name): int(count) for name, count in analysis.unsupported_ops().items()}
            stats["flops"] = total_flops
            stats["macs"] = total_flops / 2.0
            stats["unsupported_ops"] = unsupported_ops
            stats["flops_source"] = "fvcore+custom_op_handles"
            stats["flops_error"] = (
                None
                if not unsupported_ops
                else "FLOPs include custom op-handle estimates; unsupported ops remain in `unsupported_ops`."
            )
        except Exception as fv_error:
            try:
                from thop import profile as thop_profile

                with torch.no_grad():
                    macs, _ = thop_profile(profile_model, inputs=(sample,), verbose=False)
                macs = float(macs)
                stats["macs"] = macs
                stats["flops"] = macs * 2.0
                stats["unsupported_ops"] = None
                stats["flops_source"] = "thop"
                stats["flops_error"] = f"fvcore unavailable ({fv_error}); used thop fallback"
            except Exception as thop_error:
                stats["flops"] = None
                stats["macs"] = None
                stats["unsupported_ops"] = None
                stats["flops_source"] = None
                stats["flops_error"] = (
                    "FLOPs unavailable. "
                    f"fvcore error: {fv_error}. "
                    f"thop error: {thop_error}. "
                    "Try `%pip install fvcore iopath` (or `%pip install thop`) in the active notebook kernel."
                )
    finally:
        model.train(was_training)

    return stats

