from __future__ import annotations

import io
import json
import logging
import re
import tarfile
from collections import Counter
from pathlib import Path
from typing import Sequence

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm
from torchvision.transforms import functional as TVF
from torchvision.transforms.functional import InterpolationMode

from src.data.dataloaders import (
    _compute_residual_noise,
    _load_from_npz,
    _load_from_tar_npz,
    _load_image,
    _normalize,
    load_manifest,
)
from src.data.config import SampleRecord
from src.model.hybrid_ngiml import HybridNGIML
from tools.train_ngiml import _coerce_model_config, build_default_components


def _require_matplotlib() -> None:
    if plt is None:
        raise ImportError(
            "matplotlib is required for plotting helpers but is not installed in this environment."
        )

def _to_chw_rgb(image_np: np.ndarray) -> np.ndarray:
    if image_np.ndim == 2:
        image_np = np.stack([image_np, image_np, image_np], axis=-1)
    if image_np.ndim == 3 and image_np.shape[0] in (1, 3) and image_np.shape[-1] not in (1, 3):
        image_np = np.transpose(image_np, (1, 2, 0))
    if image_np.ndim != 3:
        raise ValueError(f'Unsupported image shape: {image_np.shape}')
    if image_np.shape[-1] == 1:
        image_np = np.repeat(image_np, 3, axis=-1)
    if image_np.shape[-1] > 3:
        image_np = image_np[..., :3]
    return np.transpose(image_np, (2, 0, 1))

def _to_hw_mask(mask_np: np.ndarray | None, h: int, w: int) -> np.ndarray:
    if mask_np is None:
        return np.zeros((h, w), dtype=np.uint8)
    arr = np.asarray(mask_np)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = (arr > 0).astype(np.uint8)
    if arr.shape != (h, w):
        raise ValueError(f'Mask shape {arr.shape} does not match image {(h, w)}')
    return arr

def _parse_meta(raw) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8', errors='replace')
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {'metadata_raw': raw}
    return raw if isinstance(raw, dict) else {'metadata_raw': str(raw)}

def _dataset_name(sample_uri: str, meta: dict) -> str:
    for k in ('dataset', 'dataset_name', 'source_dataset'):
        if str(meta.get(k, '')).strip():
            return str(meta[k]).strip()
    parts = Path(sample_uri.split('::')[0]).parts
    for i, part in enumerate(parts):
        if part.lower() in {'test', 'val', 'train'} and i > 0:
            return parts[i - 1]
    return parts[0] if parts else 'unknown'

def iter_prepared_samples(snapshot_root: Path):
    for npz_path in sorted(snapshot_root.rglob('*.npz')):
        with np.load(npz_path, allow_pickle=True) as blob:
            yield str(npz_path), {k: blob[k] for k in blob.files}

    for tar_path in sorted(snapshot_root.rglob('*.tar')):
        with tarfile.open(tar_path, mode='r') as tf:
            for m in tf.getmembers():
                if not m.isfile() or not m.name.lower().endswith('.npz'):
                    continue
                fobj = tf.extractfile(m)
                if fobj is None:
                    continue
                with np.load(io.BytesIO(fobj.read()), allow_pickle=True) as blob:
                    yield f'{tar_path}::{m.name}', {k: blob[k] for k in blob.files}

def compute_binary_metrics(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    *,
    empty_score_mode: str = "strict",
) -> dict[str, float]:
    pred = pred_bin.astype(bool)
    gt = gt_bin.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    tn = float(np.logical_and(~pred, ~gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    mode = str(empty_score_mode).strip().lower()
    if mode not in {"strict", "legacy"}:
        raise ValueError(f"Unsupported empty_score_mode={empty_score_mode!r}. Use 'strict' or 'legacy'.")

    precision_denom = tp + fp
    recall_denom = tp + fn
    f1_denom = (2 * tp) + fp + fn
    iou_denom = tp + fp + fn
    acc_denom = tp + tn + fp + fn

    precision = tp / (precision_denom + 1e-8)
    recall = tp / (recall_denom + 1e-8)
    if mode == "legacy":
        f1 = (2 * tp) / f1_denom if f1_denom > 0 else 1.0
        iou = tp / iou_denom if iou_denom > 0 else 1.0
        acc = (tp + tn) / acc_denom if acc_denom > 0 else 1.0
    else:
        f1 = (2 * precision * recall) / (precision + recall + 1e-8)
        iou = tp / (iou_denom + 1e-8)
        acc = (tp + tn) / (acc_denom + 1e-8)
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou, 'accuracy': acc}

def save_sample_plot(out_path: Path, image_chw: np.ndarray, gt_hw: np.ndarray, prob_hw: np.ndarray, bin05_hw: np.ndarray, title: str):
    _require_matplotlib()
    img = np.transpose(image_chw, (1, 2, 0)).astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)
    alpha = 0.45 * prob_hw[..., None].astype(np.float32)
    overlay = np.clip(img * (1.0 - alpha) + np.array([1.0, 0.0, 0.0]) * alpha, 0.0, 1.0)

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(title, fontsize=11)
    axes[0].imshow(img); axes[0].set_title('Image'); axes[0].axis('off')
    axes[1].imshow(gt_hw, cmap='gray', vmin=0, vmax=1); axes[1].set_title('Ground Truth'); axes[1].axis('off')
    axes[2].imshow(prob_hw, cmap='magma', vmin=0, vmax=1); axes[2].set_title('Pred Mask (magma)'); axes[2].axis('off')
    axes[3].imshow(bin05_hw, cmap='gray', vmin=0, vmax=1); axes[3].set_title('Pred Mask (0.5 threshold)'); axes[3].axis('off')
    axes[4].imshow(overlay); axes[4].set_title('Overlay'); axes[4].axis('off')
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def count_prepared_samples(snapshot_root: Path) -> int:
    npz_files = list(snapshot_root.rglob("*.npz"))
    tar_files = list(snapshot_root.rglob("*.tar"))
    tar_entries = 0
    for tar_path in tar_files:
        with tarfile.open(tar_path, mode="r") as tf:
            tar_entries += len([member for member in tf.getmembers() if member.isfile() and member.name.lower().endswith(".npz")])
    return int(len(npz_files) + tar_entries)


def _prepared_sample_to_inference_record(sample_uri: str, data: dict[str, object]) -> dict[str, object] | None:
    if "image" not in data:
        return None

    image_chw = _to_chw_rgb(np.asarray(data["image"]))
    h, w = int(image_chw.shape[1]), int(image_chw.shape[2])
    mask_hw = _to_hw_mask(data.get("mask"), h, w)
    meta = _parse_meta(data.get("metadata_json"))
    dataset = _dataset_name(sample_uri, meta)

    image_t = torch.from_numpy(image_chw).float()
    if image_t.max() > 1.0:
        image_t = image_t / 255.0

    return {
        "sample_uri": sample_uri,
        "image_chw": image_chw,
        "image_t": image_t,
        "mask_hw": mask_hw,
        "meta": meta,
        "dataset": dataset,
        "height": h,
        "width": w,
    }


def _append_prepared_inference_result(
    rows: list[dict[str, object]],
    plot_samples: dict[str, list[dict[str, object]]],
    sample: dict[str, object],
    prob: torch.Tensor,
    *,
    threshold_for_metrics: float,
    plot_binary_threshold: float,
    inference_strategy: str,
    normalization_mode: str,
    max_plot_samples_per_dataset: int,
) -> None:
    prob_hw = prob.detach().cpu().numpy().astype(np.float32)
    pred_bin_metric = (prob_hw >= float(threshold_for_metrics)).astype(np.uint8)
    pred_bin_plot = (prob_hw >= float(plot_binary_threshold)).astype(np.uint8)
    metrics = compute_binary_metrics(pred_bin_metric, sample["mask_hw"], empty_score_mode="strict")
    legacy_metrics = compute_binary_metrics(pred_bin_metric, sample["mask_hw"], empty_score_mode="legacy")

    meta = sample["meta"]
    raw_label = meta.get("label", int(sample["mask_hw"].max() > 0))
    if isinstance(raw_label, str):
        sample_label = 1 if raw_label.strip().lower() in {"1", "fake", "tp", "tampered", "manipulated"} else 0
    else:
        sample_label = int(raw_label)

    row = {
        "dataset": sample["dataset"],
        "sample_uri": sample["sample_uri"],
        "split": str(meta.get("split", "test")),
        "label": sample_label,
        "strategy": inference_strategy,
        "normalization_mode": normalization_mode,
        "threshold_for_metrics": float(threshold_for_metrics),
        "plot_binary_threshold": float(plot_binary_threshold),
        "height": int(sample["height"]),
        "width": int(sample["width"]),
        "mean_probability": float(prob_hw.mean()),
        "max_probability": float(prob_hw.max()),
        "pred_positive_ratio_threshold": float(pred_bin_metric.mean()),
        "pred_positive_ratio_0_5": float(pred_bin_plot.mean()),
        "gt_positive_ratio": float(sample["mask_hw"].mean()),
        "legacy_f1": float(legacy_metrics["f1"]),
        "legacy_iou": float(legacy_metrics["iou"]),
    }
    row.update(metrics)
    rows.append(row)

    if sample_label != 1:
        return

    dataset_plots = plot_samples.setdefault(str(sample["dataset"]), [])
    if len(dataset_plots) >= int(max_plot_samples_per_dataset):
        return
    dataset_plots.append(
        {
            "sample_uri": sample["sample_uri"],
            "image_chw": sample["image_chw"],
            "mask_hw": sample["mask_hw"],
            "prob_hw": prob_hw,
            "bin05_hw": pred_bin_plot,
        }
    )


def run_prepared_dataset_inference(
    *,
    checkpoint_path: str | Path,
    hf_dataset_repo_id: str,
    output_root: str | Path,
    hf_snapshot_local_dir: str | Path,
    inference_strategy: str = "direct",
    threshold_for_metrics: float | None = None,
    plot_binary_threshold: float = 0.5,
    direct_batch_size: int = 8,
    normalization_mode: str | None = None,
    max_plot_samples_per_dataset: int = 5,
) -> dict[str, object]:
    checkpoint_path = Path(checkpoint_path)
    output_root = Path(output_root)
    hf_snapshot_local_dir = Path(hf_snapshot_local_dir)

    csv_output_dir = output_root / "csv"
    plot_output_dir = output_root / "plots"
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    plot_output_dir.mkdir(parents=True, exist_ok=True)

    model, device, ckpt_info = load_model_from_checkpoint(checkpoint_path)
    resolved_normalization = (
        resolve_normalization_mode_for_inference(checkpoint_path=checkpoint_path, default_mode="imagenet")
        if normalization_mode is None
        else resolve_normalization_mode_for_inference(manual_mode=normalization_mode, checkpoint_path=checkpoint_path, default_mode="imagenet")
    )
    resolved_threshold = float(ckpt_info.get("default_threshold", 0.5) if threshold_for_metrics is None else threshold_for_metrics)

    snapshot_path = Path(
        snapshot_download(
            repo_id=hf_dataset_repo_id,
            repo_type="dataset",
            local_dir=str(hf_snapshot_local_dir),
            local_dir_use_symlinks=False,
        )
    )
    total_samples = count_prepared_samples(snapshot_path)

    rows: list[dict[str, object]] = []
    plot_samples: dict[str, list[dict[str, object]]] = {}
    inference_strategy_key = str(inference_strategy).strip().lower()
    batch_size = max(1, int(direct_batch_size)) if inference_strategy_key == "direct" else 1
    pending_batch: list[dict[str, object]] = []
    pending_shape: tuple[int, int, int] | None = None

    def _flush_pending_batch() -> None:
        nonlocal pending_batch, pending_shape
        if not pending_batch:
            return

        if inference_strategy_key == "direct" and len(pending_batch) > 1:
            probs = predict_probability_maps_batch(
                model=model,
                images=[sample["image_t"] for sample in pending_batch],
                device=device,
                normalization_mode=resolved_normalization,
            )
        else:
            probs = [
                predict_probability_map_by_strategy(
                    model=model,
                    image=sample["image_t"],
                    device=device,
                    strategy=inference_strategy_key,
                    normalization_mode=resolved_normalization,
                ).clamp(0.0, 1.0)
                for sample in pending_batch
            ]

        for sample, prob in zip(pending_batch, probs):
            _append_prepared_inference_result(
                rows,
                plot_samples,
                sample,
                prob.clamp(0.0, 1.0),
                threshold_for_metrics=resolved_threshold,
                plot_binary_threshold=plot_binary_threshold,
                inference_strategy=inference_strategy_key,
                normalization_mode=resolved_normalization,
                max_plot_samples_per_dataset=max_plot_samples_per_dataset,
            )

        pending_batch = []
        pending_shape = None

    for sample_uri, data in tqdm(iter_prepared_samples(snapshot_path), desc="Inference", total=total_samples):
        sample = _prepared_sample_to_inference_record(sample_uri, data)
        if sample is None:
            continue

        if inference_strategy_key != "direct":
            pending_batch.append(sample)
            _flush_pending_batch()
            continue

        sample_shape = tuple(int(v) for v in sample["image_t"].shape)
        if pending_shape is not None and sample_shape != pending_shape:
            _flush_pending_batch()

        pending_batch.append(sample)
        pending_shape = sample_shape
        if len(pending_batch) >= batch_size:
            _flush_pending_batch()

    _flush_pending_batch()

    results_df = pd.DataFrame(rows).sort_values(["dataset", "sample_uri"]).reset_index(drop=True)
    if results_df.empty:
        raise RuntimeError("No samples processed from HF snapshot.")

    results_csv = csv_output_dir / "ngiml_hf_test_inference_results.csv"
    summary_csv = csv_output_dir / "ngiml_hf_test_inference_summary_by_dataset.csv"
    comparison_csv = csv_output_dir / "ngiml_hf_test_inference_metric_mode_comparison.csv"
    results_df.to_csv(results_csv, index=False)

    summary_df = results_df.groupby("dataset", as_index=False).agg(
        {
            "sample_uri": "count",
            "f1": "mean",
            "iou": "mean",
            "precision": "mean",
            "recall": "mean",
            "accuracy": "mean",
            "mean_probability": "mean",
            "pred_positive_ratio_threshold": "mean",
            "gt_positive_ratio": "mean",
            "legacy_f1": "mean",
            "legacy_iou": "mean",
        }
    ).rename(columns={"sample_uri": "num_samples"})
    summary_df.to_csv(summary_csv, index=False)
    comparison_df = summary_df[
        [
            "dataset",
            "num_samples",
            "f1",
            "legacy_f1",
            "iou",
            "legacy_iou",
            "precision",
            "recall",
            "accuracy",
            "mean_probability",
            "pred_positive_ratio_threshold",
            "gt_positive_ratio",
        ]
    ].copy()
    comparison_df["f1_gap_legacy_minus_strict"] = comparison_df["legacy_f1"] - comparison_df["f1"]
    comparison_df["iou_gap_legacy_minus_strict"] = comparison_df["legacy_iou"] - comparison_df["iou"]
    comparison_df.to_csv(comparison_csv, index=False)

    for dataset_name, samples in sorted(plot_samples.items()):
        dataset_plot_dir = plot_output_dir / dataset_name
        for idx, sample in enumerate(samples, start=1):
            out_png = dataset_plot_dir / f"{dataset_name}_sample_{idx:02d}.png"
            title = f"{dataset_name} | sample {idx} | {Path(str(sample['sample_uri']).split('::')[0]).name}"
            save_sample_plot(out_png, sample["image_chw"], sample["mask_hw"], sample["prob_hw"], sample["bin05_hw"], title)

    return {
        "model": model,
        "device": device,
        "checkpoint_info": ckpt_info,
        "snapshot_path": snapshot_path,
        "normalization_mode": resolved_normalization,
        "threshold_for_metrics": float(resolved_threshold),
        "plot_binary_threshold": float(plot_binary_threshold),
        "direct_batch_size": int(batch_size),
        "results_df": results_df,
        "summary_df": summary_df,
        "comparison_df": comparison_df,
        "results_csv": results_csv,
        "summary_csv": summary_csv,
        "comparison_csv": comparison_csv,
        "plot_output_dir": plot_output_dir,
    }

def _zero_flop_jit(_inputs, _outputs) -> Counter[str]:
    return Counter()


def _build_flop_analysis(model: torch.nn.Module, sample: torch.Tensor):
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn.jit_handles import elementwise_flop_counter, generic_activation_jit

    elementwise = elementwise_flop_counter(1, 0)
    jit_logger = logging.getLogger("fvcore.nn.jit_analysis")
    previous_level = jit_logger.level
    jit_logger.setLevel(logging.ERROR)
    try:
        analysis = FlopCountAnalysis(model, sample).unsupported_ops_warnings(False).uncalled_modules_warnings(False)
    finally:
        jit_logger.setLevel(previous_level)
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

    for candidate in candidate_files:
        if not candidate.exists():
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return float(payload.get("threshold", fallback)), f"{candidate.name}:fallback"
        except Exception:
            continue

    return float(fallback), "fallback"


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
        return _coerce_model_config(model_config), "train_config.model_config"

    inferred_channels = _infer_fusion_channels_from_state_dict(checkpoint.get("model_state", {}))
    if inferred_channels:
        model_cfg.fusion.fusion_channels = inferred_channels
        return model_cfg, "state_dict.inferred_fusion_channels"

    return model_cfg, "defaults"


def _disable_pretrained_backbones(model_cfg: object) -> object:
    """Prevent backbone weight downloads when instantiating from checkpoints."""
    try:
        if hasattr(model_cfg, "efficientnet") and hasattr(model_cfg.efficientnet, "pretrained"):
            model_cfg.efficientnet.pretrained = False
    except Exception:
        pass
    try:
        if hasattr(model_cfg, "swin") and hasattr(model_cfg.swin, "pretrained"):
            model_cfg.swin.pretrained = False
    except Exception:
        pass
    return model_cfg


def _normalize_profile_input_size(value: object) -> int | None:
    if isinstance(value, int):
        return int(value)
    if isinstance(value, (tuple, list)) and value:
        try:
            if len(value) >= 2:
                return int(max(value[-2], value[-1]))
            return int(value[0])
        except Exception:
            return None
    return None


def _resolve_checkpoint_profile_input_size(train_config: dict, model_cfg: object) -> tuple[int, str]:
    train_value = _normalize_profile_input_size(train_config.get("input_size"))
    if train_value is not None:
        return train_value, "train_config.input_size"

    cfg_candidates = [
        getattr(getattr(model_cfg, "swin", None), "input_size", None),
        getattr(getattr(model_cfg, "efficientnet", None), "input_size", None),
    ]
    for candidate in cfg_candidates:
        resolved = _normalize_profile_input_size(candidate)
        if resolved is not None:
            return resolved, "model_config"

    return 448, "default"


def _select_output_head(outputs: Sequence[torch.Tensor]) -> torch.Tensor:
    if not outputs:
        raise ValueError("Model returned empty predictions list")
    # Highest-resolution decoder output is index 0 by contract.
    return outputs[0]


def _model_uses_residual_noise(model: HybridNGIML) -> bool:
    cfg = getattr(model, "cfg", None)
    cfg_flag = bool(getattr(cfg, "use_residual", True))
    return cfg_flag and (getattr(model, "noise", None) is not None)


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

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    model_cfg, config_source = _build_model_config_from_checkpoint(checkpoint)
    model_cfg = _disable_pretrained_backbones(model_cfg)
    model = HybridNGIML(model_cfg)

    missing, unexpected, skipped_mismatched = _load_state_dict_with_fallback(model, checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    resolved_threshold, threshold_source = resolve_threshold_for_checkpoint(
        Path(checkpoint_path),
        checkpoint_epoch=checkpoint_epoch,
        fallback=0.5,
    )

    train_config = checkpoint.get("train_config") or {}
    profile_input_size, profile_input_size_source = _resolve_checkpoint_profile_input_size(train_config, model_cfg)

    has_train_resize_max_side = "resize_max_side" in train_config
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
        "input_size": int(profile_input_size),
        "input_size_source": str(profile_input_size_source),
        "resize_max_side": int(train_config.get("resize_max_side", 0) or 0),
        "resize_max_side_source": "train_config" if has_train_resize_max_side else "default",
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
def resize_for_inference(
    image: torch.Tensor,
    mask: torch.Tensor | None = None,
    residual_noise: torch.Tensor | None = None,
    resize_max_side: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    cap = int(resize_max_side or 0)
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



def load_image_mask_from_record(
    record: SampleRecord,
    resize_max_side: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    image_path = str(record.image_path)
    if "::" in image_path and image_path.endswith(".npz"):
        image, mask, residual_noise = _load_from_tar_npz(image_path)
    elif image_path.endswith(".npz"):
        image, mask, residual_noise = _load_from_npz(Path(image_path).as_posix())
    else:
        image = _load_image(Path(image_path).as_posix())
        residual_noise = None
        mask = None
        if record.mask_path is not None:
            loaded = _load_image(Path(record.mask_path).as_posix())
            mask = loaded[:1] if loaded.shape[0] > 1 else loaded

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

    # Residual noise is always derived on-the-fly from RGB, independent of
    # whether a legacy manifest still contains residual_noise_path fields.
    residual_noise = _compute_residual_noise(image)

    image, mask, residual_noise = resize_for_inference(image, mask=mask, residual_noise=residual_noise, resize_max_side=resize_max_side)
    return image, mask, residual_noise


def normalize_image_for_inference(image: torch.Tensor, normalization_mode: str = "imagenet") -> torch.Tensor:
    image = image.float()
    if image.max() > 1.0:
        image = image / 255.0
    return _normalize(image, str(normalization_mode).strip().lower())


def predict_probability_map(
    model: HybridNGIML,
    image: torch.Tensor,
    device: torch.device,
    normalization_mode: str = "imagenet",
    residual_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    normalized = normalize_image_for_inference(image, normalization_mode=normalization_mode)
    x = normalized.unsqueeze(0).to(device)
    hp = None
    if _model_uses_residual_noise(model):
        hp_src = residual_noise if residual_noise is not None else _compute_residual_noise(image)
        hp = hp_src.unsqueeze(0).to(device)
    autocast_dtype = get_inference_autocast_dtype(model, device)
    use_amp = device.type == "cuda" and autocast_dtype is not None
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=autocast_dtype or torch.float16, enabled=use_amp):
            outputs = model(x, target_size=image.shape[-2:], residual_noise=hp)
            logits = _select_output_head(outputs)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu()
    return prob


def predict_probability_maps_batch(
    model: HybridNGIML,
    images: Sequence[torch.Tensor],
    device: torch.device,
    normalization_mode: str = "imagenet",
    residual_noises: Sequence[torch.Tensor | None] | None = None,
) -> list[torch.Tensor]:
    if not images:
        return []

    normalized_images: list[torch.Tensor] = []
    resolved_residuals: list[torch.Tensor] = []
    expected_shape: tuple[int, int, int] | None = None
    model_uses_residual = _model_uses_residual_noise(model)

    if residual_noises is not None and len(residual_noises) != len(images):
        raise ValueError("residual_noises must match images length when provided")

    for idx, image in enumerate(images):
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"Expected RGB CHW image tensor, got shape={tuple(image.shape)} at index={idx}")

        image = image.float()
        if image.max() > 1.0:
            image = image / 255.0
        image = image.clamp(0.0, 1.0)

        if expected_shape is None:
            expected_shape = tuple(int(v) for v in image.shape)
        elif tuple(int(v) for v in image.shape) != expected_shape:
            raise ValueError("All images in a direct batch must share the same CHW shape")

        normalized_images.append(normalize_image_for_inference(image, normalization_mode=normalization_mode))
        if model_uses_residual:
            residual = residual_noises[idx] if residual_noises is not None else None
            resolved_residuals.append((residual if residual is not None else _compute_residual_noise(image)).float())

    x = torch.stack(normalized_images, dim=0).to(device, non_blocking=True)
    hp = torch.stack(resolved_residuals, dim=0).to(device, non_blocking=True) if model_uses_residual else None
    autocast_dtype = get_inference_autocast_dtype(model, device)
    use_amp = device.type == "cuda" and autocast_dtype is not None
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=autocast_dtype or torch.float16, enabled=use_amp):
            outputs = model(x, target_size=images[0].shape[-2:], residual_noise=hp)
            logits = _select_output_head(outputs)
            probs = torch.sigmoid(logits[:, 0]).detach().cpu()
    return [probs[i] for i in range(probs.shape[0])]


def predict_binary_map(
    model: HybridNGIML,
    image: torch.Tensor,
    device: torch.device,
    threshold: float | None = None,
    normalization_mode: str = "imagenet",
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
    tile_size: int = 448,
    overlap: float = 0.25,
    tile_batch_size: int = 4,
) -> torch.Tensor:
    """Run overlap-weighted tiled inference and return full-resolution probability map."""
    model_uses_residual = _model_uses_residual_noise(model)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected RGB CHW image tensor, got shape={tuple(image.shape)}")

    image = image.float()
    if image.max() > 1.0:
        image = image / 255.0
    image = image.clamp(0.0, 1.0)

    if model_uses_residual:
        residual_noise = residual_noise if residual_noise is not None else _compute_residual_noise(image)
    else:
        residual_noise = None

    _, h, w = image.shape
    tile = max(64, int(tile_size))
    stride = max(1, int(round(tile * (1.0 - float(overlap)))))
    tile_batch = max(1, int(tile_batch_size))

    pad_h = max(0, tile - h)
    pad_w = max(0, tile - w)
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect").squeeze(0)
        if residual_noise is not None:
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
        hb = torch.stack(residual_tiles, dim=0).to(device, non_blocking=True) if residual_tiles else None

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
            hp_tile = residual_noise[:, y0 : y0 + tile, x0 : x0 + tile] if residual_noise is not None else None

            tile_images.append(normalize_image_for_inference(rgb_tile, normalization_mode=normalization_mode))
            if hp_tile is not None:
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


def _resize_prob_to_original(prob: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    if tuple(prob.shape[-2:]) == (out_h, out_w):
        return prob
    return F.interpolate(
        prob.unsqueeze(0).unsqueeze(0),
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False,
    )[0, 0]


def predict_probability_map_by_strategy(
    model: HybridNGIML,
    image: torch.Tensor,
    device: torch.device,
    strategy: str,
    normalization_mode: str = "imagenet",
    infer_size: int = 448,
    tile_size: int = 448,
    tile_overlap: float = 0.5,
    tile_batch_size: int = 16,
) -> torch.Tensor:
    """Run a single inference strategy and return probability map at original image resolution."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected RGB CHW image tensor, got shape={tuple(image.shape)}")

    image = image.float()
    if image.max() > 1.0:
        image = image / 255.0
    image = image.clamp(0.0, 1.0)

    orig_h, orig_w = int(image.shape[-2]), int(image.shape[-1])
    strategy_key = str(strategy).strip().lower()

    if strategy_key == "direct":
        return predict_probability_map(
            model=model,
            image=image,
            device=device,
            normalization_mode=normalization_mode,
        )

    if strategy_key == "sliding_window":
        return predict_probability_map_sliding_window(
            model=model,
            image=image,
            device=device,
            normalization_mode=normalization_mode,
            tile_size=tile_size,
            overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
        )

    if strategy_key == "resize_keep_aspect_center_crop":
        h, w = image.shape[-2:]
        scale = float(infer_size) / float(min(h, w))
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        resized = TVF.resize(image, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)
        top = max(0, (new_h - infer_size) // 2)
        left = max(0, (new_w - infer_size) // 2)
        cropped = TVF.crop(resized, top, left, infer_size, infer_size)
        prob = predict_probability_map(
            model=model,
            image=cropped,
            device=device,
            normalization_mode=normalization_mode,
        )
        return _resize_prob_to_original(prob, orig_h, orig_w)

    if strategy_key == "center_crop":
        h, w = image.shape[-2:]
        crop_side = min(h, w)
        top = max(0, (h - crop_side) // 2)
        left = max(0, (w - crop_side) // 2)
        cropped = TVF.crop(image, top, left, crop_side, crop_side)
        resized = TVF.resize(cropped, [infer_size, infer_size], interpolation=InterpolationMode.BILINEAR)
        prob = predict_probability_map(
            model=model,
            image=resized,
            device=device,
            normalization_mode=normalization_mode,
        )
        return _resize_prob_to_original(prob, orig_h, orig_w)

    if strategy_key == "resize":
        resized = TVF.resize(image, [infer_size, infer_size], interpolation=InterpolationMode.BILINEAR)
        prob = predict_probability_map(
            model=model,
            image=resized,
            device=device,
            normalization_mode=normalization_mode,
        )
        return _resize_prob_to_original(prob, orig_h, orig_w)

    raise ValueError(
        f"Unknown strategy: {strategy}. "
        "Supported: direct, sliding_window, resize_keep_aspect_center_crop, center_crop, resize"
    )


def run_multi_strategy_inference(
    model: HybridNGIML,
    image: torch.Tensor,
    device: torch.device,
    normalization_mode: str = "imagenet",
    threshold: float | None = None,
    infer_size: int = 448,
    tile_size: int = 448,
    tile_overlap: float = 0.5,
    tile_batch_size: int = 16,
    strategies: Sequence[str] | None = None,
    show_plot: bool = True,
    show_progress: bool = True,
) -> dict[str, object]:
    """Run configured inference strategies, optionally visualize, and return metrics/maps."""
    strategy_list = list(strategies) if strategies is not None else [
        "direct",
        "sliding_window",
        "resize_keep_aspect_center_crop",
        "center_crop",
        "resize",
    ]
    if not strategy_list:
        raise ValueError("At least one strategy must be provided")

    image = image.float()
    if image.max() > 1.0:
        image = image / 255.0
    image = image.clamp(0.0, 1.0)

    resolved_threshold = float(
        threshold if threshold is not None else getattr(model, "default_threshold", 0.5)
    )

    strategy_probs: dict[str, torch.Tensor] = {}
    strategy_bins: dict[str, torch.Tensor] = {}
    summary: dict[str, dict[str, float]] = {}

    iterator = strategy_list
    if show_progress:
        try:
            import importlib

            tqdm_auto = importlib.import_module("tqdm.auto")
            iterator = tqdm_auto.tqdm(strategy_list, desc="Inference strategies", leave=False)
        except Exception:
            iterator = strategy_list

    for strategy_name in iterator:
        prob = predict_probability_map_by_strategy(
            model=model,
            image=image,
            device=device,
            strategy=strategy_name,
            normalization_mode=normalization_mode,
            infer_size=infer_size,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
        ).clamp(0.0, 1.0)
        binary = (prob >= resolved_threshold).float()

        strategy_probs[strategy_name] = prob
        strategy_bins[strategy_name] = binary
        summary[strategy_name] = {
            "mean_probability": float(prob.mean().item()),
            "max_probability": float(prob.max().item()),
            "predicted_positive_ratio": float(binary.mean().item()),
        }

    if show_plot:
        plot_multi_strategy_inference(
            image=image,
            strategy_probs=strategy_probs,
            strategy_bins=strategy_bins,
            threshold=resolved_threshold,
            strategy_order=strategy_list,
        )

    return {
        "strategies": strategy_list,
        "normalization_mode": normalization_mode,
        "threshold": resolved_threshold,
        "infer_size": int(infer_size),
        "tile_size": int(tile_size),
        "tile_overlap": float(tile_overlap),
        "tile_batch_size": int(tile_batch_size),
        "probabilities": strategy_probs,
        "binaries": strategy_bins,
        "summary": summary,
    }


def _epoch_from_checkpoint_name(checkpoint_path: Path) -> int | None:
    match = re.search(r"checkpoint_epoch_(\d+)", checkpoint_path.name)
    if not match:
        return None
    return int(match.group(1))


def list_epoch_checkpoints(checkpoint_dir: str | Path) -> list[Path]:
    checkpoint_root = Path(checkpoint_dir)
    checkpoints = list(checkpoint_root.glob("checkpoint_epoch_*.pt"))
    if not checkpoints:
        return []

    def _sort_key(path: Path) -> tuple[int, float, str]:
        epoch = _epoch_from_checkpoint_name(path)
        epoch_key = int(epoch) if epoch is not None else -1
        try:
            mtime = float(path.stat().st_mtime)
        except OSError:
            mtime = 0.0
        return (epoch_key, mtime, path.name)

    return sorted(checkpoints, key=_sort_key)


def sweep_checkpoint_inference_for_image(
    checkpoint_dir: str | Path,
    image: torch.Tensor,
    normalization_mode: str | None = "imagenet",
    strategy: str = "direct",
    threshold: float | None = None,
    infer_size: int = 448,
    tile_size: int = 448,
    tile_overlap: float = 0.5,
    tile_batch_size: int = 16,
    show_progress: bool = True,
    show_plot: bool = True,
    plot_max_checkpoints: int | None = 8,
) -> dict[str, object]:
    """Run one strategy for one image across all epoch checkpoints in a directory."""
    checkpoint_paths = list_epoch_checkpoints(checkpoint_dir)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoint_epoch_*.pt files found in {checkpoint_dir}")

    image = image.float()
    if image.max() > 1.0:
        image = image / 255.0
    image = image.clamp(0.0, 1.0)

    records: list[dict[str, object]] = []
    strategy_key = str(strategy).strip().lower()
    checkpoint_probs: dict[str, torch.Tensor] = {}
    checkpoint_bins: dict[str, torch.Tensor] = {}
    checkpoint_thresholds: dict[str, float] = {}

    iterator = checkpoint_paths
    if show_progress:
        try:
            import importlib

            tqdm_auto = importlib.import_module("tqdm.auto")
            iterator = tqdm_auto.tqdm(checkpoint_paths, desc="Checkpoint sweep", leave=False)
        except Exception:
            iterator = checkpoint_paths

    for checkpoint_path in iterator:
        model, device, ckpt_info = load_model_from_checkpoint(checkpoint_path)
        resolved_normalization = resolve_normalization_mode_for_inference(
            manual_mode=normalization_mode,
            checkpoint_path=checkpoint_path,
            default_mode="imagenet",
        )
        run = run_multi_strategy_inference(
            model=model,
            image=image,
            device=device,
            normalization_mode=resolved_normalization,
            threshold=threshold,
            infer_size=infer_size,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
            strategies=[strategy_key],
            show_plot=False,
            show_progress=False,
        )

        summary = run["summary"][strategy_key]
        records.append(
            {
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_epoch_from_name": _epoch_from_checkpoint_name(checkpoint_path),
                "checkpoint_epoch_from_state": ckpt_info.get("epoch"),
                "strategy": strategy_key,
                "normalization_mode": resolved_normalization,
                "threshold_used": float(run["threshold"]),
                "mean_probability": float(summary["mean_probability"]),
                "max_probability": float(summary["max_probability"]),
                "predicted_positive_ratio": float(summary["predicted_positive_ratio"]),
            }
        )

        if show_plot:
            checkpoint_label = f"ep{int(ckpt_info.get('epoch', -1))}"
            checkpoint_probs[checkpoint_label] = run["probabilities"][strategy_key]
            checkpoint_bins[checkpoint_label] = run["binaries"][strategy_key]
            checkpoint_thresholds[checkpoint_label] = float(run["threshold"])

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best_by_max_probability = max(records, key=lambda row: float(row["max_probability"]))

    if show_plot and checkpoint_probs:
        labels = list(checkpoint_probs.keys())
        if isinstance(plot_max_checkpoints, int) and plot_max_checkpoints > 0 and len(labels) > plot_max_checkpoints:
            labels = labels[-plot_max_checkpoints:]
        plot_checkpoint_sweep_inference(
            image=image,
            checkpoint_probs={label: checkpoint_probs[label] for label in labels},
            checkpoint_bins={label: checkpoint_bins[label] for label in labels},
            checkpoint_thresholds={label: checkpoint_thresholds[label] for label in labels},
            strategy_name=strategy_key,
            checkpoint_order=labels,
        )

    return {
        "strategy": strategy_key,
        "normalization_mode": normalization_mode,
        "num_checkpoints": len(records),
        "records": records,
        "best_by_max_probability": best_by_max_probability,
    }


def plot_checkpoint_sweep_inference(
    image: torch.Tensor,
    checkpoint_probs: dict[str, torch.Tensor],
    checkpoint_bins: dict[str, torch.Tensor],
    checkpoint_thresholds: dict[str, float],
    strategy_name: str,
    checkpoint_order: Sequence[str] | None = None,
):
    """Render probability, binary, and overlay grids across checkpoints for one strategy."""
    import importlib

    plt = importlib.import_module("matplotlib.pyplot")
    np = importlib.import_module("numpy")

    order = list(checkpoint_order) if checkpoint_order is not None else list(checkpoint_probs.keys())
    if not order:
        raise ValueError("No checkpoints provided for sweep plotting")

    img_np = image.float().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    n = len(order)
    fig, axes = plt.subplots(3, n + 1, figsize=(4 * (n + 1), 12))

    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Uploaded RGB")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")
    axes[2, 0].axis("off")

    for idx, checkpoint_label in enumerate(order, start=1):
        prob_np = checkpoint_probs[checkpoint_label].detach().cpu().numpy()
        bin_np = checkpoint_bins[checkpoint_label].detach().cpu().numpy()
        threshold = float(checkpoint_thresholds.get(checkpoint_label, 0.5))

        overlay_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        alpha = 0.45 * prob_np[..., None]
        overlay_np = np.clip(img_np * (1.0 - alpha) + overlay_color * alpha, 0.0, 1.0)

        axes[0, idx].imshow(prob_np, cmap="magma", vmin=0.0, vmax=1.0)
        axes[0, idx].set_title(f"{checkpoint_label}\\n{strategy_name} Prob")
        axes[0, idx].axis("off")

        axes[1, idx].imshow(bin_np, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, idx].set_title(f"{checkpoint_label}\\nBinary (t={threshold:.2f})")
        axes[1, idx].axis("off")

        axes[2, idx].imshow(overlay_np)
        axes[2, idx].set_title(f"{checkpoint_label}\\nOverlay")
        axes[2, idx].axis("off")

    plt.tight_layout()
    plt.show()
    return fig, axes


def plot_multi_strategy_inference(
    image: torch.Tensor,
    strategy_probs: dict[str, torch.Tensor],
    strategy_bins: dict[str, torch.Tensor],
    threshold: float,
    strategy_order: Sequence[str] | None = None,
):
    """Render probability, binary, and overlay grids for each strategy."""
    import importlib

    plt = importlib.import_module("matplotlib.pyplot")
    np = importlib.import_module("numpy")

    order = list(strategy_order) if strategy_order is not None else list(strategy_probs.keys())
    if not order:
        raise ValueError("No strategies provided for plotting")

    img_np = image.float().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    n = len(order)
    fig, axes = plt.subplots(3, n + 1, figsize=(4 * (n + 1), 12))

    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Uploaded RGB")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")
    axes[2, 0].axis("off")

    for idx, strategy_name in enumerate(order, start=1):
        prob_np = strategy_probs[strategy_name].detach().cpu().numpy()
        bin_np = strategy_bins[strategy_name].detach().cpu().numpy()

        overlay_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        alpha = 0.45 * prob_np[..., None]
        overlay_np = np.clip(img_np * (1.0 - alpha) + overlay_color * alpha, 0.0, 1.0)

        title_name = strategy_name.replace("_", " ").title()

        axes[0, idx].imshow(prob_np, cmap="magma", vmin=0.0, vmax=1.0)
        axes[0, idx].set_title(f"{title_name}\\nProbability")
        axes[0, idx].axis("off")

        axes[1, idx].imshow(bin_np, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, idx].set_title(f"{title_name}\\nBinary (t={threshold:.2f})")
        axes[1, idx].axis("off")

        axes[2, idx].imshow(overlay_np)
        axes[2, idx].set_title(f"{title_name}\\nOverlay")
        axes[2, idx].axis("off")

    plt.tight_layout()
    plt.show()
    return fig, axes


def get_model_complexity_stats(
    model: HybridNGIML,
    input_size: tuple[int, int, int, int] = (1, 3, 448, 448),
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
            from thop import profile as thop_profile

            with torch.no_grad():
                macs, _ = thop_profile(profile_model, inputs=(sample,), verbose=False)
            macs = float(macs)
            stats["macs"] = macs
            stats["flops"] = macs * 2.0
            stats["unsupported_ops"] = None
            stats["flops_source"] = "thop"
            stats["flops_error"] = None
        except Exception as thop_error:
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
                    else "THOP unavailable; fvcore fallback may undercount unsupported ops listed in `unsupported_ops`."
                )
            except Exception as fv_error:
                stats["flops"] = None
                stats["macs"] = None
                stats["unsupported_ops"] = None
                stats["flops_source"] = None
                stats["flops_error"] = (
                    "FLOPs unavailable. "
                    f"thop error: {thop_error}. "
                    f"fvcore error: {fv_error}. "
                    "Try `%pip install thop` (or `%pip install fvcore iopath`) in the active notebook kernel."
                )
    finally:
        model.train(was_training)

    return stats

