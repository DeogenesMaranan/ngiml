from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
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
from src.model.hybrid_ngiml import HybridNGIML
from tools.train_ngiml import build_default_components
import numpy as np
import json
import tarfile
import io
import matplotlib.pyplot as plt

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

def compute_binary_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray) -> dict[str, float]:
    pred = pred_bin.astype(bool)
    gt = gt_bin.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    tn = float(np.logical_and(~pred, ~gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou, 'accuracy': acc}

def save_sample_plot(out_path: Path, image_chw: np.ndarray, gt_hw: np.ndarray, prob_hw: np.ndarray, bin05_hw: np.ndarray, title: str):
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
    snapshot_root = Path(snapshot_root)
    npz_count = sum(1 for _ in snapshot_root.rglob('*.npz'))
    tar_member_count = 0
    for tar_path in snapshot_root.rglob('*.tar'):
        with tarfile.open(tar_path, mode='r') as tf:
            tar_member_count += sum(
                1
                for member in tf.getmembers()
                if member.isfile() and member.name.lower().endswith('.npz')
            )
    return int(npz_count + tar_member_count)


def download_hf_snapshot(
    repo_id: str,
    local_dir: str | Path,
    repo_type: str = 'dataset',
    local_dir_use_symlinks: bool = False,
) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(local_dir),
        local_dir_use_symlinks=local_dir_use_symlinks,
    )
    return Path(snapshot_path)


def _sample_label_from_meta(meta: dict, mask_hw: np.ndarray) -> int:
    raw_label = meta.get('label', int(mask_hw.max() > 0))
    if isinstance(raw_label, str):
        return 1 if raw_label.strip().lower() in {'1', 'fake', 'tp', 'tampered', 'manipulated'} else 0
    return int(raw_label)


def select_prepared_test_sample(
    snapshot_root: str | Path,
    fake_only: bool = True,
    sample_index: int = 0,
    dataset_name: str | None = None,
) -> dict[str, object]:
    snapshot_root = Path(snapshot_root)
    matches: list[dict[str, object]] = []

    for sample_uri, data in iter_prepared_samples(snapshot_root):
        if 'image' not in data:
            continue

        image_chw = _to_chw_rgb(np.asarray(data['image']))
        h, w = int(image_chw.shape[1]), int(image_chw.shape[2])
        mask_hw = _to_hw_mask(data.get('mask'), h, w)
        meta = _parse_meta(data.get('metadata_json'))
        sample_dataset = _dataset_name(sample_uri, meta)
        sample_label = _sample_label_from_meta(meta, mask_hw)

        if dataset_name and str(sample_dataset).strip().lower() != str(dataset_name).strip().lower():
            continue
        if fake_only and sample_label != 1:
            continue

        image_t = torch.from_numpy(image_chw).float()
        if image_t.max() > 1.0:
            image_t = image_t / 255.0

        matches.append(
            {
                'sample_uri': sample_uri,
                'dataset': sample_dataset,
                'label': sample_label,
                'image': image_t.clamp(0.0, 1.0),
                'image_chw': image_chw,
                'mask_hw': mask_hw,
                'metadata': meta,
            }
        )

        if len(matches) > sample_index:
            return matches[sample_index]

    sample_desc = f"dataset={dataset_name!r}, fake_only={fake_only}, sample_index={sample_index}"
    raise RuntimeError(f'No prepared test sample matched selection: {sample_desc}')


def _predict_probability_maps_batch_direct(
    model: HybridNGIML,
    images: list[torch.Tensor],
    device: torch.device,
    normalization_mode: str = 'zero_one',
) -> list[torch.Tensor]:
    if not images:
        return []

    # Preserve old direct-inference semantics by only batching images that already
    # have identical spatial sizes. Padding mixed sizes changed model inputs and
    # produced different scores for older checkpoints.
    grouped_indices: dict[tuple[int, int], list[int]] = {}
    for idx, image in enumerate(images):
        key = (int(image.shape[-2]), int(image.shape[-1]))
        grouped_indices.setdefault(key, []).append(idx)

    results: list[torch.Tensor | None] = [None] * len(images)
    autocast_dtype = get_inference_autocast_dtype(model, device)
    use_amp = device.type == 'cuda' and autocast_dtype is not None

    for (_, _), indices in grouped_indices.items():
        normalized_batch = [
            normalize_image_for_inference(images[idx], normalization_mode=normalization_mode)
            for idx in indices
        ]
        xb = torch.stack(normalized_batch, dim=0).to(device, non_blocking=True)

        hb = None
        if _model_uses_residual_noise(model):
            residual_batch = [_compute_residual_noise(images[idx]).float() for idx in indices]
            hb = torch.stack(residual_batch, dim=0).to(device, non_blocking=True)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=autocast_dtype or torch.float16, enabled=use_amp):
                outputs = model(xb, target_size=xb.shape[-2:], residual_noise=hb)
                logits = _select_output_head(outputs)
                probs = torch.sigmoid(logits[:, 0]).float().cpu()

        for batch_idx, image_idx in enumerate(indices):
            results[image_idx] = probs[batch_idx].clone()

    return [result for result in results if result is not None]


def predict_probability_maps_by_strategy_batch(
    model: HybridNGIML,
    images: list[torch.Tensor],
    device: torch.device,
    strategy: str,
    normalization_mode: str = 'imagenet',
    infer_size: int = 448,
    tile_size: int = 448,
    tile_overlap: float = 0.5,
    tile_batch_size: int = 16,
) -> list[torch.Tensor]:
    if not images:
        return []

    strategy_key = str(strategy).strip().lower()

    if strategy_key == 'direct':
        return _predict_probability_maps_batch_direct(
            model=model,
            images=images,
            device=device,
            normalization_mode=normalization_mode,
        )

    if strategy_key in {'resize', 'center_crop', 'resize_keep_aspect_center_crop'}:
        prepared: list[torch.Tensor] = []
        original_sizes: list[tuple[int, int]] = []
        for image in images:
            image = image.float()
            if image.max() > 1.0:
                image = image / 255.0
            image = image.clamp(0.0, 1.0)
            orig_h, orig_w = int(image.shape[-2]), int(image.shape[-1])
            original_sizes.append((orig_h, orig_w))

            if strategy_key == 'resize':
                transformed = TVF.resize(image, [infer_size, infer_size], interpolation=InterpolationMode.BILINEAR)
            elif strategy_key == 'center_crop':
                crop_side = min(orig_h, orig_w)
                top = max(0, (orig_h - crop_side) // 2)
                left = max(0, (orig_w - crop_side) // 2)
                cropped = TVF.crop(image, top, left, crop_side, crop_side)
                transformed = TVF.resize(cropped, [infer_size, infer_size], interpolation=InterpolationMode.BILINEAR)
            else:
                scale = float(infer_size) / float(min(orig_h, orig_w))
                new_h = max(1, int(round(orig_h * scale)))
                new_w = max(1, int(round(orig_w * scale)))
                resized = TVF.resize(image, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)
                top = max(0, (new_h - infer_size) // 2)
                left = max(0, (new_w - infer_size) // 2)
                transformed = TVF.crop(resized, top, left, infer_size, infer_size)

            prepared.append(transformed)

        probs = _predict_probability_maps_batch_direct(
            model=model,
            images=prepared,
            device=device,
            normalization_mode=normalization_mode,
        )
        return [
            _resize_prob_to_original(prob, out_h, out_w)
            for prob, (out_h, out_w) in zip(probs, original_sizes)
        ]

    return [
        predict_probability_map_by_strategy(
            model=model,
            image=image,
            device=device,
            strategy=strategy_key,
            normalization_mode=normalization_mode,
            infer_size=infer_size,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
        )
        for image in images
    ]


def run_prepared_test_inference(
    model: HybridNGIML,
    device: torch.device,
    snapshot_root: str | Path,
    inference_strategy: str = 'direct',
    normalization_mode: str = 'imagenet',
    inference_batch_size: int = 64,
    threshold_for_metrics: float | None = None,
    plot_binary_threshold: float = 0.5,
    csv_output_dir: str | Path | None = None,
    plot_output_dir: str | Path | None = None,
    max_plot_samples_per_dataset: int = 5,
    show_progress: bool = True,
) -> dict[str, object]:
    import pandas as pd

    snapshot_root = Path(snapshot_root)
    total_samples = count_prepared_samples(snapshot_root)
    rows: list[dict[str, object]] = []
    plot_samples: dict[str, list[dict[str, object]]] = {}
    pending_samples: list[tuple[str, dict]] = []
    resolved_batch_size = max(1, int(inference_batch_size))
    threshold_used = float(
        threshold_for_metrics if threshold_for_metrics is not None else getattr(model, 'default_threshold', 0.5)
    )

    iterator = iter_prepared_samples(snapshot_root)
    if show_progress:
        try:
            import importlib

            tqdm_auto = importlib.import_module('tqdm.auto')
            iterator = tqdm_auto.tqdm(iterator, desc='Inference', total=total_samples)
        except Exception:
            iterator = iter_prepared_samples(snapshot_root)

    def _flush_pending() -> None:
        nonlocal pending_samples
        if not pending_samples:
            return
        batch_records: list[tuple[str, dict, np.ndarray, np.ndarray, dict, str, int]] = []
        batch_images: list[torch.Tensor] = []
        for pending_uri, pending_data in pending_samples:
            image_chw = _to_chw_rgb(np.asarray(pending_data['image']))
            h, w = int(image_chw.shape[1]), int(image_chw.shape[2])
            mask_hw = _to_hw_mask(pending_data.get('mask'), h, w)
            meta = _parse_meta(pending_data.get('metadata_json'))
            dataset = _dataset_name(pending_uri, meta)
            sample_label = _sample_label_from_meta(meta, mask_hw)
            image_t = torch.from_numpy(image_chw).float()
            if image_t.max() > 1.0:
                image_t = image_t / 255.0
            batch_records.append((pending_uri, pending_data, image_chw, mask_hw, meta, dataset, sample_label))
            batch_images.append(image_t.clamp(0.0, 1.0))

        batch_probs = predict_probability_maps_by_strategy_batch(
            model=model,
            images=batch_images,
            device=device,
            strategy=inference_strategy,
            normalization_mode=normalization_mode,
        )

        for (pending_uri, _pending_data, image_chw, mask_hw, meta, dataset, sample_label), prob in zip(batch_records, batch_probs):
            prob_hw = prob.detach().cpu().numpy().astype(np.float32)
            pred_bin_metric = (prob_hw >= threshold_used).astype(np.uint8)
            pred_bin_plot = (prob_hw >= float(plot_binary_threshold)).astype(np.uint8)
            metrics = compute_binary_metrics(pred_bin_metric, mask_hw)
            row = {
                'dataset': dataset,
                'sample_uri': pending_uri,
                'split': str(meta.get('split', 'test')),
                'label': sample_label,
                'strategy': str(inference_strategy),
                'normalization_mode': normalization_mode,
                'threshold_for_metrics': threshold_used,
                'plot_binary_threshold': float(plot_binary_threshold),
                'height': int(image_chw.shape[1]),
                'width': int(image_chw.shape[2]),
                'mean_probability': float(prob_hw.mean()),
                'max_probability': float(prob_hw.max()),
                'pred_positive_ratio_threshold': float(pred_bin_metric.mean()),
                'pred_positive_ratio_0_5': float(pred_bin_plot.mean()),
                'gt_positive_ratio': float(mask_hw.mean()),
            }
            row.update(metrics)
            rows.append(row)

            if sample_label == 1:
                plot_record = {
                    'dataset': dataset,
                    'sample_uri': pending_uri,
                    'image_chw': image_chw,
                    'mask_hw': mask_hw,
                    'prob_hw': prob_hw,
                    'bin05_hw': pred_bin_plot,
                }
                ds_bucket = plot_samples.setdefault(str(plot_record['dataset']), [])
                if len(ds_bucket) < int(max_plot_samples_per_dataset):
                    ds_bucket.append(plot_record)
        pending_samples = []

    for sample_uri, data in iterator:
        if 'image' not in data:
            continue
        pending_samples.append((sample_uri, data))
        if len(pending_samples) >= resolved_batch_size:
            _flush_pending()

    _flush_pending()

    results_df = pd.DataFrame(rows).sort_values(['dataset', 'sample_uri']).reset_index(drop=True)
    if results_df.empty:
        raise RuntimeError(f'No samples processed from snapshot: {snapshot_root}')

    summary_df = results_df.groupby('dataset', as_index=False).agg(
        {
            'sample_uri': 'count',
            'f1': 'mean',
            'iou': 'mean',
            'precision': 'mean',
            'recall': 'mean',
            'accuracy': 'mean',
            'mean_probability': 'mean',
            'pred_positive_ratio_threshold': 'mean',
            'gt_positive_ratio': 'mean',
        }
    ).rename(columns={'sample_uri': 'num_samples'})

    results_csv_path: Path | None = None
    summary_csv_path: Path | None = None
    if csv_output_dir is not None:
        csv_output_dir = Path(csv_output_dir)
        csv_output_dir.mkdir(parents=True, exist_ok=True)
        results_csv_path = csv_output_dir / 'ngiml_hf_test_inference_results.csv'
        summary_csv_path = csv_output_dir / 'ngiml_hf_test_inference_summary_by_dataset.csv'
        results_df.to_csv(results_csv_path, index=False)
        summary_df.to_csv(summary_csv_path, index=False)

    if plot_output_dir is not None:
        plot_output_dir = Path(plot_output_dir)
        plot_output_dir.mkdir(parents=True, exist_ok=True)
        for ds_name, samples in sorted(plot_samples.items()):
            ds_dir = plot_output_dir / ds_name
            for i, sample in enumerate(samples, start=1):
                out_png = ds_dir / f'{ds_name}_sample_{i:02d}.png'
                title = f"{ds_name} | sample {i} | {Path(str(sample['sample_uri']).split('::')[0]).name}"
                save_sample_plot(
                    out_png,
                    sample['image_chw'],
                    sample['mask_hw'],
                    sample['prob_hw'],
                    sample['bin05_hw'],
                    title,
                )

    return {
        'snapshot_path': snapshot_root,
        'num_samples': int(len(results_df)),
        'device': str(device),
        'strategy': str(inference_strategy),
        'inference_batch_size': resolved_batch_size,
        'normalization_mode': normalization_mode,
        'threshold_used': threshold_used,
        'plot_binary_threshold': float(plot_binary_threshold),
        'results_df': results_df,
        'summary_df': summary_df,
        'results_csv': results_csv_path,
        'summary_csv': summary_csv_path,
        'plot_output_dir': Path(plot_output_dir) if plot_output_dir is not None else None,
    }


def run_prepared_test_inference_from_hf_dataset(
    checkpoint_path: str | Path,
    hf_dataset_repo_id: str,
    snapshot_local_dir: str | Path,
    output_root: str | Path,
    inference_strategy: str = 'direct',
    inference_batch_size: int = 64,
    threshold_for_metrics: float | None = None,
    plot_binary_threshold: float = 0.5,
    normalization_mode: str | None = None,
    max_plot_samples_per_dataset: int = 5,
    local_dir_use_symlinks: bool = False,
    show_progress: bool = True,
) -> dict[str, object]:
    checkpoint_path = Path(checkpoint_path)
    output_root = Path(output_root)
    csv_output_dir = output_root / 'csv'
    plot_output_dir = output_root / 'plots'
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    plot_output_dir.mkdir(parents=True, exist_ok=True)

    model, device, ckpt_info = load_model_from_checkpoint(checkpoint_path)
    resolved_normalization = normalization_mode or resolve_normalization_mode_for_inference(
        checkpoint_path=checkpoint_path,
        default_mode='imagenet',
    )
    snapshot_path = download_hf_snapshot(
        repo_id=hf_dataset_repo_id,
        local_dir=snapshot_local_dir,
        repo_type='dataset',
        local_dir_use_symlinks=local_dir_use_symlinks,
    )

    run = run_prepared_test_inference(
        model=model,
        device=device,
        snapshot_root=snapshot_path,
        inference_strategy=inference_strategy,
        normalization_mode=resolved_normalization,
        inference_batch_size=inference_batch_size,
        threshold_for_metrics=threshold_for_metrics,
        plot_binary_threshold=plot_binary_threshold,
        csv_output_dir=csv_output_dir,
        plot_output_dir=plot_output_dir,
        max_plot_samples_per_dataset=max_plot_samples_per_dataset,
        show_progress=show_progress,
    )
    run['checkpoint_path'] = checkpoint_path
    run['checkpoint_info'] = ckpt_info
    return run


def sweep_prepared_test_inference_across_checkpoints_from_hf_dataset(
    checkpoint_dir: str | Path,
    hf_dataset_repo_id: str,
    snapshot_local_dir: str | Path,
    inference_strategy: str = 'direct',
    inference_batch_size: int = 64,
    threshold_for_metrics: float | None = None,
    plot_binary_threshold: float = 0.5,
    normalization_mode: str | None = None,
    local_dir_use_symlinks: bool = False,
    show_progress: bool = True,
) -> dict[str, object]:
    import pandas as pd

    checkpoint_paths = list_epoch_checkpoints(checkpoint_dir)
    if not checkpoint_paths:
        raise FileNotFoundError(f'No checkpoint_epoch_*.pt files found in {checkpoint_dir}')

    snapshot_path = download_hf_snapshot(
        repo_id=hf_dataset_repo_id,
        local_dir=snapshot_local_dir,
        repo_type='dataset',
        local_dir_use_symlinks=local_dir_use_symlinks,
    )

    records: list[dict[str, object]] = []
    iterator = checkpoint_paths
    if show_progress:
        try:
            import importlib

            tqdm_auto = importlib.import_module('tqdm.auto')
            iterator = tqdm_auto.tqdm(checkpoint_paths, desc='Prepared test epoch sweep', leave=False)
        except Exception:
            iterator = checkpoint_paths

    for checkpoint_path in iterator:
        model, device, ckpt_info = load_model_from_checkpoint(checkpoint_path)
        resolved_normalization = normalization_mode or resolve_normalization_mode_for_inference(
            checkpoint_path=checkpoint_path,
            default_mode='imagenet',
        )
        run = run_prepared_test_inference(
            model=model,
            device=device,
            snapshot_root=snapshot_path,
            inference_strategy=inference_strategy,
            normalization_mode=resolved_normalization,
            inference_batch_size=inference_batch_size,
            threshold_for_metrics=threshold_for_metrics,
            plot_binary_threshold=plot_binary_threshold,
            csv_output_dir=None,
            plot_output_dir=None,
            max_plot_samples_per_dataset=0,
            show_progress=False,
        )

        summary_df = run['summary_df']
        for row in summary_df.to_dict(orient='records'):
            records.append(
                {
                    'checkpoint_path': str(checkpoint_path),
                    'checkpoint_epoch_from_name': _epoch_from_checkpoint_name(Path(checkpoint_path)),
                    'checkpoint_epoch_from_state': ckpt_info.get('epoch'),
                    'dataset': row.get('dataset'),
                    'num_samples': int(row.get('num_samples', 0)),
                    'f1': float(row.get('f1', 0.0)),
                    'iou': float(row.get('iou', 0.0)),
                    'precision': float(row.get('precision', 0.0)),
                    'recall': float(row.get('recall', 0.0)),
                    'accuracy': float(row.get('accuracy', 0.0)),
                    'mean_probability': float(row.get('mean_probability', 0.0)),
                    'pred_positive_ratio_threshold': float(row.get('pred_positive_ratio_threshold', 0.0)),
                    'gt_positive_ratio': float(row.get('gt_positive_ratio', 0.0)),
                    'normalization_mode': resolved_normalization,
                    'threshold_used': float(run['threshold_used']),
                    'strategy': str(inference_strategy),
                    'inference_batch_size': int(run['inference_batch_size']),
                }
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(records).sort_values(
        ['checkpoint_epoch_from_name', 'checkpoint_epoch_from_state', 'dataset']
    ).reset_index(drop=True)
    if results_df.empty:
        raise RuntimeError('No records produced during prepared test checkpoint sweep')

    pivot_f1 = results_df.pivot_table(
        index='checkpoint_epoch_from_name',
        columns='dataset',
        values='f1',
        aggfunc='first',
    ).sort_index()
    pivot_iou = results_df.pivot_table(
        index='checkpoint_epoch_from_name',
        columns='dataset',
        values='iou',
        aggfunc='first',
    ).sort_index()

    return {
        'snapshot_path': snapshot_path,
        'num_checkpoints': len(checkpoint_paths),
        'results_df': results_df,
        'f1_by_dataset': pivot_f1,
        'iou_by_dataset': pivot_iou,
    }


def sweep_checkpoint_inference_for_prepared_sample_from_hf_dataset(
    checkpoint_dir: str | Path,
    hf_dataset_repo_id: str,
    snapshot_local_dir: str | Path,
    normalization_mode: str = 'imagenet',
    strategy: str = 'direct',
    threshold: float | None = None,
    infer_size: int = 448,
    tile_size: int = 448,
    tile_overlap: float = 0.5,
    tile_batch_size: int = 16,
    fake_only: bool = True,
    sample_index: int = 0,
    dataset_name: str | None = None,
    local_dir_use_symlinks: bool = False,
    show_progress: bool = True,
    show_plot: bool = True,
    plot_max_checkpoints: int | None = 8,
) -> dict[str, object]:
    snapshot_path = download_hf_snapshot(
        repo_id=hf_dataset_repo_id,
        local_dir=snapshot_local_dir,
        repo_type='dataset',
        local_dir_use_symlinks=local_dir_use_symlinks,
    )
    selected = select_prepared_test_sample(
        snapshot_root=snapshot_path,
        fake_only=fake_only,
        sample_index=sample_index,
        dataset_name=dataset_name,
    )
    run = sweep_checkpoint_inference_for_image(
        checkpoint_dir=checkpoint_dir,
        image=selected['image'],
        normalization_mode=normalization_mode,
        strategy=strategy,
        threshold=threshold,
        infer_size=infer_size,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        tile_batch_size=tile_batch_size,
        show_progress=show_progress,
        show_plot=show_plot,
        plot_max_checkpoints=plot_max_checkpoints,
    )
    run['snapshot_path'] = snapshot_path
    run['selected_sample'] = selected
    return run

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

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    model_cfg, config_source = _build_model_config_from_checkpoint(checkpoint)
    model_cfg = _disable_pretrained_backbones(model_cfg)
    model = HybridNGIML(model_cfg).to(device)

    missing, unexpected, skipped_mismatched = _load_state_dict_with_fallback(model, checkpoint["model_state"])
    model.eval()
    resolved_threshold, threshold_source = resolve_threshold_for_checkpoint(
        Path(checkpoint_path),
        checkpoint_epoch=checkpoint_epoch,
        fallback=0.5,
    )

    train_config = checkpoint.get("train_config") or {}

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


def _resolve_possible_local_path(path_str: str) -> str:
    path = Path(path_str)
    return path.as_posix()


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
        image, mask, residual_noise = _load_from_npz(_resolve_possible_local_path(image_path))
    else:
        image = _load_image(_resolve_possible_local_path(image_path))
        residual_noise = None
        mask = None
        if record.mask_path is not None:
            loaded = _load_image(_resolve_possible_local_path(record.mask_path))
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
    normalization_mode: str = "imagenet",
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
        run = run_multi_strategy_inference(
            model=model,
            image=image,
            device=device,
            normalization_mode=normalization_mode,
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
                "normalization_mode": normalization_mode,
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



