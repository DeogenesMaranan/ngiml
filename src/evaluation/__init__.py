"""Standalone evaluation utilities for NGIML models.

Provides reusable metric computation, threshold optimisation, and model
evaluation independent of the training loop.  Modelled after LIFD's
``evaluation/eval_utils.py`` for parity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# ---------------------------------------------------------------------------
#  Core metric helpers
# ---------------------------------------------------------------------------

def segmentation_counts(
    logits: Tensor,
    target: Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute pixel-level TP / TN / FP / FN counts."""
    probs = torch.sigmoid(logits)
    pred = (probs >= float(threshold)).float()
    target = target.float()
    tp = torch.sum(pred * target).item()
    tn = torch.sum((1.0 - pred) * (1.0 - target)).item()
    fp = torch.sum(pred * (1.0 - target)).item()
    fn = torch.sum((1.0 - pred) * target).item()
    return {"tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn)}


def metrics_from_counts(
    tp: float,
    tn: float,
    fp: float,
    fn: float,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Derive IoU, Dice, Precision, Recall, Accuracy from counts."""
    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps) if (precision + recall) > 0 else 0.0
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "f1": float(f1),
    }


def build_threshold_grid(
    start: float = 0.1,
    end: float = 0.9,
    step: float = 0.1,
) -> List[float]:
    """Build a sorted grid of thresholds, always including 0.5."""
    start = float(min(max(start, 0.0), 1.0))
    end = float(min(max(end, 0.0), 1.0))
    step = float(max(step, 1e-6))
    if end < start:
        start, end = end, start

    values: list[float] = []
    t = start
    while t <= (end + 1e-9):
        values.append(round(t, 4))
        t += step

    if not values:
        values = [0.5]
    if 0.5 not in values:
        values.append(0.5)
    return sorted(set(values))


# ---------------------------------------------------------------------------
#  Evaluation summary
# ---------------------------------------------------------------------------

@dataclass
class EvaluationSummary:
    """Container for evaluation results."""

    metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    per_threshold: Dict[float, Dict[str, float]]
    samples: int


def _stats_to_confusion(counts: Dict[str, float]) -> np.ndarray:
    return np.array(
        [
            [counts.get("tn", 0.0), counts.get("fp", 0.0)],
            [counts.get("fn", 0.0), counts.get("tp", 0.0)],
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------
#  High-level evaluation routine
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_loader(
    model: "torch.nn.Module",
    loader: "torch.utils.data.DataLoader",
    *,
    device: torch.device | str | None = None,
    loss_fn: Optional["torch.nn.Module"] = None,
    thresholds: Sequence[float] | None = None,
    optimize_metric: str = "iou",
    amp: bool = False,
    channels_last: bool = False,
    max_batches: int | None = None,
    image_key: str = "images",
    mask_key: str = "masks",
    high_pass_key: str = "high_pass",
) -> EvaluationSummary:
    """Evaluate an NGIML model on a DataLoader.

    Works independently of the training harness, making it suitable for
    standalone evaluation scripts, CI tests, and inference notebooks.

    Args:
        model: An NGIML model (or any model whose ``forward`` returns
            ``List[Tensor]`` with the final prediction last).
        loader: DataLoader yielding dicts with at least ``image_key`` and
            ``mask_key`` tensors.
        device: Device to run evaluation on.
        loss_fn: Optional loss function accepting ``(preds, masks)``; when
            provided the mean loss is included in the returned metrics.
        thresholds: Explicit threshold grid.  Defaults to
            ``build_threshold_grid()``.
        optimize_metric: Metric name used to select the best threshold
            (``"iou"``, ``"dice"``, ``"f1"``).
        amp: Whether to use automatic mixed precision.
        channels_last: Whether to convert images to ``channels_last`` layout.
        max_batches: Cap evaluation at this many batches (useful for quick
            smoke tests).
        image_key / mask_key / high_pass_key: Dict keys produced by the
            dataloader.

    Returns:
        An :class:`EvaluationSummary` containing metrics at the best
        threshold, per-threshold metrics, and a confusion matrix.
    """
    resolved_device = _resolve_device(device)
    model.eval()

    if thresholds is None:
        thresholds = build_threshold_grid()
    thresholds = sorted(set(float(t) for t in thresholds))

    threshold_stats: Dict[float, Dict[str, float]] = {
        t: {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0} for t in thresholds
    }
    total_loss = 0.0
    batches = 0

    iterator = loader
    if tqdm:
        iterator = tqdm(iterator, desc="Evaluating", leave=False)

    for batch_idx, batch in enumerate(iterator):
        images = batch[image_key].to(resolved_device, non_blocking=True)
        masks = batch[mask_key].to(resolved_device, non_blocking=True)
        high_pass = batch.get(high_pass_key)
        if isinstance(high_pass, Tensor):
            high_pass = high_pass.to(resolved_device, non_blocking=True)
        else:
            high_pass = None

        if channels_last and resolved_device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
            if high_pass is not None:
                high_pass = high_pass.contiguous(memory_format=torch.channels_last)

        with torch.amp.autocast(device_type=resolved_device.type, enabled=amp):
            preds = model(images, target_size=masks.shape[-2:], high_pass=high_pass)
            if loss_fn is not None:
                loss = loss_fn(preds, masks)
                if torch.isfinite(loss):
                    total_loss += loss.item()

        logits = preds[-1]
        if not torch.isfinite(logits).all():
            continue  # skip non-finite batches

        for thr in thresholds:
            counts = segmentation_counts(logits, masks, threshold=thr)
            for k in ("tp", "tn", "fp", "fn"):
                threshold_stats[thr][k] += counts[k]

        batches += 1
        if max_batches is not None and batches >= max_batches:
            break

    if batches == 0:
        raise RuntimeError("No valid batches evaluated; check data / model output.")

    # Compute per-threshold metrics
    per_threshold: Dict[float, Dict[str, float]] = {}
    for thr, stats in threshold_stats.items():
        per_threshold[thr] = metrics_from_counts(stats["tp"], stats["tn"], stats["fp"], stats["fn"])

    # Select best threshold
    opt_key = optimize_metric.lower()
    if opt_key not in {"iou", "dice", "f1", "precision", "recall"}:
        opt_key = "iou"

    best_thr = max(per_threshold, key=lambda t: per_threshold[t].get(opt_key, 0.0))
    best_metrics = dict(per_threshold[best_thr])
    best_metrics["threshold"] = best_thr
    if loss_fn is not None:
        best_metrics["loss"] = total_loss / batches

    confusion = _stats_to_confusion(threshold_stats[best_thr])

    return EvaluationSummary(
        metrics=best_metrics,
        confusion_matrix=confusion,
        per_threshold=per_threshold,
        samples=batches,
    )


def select_best_threshold(
    per_threshold: Dict[float, Dict[str, float]],
    metric: str = "iou",
) -> Tuple[float, Dict[str, float]]:
    """Given per-threshold metrics, return ``(best_threshold, metrics)``."""
    metric = metric.lower()
    best_thr = max(per_threshold, key=lambda t: per_threshold[t].get(metric, 0.0))
    return best_thr, per_threshold[best_thr]


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


__all__ = [
    "EvaluationSummary",
    "build_threshold_grid",
    "evaluate_loader",
    "metrics_from_counts",
    "segmentation_counts",
    "select_best_threshold",
]
