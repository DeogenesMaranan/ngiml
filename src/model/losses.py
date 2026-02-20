"""Training losses for NGIML multi-stage localization."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SoftDiceLoss(nn.Module):
    """Soft Dice operating on logits for stable gradients."""

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        probs = torch.sigmoid(logits)
        target = target.float()
        dims = (1, 2, 3)
        intersection = torch.sum(probs * target, dim=dims)
        denom = torch.sum(probs, dim=dims) + torch.sum(target, dim=dims)
        dice = (2 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class FocalWithLogitsLoss(nn.Module):
    """Binary focal loss operating on logits for class-imbalance robustness."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = float(max(0.0, gamma))
        self.alpha = float(min(max(alpha, 0.0), 1.0))
        self.eps = eps

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        target = target.float()
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)

        pt = target * probs + (1.0 - target) * (1.0 - probs)
        alpha_t = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)
        focal_weight = alpha_t * torch.pow(1.0 - pt, self.gamma)

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = focal_weight * bce
        return loss.mean()


class TverskyLoss(nn.Module):
    """Tversky loss with logits input; beta > alpha emphasizes recall."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6) -> None:
        super().__init__()
        self.alpha = float(max(0.0, alpha))
        self.beta = float(max(0.0, beta))
        self.smooth = smooth

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        probs = torch.sigmoid(logits)
        target = target.float()

        dims = (1, 2, 3)
        tp = torch.sum(probs * target, dim=dims)
        fp = torch.sum(probs * (1.0 - target), dim=dims)
        fn = torch.sum((1.0 - probs) * target, dim=dims)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


@dataclass
class MultiStageLossConfig:
    """Configuration flags for the combined Dice + weighted BCE loss."""

    dice_weight: float = 1.0
    bce_weight: float = 1.0
    pos_weight: float = 2.0
    stage_weights: Optional[Sequence[float]] = None
    smooth: float = 1e-6
    hybrid_mode: str = "dice_bce"  # one of: dice_bce, dice_focal
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    tversky_weight: float = 0.0
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7


class MultiStageManipulationLoss(nn.Module):
    """Applies configurable hybrid segmentation supervision at every prediction stage."""

    def __init__(self, config: MultiStageLossConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or MultiStageLossConfig()
        self.dice = SoftDiceLoss(smooth=self.cfg.smooth)
        self.focal = FocalWithLogitsLoss(gamma=self.cfg.focal_gamma, alpha=self.cfg.focal_alpha)
        self.tversky = TverskyLoss(
            alpha=self.cfg.tversky_alpha,
            beta=self.cfg.tversky_beta,
            smooth=self.cfg.smooth,
        )

        mode = self.cfg.hybrid_mode.strip().lower()
        if mode not in {"dice_bce", "dice_focal"}:
            raise ValueError("hybrid_mode must be one of: 'dice_bce', 'dice_focal'")
        self.hybrid_mode = mode

    def _stage_weights(self, num_stages: int) -> List[float]:
        if self.cfg.stage_weights is None:
            return [float(i + 1) / float(num_stages) for i in range(num_stages)]
        if len(self.cfg.stage_weights) < num_stages:
            raise ValueError("Provided stage_weights shorter than number of stages")
        return list(self.cfg.stage_weights[:num_stages])

    def forward(self, preds: List[Tensor], target: Tensor) -> Tensor:
        if not preds:
            raise ValueError("Loss received empty predictions list")
        target = target.float()
        stage_weights = self._stage_weights(len(preds))
        pos_weight = torch.as_tensor(
            self.cfg.pos_weight,
            dtype=target.dtype,
            device=target.device,
        )

        total_loss = 0.0
        normalizer = 0.0
        for stage_weight, logits in zip(stage_weights, preds):
            if logits.shape[-2:] != target.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            dice = self.dice(logits, target)
            if self.hybrid_mode == "dice_bce":
                bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
                hybrid_term = self.cfg.bce_weight * bce
            else:
                focal = self.focal(logits, target)
                hybrid_term = self.cfg.bce_weight * focal

            stage_loss = self.cfg.dice_weight * dice + hybrid_term
            if self.cfg.tversky_weight > 0:
                stage_loss = stage_loss + self.cfg.tversky_weight * self.tversky(logits, target)

            total_loss += stage_weight * stage_loss
            normalizer += stage_weight

        return total_loss / max(normalizer, 1e-6)


__all__ = [
    "SoftDiceLoss",
    "FocalWithLogitsLoss",
    "TverskyLoss",
    "MultiStageLossConfig",
    "MultiStageManipulationLoss",
]
