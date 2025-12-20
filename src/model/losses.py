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


@dataclass
class MultiStageLossConfig:
    """Configuration flags for the combined Dice + weighted BCE loss."""

    dice_weight: float = 1.0
    bce_weight: float = 1.0
    pos_weight: float = 2.0
    stage_weights: Optional[Sequence[float]] = None
    smooth: float = 1e-6


class MultiStageManipulationLoss(nn.Module):
    """Applies Dice + weighted BCE supervision at every prediction stage."""

    def __init__(self, config: MultiStageLossConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or MultiStageLossConfig()
        self.dice = SoftDiceLoss(smooth=self.cfg.smooth)

    def _stage_weights(self, num_stages: int) -> List[float]:
        if self.cfg.stage_weights is None:
            return [1.0 for _ in range(num_stages)]
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

            bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            dice = self.dice(logits, target)
            stage_loss = self.cfg.bce_weight * bce + self.cfg.dice_weight * dice

            total_loss += stage_weight * stage_loss
            normalizer += stage_weight

        return total_loss / max(normalizer, 1e-6)


__all__ = [
    "SoftDiceLoss",
    "MultiStageLossConfig",
    "MultiStageManipulationLoss",
]
