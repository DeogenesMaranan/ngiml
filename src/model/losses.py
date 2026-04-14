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


class LovaszHingeLoss(nn.Module):
    """Lovasz Hinge Loss for optimizing IoU directly."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        signs = targets * 2 - 1
        errors = 1 - logits * signs
        errors_sorted, perm = torch.sort(errors.view(errors.size(0), -1), dim=1, descending=True)
        perm = perm.detach()
        targets_sorted = targets.view(targets.size(0), -1).gather(1, perm)
        grad = self._lovasz_grad(targets_sorted)
        return (F.relu(errors_sorted) * grad).mean()

    @staticmethod
    def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        """Compute gradient of the Lovasz extension w.r.t sorted errors."""
        gts = gt_sorted.sum(dim=1, keepdim=True)
        intersection = gts - gt_sorted.cumsum(dim=1)
        union = gts + (1 - gt_sorted).cumsum(dim=1)
        jacc = 1 - intersection / union.clamp_min(1.0)
        return torch.cat([jacc[:, :1], jacc[:, 1:] - jacc[:, :-1]], dim=1)


@dataclass
class MultiStageLossConfig:
    """Configuration for multi-stage segmentation loss."""

    dice_weight: float = 1.0
    bce_weight: float = 1.0
    focal_weight: float = 0.0
    pos_weight: float = 1.0
    stage_weights: Optional[Sequence[float]] = field(default_factory=lambda: [0.05, 0.1, 0.2, 1.0])
    smooth: float = 1e-6
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    tversky_weight: float = 0.0
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.8
    lovasz_weight: float = 0.0
    boundary_weight: float = 0.03
    hard_pixel_mining: bool = False
    



class MultiStageManipulationLoss(nn.Module):
    """Hybrid multi-stage loss with optional boundary supervision."""
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
        self.lovasz = LovaszHingeLoss()
        self.boundary_weight = float(max(0.0, getattr(self.cfg, "boundary_weight", 0.0)))
        self.boundary_loss = SobelBoundaryLoss() if self.boundary_weight > 0.0 else None

    def _stage_weights(self, num_stages: int) -> List[float]:
        if self.cfg.stage_weights is None:
            return [float(i + 1) / float(num_stages) for i in range(num_stages)]
        if len(self.cfg.stage_weights) < num_stages:
            raise ValueError("Provided stage_weights shorter than number of stages")
        return list(self.cfg.stage_weights[:num_stages])

    def forward(
        self,
        preds: List[Tensor],
        target: Tensor,
    ) -> Tensor:
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

            hybrid_term = torch.zeros((), dtype=target.dtype, device=target.device)
            bce_map = None
            if float(self.cfg.bce_weight) > 0.0:
                bce_map = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")

            focal_term = None
            if float(getattr(self.cfg, "focal_weight", 0.0)) > 0.0:
                focal_term = self.focal(logits, target)

            if getattr(self.cfg, "hard_pixel_mining", False):
                with torch.no_grad():
                    pred_prob = torch.sigmoid(logits)
                    difficulty = torch.abs(pred_prob - target)
                    weight = 1.0 + 2.0 * (difficulty > 0.3).float()
                if bce_map is not None:
                    hybrid_term = hybrid_term + float(self.cfg.bce_weight) * (bce_map * weight).mean()
                if focal_term is not None:
                    hybrid_term = hybrid_term + float(getattr(self.cfg, "focal_weight", 0.0)) * focal_term
                dice = (1.0 - ((1.0 - dice) * weight).mean())
            else:
                if bce_map is not None:
                    hybrid_term = hybrid_term + float(self.cfg.bce_weight) * bce_map.mean()
                if focal_term is not None:
                    hybrid_term = hybrid_term + float(getattr(self.cfg, "focal_weight", 0.0)) * focal_term

            stage_loss = float(self.cfg.dice_weight) * dice + hybrid_term
            if self.cfg.tversky_weight > 0:
                stage_loss = stage_loss + self.cfg.tversky_weight * self.tversky(logits, target)
            if getattr(self.cfg, "lovasz_weight", 0) > 0:
                stage_loss = stage_loss + self.cfg.lovasz_weight * self.lovasz(logits, target)

            total_loss += stage_weight * stage_loss
            normalizer += stage_weight

        if self.boundary_loss is not None and preds:
            boundary = self.boundary_loss(preds[0], target)
            total_loss += self.boundary_weight * boundary

        return total_loss / max(normalizer, 1e-6)

class SobelBoundaryLoss(nn.Module):
    """Boundary loss based on Sobel gradient magnitude matching."""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
    ) -> Tensor:
        pred = torch.sigmoid(pred)
        target = target.float()
        sobel_x = self.sobel_x.to(dtype=pred.dtype, device=pred.device)
        sobel_y = self.sobel_y.to(dtype=pred.dtype, device=pred.device)
        grad_pred_x = F.conv2d(pred, sobel_x, padding=1)
        grad_pred_y = F.conv2d(pred, sobel_y, padding=1)
        grad_target_x = F.conv2d(target, sobel_x, padding=1)
        grad_target_y = F.conv2d(target, sobel_y, padding=1)
        grad_pred = torch.sqrt(grad_pred_x ** 2 + grad_pred_y ** 2 + 1e-6)
        grad_target = torch.sqrt(grad_target_x ** 2 + grad_target_y ** 2 + 1e-6)
        return F.l1_loss(grad_pred, grad_target)

__all__ = [
    "SoftDiceLoss",
    "FocalWithLogitsLoss",
    "TverskyLoss",
    "MultiStageLossConfig",
    "MultiStageManipulationLoss",
    "SobelBoundaryLoss",
    "LovaszHingeLoss",
]
