from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .imbalance_losses import compute_effective_num_weights


def _dense_class_weights(
    class_counts: Optional[Sequence[int] | Mapping[int, int] | torch.Tensor],
    weight_mode: str,
    beta: float,
) -> Optional[torch.Tensor]:
    if class_counts is None or weight_mode == "none":
        return None

    if isinstance(class_counts, Mapping):
        if not class_counts:
            return None
        num_classes = max(int(key) for key in class_counts.keys()) + 1
        dense_counts = torch.zeros(num_classes, dtype=torch.float32)
        for key, value in class_counts.items():
            dense_counts[int(key)] = float(value)
        counts = dense_counts
    else:
        counts = torch.as_tensor(class_counts, dtype=torch.float32)

    counts = counts.clamp(min=1.0)
    if weight_mode == "effective_num":
        return compute_effective_num_weights(counts, beta=beta, normalize=True)
    if weight_mode == "inverse":
        weights = counts.reciprocal()
    elif weight_mode == "sqrt_inv":
        weights = counts.rsqrt()
    else:
        raise ValueError("weight_mode must be 'none', 'inverse', 'sqrt_inv', or 'effective_num'.")
    return weights / weights.mean().clamp(min=1e-12)


@dataclass(frozen=True)
class ContrastiveLossInfo:
    class_weights: Optional[torch.Tensor]
    temperature: float


class TailAwareSupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss with optional class-aware anchor weighting.

    Why this module exists:
    - The upgraded classification head exposes a projection branch.
    - Contrastive supervision helps tail classes occupy a cleaner, tighter
      region in feature space instead of being absorbed by head classes.
    """

    def __init__(
        self,
        *,
        temperature: float = 0.07,
        class_counts: Optional[Sequence[int] | Mapping[int, int] | torch.Tensor] = None,
        weight_mode: str = "effective_num",
        beta: float = 0.999,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0.")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'.")

        self.temperature = temperature
        self.reduction = reduction
        class_weights = _dense_class_weights(class_counts, weight_mode=weight_mode, beta=beta)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"features must have shape [B, D], got {tuple(features.shape)}.")
        labels = torch.as_tensor(labels, dtype=torch.long, device=features.device)
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError("labels must have shape [B] and match the batch size.")
        if features.shape[0] < 2:
            return features.new_zeros(())

        features = F.normalize(features, dim=1)
        logits = features @ features.t() / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        logits_mask = ~torch.eye(features.shape[0], device=features.device, dtype=torch.bool)
        positive_mask = label_mask & logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-12))

        pos_count = positive_mask.sum(dim=1)
        valid_mask = pos_count > 0
        if not bool(valid_mask.any().item()):
            return features.new_zeros(())

        loss = torch.zeros(features.shape[0], device=features.device, dtype=features.dtype)
        positive_mask_float = positive_mask.float()
        loss[valid_mask] = -(
            positive_mask_float[valid_mask] * log_prob[valid_mask]
        ).sum(dim=1) / pos_count[valid_mask].float()

        if self.class_weights is not None:
            anchor_weights = self.class_weights.to(device=features.device).index_select(0, labels)
            anchor_weights = anchor_weights / anchor_weights[valid_mask].mean().clamp(min=1e-12)
            loss = anchor_weights * loss

        loss = loss[valid_mask]
        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


def build_recommended_hsi_contrastive_loss(
    class_counts: Optional[Sequence[int] | Mapping[int, int] | torch.Tensor] = None,
    *,
    temperature: float = 0.07,
    weight_mode: str = "effective_num",
    beta: float = 0.999,
) -> tuple[TailAwareSupervisedContrastiveLoss, ContrastiveLossInfo]:
    """
    Recommended contrastive preset for the current long-tail HSI project.
    """

    loss = TailAwareSupervisedContrastiveLoss(
        temperature=temperature,
        class_counts=class_counts,
        weight_mode=weight_mode,
        beta=beta,
    )
    class_weights = None if getattr(loss, "class_weights", None) is None else loss.class_weights.detach().clone()
    return loss, ContrastiveLossInfo(class_weights=class_weights, temperature=temperature)
