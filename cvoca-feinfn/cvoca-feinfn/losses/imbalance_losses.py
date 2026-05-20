from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_class_count_tensor(class_counts: Iterable[int] | Mapping[int, int] | torch.Tensor) -> torch.Tensor:
    if isinstance(class_counts, Mapping):
        if not class_counts:
            raise ValueError("class_counts mapping cannot be empty.")
        max_key = max(int(key) for key in class_counts.keys())
        dense_counts = [0.0] * (max_key + 1)
        for key, value in class_counts.items():
            dense_counts[int(key)] = float(value)
        counts = torch.as_tensor(dense_counts, dtype=torch.float32)
    else:
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
    if counts.ndim != 1:
        raise ValueError(f"class_counts must be 1D, got shape {tuple(counts.shape)}.")
    if counts.numel() == 0:
        raise ValueError("class_counts must contain at least one class.")
    if (counts < 0).any():
        raise ValueError("class_counts cannot contain negative values.")
    return counts


def compute_effective_num_weights(
    class_counts: Iterable[int] | torch.Tensor,
    beta: float = 0.999,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Effective-number class weights from:
    "Class-Balanced Loss Based on Effective Number of Samples".

    Why this function exists:
    - Direct inverse-frequency weighting is often too aggressive on long-tail HSI.
    - Effective-number weighting softens the tail emphasis by accounting for
      diminishing returns from redundant majority samples.
    """

    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be in (0, 1).")

    counts = _to_class_count_tensor(class_counts)
    safe_counts = counts.clamp(min=1.0)
    beta_tensor = torch.full_like(safe_counts, fill_value=beta)
    effective_num = (1.0 - torch.pow(beta_tensor, safe_counts)) / (1.0 - beta)
    weights = effective_num.clamp(min=1e-12).reciprocal()
    weights = torch.where(counts > 0, weights, torch.zeros_like(weights))

    if normalize:
        positive_mask = counts > 0
        positive_mean = weights[positive_mask].mean().clamp(min=1e-12)
        weights = weights / positive_mean
    return weights


def compute_ldam_margins(
    class_counts: Iterable[int] | torch.Tensor,
    max_margin: float = 0.5,
) -> torch.Tensor:
    """
    Label-distribution-aware margins from:
    "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss".

    Margin formula:
        m_j \propto 1 / n_j^{1/4}
    then rescaled so max(m_j) == max_margin.
    """

    if max_margin <= 0.0:
        raise ValueError("max_margin must be > 0.")

    counts = _to_class_count_tensor(class_counts)
    safe_counts = counts.clamp(min=1.0)
    margins = safe_counts.pow(-0.25)
    margins = margins * (max_margin / margins.max().clamp(min=1e-12))
    margins = torch.where(counts > 0, margins, torch.zeros_like(margins))
    return margins


def suggest_effective_num_beta(class_counts: Iterable[int] | torch.Tensor) -> float:
    """
    Heuristic beta for HSI long-tail training.

    Effective-number weighting needs beta close to 1, but the best scale depends
    on how many samples each class contains. This heuristic keeps the weighting
    mild on small HSI datasets and stronger on large scenes.
    """

    counts = _to_class_count_tensor(class_counts)
    max_count = int(counts.max().item())
    if max_count < 100:
        return 0.99
    if max_count < 1000:
        return 0.999
    return 0.9999


@dataclass(frozen=True)
class ImbalanceLossInfo:
    """
    Useful metadata returned by the factory helper.
    """

    class_counts: torch.Tensor
    margins: torch.Tensor
    effective_num_weights: torch.Tensor
    drw_start_epoch: int
    beta: float


class CBLDAMLoss(nn.Module):
    """
    Recommended loss for the current long-tail HSI pipeline.

    Design:
    1. LDAM margin reshapes the decision boundary in favor of minority classes.
    2. Effective-number weights are introduced later by DRW so the backbone can
       first learn a stable representation before class reweighting kicks in.
    3. Optional focal modulation is exposed but disabled by default because
       weighted sampling + focal + margin can over-focus tiny HSI classes.

    This makes the loss a practical hybrid of the two canonical papers:
    - Class-Balanced Loss Based on Effective Number of Samples
    - LDAM Loss with Deferred Re-Weighting
    """

    def __init__(
        self,
        class_counts: Iterable[int] | torch.Tensor,
        *,
        max_margin: float = 0.5,
        scale: float = 1.0,
        beta: float = 0.999,
        gamma: float = 0.0,
        drw_start_epoch: int = 0,
        margin_mode: str = "linear",
        cosine_eps: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if scale <= 0.0:
            raise ValueError("scale must be > 0.")
        if gamma < 0.0:
            raise ValueError("gamma must be >= 0.")
        if drw_start_epoch < 0:
            raise ValueError("drw_start_epoch must be >= 0.")
        if margin_mode not in {"linear", "additive_cosine", "angular"}:
            raise ValueError("margin_mode must be 'linear', 'additive_cosine', or 'angular'.")
        if not 0.0 < cosine_eps < 1.0:
            raise ValueError("cosine_eps must be in (0, 1).")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'.")

        counts = _to_class_count_tensor(class_counts)
        self.num_classes = int(counts.numel())
        self.default_logits_scale = scale
        self.gamma = gamma
        self.drw_start_epoch = drw_start_epoch
        self.margin_mode = margin_mode
        self.cosine_eps = cosine_eps
        self.reduction = reduction
        self.current_epoch = 0

        self.register_buffer("class_counts", counts)
        self.register_buffer("margins", compute_ldam_margins(counts, max_margin=max_margin))
        self.register_buffer("effective_num_weights", compute_effective_num_weights(counts, beta=beta, normalize=True))

    def set_epoch(self, epoch: int) -> None:
        """
        Update the current epoch so DRW knows when to activate class weighting.
        """

        if epoch < 0:
            raise ValueError("epoch must be >= 0.")
        self.current_epoch = epoch

    def get_active_class_weights(self) -> Optional[torch.Tensor]:
        """
        Return current class weights after applying the DRW schedule.
        """

        if self.current_epoch < self.drw_start_epoch:
            return None
        return self.effective_num_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, logits_scale: Optional[float | torch.Tensor] = None) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"logits must have shape [B, C], got {tuple(logits.shape)}.")
        if logits.shape[1] != self.num_classes:
            raise ValueError(
                f"logits has {logits.shape[1]} classes, but the loss was built for {self.num_classes} classes."
            )

        targets = torch.as_tensor(targets, dtype=torch.long, device=logits.device)
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError("targets must have shape [B] and match the batch size.")

        resolved_scale = self._resolve_logits_scale(logits, logits_scale)
        margins = self.margins.to(device=logits.device).index_select(0, targets)
        adjusted_logits = self._apply_margin(logits, targets, margins, resolved_scale)

        log_probs = F.log_softmax(adjusted_logits, dim=1)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        nll = -target_log_probs
        pt = target_log_probs.exp()

        loss = nll
        if self.gamma > 0.0:
            loss = torch.pow(1.0 - pt, self.gamma) * loss

        class_weights = self.get_active_class_weights()
        sample_weights = None
        if class_weights is not None:
            sample_weights = class_weights.to(device=logits.device).index_select(0, targets)
            # Batch-wise normalization keeps the weighted reduction numerically
            # stable and is mathematically equivalent to dividing by
            # sample_weights.sum() after the loss accumulation.
            sample_weights = sample_weights / sample_weights.mean().clamp(min=1e-12)
            loss = sample_weights * loss

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()

    def _apply_margin(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        margins: torch.Tensor,
        logits_scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply LDAM in either linear-logit space or cosine-based space.

        linear:
            z_y <- z_y - m_y
        additive_cosine:
            s cos(theta_y) <- s (cos(theta_y) - m_y)
        angular:
            cos(theta_y) <- cos(theta_y + m_y)

        additive_cosine is the most appropriate mode for the current head because
        the head fuses cosine classifier scores with prototype-based similarities.
        """

        target_index = targets.unsqueeze(1)
        target_logits = logits.gather(1, target_index)

        if self.margin_mode == "linear":
            adjusted_target_logits = target_logits - margins.unsqueeze(1)
        elif self.margin_mode == "additive_cosine":
            adjusted_target_logits = target_logits - margins.unsqueeze(1) * logits_scale
        else:
            raw_target_logits = target_logits / logits_scale
            clamped_target_logits = raw_target_logits.clamp(min=-1.0 + self.cosine_eps, max=1.0 - self.cosine_eps)
            target_angles = torch.acos(clamped_target_logits)
            adjusted_target_logits = torch.cos(target_angles + margins.unsqueeze(1)) * logits_scale

        # Use out-of-place scatter so autograd can still see the pre-margin
        # logits that were read by gather() above.
        return logits.scatter(1, target_index, adjusted_target_logits)

    def _resolve_logits_scale(
        self,
        logits: torch.Tensor,
        logits_scale: Optional[float | torch.Tensor],
    ) -> torch.Tensor:
        if logits_scale is None:
            return logits.new_tensor(float(self.default_logits_scale))
        if isinstance(logits_scale, torch.Tensor):
            return logits_scale.to(device=logits.device, dtype=logits.dtype)
        return logits.new_tensor(float(logits_scale))


def build_recommended_hsi_imbalance_loss(
    class_counts: Iterable[int] | torch.Tensor,
    total_epochs: int,
    *,
    max_margin: float = 0.5,
    scale: float = 1.0,
    beta: Optional[float] = None,
    gamma: float = 0.0,
    drw_ratio: float = 0.5,
    margin_mode: str = "linear",
    cosine_eps: float = 1e-6,
    reduction: str = "mean",
) -> tuple[CBLDAMLoss, ImbalanceLossInfo]:
    """
    Build the recommended loss preset for the current SPARC-Net project.

    Recommended preset:
    - LDAM margin from epoch 0.
    - Effective-number reweighting enabled after the first half of training.
    - Focal modulation disabled by default.
    - The actual cosine-logit scale is expected to come from the classifier head.

    Why this is a good fit for the current project:
    - Your backbone is already strong and frequency-aware, so the biggest risk is
      tail-class overfitting rather than insufficient hardness mining.
    - LDAM improves the boundary directly, while DRW prevents early training from
      being dominated by noisy tail gradients.
    """

    if total_epochs <= 0:
        raise ValueError("total_epochs must be > 0.")
    if not 0.0 <= drw_ratio < 1.0:
        raise ValueError("drw_ratio must be in [0, 1).")

    counts = _to_class_count_tensor(class_counts)
    chosen_beta = suggest_effective_num_beta(counts) if beta is None else beta
    drw_start_epoch = int(total_epochs * drw_ratio)

    loss = CBLDAMLoss(
        class_counts=counts,
        max_margin=max_margin,
        scale=scale,
        beta=chosen_beta,
        gamma=gamma,
        drw_start_epoch=drw_start_epoch,
        margin_mode=margin_mode,
        cosine_eps=cosine_eps,
        reduction=reduction,
    )
    info = ImbalanceLossInfo(
        class_counts=counts.clone(),
        margins=loss.margins.detach().clone(),
        effective_num_weights=loss.effective_num_weights.detach().clone(),
        drw_start_epoch=drw_start_epoch,
        beta=chosen_beta,
    )
    return loss, info
