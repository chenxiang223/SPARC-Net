from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

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


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if labels.numel() == 0:
        return 0.0
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


def _macro_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if labels.numel() == 0:
        return 0.0
    preds = logits.argmax(dim=1)
    num_classes = int(max(int(preds.max().item()), int(labels.max().item())) + 1)
    correct_sum = torch.zeros(num_classes, device=labels.device)
    count_sum = torch.zeros(num_classes, device=labels.device)
    for class_id in range(num_classes):
        mask = labels == class_id
        if mask.any():
            correct_sum[class_id] = (preds[mask] == labels[mask]).float().mean()
            count_sum[class_id] = 1.0
    valid = count_sum > 0
    if not bool(valid.any().item()):
        return 0.0
    return float(correct_sum[valid].mean().item())


def compute_prior_vector(
    class_counts: Iterable[int] | Mapping[int, int] | torch.Tensor,
    *,
    mode: str = "frequency",
    effective_num_beta: float = 0.999,
) -> torch.Tensor:
    """
    Convert training-set class counts into a prior distribution.

    frequency:
        Raw empirical class prior p(y) from the long-tail training set.
    effective_num:
        A softer prior derived from effective sample counts, which is often more
        stable for HSI scenes where many head-class pixels are redundant.
    """

    counts = _to_class_count_tensor(class_counts)
    positive_mask = counts > 0
    safe_counts = counts.clamp(min=1.0)

    if mode == "frequency":
        masses = counts
    elif mode == "effective_num":
        if not 0.0 < effective_num_beta < 1.0:
            raise ValueError("effective_num_beta must be in (0, 1).")
        beta_tensor = torch.full_like(safe_counts, fill_value=effective_num_beta)
        masses = (1.0 - torch.pow(beta_tensor, safe_counts)) / (1.0 - effective_num_beta)
    else:
        raise ValueError("mode must be 'frequency' or 'effective_num'.")

    masses = torch.where(positive_mask, masses, torch.zeros_like(masses))
    normalizer = masses.sum().clamp(min=1e-12)
    prior = masses / normalizer
    return prior


def build_prior_bias(
    class_counts: Iterable[int] | Mapping[int, int] | torch.Tensor,
    *,
    mode: str,
    alpha: float,
    effective_num_beta: float = 0.999,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Build an additive bias vector for prior calibration.

    We subtract alpha * log p(y) from logits to counter the head-class prior
    learned during long-tail training. The bias is centered because softmax is
    invariant to a global offset.
    """

    counts = _to_class_count_tensor(class_counts)
    if mode == "none" or alpha == 0.0:
        return torch.zeros_like(counts)
    if alpha < 0.0:
        raise ValueError("alpha must be >= 0.")

    prior = compute_prior_vector(
        counts,
        mode=mode,
        effective_num_beta=effective_num_beta,
    )
    positive_mask = prior > 0
    safe_prior = prior.clamp(min=eps)
    bias = -float(alpha) * safe_prior.log()
    bias = torch.where(positive_mask, bias, torch.zeros_like(bias))
    if bool(positive_mask.any().item()):
        bias = bias - bias[positive_mask].mean()
    return bias


@dataclass(frozen=True)
class RTPCConfig:
    """
    Stage-3 post-hoc calibration for decoupled long-tail training.

    Why this module exists:
    - The decoupling paper shows that a fixed representation can still benefit
      from classifier-only adjustment.
    - The prior-gap paper highlights that long-tail models often retain a bias
      toward the training prior even after the representation is learned.

    For the current HSI model we use a logit-space variant:
    1. learn class-wise positive logit scales on top of the frozen model;
    2. search an additive prior-correction bias on the validation set.
    """

    epochs: int = 20
    lr: float = 5e-3
    weight_decay: float = 0.0
    learn_class_scales: bool = True
    learn_logit_mixer: bool = True
    normalize_scales: bool = True
    max_log_scale: Optional[float] = 0.25
    max_logit_mixer_residual: float = 0.08
    train_loader_preference: str = "balanced"
    prior_modes: Sequence[str] = ("none", "frequency", "effective_num")
    prior_alpha_candidates: Sequence[float] = (0.0, 0.1, 0.2, 0.35, 0.5)
    default_prior_mode: str = "none"
    default_prior_alpha: float = 0.0
    effective_num_beta: float = 0.999
    monitor: str = "acc"
    blend_candidates: Sequence[float] = (0.0, 0.20, 0.40, 0.60, 0.80, 1.0)
    min_val_gain: float = 0.0002
    min_val_loss_gain: float = 1e-4


@dataclass(frozen=True)
class CalibrationResult:
    train_loss: float
    train_acc: float
    train_macro_acc: float
    val_loss: Optional[float]
    val_acc: Optional[float]
    val_macro_acc: Optional[float]
    prior_mode: str
    prior_alpha: float
    blend_strength: float
    baseline_val_acc: Optional[float] = None
    baseline_val_macro_acc: Optional[float] = None


@dataclass(frozen=True)
class PriorSearchResult:
    prior_mode: str
    prior_alpha: float
    val_loss: float
    val_acc: float
    val_macro_acc: float


class ClassWiseLogitScaler(nn.Module):
    """
    Lightweight LWS-style adapter in logit space.

    The original LWS idea learns per-class scaling factors after the backbone is
    frozen. Because the current head fuses cosine logits and prototype logits,
    a post-hoc logit scaler is safer than directly manipulating weight vectors.
    """

    def __init__(
        self,
        num_classes: int,
        normalize_scales: bool = True,
        max_log_scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(num_classes))
        self.normalize_scales = normalize_scales
        self.max_log_scale = max_log_scale

    def get_scales(self) -> torch.Tensor:
        log_scale = self.log_scale
        if self.max_log_scale is not None:
            if self.max_log_scale < 0.0:
                raise ValueError("max_log_scale must be >= 0.")
            log_scale = log_scale.clamp(min=-self.max_log_scale, max=self.max_log_scale)
        scales = torch.exp(log_scale)
        if self.normalize_scales:
            scales = scales / scales.mean().clamp(min=1e-12)
        return scales

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * self.get_scales().to(device=logits.device, dtype=logits.dtype)


class ReversibleTailPriorCalibrator(nn.Module):
    """
    Final calibration layer applied after the frozen model emits logits.

    Order:
    1. optional class-wise positive scaling;
    2. additive prior-correction bias.
    """

    def __init__(
        self,
        num_classes: int,
        class_counts: Iterable[int] | Mapping[int, int] | torch.Tensor,
        *,
        learn_class_scales: bool = True,
        learn_logit_mixer: bool = True,
        normalize_scales: bool = True,
        max_log_scale: Optional[float] = None,
        max_logit_mixer_residual: float = 0.08,
        effective_num_beta: float = 0.999,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.effective_num_beta = float(effective_num_beta)
        self.max_logit_mixer_residual = float(max_logit_mixer_residual)
        counts = _to_class_count_tensor(class_counts)
        if counts.numel() != self.num_classes:
            raise ValueError(f"class_counts has {counts.numel()} classes, expected {self.num_classes}.")

        self.register_buffer("class_counts", counts)
        self.register_buffer("prior_bias", torch.zeros(self.num_classes, dtype=torch.float32))
        self.register_buffer("blend_strength", torch.ones((), dtype=torch.float32))
        self.scaler = (
            ClassWiseLogitScaler(
                self.num_classes,
                normalize_scales=normalize_scales,
                max_log_scale=max_log_scale,
            )
            if learn_class_scales
            else None
        )
        self.logit_mixer_residual = nn.Parameter(torch.zeros(self.num_classes, self.num_classes)) if learn_logit_mixer else None
        self.prior_mode = "none"
        self.prior_alpha = 0.0

    def forward_without_prior(self, logits: torch.Tensor) -> torch.Tensor:
        if self.scaler is not None:
            logits = self.scaler(logits)
        if self.logit_mixer_residual is not None and self.max_logit_mixer_residual > 0.0:
            residual = self.max_logit_mixer_residual * torch.tanh(self.logit_mixer_residual)
            logits = logits + logits @ residual.to(device=logits.device, dtype=logits.dtype).t()
        return logits

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        calibrated_logits = self.forward_without_prior(logits)
        calibrated_logits = calibrated_logits + self.prior_bias.to(device=logits.device, dtype=logits.dtype).view(1, -1)
        blend = self.blend_strength.to(device=logits.device, dtype=logits.dtype)
        return logits + blend * (calibrated_logits - logits)

    @torch.no_grad()
    def set_prior_correction(self, mode: str, alpha: float) -> None:
        bias = build_prior_bias(
            self.class_counts,
            mode=mode,
            alpha=alpha,
            effective_num_beta=self.effective_num_beta,
        )
        self.prior_bias.copy_(bias.to(device=self.prior_bias.device, dtype=self.prior_bias.dtype))
        self.prior_mode = mode
        self.prior_alpha = float(alpha)

    @torch.no_grad()
    def set_blend_strength(self, strength: float) -> None:
        clipped = min(1.0, max(0.0, float(strength)))
        self.blend_strength.fill_(clipped)


@torch.no_grad()
def search_prior_correction(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_counts: Iterable[int] | Mapping[int, int] | torch.Tensor,
    *,
    prior_modes: Sequence[str],
    prior_alpha_candidates: Sequence[float],
    effective_num_beta: float = 0.999,
    monitor: str = "macro_acc",
) -> PriorSearchResult:
    """
    Grid-search the additive prior correction on a validation split.
    """

    if monitor not in {"acc", "macro_acc"}:
        raise ValueError("monitor must be 'acc' or 'macro_acc'.")

    best_result: Optional[PriorSearchResult] = None
    best_metric = float("-inf")

    for mode in prior_modes:
        for alpha in prior_alpha_candidates:
            bias = build_prior_bias(
                class_counts,
                mode=mode,
                alpha=float(alpha),
                effective_num_beta=effective_num_beta,
            ).to(device=logits.device, dtype=logits.dtype)
            calibrated_logits = logits + bias.view(1, -1)
            val_loss = float(F.cross_entropy(calibrated_logits, labels).item())
            val_acc = _accuracy(calibrated_logits, labels)
            val_macro_acc = _macro_accuracy(calibrated_logits, labels)
            monitor_value = val_macro_acc if monitor == "macro_acc" else val_acc
            if monitor_value > best_metric:
                best_metric = monitor_value
                best_result = PriorSearchResult(
                    prior_mode=mode,
                    prior_alpha=float(alpha),
                    val_loss=val_loss,
                    val_acc=val_acc,
                    val_macro_acc=val_macro_acc,
                )

    if best_result is None:
        raise RuntimeError("Prior search failed to produce a valid result.")
    return best_result


# Backward-compatible aliases for older Windows experiment scripts.
PostHocCalibrationConfig = RTPCConfig
PostHocLogitCalibrator = ReversibleTailPriorCalibrator
