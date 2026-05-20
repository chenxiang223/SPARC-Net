from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _dense_class_count_tensor(
    class_counts: Optional[Sequence[int] | Mapping[int, int] | torch.Tensor],
    num_classes: int,
) -> torch.Tensor:
    """
    Convert class-count metadata to a dense [C] tensor.
    """

    if class_counts is None:
        return torch.ones(num_classes, dtype=torch.float32)

    if isinstance(class_counts, Mapping):
        counts = torch.zeros(num_classes, dtype=torch.float32)
        keys = [int(key) for key in class_counts]
        key_offset = 1 if keys and 0 not in keys and min(keys) >= 1 and max(keys) <= num_classes else 0
        for key, value in class_counts.items():
            key = int(key) - key_offset
            if 0 <= key < num_classes:
                counts[key] = float(value)
        return counts

    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    if counts.ndim != 1:
        raise ValueError(f"class_counts must be 1D, got shape {tuple(counts.shape)}.")
    if counts.numel() != num_classes:
        raise ValueError(f"class_counts has {counts.numel()} classes, expected {num_classes}.")
    return counts


def _normalize_rows(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)


class ClassificationHead(nn.Module):
    """
    Baseline head kept for ablation and compatibility.
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)
        self.margin_mode = "linear"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x).flatten(1))

    def reset_classifier(self) -> None:
        self.fc.reset_parameters()

    def apply_tau_normalization(self, tau: float = 1.0) -> None:
        # Tau-normalization is designed for normalized classifiers, so the
        # baseline linear head leaves this as a no-op.
        _ = tau


class CosineClassifier(nn.Module):
    """
    Cosine classifier for long-tail recognition.

    Why this module exists:
    - It aligns naturally with margin-based long-tail losses.
    - It reduces the classifier-norm bias highlighted in decoupled training work.
    """

    def __init__(self, in_features: int, num_classes: int, scale: float = 30.0, eps: float = 1e-6) -> None:
        super().__init__()
        if scale <= 0.0:
            raise ValueError("scale must be > 0.")
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        self.scale = float(scale)
        self.eps = eps
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        x = _normalize_rows(x, eps=self.eps)
        weight = _normalize_rows(self.weight, eps=self.eps)
        return x @ weight.t()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_raw(x) * self.scale

    @torch.no_grad()
    def apply_tau_normalization(self, tau: float = 1.0) -> None:
        if tau < 0.0:
            raise ValueError("tau must be >= 0.")
        norms = self.weight.norm(dim=1, keepdim=True).clamp(min=self.eps)
        self.weight.div_(norms.pow(tau))


class AuxiliaryCosineHead(nn.Module):
    """
    Lightweight deep-supervision head for intermediate feature maps.
    """

    def __init__(self, in_channels: int, embed_dim: int, num_classes: int, scale: float = 30.0) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(in_channels, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.classifier = CosineClassifier(embed_dim, num_classes, scale=scale)

    def forward_raw(self, feat: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(feat).flatten(1)
        embedding = self.proj(pooled)
        return self.classifier.forward_raw(embedding)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.forward_raw(feat) * self.classifier.scale

    def reset_classifier(self) -> None:
        self.classifier.reset_parameters()

    def apply_tau_normalization(self, tau: float = 1.0) -> None:
        self.classifier.apply_tau_normalization(tau=tau)


@dataclass
class LongTailHeadOutput:
    # `logits` is the final scaled score used for classification.
    logits: torch.Tensor
    raw_logits: torch.Tensor
    pooled_feature: torch.Tensor
    embedding: torch.Tensor
    projection: torch.Tensor
    # `main_logits` keeps the scaled cosine-classifier branch for ablations.
    main_logits: torch.Tensor
    raw_main_logits: torch.Tensor
    prototype_logits: torch.Tensor
    raw_prototype_logits: torch.Tensor
    dynamic_gate: torch.Tensor
    logit_scale: torch.Tensor
    fusion_weights: Optional[torch.Tensor] = None
    head_regularizer: Optional[torch.Tensor] = None
    aux_logits_mid: Optional[torch.Tensor] = None
    aux_logits_early: Optional[torch.Tensor] = None


class BaselineCosineHead(nn.Module):
    """
    Plain cosine head used for strict Innovation2 ablation.

    This head intentionally removes the prototype-relation branch and keeps only:
    - pooled final-feature embedding;
    - a standard cosine classifier;
    - optional auxiliary supervision and projection for Innovation3.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        embed_dim: Optional[int] = None,
        contrast_dim: int = 128,
        scale: float = 30.0,
        use_auxiliary_heads: bool = True,
    ) -> None:
        super().__init__()
        embed_dim = in_channels if embed_dim is None else embed_dim
        self.margin_mode = "additive_cosine"
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_neck = nn.Sequential(
            nn.Linear(in_channels, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.classifier = CosineClassifier(embed_dim, num_classes, scale=scale)
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, contrast_dim, bias=True),
        )
        self.use_auxiliary_heads = use_auxiliary_heads
        if use_auxiliary_heads:
            self.aux_mid = AuxiliaryCosineHead(in_channels, embed_dim, num_classes, scale=scale)
            self.aux_early = AuxiliaryCosineHead(in_channels, embed_dim, num_classes, scale=scale)
        else:
            self.aux_mid = None
            self.aux_early = None

    def reset_classifier(self) -> None:
        self.classifier.reset_parameters()
        if self.aux_mid is not None:
            self.aux_mid.reset_classifier()
        if self.aux_early is not None:
            self.aux_early.reset_classifier()

    @torch.no_grad()
    def apply_tau_normalization(self, tau: float = 1.0) -> None:
        self.classifier.apply_tau_normalization(tau=tau)
        if self.aux_mid is not None:
            self.aux_mid.apply_tau_normalization(tau=tau)
        if self.aux_early is not None:
            self.aux_early.apply_tau_normalization(tau=tau)

    def forward(
        self,
        final_feat: torch.Tensor,
        *,
        mid_feat: Optional[torch.Tensor] = None,
        early_feat: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> LongTailHeadOutput:
        del targets
        pooled = self.pool(final_feat).flatten(1)
        embedding = self.feature_neck(pooled)
        normalized_embedding = _normalize_rows(embedding)
        raw_logits = self.classifier.forward_raw(normalized_embedding)
        logits = raw_logits * self.classifier.scale
        projection = _normalize_rows(self.projection_head(embedding))
        aux_logits_mid = self.aux_mid(mid_feat) if self.aux_mid is not None and mid_feat is not None else None
        aux_logits_early = self.aux_early(early_feat) if self.aux_early is not None and early_feat is not None else None

        zero_logits = raw_logits.new_zeros(raw_logits.shape)
        zero_gate = raw_logits.new_zeros(raw_logits.shape)
        return LongTailHeadOutput(
            logits=logits,
            raw_logits=raw_logits,
            pooled_feature=pooled,
            embedding=normalized_embedding,
            projection=projection,
            main_logits=logits,
            raw_main_logits=raw_logits,
            prototype_logits=zero_logits,
            raw_prototype_logits=zero_logits,
            dynamic_gate=zero_gate,
            logit_scale=normalized_embedding.new_tensor(self.classifier.scale),
            fusion_weights=None,
            head_regularizer=None,
            aux_logits_mid=aux_logits_mid,
            aux_logits_early=aux_logits_early,
        )


class LongTailDynamicHead(nn.Module):
    """
    Adaptive long-tail head for the current HSI backbone.

    The current revision keeps the cosine classifier as the dominant decision
    branch and uses a reliability-gated momentum dual-prototype relation path
    as an auxiliary long-tail stabilizer.

    In practice this means:
    1. the final-feature cosine classifier remains the main prediction source;
    2. multi-level cues update stable/context prototype memories with
       confidence-aware EMA; and
    3. prototype relations supervise and correct only after the prototype bank
       is mature enough to be trusted.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        embed_dim: Optional[int] = None,
        contrast_dim: int = 128,
        scale: float = 30.0,
        class_counts: Optional[Sequence[int] | Mapping[int, int] | torch.Tensor] = None,
        prototype_momentum: float = 0.9,
        prototype_blend_min: float = 0.05,
        prototype_blend_max: float = 0.45,
        use_prototype_branch: bool = True,
        use_dynamic_gate: bool = True,
        use_auxiliary_heads: bool = True,
        alignment_scale_limit: float = 0.35,
        alignment_bias_limit: float = 0.20,
        freeze_source_classifier_in_stage2: bool = True,
    ) -> None:
        super().__init__()
        if alignment_scale_limit < 0.0:
            raise ValueError("alignment_scale_limit must be >= 0.")
        if alignment_bias_limit < 0.0:
            raise ValueError("alignment_bias_limit must be >= 0.")

        embed_dim = in_channels if embed_dim is None else embed_dim
        self.margin_mode = "additive_cosine"
        self.num_classes = num_classes
        self.prototype_momentum = float(prototype_momentum)
        self.prototype_blend_min = float(prototype_blend_min)
        self.prototype_blend_max = float(prototype_blend_max)
        if not 0.0 <= self.prototype_blend_min <= self.prototype_blend_max <= 1.0:
            raise ValueError("prototype_blend_min/max must satisfy 0 <= min <= max <= 1.")
        self.num_prototypes_per_class = 3
        self.prototype_temperature = 0.25
        self.prototype_new_slot_threshold = 0.72
        self.prototype_min_update_weight = 0.05
        self.prototype_new_slot_min_weight = 0.35
        self.prototype_reliability_tau = 4.0
        # Keep the original switch names for backward compatibility with the
        # ablation/config plumbing. Semantically this now enables the
        # distribution-alignment branch rather than an online prototype bank.
        self.use_prototype_branch = use_prototype_branch
        self.use_dynamic_gate = use_dynamic_gate and use_prototype_branch
        self.alignment_scale_limit = float(alignment_scale_limit)
        self.alignment_bias_limit = float(alignment_bias_limit)
        self.freeze_source_classifier_in_stage2 = bool(freeze_source_classifier_in_stage2)
        self.fusion_residual_scale = 0.18
        self.prototype_margin = 0.12
        self.prototype_correction_scale = 0.18
        self.prototype_delta_clip = 0.45
        self.training_output_mode = "main"
        self.prototype_updates_enabled = True
        self.guided_context_enabled = False
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_neck = nn.Sequential(
            nn.Linear(in_channels, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.classifier = CosineClassifier(embed_dim, num_classes, scale=scale)
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, contrast_dim, bias=True),
        )
        self.register_buffer(
            "stable_prototypes",
            torch.zeros(num_classes, self.num_prototypes_per_class, embed_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "stable_proto_counts",
            torch.zeros(num_classes, self.num_prototypes_per_class, dtype=torch.float32),
        )
        self.register_buffer(
            "context_prototypes",
            torch.zeros(num_classes, self.num_prototypes_per_class, embed_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "context_proto_counts",
            torch.zeros(num_classes, self.num_prototypes_per_class, dtype=torch.float32),
        )

        self.use_auxiliary_heads = use_auxiliary_heads
        if use_auxiliary_heads:
            self.aux_mid = AuxiliaryCosineHead(in_channels, embed_dim, num_classes, scale=scale)
            self.aux_early = AuxiliaryCosineHead(in_channels, embed_dim, num_classes, scale=scale)
        else:
            self.aux_mid = None
            self.aux_early = None

        if self.use_prototype_branch:
            self.mid_neck = nn.Sequential(
                nn.Linear(in_channels, embed_dim, bias=False),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
            )
            self.early_neck = nn.Sequential(
                nn.Linear(in_channels, embed_dim, bias=False),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
            )
            self.mid_guidance = nn.LayerNorm(embed_dim)
            self.early_guidance = nn.LayerNorm(embed_dim)
            relation_dim = 3 * embed_dim
            self.prototype_mix_gate = nn.Sequential(
                nn.Linear(relation_dim, embed_dim, bias=False),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 1, bias=True),
                nn.Sigmoid(),
            )
            if self.use_dynamic_gate:
                self.dynamic_gate = nn.Sequential(
                    nn.Linear(relation_dim + 2, embed_dim, bias=False),
                    nn.LayerNorm(embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, 1, bias=True),
                    nn.Sigmoid(),
                )
            else:
                self.dynamic_gate = None
            self.alignment_scale = nn.Parameter(torch.zeros(num_classes))
            self.alignment_bias = nn.Parameter(torch.zeros(num_classes))
        else:
            self.mid_neck = None
            self.early_neck = None
            self.mid_guidance = None
            self.early_guidance = None
            self.prototype_mix_gate = None
            self.dynamic_gate = None
            self.alignment_scale = None
            self.alignment_bias = None

        counts = _dense_class_count_tensor(class_counts, num_classes=num_classes).clamp(min=1.0)
        self.register_buffer("class_counts", counts)

    def reset_classifier(self) -> None:
        self.classifier.reset_parameters()
        self.stable_prototypes.zero_()
        self.stable_proto_counts.zero_()
        self.context_prototypes.zero_()
        self.context_proto_counts.zero_()
        if self.alignment_scale is not None:
            nn.init.zeros_(self.alignment_scale)
        if self.alignment_bias is not None:
            nn.init.zeros_(self.alignment_bias)
        for module in (self.mid_neck, self.early_neck):
            if module is None:
                continue
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    submodule.reset_parameters()
        if self.mid_guidance is not None:
            self.mid_guidance.reset_parameters()
        if self.early_guidance is not None:
            self.early_guidance.reset_parameters()
        if self.prototype_mix_gate is not None:
            for module in self.prototype_mix_gate.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()
            final_linear = self.prototype_mix_gate[-2] if isinstance(self.prototype_mix_gate[-2], nn.Linear) else None
            if final_linear is not None and final_linear.bias is not None:
                nn.init.constant_(final_linear.bias, 0.0)
        if self.dynamic_gate is not None:
            for module in self.dynamic_gate.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()
            final_linear = self.dynamic_gate[-2] if isinstance(self.dynamic_gate[-2], nn.Linear) else None
            if final_linear is not None and final_linear.bias is not None:
                nn.init.constant_(final_linear.bias, -2.8)
        if self.aux_mid is not None:
            self.aux_mid.reset_classifier()
        if self.aux_early is not None:
            self.aux_early.reset_classifier()

    @torch.no_grad()
    def apply_tau_normalization(self, tau: float = 1.0) -> None:
        self.classifier.apply_tau_normalization(tau=tau)
        if self.aux_mid is not None:
            self.aux_mid.apply_tau_normalization(tau=tau)
        if self.aux_early is not None:
            self.aux_early.apply_tau_normalization(tau=tau)

    def set_prototype_correction_scale(self, scale: float) -> None:
        self.prototype_correction_scale = max(0.0, float(scale))

    def get_prototype_correction_scale(self) -> float:
        return float(self.prototype_correction_scale)

    def set_head_training_mode(
        self,
        *,
        output_mode: str = "main",
        update_prototypes: bool = True,
        guided_context: bool = False,
    ) -> None:
        valid_modes = {"main", "prototype", "corrected"}
        if output_mode not in valid_modes:
            raise ValueError(f"output_mode must be one of {valid_modes}, got {output_mode}.")
        self.training_output_mode = output_mode
        self.prototype_updates_enabled = bool(update_prototypes)
        self.guided_context_enabled = bool(guided_context)

    def forward(
        self,
        final_feat: torch.Tensor,
        *,
        mid_feat: Optional[torch.Tensor] = None,
        early_feat: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> LongTailHeadOutput:
        pooled = self.pool(final_feat).flatten(1)
        embedding = self.feature_neck(pooled)
        normalized_embedding = _normalize_rows(embedding)
        raw_main_logits = self.classifier.forward_raw(normalized_embedding)

        if self.use_prototype_branch:
            # The prototype path must not perturb representation learning on
            # easy/saturated splits. Use the trained main embedding as the
            # reliable anchor; the context bank remains available for future
            # variants but falls back to the same stable feature space here.
            if self.guided_context_enabled or self.training_output_mode in {"prototype", "corrected"}:
                context_embedding, fusion_weights = self._compute_guided_fusion_embedding(
                    final_embedding=embedding,
                    mid_feat=mid_feat,
                    early_feat=early_feat,
                )
            else:
                context_embedding = embedding
                fusion_weights = torch.cat(
                    [
                        torch.ones(normalized_embedding.shape[0], 1, device=normalized_embedding.device, dtype=normalized_embedding.dtype),
                        torch.ones(normalized_embedding.shape[0], 1, device=normalized_embedding.device, dtype=normalized_embedding.dtype),
                        torch.zeros(normalized_embedding.shape[0], 1, device=normalized_embedding.device, dtype=normalized_embedding.dtype),
                    ],
                    dim=1,
                )
            normalized_context_embedding = _normalize_rows(context_embedding)

            pending_update_reliability = None
            if self.training and targets is not None and self.prototype_updates_enabled:
                update_reliability = self._sample_update_reliability(
                    raw_main_logits.detach(),
                    targets.detach(),
                )
                pending_update_reliability = update_reliability

            stable_bank = self.stable_prototypes.detach().clone()
            stable_counts = self.stable_proto_counts.detach().clone()
            context_bank = self.context_prototypes.detach().clone()
            context_counts = self.context_proto_counts.detach().clone()
            stable_proto_logits = self._prototype_logits_from_bank(
                normalized_embedding,
                bank=stable_bank,
                bank_counts=stable_counts,
            )
            context_proto_logits = self._prototype_logits_from_bank(
                normalized_context_embedding,
                bank=context_bank,
                bank_counts=context_counts,
            )

            relation_input = torch.cat(
                [
                    normalized_embedding,
                    normalized_context_embedding,
                    (normalized_embedding - normalized_context_embedding).abs(),
                ],
                dim=1,
            )
            if self.prototype_mix_gate is not None:
                proto_mix = self.prototype_mix_gate(relation_input)
            else:
                proto_mix = normalized_embedding.new_zeros((normalized_embedding.shape[0], 1))

            raw_fusion_logits = (1.0 - proto_mix) * stable_proto_logits + proto_mix * context_proto_logits
            raw_alignment_logits = self._apply_alignment_affine(raw_fusion_logits)
            prototype_reliability = self._prototype_reliability(
                stable_counts=stable_counts,
                context_counts=context_counts,
            ).to(
                device=raw_alignment_logits.device,
                dtype=raw_alignment_logits.dtype,
            )

            main_logits = raw_main_logits * self.classifier.scale
            prototype_logits = raw_alignment_logits * self.classifier.scale
            main_prob = torch.softmax(main_logits, dim=1)
            prototype_prob = torch.softmax(prototype_logits, dim=1)
            main_conf = main_prob.amax(dim=1, keepdim=True)
            prototype_conf = prototype_prob.amax(dim=1, keepdim=True)
            agreement = (main_logits.argmax(dim=1) == prototype_logits.argmax(dim=1)).float().unsqueeze(1)

            main_uncertainty = (1.0 - main_conf).clamp(min=0.0, max=1.0)
            prototype_support = (prototype_prob - main_prob).clamp(min=0.0)
            agreement_support = 0.25 + 0.35 * agreement
            uncertainty_support = 0.25 * prototype_prob * (0.30 + main_uncertainty)
            prototype_advantage = (prototype_conf > (main_conf + 0.05)).float()
            main_is_uncertain = (main_conf < 0.80).float()
            prototype_override = (1.0 - agreement) * prototype_advantage * main_is_uncertain
            trust_gate = (
                agreement_support * (0.35 + main_uncertainty) * (prototype_support + uncertainty_support)
                + 0.50 * prototype_override * (0.35 + main_uncertainty) * prototype_prob
            ).clamp(min=0.0, max=1.0)
            if self.dynamic_gate is not None:
                learned_gate = self.dynamic_gate(torch.cat([relation_input, main_conf, prototype_conf], dim=1))
                trust_gate = (trust_gate * (0.50 + learned_gate)).clamp(min=0.0, max=1.0)
            class_gate = self._class_balance_gate().view(1, -1)
            prototype_delta = (raw_alignment_logits - raw_main_logits).clamp(
                min=-self.prototype_delta_clip,
                max=self.prototype_delta_clip,
            )
            dynamic_gate = trust_gate * class_gate * prototype_reliability.view(1, -1)
            corrected_raw_logits = raw_main_logits + self.prototype_correction_scale * dynamic_gate * prototype_delta
            if self.training:
                if self.training_output_mode == "prototype":
                    raw_logits = raw_alignment_logits
                elif self.training_output_mode == "corrected":
                    raw_logits = corrected_raw_logits
                else:
                    raw_logits = raw_main_logits
            else:
                raw_logits = corrected_raw_logits
            if targets is not None:
                target_mask = F.one_hot(targets, num_classes=self.num_classes).bool()
                positive_proto = raw_alignment_logits[target_mask]
                negative_proto = raw_alignment_logits.masked_fill(target_mask, -1e4).max(dim=1).values
                target_reliability = prototype_reliability[targets].detach()
                main_target_prob = torch.softmax(main_logits.detach(), dim=1).gather(1, targets.view(-1, 1)).squeeze(1)
                regularizer_weight = target_reliability * (0.50 + 0.50 * main_target_prob)
                normalizer = regularizer_weight.sum().clamp(min=1e-6)
                prototype_margin = (
                    F.relu(self.prototype_margin - (positive_proto - negative_proto)) * regularizer_weight
                ).sum() / normalizer
                prototype_ce = (F.cross_entropy(prototype_logits, targets, reduction="none") * regularizer_weight).sum()
                prototype_ce = prototype_ce / normalizer
                relation_consistency = 0.5 * (
                    F.mse_loss(stable_proto_logits, context_proto_logits)
                    + F.mse_loss(raw_alignment_logits, raw_main_logits.detach())
                )
                classifier_alignment = self._classifier_prototype_alignment_loss(
                    stable_bank=stable_bank,
                    stable_counts=stable_counts,
                    context_bank=context_bank,
                    context_counts=context_counts,
                )
                correction_penalty = (dynamic_gate * prototype_delta).abs().mean()
                head_regularizer = self._imbalance_strength().to(
                    device=raw_main_logits.device,
                    dtype=raw_main_logits.dtype,
                ) * (
                    0.10 * prototype_ce
                    + 0.20 * prototype_margin
                    + 0.03 * classifier_alignment
                    + 0.01 * relation_consistency
                    + 0.01 * correction_penalty
                )
            else:
                head_regularizer = 0.10 * (stable_proto_logits - context_proto_logits).pow(2).mean()
            projection_source = embedding
            fusion_weights = torch.cat([fusion_weights[:, :1], 1.0 - proto_mix, proto_mix], dim=1)
            if pending_update_reliability is not None and targets is not None:
                self._update_prototype_bank(
                    normalized_embedding.detach(),
                    targets.detach(),
                    bank=self.stable_prototypes,
                    bank_counts=self.stable_proto_counts,
                    update_weights=pending_update_reliability,
                )
                self._update_prototype_bank(
                    normalized_context_embedding.detach(),
                    targets.detach(),
                    bank=self.context_prototypes,
                    bank_counts=self.context_proto_counts,
                    update_weights=pending_update_reliability,
                )
        else:
            raw_alignment_logits = normalized_embedding.new_zeros(normalized_embedding.shape[0], self.num_classes)
            dynamic_gate = normalized_embedding.new_zeros(normalized_embedding.shape[0], self.num_classes)
            raw_logits = raw_main_logits
            fusion_weights = None
            head_regularizer = None
            projection_source = embedding
        logits = raw_logits * self.classifier.scale

        projection = _normalize_rows(self.projection_head(projection_source))
        aux_logits_mid = self.aux_mid(mid_feat) if self.aux_mid is not None and mid_feat is not None else None
        aux_logits_early = (
            self.aux_early(early_feat) if self.aux_early is not None and early_feat is not None else None
        )

        return LongTailHeadOutput(
            logits=logits,
            raw_logits=raw_logits,
            pooled_feature=pooled,
            embedding=normalized_embedding,
            projection=projection,
            main_logits=raw_main_logits * self.classifier.scale,
            raw_main_logits=raw_main_logits,
            # Keep legacy field names so the rest of the pipeline can stay
            # unchanged. They now store the selective-fusion branch.
            prototype_logits=raw_alignment_logits * self.classifier.scale,
            raw_prototype_logits=raw_alignment_logits,
            dynamic_gate=dynamic_gate,
            logit_scale=normalized_embedding.new_tensor(self.classifier.scale),
            fusion_weights=fusion_weights,
            head_regularizer=head_regularizer,
            aux_logits_mid=aux_logits_mid,
            aux_logits_early=aux_logits_early,
        )

    def _apply_alignment_affine(self, raw_fusion_logits: torch.Tensor) -> torch.Tensor:
        if self.alignment_scale is None or self.alignment_bias is None:
            return raw_fusion_logits

        scale_residual = 0.20 * self.alignment_scale_limit * torch.tanh(self.alignment_scale).view(1, -1)
        bias_residual = 0.20 * self.alignment_bias_limit * torch.tanh(self.alignment_bias).view(1, -1)
        return raw_fusion_logits * (1.0 + scale_residual) + bias_residual

    def _class_balance_gate(self) -> torch.Tensor:
        num_classes = int(self.class_counts.numel())
        if num_classes <= 1:
            return self.class_counts.new_zeros(num_classes)

        counts = self.class_counts.clamp(min=1.0)
        max_count = counts.max()
        min_count = counts.min()
        imbalance_strength = self._imbalance_strength()
        if float((max_count - min_count).item()) < 1e-6:
            return counts.new_zeros(num_classes)

        rarity = torch.log(max_count / counts) / torch.log(max_count / min_count).clamp(min=1e-6)
        rarity = rarity.clamp(min=0.0, max=1.0)
        return imbalance_strength * (0.12 + 0.48 * rarity)

    def _imbalance_strength(self) -> torch.Tensor:
        counts = self.class_counts.clamp(min=1.0)
        ratio = counts.max() / counts.min()
        # Prototype regularization is designed for long-tail splits. On
        # balanced splits, keep the upgraded head behaviorally close to the
        # baseline cosine head instead of injecting unnecessary bias.
        return (torch.log(ratio) / torch.log(counts.new_tensor(10.0))).clamp(min=0.0, max=1.0)

    @torch.no_grad()
    def _sample_update_reliability(self, raw_main_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.numel() == 0:
            return raw_main_logits.new_zeros((0,))

        scaled_logits = raw_main_logits * self.classifier.scale
        probs = torch.softmax(scaled_logits, dim=1)
        target_prob = probs.gather(1, targets.view(-1, 1)).squeeze(1)

        target_logits = scaled_logits.gather(1, targets.view(-1, 1)).squeeze(1)
        negative_logits = scaled_logits.masked_fill(
            F.one_hot(targets, num_classes=self.num_classes).bool(),
            -1e4,
        ).max(dim=1).values
        margin_score = torch.sigmoid((target_logits - negative_logits) / 2.0)
        pred_matches_target = scaled_logits.argmax(dim=1) == targets

        reliability = 0.65 * target_prob + 0.35 * margin_score
        reliability = torch.where(pred_matches_target, reliability, 0.25 * reliability)
        return reliability.clamp(min=self.prototype_min_update_weight, max=1.0)

    def _prototype_reliability(
        self,
        *,
        stable_counts: Optional[torch.Tensor] = None,
        context_counts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        stable_counts = self.stable_proto_counts if stable_counts is None else stable_counts
        context_counts = self.context_proto_counts if context_counts is None else context_counts
        stable_active = stable_counts > 0
        context_active = context_counts > 0
        stable_count = stable_counts.sum(dim=1)
        context_count = context_counts.sum(dim=1)

        stable_maturity = 1.0 - torch.exp(-stable_count / self.prototype_reliability_tau)
        context_maturity = 1.0 - torch.exp(-context_count / self.prototype_reliability_tau)
        occupancy = 0.5 * (stable_active.float().mean(dim=1) + context_active.float().mean(dim=1))
        maturity = 0.5 * (stable_maturity + context_maturity)
        return (0.65 * maturity + 0.35 * occupancy).clamp(min=0.0, max=1.0)

    def _classifier_prototype_alignment_loss(
        self,
        *,
        stable_bank: Optional[torch.Tensor] = None,
        stable_counts: Optional[torch.Tensor] = None,
        context_bank: Optional[torch.Tensor] = None,
        context_counts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        classifier_weight = _normalize_rows(self.classifier.weight)
        stable_bank = self.stable_prototypes.detach().clone() if stable_bank is None else stable_bank
        stable_counts = self.stable_proto_counts.detach().clone() if stable_counts is None else stable_counts
        context_bank = self.context_prototypes.detach().clone() if context_bank is None else context_bank
        context_counts = self.context_proto_counts.detach().clone() if context_counts is None else context_counts

        stable_mask = stable_counts > 0
        context_mask = context_counts > 0
        if not bool(stable_mask.any().item() or context_mask.any().item()):
            return classifier_weight.new_zeros(())

        stable_sum = (stable_bank * stable_mask.unsqueeze(-1)).sum(dim=1)
        stable_denom = stable_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        stable_mean = stable_sum / stable_denom

        context_sum = (context_bank * context_mask.unsqueeze(-1)).sum(dim=1)
        context_denom = context_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        context_mean = context_sum / context_denom

        active = stable_mask.any(dim=1) | context_mask.any(dim=1)
        prototype_mean = 0.5 * (stable_mean + context_mean)
        prototype_mean = _normalize_rows(prototype_mean)
        if not bool(active.any().item()):
            return classifier_weight.new_zeros(())
        return (1.0 - (classifier_weight[active] * prototype_mean[active]).sum(dim=1)).mean()

    @torch.no_grad()
    def _update_prototype_bank(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        *,
        bank: torch.Tensor,
        bank_counts: torch.Tensor,
        update_weights: Optional[torch.Tensor] = None,
    ) -> None:
        if embeddings.numel() == 0 or labels.numel() == 0:
            return

        embeddings = _normalize_rows(embeddings)
        if update_weights is None:
            update_weights = embeddings.new_ones((embeddings.shape[0],))
        update_weights = update_weights.to(device=embeddings.device, dtype=embeddings.dtype).clamp(min=0.0, max=1.0)

        for feat, label, update_weight in zip(embeddings, labels, update_weights):
            class_id = int(label.item())
            class_bank = bank[class_id]
            class_counts = bank_counts[class_id]
            active = class_counts > 0

            if not bool(active.any().item()):
                slot = 0
            else:
                active_indices = active.nonzero(as_tuple=False).flatten()
                active_bank = class_bank[active_indices]
                sims = torch.matmul(active_bank, feat)
                best_active_slot = int(active_indices[int(sims.argmax().item())].item())
                should_open_slot = (
                    int(active.sum().item()) < self.num_prototypes_per_class
                    and float(sims.max().item()) < self.prototype_new_slot_threshold
                    and float(update_weight.item()) >= self.prototype_new_slot_min_weight
                )
                if should_open_slot:
                    slot = int((~active).nonzero(as_tuple=False).flatten()[0].item())
                else:
                    slot = best_active_slot

            if float(class_counts[slot].item()) <= 0.0:
                updated = feat
            else:
                adaptive_momentum = 1.0 - (1.0 - self.prototype_momentum) * float(update_weight.item())
                updated = adaptive_momentum * class_bank[slot] + (1.0 - adaptive_momentum) * feat
                updated = _normalize_rows(updated.unsqueeze(0)).squeeze(0)
            class_bank[slot].copy_(updated)
            class_counts[slot] += update_weight.clamp(min=self.prototype_min_update_weight)

    def _prototype_logits_from_bank(
        self,
        query: torch.Tensor,
        *,
        bank: torch.Tensor,
        bank_counts: torch.Tensor,
    ) -> torch.Tensor:
        mask = bank_counts > 0
        if not bool(mask.any().item()):
            return query.new_zeros((query.shape[0], self.num_classes))

        sims = torch.einsum("bd,ckd->bck", query, bank)
        sims = sims.masked_fill(~mask.unsqueeze(0), -1e4)
        weights = torch.softmax(sims / self.prototype_temperature, dim=-1)
        weights = torch.where(mask.unsqueeze(0), weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        relation_score = (weights * sims).sum(dim=-1)
        best_score = sims.max(dim=-1).values
        occupancy = mask.float().mean(dim=-1).view(1, -1)
        logits = 0.5 * (relation_score + best_score) + 0.05 * occupancy
        logits = torch.where(mask.any(dim=-1).view(1, -1), logits, torch.zeros_like(logits))
        return logits

    def _compute_guided_fusion_embedding(
        self,
        *,
        final_embedding: torch.Tensor,
        mid_feat: Optional[torch.Tensor],
        early_feat: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mid_neck is None or self.early_neck is None or self.mid_guidance is None or self.early_guidance is None:
            batch_size = final_embedding.shape[0]
            default_weights = final_embedding.new_zeros((batch_size, 3))
            default_weights[:, 0] = 1.0
            return final_embedding, default_weights

        if mid_feat is None:
            mid_embedding = final_embedding
        else:
            mid_embedding = self.mid_neck(self.pool(mid_feat).flatten(1))

        if early_feat is None:
            early_embedding = final_embedding
        else:
            early_embedding = self.early_neck(self.pool(early_feat).flatten(1))

        mid_gate = torch.sigmoid(self.mid_guidance(final_embedding + mid_embedding))
        early_gate = torch.sigmoid(self.early_guidance(final_embedding + early_embedding))
        fused_embedding = final_embedding + self.fusion_residual_scale * (
            mid_gate * mid_embedding + early_gate * early_embedding
        )
        fusion_weights = torch.stack(
            [
                torch.ones(final_embedding.shape[0], device=final_embedding.device, dtype=final_embedding.dtype),
                mid_gate.mean(dim=1),
                early_gate.mean(dim=1),
            ],
            dim=1,
        )
        return fused_embedding, fusion_weights

    def freeze_source_classifier(self) -> None:
        if not self.freeze_source_classifier_in_stage2 or not self.use_prototype_branch:
            return
        for module in (self.feature_neck, self.classifier, self.projection_head, self.aux_mid, self.aux_early):
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = False

    def release_source_classifier(self) -> None:
        for module in (self.feature_neck, self.classifier, self.projection_head, self.aux_mid, self.aux_early):
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = True
