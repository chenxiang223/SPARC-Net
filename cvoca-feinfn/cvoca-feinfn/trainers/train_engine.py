from __future__ import annotations

from contextlib import nullcontext
import copy
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.amp import GradScaler, autocast
    _AUTOCAST_SUPPORTS_DEVICE_TYPE = True
except ImportError:  # pragma: no cover - compatibility for older PyTorch
    from torch.cuda.amp import GradScaler, autocast
    _AUTOCAST_SUPPORTS_DEVICE_TYPE = False

from metrics import compute_classification_metrics

from .calibration import (
    CalibrationResult,
    RTPCConfig,
    ReversibleTailPriorCalibrator,
    search_prior_correction,
)


@dataclass(frozen=True)
class TrainingStageConfig:
    """
    Configuration for one training stage.
    """

    name: str
    epochs: int
    lr: float
    weight_decay: float = 1e-4
    freeze_backbone: bool = False
    reset_classifier: bool = False
    use_aux_loss: bool = True
    use_contrastive_loss: bool = True
    use_balanced_loader: bool = False
    head_regularization_weight: float = 0.0
    grad_clip_norm: Optional[float] = None
    head_output_mode: str = "main"
    update_prototypes: bool = True
    guided_context: bool = False


@dataclass(frozen=True)
class StagedTrainerConfig:
    """
    Two-stage training configuration for long-tail recognition.
    """

    stage1: TrainingStageConfig
    stage2: TrainingStageConfig
    aux_mid_weight: float = 0.3
    aux_early_weight: float = 0.15
    contrastive_weight: float = 0.1
    amp: bool = True
    stage2_tau_norm: Optional[float] = None
    monitor: str = "macro_acc"
    restore_best_state: bool = True
    stage2_start_from_best_stage1: bool = True
    stage2_revert_if_no_val_gain: bool = True
    stage2_min_val_gain: float = 0.0002
    stage2_skip_if_stage1_val_at_least: Optional[float] = None
    calibration: Optional[RTPCConfig] = None
    prototype_correction_candidates: Tuple[float, ...] = (0.0, 0.15, 0.30, 0.50, 0.80, 1.20, 1.60)
    prototype_correction_min_gain: float = 0.0003
    prototype_correction_head_drop_tolerance: float = 0.0015
    verbose: bool = True
    print_every: int = 1


@dataclass(frozen=True)
class EpochStats:
    stage: str
    epoch: int
    train_loss: float
    train_main_loss: float
    train_aux_loss: float
    train_contrastive_loss: float
    train_acc: float
    train_macro_acc: float
    train_metrics: Dict[str, float] = field(default_factory=dict)
    val_loss: Optional[float] = None
    val_acc: Optional[float] = None
    val_macro_acc: Optional[float] = None
    val_metrics: Optional[Dict[str, float]] = None
    test_loss: Optional[float] = None
    test_acc: Optional[float] = None
    test_macro_acc: Optional[float] = None
    test_metrics: Optional[Dict[str, float]] = None


@dataclass
class TrainingHistory:
    logs: List[EpochStats] = field(default_factory=list)
    best_metric: float = float("-inf")
    best_stage: Optional[str] = None
    best_epoch: Optional[int] = None
    best_test_metric: float = float("-inf")
    best_test_stage: Optional[str] = None
    best_test_epoch: Optional[int] = None
    best_test_eval: Optional[Dict[str, object]] = None


def _move_to_device(batch, device: torch.device):
    if isinstance(batch, dict):
        patches = batch["patch"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        return patches, labels
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        patches = batch[0].to(device, non_blocking=True)
        labels = batch[1].to(device, non_blocking=True)
        return patches, labels
    raise ValueError("Batch must be a dict with patch/label or a tuple like (patches, labels).")


def _safe_item(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def _make_amp_context(enabled: bool):
    if not enabled:
        return nullcontext()
    if _AUTOCAST_SUPPORTS_DEVICE_TYPE:
        return autocast(device_type="cuda", enabled=True)
    return autocast(enabled=True)


def _call_classification_loss(loss_fn, logits: torch.Tensor, labels: torch.Tensor, logits_scale):
    try:
        return loss_fn(logits, labels, logits_scale=logits_scale)
    except TypeError:
        return loss_fn(logits, labels)


def _macro_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    num_classes = int(max(int(preds.max().item()), int(labels.max().item())) + 1) if labels.numel() > 0 else 0
    if num_classes == 0:
        return 0.0
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


def format_metric_summary_from_dict(summary: Dict[str, float], prefix: str = "") -> str:
    parts = []
    if prefix:
        parts.append(prefix)
    for key in ("oa", "aa", "kappa", "macro_f1", "head_aa", "medium_aa", "tail_aa"):
        if key in summary:
            parts.append(f"{key.upper()}={summary[key]:.4f}")
    return " ".join(parts)


class StagedLongTailTrainer:
    """
    Long-tail trainer for the upgraded HSI pipeline.

    Stage 1:
    - train the full backbone
    - use main classification + auxiliary heads + contrastive loss

    Stage 2:
    - freeze the backbone
    - optionally reset the classifier
    - train only the head, usually with a more balanced loader

    Optional Stage 3:
    - keep the model frozen
    - fit a lightweight post-hoc logit calibrator
    - search prior correction on the validation split
    """

    def __init__(
        self,
        model: nn.Module,
        classification_loss: nn.Module,
        *,
        contrastive_loss: Optional[nn.Module] = None,
        config: Optional[StagedTrainerConfig] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        if config is None:
            config = StagedTrainerConfig(
                stage1=TrainingStageConfig(name="stage1", epochs=100, lr=3e-4, use_aux_loss=True, use_contrastive_loss=True),
                stage2=TrainingStageConfig(
                    name="stage2",
                    epochs=50,
                    lr=1e-3,
                    weight_decay=0.0,
                    freeze_backbone=True,
                    reset_classifier=True,
                    use_aux_loss=False,
                    use_contrastive_loss=False,
                    use_balanced_loader=True,
                ),
            )
        self.model = model
        self.classification_loss = classification_loss
        self.contrastive_loss = contrastive_loss
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.use_amp = bool(config.amp and self.device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)
        self.logit_calibrator: Optional[ReversibleTailPriorCalibrator] = None
        self.calibration_result: Optional[CalibrationResult] = None
        if self.config.verbose:
            print(
                f"[Trainer Device] {self.device.type}"
                + (f":{self.device.index}" if self.device.index is not None else "")
                + f" | AMP={self.use_amp}",
                flush=True,
            )

    def fit(
        self,
        train_loader,
        *,
        val_loader=None,
        test_loader=None,
        balanced_train_loader=None,
    ) -> TrainingHistory:
        history = TrainingHistory()
        best_state_dict = None
        global_epoch = 0

        for stage_cfg in (self.config.stage1, self.config.stage2):
            if stage_cfg.epochs <= 0:
                continue

            stage_entry_metric = history.best_metric
            stage_entry_stage = history.best_stage
            stage_entry_epoch = history.best_epoch
            stage_entry_state = copy.deepcopy(best_state_dict) if best_state_dict is not None else None

            if (
                stage_cfg.name == self.config.stage2.name
                and self.config.stage2_start_from_best_stage1
                and best_state_dict is not None
            ):
                self.model.load_state_dict(best_state_dict)
                stage_entry_state = copy.deepcopy(best_state_dict)

            if (
                stage_cfg.name == self.config.stage2.name
                and val_loader is not None
                and self.config.stage2_skip_if_stage1_val_at_least is not None
                and stage_entry_metric >= float(self.config.stage2_skip_if_stage1_val_at_least)
            ):
                if self.config.verbose:
                    print(
                        f"[stage2_gate] skipped because best stage1 val "
                        f"{stage_entry_metric:.4f} >= {self.config.stage2_skip_if_stage1_val_at_least:.4f}",
                        flush=True,
                    )
                continue

            self._configure_stage(stage_cfg)
            optimizer = self._build_optimizer(stage_cfg)
            scheduler = self._build_scheduler(optimizer, stage_cfg)
            stage_loader = (
                balanced_train_loader
                if stage_cfg.use_balanced_loader and balanced_train_loader is not None
                else train_loader
            )

            for local_epoch in range(stage_cfg.epochs):
                self._set_loss_epoch(global_epoch)
                train_metrics = self._run_one_epoch(
                    stage_loader,
                    optimizer=optimizer,
                    stage_cfg=stage_cfg,
                    training=True,
                )

                if scheduler is not None:
                    scheduler.step()

                val_metrics = None
                if val_loader is not None:
                    val_metrics = self._run_one_epoch(
                        val_loader,
                        optimizer=None,
                        stage_cfg=stage_cfg,
                        training=False,
                    )
                test_metrics = None
                if test_loader is not None:
                    test_metrics = self._run_one_epoch(
                        test_loader,
                        optimizer=None,
                        stage_cfg=stage_cfg,
                        training=False,
                    )

                stats = EpochStats(
                    stage=stage_cfg.name,
                    epoch=global_epoch,
                    train_loss=train_metrics["loss"],
                    train_main_loss=train_metrics["main_loss"],
                    train_aux_loss=train_metrics["aux_loss"],
                    train_contrastive_loss=train_metrics["contrastive_loss"],
                    train_acc=train_metrics["acc"],
                    train_macro_acc=train_metrics["macro_acc"],
                    train_metrics=train_metrics["summary_metrics"],
                    val_loss=None if val_metrics is None else val_metrics["loss"],
                    val_acc=None if val_metrics is None else val_metrics["acc"],
                    val_macro_acc=None if val_metrics is None else val_metrics["macro_acc"],
                    val_metrics=None if val_metrics is None else val_metrics["summary_metrics"],
                    test_loss=None if test_metrics is None else test_metrics["loss"],
                    test_acc=None if test_metrics is None else test_metrics["acc"],
                    test_macro_acc=None if test_metrics is None else test_metrics["macro_acc"],
                    test_metrics=None if test_metrics is None else test_metrics["summary_metrics"],
                )
                history.logs.append(stats)

                if self.config.verbose and self.config.print_every > 0 and (local_epoch + 1) % self.config.print_every == 0:
                    self._print_epoch_summary(stats)

                monitor_value = self._get_monitor_value(stats)
                if monitor_value > history.best_metric:
                    history.best_metric = monitor_value
                    history.best_stage = stage_cfg.name
                    history.best_epoch = global_epoch
                    if self.config.restore_best_state:
                        best_state_dict = copy.deepcopy(self.model.state_dict())

                if test_metrics is not None:
                    self._update_best_test_history(history, test_metrics, stage_cfg.name, global_epoch)

                global_epoch += 1

            if (
                stage_cfg.name == self.config.stage2.name
                and val_loader is not None
                and self.config.stage2_revert_if_no_val_gain
                and stage_entry_state is not None
                and history.best_metric < stage_entry_metric + self.config.stage2_min_val_gain
            ):
                self.model.load_state_dict(stage_entry_state)
                best_state_dict = copy.deepcopy(stage_entry_state)
                history.best_metric = stage_entry_metric
                history.best_stage = stage_entry_stage
                history.best_epoch = stage_entry_epoch
                if self.config.verbose:
                    print(
                        f"[stage2_gate] reverted to {stage_entry_stage} "
                        f"because val gain < {self.config.stage2_min_val_gain:.5f}",
                        flush=True,
                    )

        if self.config.stage2_tau_norm is not None:
            self.model.apply_classifier_tau_normalization(tau=self.config.stage2_tau_norm)
            if val_loader is not None:
                tau_metrics = self.evaluate(val_loader)
                tau_test_metrics = self.evaluate(test_loader) if test_loader is not None else None
                history.logs.append(
                    EpochStats(
                        stage="stage2_tau_norm",
                        epoch=global_epoch,
                        train_loss=0.0,
                        train_main_loss=0.0,
                        train_aux_loss=0.0,
                        train_contrastive_loss=0.0,
                        train_acc=0.0,
                        train_macro_acc=0.0,
                        train_metrics={},
                        val_loss=tau_metrics["loss"],
                        val_acc=tau_metrics["acc"],
                        val_macro_acc=tau_metrics["macro_acc"],
                        val_metrics=tau_metrics["summary_metrics"],
                        test_loss=None if tau_test_metrics is None else tau_test_metrics["loss"],
                        test_acc=None if tau_test_metrics is None else tau_test_metrics["acc"],
                        test_macro_acc=None if tau_test_metrics is None else tau_test_metrics["macro_acc"],
                        test_metrics=None if tau_test_metrics is None else tau_test_metrics["summary_metrics"],
                    )
                )
                if self.config.verbose:
                    self._print_epoch_summary(history.logs[-1])
                monitor_value = tau_metrics["macro_acc"] if self.config.monitor == "macro_acc" else tau_metrics["acc"]
                if monitor_value > history.best_metric:
                    history.best_metric = monitor_value
                    history.best_stage = "stage2_tau_norm"
                    history.best_epoch = global_epoch
                    if self.config.restore_best_state:
                        best_state_dict = copy.deepcopy(self.model.state_dict())
                if tau_test_metrics is not None:
                    self._update_best_test_history(history, tau_test_metrics, "stage2_tau_norm", global_epoch)

        if self.config.restore_best_state and best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        self._set_prototype_correction_scale(0.0)
        if hasattr(self.model, "set_head_training_mode"):
            self.model.set_head_training_mode(
                output_mode="corrected",
                update_prototypes=False,
                guided_context=True,
            )
        self._tune_prototype_correction_scale(val_loader)

        if self.config.calibration is not None:
            self.logit_calibrator, self.calibration_result = self._fit_rtpc_calibration(
                train_loader,
                val_loader=val_loader,
                balanced_train_loader=balanced_train_loader,
            )
            if self.calibration_result is not None:
                history.logs.append(
                    EpochStats(
                        stage="stage3_calibration",
                        epoch=global_epoch,
                        train_loss=self.calibration_result.train_loss,
                        train_main_loss=self.calibration_result.train_loss,
                        train_aux_loss=0.0,
                        train_contrastive_loss=0.0,
                        train_acc=self.calibration_result.train_acc,
                        train_macro_acc=self.calibration_result.train_macro_acc,
                        train_metrics={},
                        val_loss=self.calibration_result.val_loss,
                        val_acc=self.calibration_result.val_acc,
                        val_macro_acc=self.calibration_result.val_macro_acc,
                        val_metrics=None,
                        test_loss=None,
                        test_acc=None,
                        test_macro_acc=None,
                        test_metrics=None,
                    )
                )
                if test_loader is not None:
                    calibration_test_metrics = self.evaluate(test_loader)
                    history.logs[-1] = EpochStats(
                        stage=history.logs[-1].stage,
                        epoch=history.logs[-1].epoch,
                        train_loss=history.logs[-1].train_loss,
                        train_main_loss=history.logs[-1].train_main_loss,
                        train_aux_loss=history.logs[-1].train_aux_loss,
                        train_contrastive_loss=history.logs[-1].train_contrastive_loss,
                        train_acc=history.logs[-1].train_acc,
                        train_macro_acc=history.logs[-1].train_macro_acc,
                        train_metrics=history.logs[-1].train_metrics,
                        val_loss=history.logs[-1].val_loss,
                        val_acc=history.logs[-1].val_acc,
                        val_macro_acc=history.logs[-1].val_macro_acc,
                        val_metrics=history.logs[-1].val_metrics,
                        test_loss=calibration_test_metrics["loss"],
                        test_acc=calibration_test_metrics["acc"],
                        test_macro_acc=calibration_test_metrics["macro_acc"],
                        test_metrics=calibration_test_metrics["summary_metrics"],
                    )
                    self._update_best_test_history(history, calibration_test_metrics, "stage3_calibration", global_epoch)
                if self.config.verbose:
                    self._print_calibration_summary(self.calibration_result)
                calibration_metric = (
                    self.calibration_result.val_macro_acc
                    if self.config.monitor == "macro_acc"
                    else self.calibration_result.val_acc
                )
                if calibration_metric is not None and calibration_metric > history.best_metric:
                    history.best_metric = calibration_metric
                    history.best_stage = "stage3_calibration"
                    history.best_epoch = global_epoch

        if test_loader is not None:
            self.test_metrics = self.evaluate(test_loader)
        else:
            self.test_metrics = None
        return history

    def _set_prototype_correction_scale(self, scale: float) -> None:
        if hasattr(self.model, "set_prototype_correction_scale"):
            self.model.set_prototype_correction_scale(scale)

    def _tune_prototype_correction_scale(self, val_loader) -> None:
        candidates = tuple(float(x) for x in self.config.prototype_correction_candidates)
        if val_loader is None or not candidates or not hasattr(self.model, "set_prototype_correction_scale"):
            return

        baseline_scale = 0.0 if 0.0 in candidates else candidates[0]
        best_scale = baseline_scale
        best_score = float("-inf")
        baseline_score = None
        baseline_head_aa = None
        candidate_scores = []

        for scale in candidates:
            self.model.set_prototype_correction_scale(scale)
            metrics = self.evaluate(val_loader)
            summary = metrics.get("summary_metrics", {})
            macro_f1 = float(summary.get("macro_f1", metrics["macro_acc"]))
            head_aa = float(summary.get("head_aa", metrics["acc"]))
            medium_aa = float(summary.get("medium_aa", metrics["macro_acc"]))
            tail_aa = float(summary.get("tail_aa", metrics["macro_acc"]))
            score = (
                0.50 * float(metrics["acc"])
                + 0.20 * medium_aa
                + 0.20 * tail_aa
                + 0.10 * macro_f1
            )
            candidate_scores.append((scale, score))
            if scale == baseline_scale:
                baseline_score = score
                baseline_head_aa = head_aa
            if (
                baseline_head_aa is not None
                and head_aa < baseline_head_aa - self.config.prototype_correction_head_drop_tolerance
            ):
                continue
            if score > best_score:
                best_score = score
                best_scale = scale

        if baseline_score is not None and best_scale != baseline_scale:
            if best_score < baseline_score + self.config.prototype_correction_min_gain:
                best_scale = baseline_scale
                best_score = baseline_score

        self.model.set_prototype_correction_scale(best_scale)
        score_text = ", ".join(f"{scale:.2f}:{score:.5f}" for scale, score in candidate_scores)
        print(f"[prototype_correction_candidates] {score_text}", flush=True)
        print(
            f"[prototype_correction] selected_scale={best_scale:.4f} "
            f"val_balanced_score={best_score:.4f}",
            flush=True,
        )

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        eval_stage = TrainingStageConfig(name="eval", epochs=1, lr=0.0, use_aux_loss=False, use_contrastive_loss=False)
        return self._run_one_epoch(loader, optimizer=None, stage_cfg=eval_stage, training=False)

    def _configure_stage(self, stage_cfg: TrainingStageConfig) -> None:
        if stage_cfg.freeze_backbone and hasattr(self.model, "freeze_backbone"):
            self.model.freeze_backbone()
        elif not stage_cfg.freeze_backbone and hasattr(self.model, "unfreeze_all"):
            self.model.unfreeze_all()
        else:
            for param in self.model.parameters():
                param.requires_grad = True

        if stage_cfg.reset_classifier and hasattr(self.model, "reset_classifier"):
            self.model.reset_classifier()

        if hasattr(self.model, "set_head_training_mode"):
            self.model.set_head_training_mode(
                output_mode=stage_cfg.head_output_mode,
                update_prototypes=stage_cfg.update_prototypes,
                guided_context=stage_cfg.guided_context,
            )

    def _build_optimizer(self, stage_cfg: TrainingStageConfig):
        params = [param for param in self.model.parameters() if param.requires_grad]
        if not params:
            raise RuntimeError(f"No trainable parameters were found for stage {stage_cfg.name}.")
        return torch.optim.AdamW(params, lr=stage_cfg.lr, weight_decay=stage_cfg.weight_decay)

    @staticmethod
    def _build_scheduler(optimizer, stage_cfg: TrainingStageConfig):
        if stage_cfg.epochs <= 1:
            return None
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_cfg.epochs)

    def _set_loss_epoch(self, epoch: int) -> None:
        if hasattr(self.classification_loss, "set_epoch"):
            self.classification_loss.set_epoch(epoch)

    def _run_one_epoch(self, loader, *, optimizer, stage_cfg: TrainingStageConfig, training: bool) -> Dict[str, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_main_loss = 0.0
        total_aux_loss = 0.0
        total_contrastive_loss = 0.0
        total_correct = 0.0
        total_samples = 0
        all_logits = []
        all_labels = []

        for batch in loader:
            patches, labels = _move_to_device(batch, self.device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            amp_context = _make_amp_context(self.use_amp)
            with amp_context:
                outputs = self.model(patches, targets=labels if training else None, return_aux=True)
                if not training and self.logit_calibrator is not None:
                    outputs = dict(outputs)
                    outputs["logits"] = self.logit_calibrator(outputs["logits"])
                    main_loss = F.cross_entropy(outputs["logits"], labels)
                    zero = outputs["logits"].new_zeros(())
                    loss_dict = {
                        "loss": main_loss,
                        "main_loss": main_loss,
                        "aux_loss": zero,
                        "contrastive_loss": zero,
                    }
                else:
                    loss_dict = self._compute_loss_dict(outputs, labels, stage_cfg)
                total_batch_loss = loss_dict["loss"]

            if training:
                if self.use_amp:
                    self.scaler.scale(total_batch_loss).backward()
                    if stage_cfg.grad_clip_norm is not None:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), stage_cfg.grad_clip_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    total_batch_loss.backward()
                    if stage_cfg.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), stage_cfg.grad_clip_norm)
                    optimizer.step()

            logits = outputs["logits"].detach()
            all_logits.append(logits)
            all_labels.append(labels.detach())
            total_samples += int(labels.shape[0])
            total_correct += float((logits.argmax(dim=1) == labels).sum().item())
            total_loss += _safe_item(total_batch_loss) * labels.shape[0]
            total_main_loss += _safe_item(loss_dict["main_loss"]) * labels.shape[0]
            total_aux_loss += _safe_item(loss_dict["aux_loss"]) * labels.shape[0]
            total_contrastive_loss += _safe_item(loss_dict["contrastive_loss"]) * labels.shape[0]

        all_logits = torch.cat(all_logits, dim=0) if all_logits else torch.empty(0, 0, device=self.device)
        all_labels = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long, device=self.device)
        avg_loss = total_loss / max(total_samples, 1)
        avg_main_loss = total_main_loss / max(total_samples, 1)
        avg_aux_loss = total_aux_loss / max(total_samples, 1)
        avg_contrastive_loss = total_contrastive_loss / max(total_samples, 1)
        acc = total_correct / max(total_samples, 1)
        macro_acc = _macro_accuracy(all_logits, all_labels) if total_samples > 0 else 0.0
        class_counts = self._get_metric_class_counts()
        classification_metrics = compute_classification_metrics(
            logits=all_logits,
            labels=all_labels,
            num_classes=all_logits.shape[1] if all_logits.ndim == 2 and all_logits.numel() > 0 else None,
            class_counts=class_counts,
        )

        return {
            "loss": avg_loss,
            "main_loss": avg_main_loss,
            "aux_loss": avg_aux_loss,
            "contrastive_loss": avg_contrastive_loss,
            "acc": acc,
            "macro_acc": macro_acc,
            "classification_metrics": classification_metrics,
            "summary_metrics": classification_metrics.to_summary_dict(),
        }

    def _compute_loss_dict(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor, stage_cfg: TrainingStageConfig):
        logits_scale = outputs.get("logit_scale", 1.0)
        main_loss = _call_classification_loss(self.classification_loss, outputs["logits"], labels, logits_scale)

        aux_loss = outputs["logits"].new_zeros(())
        if stage_cfg.use_aux_loss:
            if "aux_logits_mid" in outputs:
                aux_loss = aux_loss + self.config.aux_mid_weight * _call_classification_loss(
                    self.classification_loss,
                    outputs["aux_logits_mid"],
                    labels,
                    logits_scale,
                )
            if "aux_logits_early" in outputs:
                aux_loss = aux_loss + self.config.aux_early_weight * _call_classification_loss(
                    self.classification_loss,
                    outputs["aux_logits_early"],
                    labels,
                    logits_scale,
                )

        contrastive_loss = outputs["logits"].new_zeros(())
        if stage_cfg.use_contrastive_loss and self.contrastive_loss is not None and "projection" in outputs:
            contrastive_loss = self.config.contrastive_weight * self.contrastive_loss(outputs["projection"], labels)

        head_regularization_loss = outputs["logits"].new_zeros(())
        if stage_cfg.head_regularization_weight > 0.0 and "head_regularizer" in outputs:
            head_regularization_loss = stage_cfg.head_regularization_weight * outputs["head_regularizer"]

        total_loss = main_loss + aux_loss + contrastive_loss + head_regularization_loss
        return {
            "loss": total_loss,
            "main_loss": main_loss,
            "aux_loss": aux_loss + head_regularization_loss,
            "contrastive_loss": contrastive_loss,
        }

    def _get_monitor_value(self, stats: EpochStats) -> float:
        if self.config.monitor == "acc":
            return stats.train_acc if stats.val_acc is None else stats.val_acc
        if self.config.monitor == "macro_acc":
            return stats.train_macro_acc if stats.val_macro_acc is None else stats.val_macro_acc
        raise ValueError("monitor must be 'acc' or 'macro_acc'.")

    def _get_metric_class_counts(self):
        try:
            return self._infer_class_counts()
        except RuntimeError:
            return None

    @staticmethod
    def _print_epoch_summary(stats: EpochStats) -> None:
        train_parts = [
            f"[{stats.stage}]",
            f"epoch={stats.epoch + 1}",
            f"train_loss={stats.train_loss:.4f}",
        ]
        if stats.train_metrics:
            train_parts.append(format_metric_summary_from_dict(stats.train_metrics, prefix="train"))
        if stats.val_metrics is not None and stats.val_loss is not None:
            train_parts.append(f"val_loss={stats.val_loss:.4f}")
            train_parts.append(format_metric_summary_from_dict(stats.val_metrics, prefix="val"))
        if stats.test_metrics is not None and stats.test_loss is not None:
            train_parts.append(f"test_loss={stats.test_loss:.4f}")
            train_parts.append(format_metric_summary_from_dict(stats.test_metrics, prefix="test"))
        print(" | ".join(train_parts), flush=True)

    def _update_best_test_history(
        self,
        history: TrainingHistory,
        test_metrics: Dict[str, object],
        stage_name: str,
        epoch: int,
    ) -> None:
        monitor_key = "macro_acc" if self.config.monitor == "macro_acc" else "acc"
        monitor_value = float(test_metrics[monitor_key])
        if monitor_value > history.best_test_metric:
            history.best_test_metric = monitor_value
            history.best_test_stage = stage_name
            history.best_test_epoch = epoch
            history.best_test_eval = dict(test_metrics)

    @staticmethod
    def _print_calibration_summary(calibration_result: CalibrationResult) -> None:
        parts = [
            "[stage3_calibration]",
            f"train_loss={calibration_result.train_loss:.4f}",
            f"train_macro_acc={calibration_result.train_macro_acc:.4f}",
        ]
        if calibration_result.baseline_val_macro_acc is not None:
            parts.append(f"base_val_macro_acc={calibration_result.baseline_val_macro_acc:.4f}")
        if calibration_result.val_macro_acc is not None:
            parts.append(f"val_macro_acc={calibration_result.val_macro_acc:.4f}")
        parts.append(f"prior_mode={calibration_result.prior_mode}")
        parts.append(f"prior_alpha={calibration_result.prior_alpha:.4f}")
        parts.append(f"reversibility={calibration_result.reversibility_strength:.2f}")
        print(" | ".join(parts), flush=True)

    def _infer_class_counts(self) -> torch.Tensor:
        if hasattr(self.classification_loss, "class_counts"):
            return torch.as_tensor(self.classification_loss.class_counts, dtype=torch.float32).detach().cpu()
        classifier = getattr(self.model, "classifier", None)
        if classifier is not None and hasattr(classifier, "class_counts"):
            return torch.as_tensor(classifier.class_counts, dtype=torch.float32).detach().cpu()
        raise RuntimeError(
            "Could not infer class_counts for calibration. "
            "Build the trainer with a loss or classifier that exposes class_counts."
        )

    @torch.no_grad()
    def _collect_logits(self, loader, *, apply_calibrator: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        all_logits = []
        all_labels = []
        for batch in loader:
            patches, labels = _move_to_device(batch, self.device)
            outputs = self.model(patches, return_aux=True)
            logits = outputs["logits"].detach()
            if apply_calibrator and self.logit_calibrator is not None:
                logits = self.logit_calibrator(logits)
            all_logits.append(logits)
            all_labels.append(labels.detach())
        if not all_logits:
            return (
                torch.empty(0, 0, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device),
            )
        return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)

    def _fit_rtpc_calibration(self, train_loader, *, val_loader=None, balanced_train_loader=None):
        cfg = self.config.calibration
        if cfg is None:
            return None, None

        class_counts = self._infer_class_counts()
        calibrator = ReversibleTailPriorCalibrator(
            num_classes=int(class_counts.numel()),
            class_counts=class_counts,
            learn_class_scales=cfg.learn_class_scales,
            learn_logit_residual_corrector=cfg.learn_logit_residual_corrector,
            normalize_scales=cfg.normalize_scales,
            max_log_scale=cfg.max_log_scale,
            max_logit_residual=cfg.max_logit_residual,
            effective_num_beta=cfg.effective_num_beta,
        ).to(self.device)

        if cfg.train_loader_preference == "val" and val_loader is not None:
            calibration_loader = val_loader
        elif cfg.train_loader_preference == "balanced" and balanced_train_loader is not None:
            calibration_loader = balanced_train_loader
        else:
            calibration_loader = train_loader

        def measure_logits(logits: torch.Tensor, labels: torch.Tensor):
            if labels.numel() == 0:
                return 0.0, 0.0, 0.0, float("-inf")
            loss = float(F.cross_entropy(logits, labels).item())
            acc = float((logits.argmax(dim=1) == labels).float().mean().item())
            macro_acc = _macro_accuracy(logits, labels)
            if cfg.monitor == "macro_acc":
                score = macro_acc
            elif cfg.monitor == "acc":
                score = acc
            else:
                raise ValueError("calibration monitor must be 'acc' or 'macro_acc'.")
            return loss, acc, macro_acc, score

        if calibrator.scaler is not None and cfg.epochs > 0:
            optimizer = torch.optim.AdamW(calibrator.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            self.model.eval()
            for _ in range(cfg.epochs):
                for batch in calibration_loader:
                    patches, labels = _move_to_device(batch, self.device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        base_logits = self.model(patches, return_aux=True)["logits"].detach()
                    scaled_logits = calibrator.forward_without_prior(base_logits)
                    loss = F.cross_entropy(scaled_logits, labels)
                    loss.backward()
                    optimizer.step()

        baseline_val_acc = None
        baseline_val_macro_acc = None
        if val_loader is not None:
            base_val_logits, val_labels = self._collect_logits(val_loader, apply_calibrator=False)
            base_val_loss, base_val_acc, base_val_macro_acc, base_score = measure_logits(base_val_logits, val_labels)
            baseline_val_acc = base_val_acc
            baseline_val_macro_acc = base_val_macro_acc
            scaled_val_logits = calibrator.forward_without_prior(base_val_logits)
            prior_search = search_prior_correction(
                scaled_val_logits,
                val_labels,
                class_counts,
                prior_modes=cfg.prior_modes,
                prior_alpha_candidates=cfg.prior_alpha_candidates,
                effective_num_beta=cfg.effective_num_beta,
                monitor=cfg.monitor,
            )
            calibrator.set_prior_correction(prior_search.prior_mode, prior_search.prior_alpha)
            full_val_logits = scaled_val_logits + calibrator.prior_bias.to(
                device=scaled_val_logits.device,
                dtype=scaled_val_logits.dtype,
            ).view(1, -1)

            reversibility_candidates = sorted(
                {min(1.0, max(0.0, float(x))) for x in cfg.reversibility_candidates} | {0.0, 1.0}
            )
            best_reversibility = 0.0
            best_loss = base_val_loss
            best_acc = base_val_acc
            best_macro_acc = base_val_macro_acc
            best_score = base_score
            reversibility_scores = []
            for reversibility in reversibility_candidates:
                reversible_logits = base_val_logits + reversibility * (full_val_logits - base_val_logits)
                cand_loss, cand_acc, cand_macro_acc, cand_score = measure_logits(reversible_logits, val_labels)
                reversibility_scores.append((reversibility, cand_score))
                improves_score = cand_score > best_score
                preserves_score_and_improves_loss = (
                    cand_score >= best_score - 1e-12
                    and cand_loss < best_loss - cfg.min_val_loss_gain
                    and reversibility > 0.0
                )
                if improves_score or preserves_score_and_improves_loss:
                    best_reversibility = reversibility
                    best_loss = cand_loss
                    best_acc = cand_acc
                    best_macro_acc = cand_macro_acc
                    best_score = cand_score

            if best_reversibility > 0.0 and best_score < base_score + cfg.min_val_gain:
                loss_gain = base_val_loss - best_loss
                if loss_gain < cfg.min_val_loss_gain:
                    best_reversibility = 0.0
                    best_loss = base_val_loss
                    best_acc = base_val_acc
                    best_macro_acc = base_val_macro_acc
            if best_reversibility == 0.0:
                calibrator.set_prior_correction("none", 0.0)
            calibrator.set_reversibility_strength(best_reversibility)
            val_loss = best_loss
            val_acc = best_acc
            val_macro_acc = best_macro_acc

            score_text = ", ".join(f"{strength:.2f}:{score:.5f}" for strength, score in reversibility_scores)
            print(f"[rtpc_reversibility_candidates] {score_text}", flush=True)
        else:
            calibrator.set_prior_correction(cfg.default_prior_mode, cfg.default_prior_alpha)
            calibrator.set_reversibility_strength(1.0)
            val_loss = None
            val_acc = None
            val_macro_acc = None

        base_train_logits, train_labels = self._collect_logits(calibration_loader, apply_calibrator=False)
        calibrated_train_logits = calibrator(base_train_logits)
        train_loss, train_acc, train_macro_acc, _ = measure_logits(calibrated_train_logits, train_labels)

        return calibrator, CalibrationResult(
            train_loss=train_loss,
            train_acc=train_acc,
            train_macro_acc=train_macro_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            val_macro_acc=val_macro_acc,
            prior_mode=calibrator.prior_mode,
            prior_alpha=calibrator.prior_alpha,
            reversibility_strength=float(calibrator.reversibility_strength.detach().cpu().item()),
            baseline_val_acc=baseline_val_acc,
            baseline_val_macro_acc=baseline_val_macro_acc,
        )


# Backward-compatible aliases for older experiment scripts.
DecoupledTrainerConfig = StagedTrainerConfig
DecoupledLongTailTrainer = StagedLongTailTrainer
