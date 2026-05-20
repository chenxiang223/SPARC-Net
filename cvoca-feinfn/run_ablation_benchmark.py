from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

import main as exp_main
from losses import build_recommended_hsi_contrastive_loss, build_recommended_hsi_imbalance_loss
from models import SPARCNet
from trainers import (
    DecoupledLongTailTrainer,
    DecoupledTrainerConfig,
    PostHocCalibrationConfig,
    TrainingStageConfig,
)


PROJECT_DIR = Path(__file__).resolve().parent


DATASET_PRESETS: Dict[str, Dict[str, Any]] = {
    "IndianPines": {
        "name": "IndianPines",
        "cube_path": str(PROJECT_DIR / "datasets" / "IndianPines" / "Indian_pines_corrected.npy"),
        "gt_path": str(PROJECT_DIR / "datasets" / "IndianPines" / "Indian_pines_gt.npy"),
        "cube_key": None,
        "gt_key": None,
        "input_layout": "HWC",
    },
    "PaviaU": {
        "name": "PaviaU",
        "cube_path": str(PROJECT_DIR / "datasets" / "PaviaU" / "PaviaU.mat"),
        "gt_path": str(PROJECT_DIR / "datasets" / "PaviaU" / "PaviaU_gt.mat"),
        "cube_key": "paviaU",
        "gt_key": "paviaU_gt",
        "input_layout": "HWC",
    },
    "Salinas": {
        "name": "Salinas",
        "cube_path": str(PROJECT_DIR / "datasets" / "Salinas" / "Salinas_corrected.mat"),
        "gt_path": str(PROJECT_DIR / "datasets" / "Salinas" / "Salinas_gt.mat"),
        "cube_key": "salinas_corrected",
        "gt_key": "salinas_gt",
        "input_layout": "HWC",
    },
    "Houston2018": {
        "name": "Houston2018",
        "cube_path": str(PROJECT_DIR / "datasets" / "HOU2018" / "houston2018hsi.npy"),
        "gt_path": str(PROJECT_DIR / "datasets" / "HOU2018" / "houston2018_gt.npy"),
        "cube_key": None,
        "gt_key": None,
        "input_layout": "HWC",
    },
    "Houston2013": {
        "name": "Houston2013",
        "cube_path": str(PROJECT_DIR / "datasets" / "Houston13" / "Houston_recovered.mat"),
        "gt_path": str(PROJECT_DIR / "datasets" / "Houston13" / "Houston_recovered_gt.mat"),
        "cube_key": "Houston",
        "gt_key": "Houston_gt",
        "input_layout": "HWC",
    },
    "Botswana": {
        "name": "Botswana",
        "cube_path": str(PROJECT_DIR / "datasets" / "Botswana" / "Botswana.npy"),
        "gt_path": str(PROJECT_DIR / "datasets" / "Botswana" / "Botswana_gt.npy"),
        "cube_key": None,
        "gt_key": None,
        "input_layout": "HWC",
    },
    "KSC": {
        "name": "KSC",
        "cube_path": str(PROJECT_DIR / "datasets" / "KSC" / "KSC.mat"),
        "gt_path": str(PROJECT_DIR / "datasets" / "KSC" / "KSC_gt.mat"),
        "cube_key": "KSC",
        "gt_key": "KSC_gt",
        "input_layout": "HWC",
    },
}


VARIANT_PRESETS: Dict[str, Dict[str, Any]] = {
    "full": {
        "label": "Full",
        "description": "Full SPARC-Net after removing the negative APTA branch.",
        "switches": {
            "use_phase_amplitude_fusion_backbone": True,
            "use_dynamic_prototype_head": True,
            "use_decoupled_training_engine": True,
            "use_adaptive_posthoc_calibration": True,
        },
    },
    "wo_apcr": {
        "label": "w/o APCR",
        "description": "Disable amplitude-phase channel recalibration inside ACSE.",
        "switches": {
            "use_phase_amplitude_fusion_backbone": True,
            "use_dynamic_prototype_head": True,
            "use_decoupled_training_engine": True,
            "use_adaptive_posthoc_calibration": True,
            "backbone_detail": {"use_complex_attention": False},
        },
    },
    "with_apta": {
        "label": "w/ APTA",
        "description": "Legacy comparison that re-enables the removed amplitude-phase token adapter.",
        "switches": {
            "use_phase_amplitude_fusion_backbone": True,
            "use_dynamic_prototype_head": True,
            "use_decoupled_training_engine": True,
            "use_adaptive_posthoc_calibration": True,
            "backbone_detail": {"use_apta": True},
        },
    },
    "wo_sffc": {
        "label": "w/o SFFC",
        "description": "Disable the spatial-frequency fusion core by removing spatial, frequency, and TSFI paths.",
        "switches": {
            "use_phase_amplitude_fusion_backbone": True,
            "use_dynamic_prototype_head": True,
            "use_decoupled_training_engine": True,
            "use_adaptive_posthoc_calibration": True,
            "backbone_detail": {
                "use_spatial_branch": False,
                "use_frequency_branch": False,
                "use_tsfi": False,
            },
        },
    },
    "wo_tsfi": {
        "label": "w/o TSFI",
        "description": "Disable token-guided spatial-frequency interaction while keeping the two branches.",
        "switches": {
            "use_phase_amplitude_fusion_backbone": True,
            "use_dynamic_prototype_head": True,
            "use_decoupled_training_engine": True,
            "use_adaptive_posthoc_calibration": True,
            "backbone_detail": {"use_tsfi": False},
        },
    },
    "wo_refinement_spectral_fidelity": {
        "label": "w/o Refinement and Spectral Fidelity",
        "description": "Disable refinement and spectral-fidelity modules after SFFC.",
        "switches": {
            "use_phase_amplitude_fusion_backbone": True,
            "use_dynamic_prototype_head": True,
            "use_decoupled_training_engine": True,
            "use_adaptive_posthoc_calibration": True,
            "backbone_detail": {
                "use_transformer": False,
                "use_multi_scale": False,
                "use_sff_gate": False,
                "use_spectral_bypass": False,
            },
        },
    },
    "wo_innovation1": {
        "label": "w/o SPAP Backbone",
        "description": "Replace the SPARC-Net SPAP backbone with the baseline backbone.",
        "switches": {
            "use_phase_amplitude_fusion_backbone": False,
            "use_dynamic_prototype_head": True,
            "use_decoupled_training_engine": True,
            "use_adaptive_posthoc_calibration": True,
        },
    },
    "wo_innovation2": {
        "label": "w.o Innovation2",
        "description": "Replace the momentum dual-prototype relation head with the baseline cosine head.",
        "switches": {
            "use_phase_amplitude_fusion_backbone": True,
            "use_dynamic_prototype_head": False,
            "use_decoupled_training_engine": True,
            "use_adaptive_posthoc_calibration": True,
        },
    },
    "wo_innovation3": {
        "label": "w.o Innovation3",
        "description": "Disable the decoupled two-stage training engine and train in a single stage.",
        "switches": {
            "use_phase_amplitude_fusion_backbone": True,
            "use_dynamic_prototype_head": True,
            "use_decoupled_training_engine": False,
            "use_adaptive_posthoc_calibration": True,
        },
    },
    "wo_innovation4": {
        "label": "w.o Innovation4",
        "description": "Disable the adaptive post-hoc calibration module.",
        "switches": {
            "use_phase_amplitude_fusion_backbone": True,
            "use_dynamic_prototype_head": True,
            "use_decoupled_training_engine": True,
            "use_adaptive_posthoc_calibration": False,
        },
    },
    "wo_innovation2_3": {
        "label": "w.o Innovation2&3",
        "description": "Disable both the dynamic prototype head and the decoupled training strategy.",
        "switches": {
            "use_phase_amplitude_fusion_backbone": True,
            "use_dynamic_prototype_head": False,
            "use_decoupled_training_engine": False,
            "use_adaptive_posthoc_calibration": True,
        },
    },
    "baseline": {
        "label": "Baseline",
        "description": "Disable all four innovations.",
        "switches": {
            "use_phase_amplitude_fusion_backbone": False,
            "use_dynamic_prototype_head": False,
            "use_decoupled_training_engine": False,
            "use_adaptive_posthoc_calibration": False,
        },
    },
}

DEFAULT_VARIANT_KEYS = [key for key in VARIANT_PRESETS if key != "with_apta"]


SUMMARY_METRIC_KEYS = ("oa", "aa", "kappa", "macro_f1", "head_aa", "medium_aa", "tail_aa")


TRAIN_COUNT_PRESETS: Dict[str, Dict[str, Any]] = {
    "houston2018_table": {
        "dataset": "Houston2018",
        "train_counts_by_original_label": {
            1: 29,
            2: 98,
            3: 2,
            4: 41,
            5: 15,
            6: 14,
            7: 2,
            8: 119,
            9: 671,
            10: 138,
            11: 102,
            12: 5,
            13: 139,
            14: 30,
            15: 21,
            16: 35,
            17: 2,
            18: 20,
            19: 16,
            20: 20,
        },
        "val_ratio": 0.0,
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generic ablation benchmark runner for the current HSI repository. "
            "Supports built-in datasets and custom dataset paths."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="IndianPines",
        help=(
            "Built-in dataset name. Supported presets: "
            + ", ".join(sorted(DATASET_PRESETS))
            + ". Use 'custom' together with --cube-path/--gt-path for a custom dataset."
        ),
    )
    parser.add_argument("--cube-path", type=str, default=None, help="Custom cube path when --dataset=custom.")
    parser.add_argument("--gt-path", type=str, default=None, help="Custom ground-truth path when --dataset=custom.")
    parser.add_argument("--cube-key", type=str, default=None, help="MAT/HDF5 key for the custom cube.")
    parser.add_argument("--gt-key", type=str, default=None, help="MAT/HDF5 key for the custom ground truth.")
    parser.add_argument("--input-layout", type=str, default="HWC", help="Input array layout, usually HWC.")
    parser.add_argument("--train-mode", choices=("ratio", "fixed_per_class", "custom_per_class"), default="ratio")
    parser.add_argument("--train-ratio", type=float, default=0.05)
    parser.add_argument("--train-samples-per-class", type=int, default=10)
    parser.add_argument(
        "--train-counts-preset",
        type=str,
        choices=sorted(TRAIN_COUNT_PRESETS),
        default=None,
        help="Named preset for custom_per_class splits.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Optional val ratio override. Use 0 to disable the validation split.",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANT_PRESETS),
        default=DEFAULT_VARIANT_KEYS,
        help=(
            "Ablation variants to run. Default uses the paper variants after APTA removal; "
            "pass 'with_apta' explicitly for the legacy negative-module comparison."
        ),
    )
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index. Use -1 to let the trainer auto-select.")
    parser.add_argument("--stage1-epochs", type=int, default=None)
    parser.add_argument("--stage2-epochs", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=0, help="Epoch print frequency. 0 disables trainer logging.")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose trainer output.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing results.json if present.")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def resolve_dataset_config(args: argparse.Namespace) -> Dict[str, Any]:
    if args.dataset == "custom":
        if not args.cube_path or not args.gt_path:
            raise ValueError("When --dataset=custom, both --cube-path and --gt-path are required.")
        return {
            "name": "custom",
            "cube_path": args.cube_path,
            "gt_path": args.gt_path,
            "cube_key": args.cube_key,
            "gt_key": args.gt_key,
            "input_layout": args.input_layout,
        }
    if args.dataset not in DATASET_PRESETS:
        raise ValueError(
            f"Unknown dataset '{args.dataset}'. Supported presets: {', '.join(sorted(DATASET_PRESETS))}, or use custom."
        )
    return copy.deepcopy(DATASET_PRESETS[args.dataset])


def build_default_ablation_block() -> Dict[str, Any]:
    return {
        **exp_main.DEFAULT_INNOVATION_SWITCHES,
        "backbone_detail": exp_main.DEFAULT_BACKBONE_DETAIL_SWITCHES.copy(),
        "head_detail": exp_main.DEFAULT_HEAD_DETAIL_SWITCHES.copy(),
        "training_detail": exp_main.DEFAULT_TRAINING_DETAIL_SWITCHES.copy(),
    }


def resolve_train_count_preset(preset_name: Optional[str], dataset_name: str) -> Optional[Dict[str, Any]]:
    if preset_name is None:
        return None
    if preset_name not in TRAIN_COUNT_PRESETS:
        raise ValueError(
            f"Unknown train-count preset '{preset_name}'. Supported presets: {', '.join(sorted(TRAIN_COUNT_PRESETS))}."
        )
    preset = copy.deepcopy(TRAIN_COUNT_PRESETS[preset_name])
    expected_dataset = preset.get("dataset")
    if expected_dataset is not None and expected_dataset != dataset_name:
        raise ValueError(
            f"Train-count preset '{preset_name}' is defined for dataset '{expected_dataset}', "
            f"but the current dataset is '{dataset_name}'."
        )
    return preset


def build_config(
    *,
    dataset_cfg: Dict[str, Any],
    train_mode: str,
    train_ratio: float,
    train_samples_per_class: int,
    train_count_preset: Optional[str],
    val_ratio: Optional[float],
    seed: int,
    gpu: Optional[int],
    variant_key: str,
    stage1_epochs: Optional[int],
    stage2_epochs: Optional[int],
    amp: bool,
    print_every: int,
    verbose: bool,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(exp_main.CONFIG)
    preset_cfg = resolve_train_count_preset(train_count_preset, dataset_cfg["name"])

    cfg["dataset"]["name"] = dataset_cfg["name"]
    cfg["dataset"]["cube_path"] = dataset_cfg["cube_path"]
    cfg["dataset"]["cube_key"] = dataset_cfg.get("cube_key")
    cfg["dataset"]["gt_path"] = dataset_cfg["gt_path"]
    cfg["dataset"]["gt_key"] = dataset_cfg.get("gt_key")
    cfg["dataset"]["input_layout"] = dataset_cfg.get("input_layout", "HWC")

    cfg["split"]["train_mode"] = train_mode
    cfg["split"]["train_samples_per_class_by_original_label"] = None
    if train_mode == "ratio":
        cfg["split"]["train_ratio"] = float(train_ratio)
    elif train_mode == "fixed_per_class":
        cfg["split"]["train_samples_per_class"] = int(train_samples_per_class)
    else:
        if preset_cfg is None:
            raise ValueError("custom_per_class mode requires --train-counts-preset.")
        cfg["split"]["train_samples_per_class_by_original_label"] = {
            int(key): int(value) for key, value in preset_cfg["train_counts_by_original_label"].items()
        }
    if val_ratio is not None:
        cfg["split"]["val_ratio"] = float(val_ratio)
    elif preset_cfg is not None and "val_ratio" in preset_cfg:
        cfg["split"]["val_ratio"] = float(preset_cfg["val_ratio"])
    cfg["split"]["seed"] = int(seed)

    cfg["runtime"]["seed"] = int(seed)
    cfg["runtime"]["device"] = None if gpu is None else f"cuda:{gpu}"

    cfg["trainer"]["amp"] = bool(amp)
    cfg["trainer"]["print_every"] = int(print_every)
    cfg["trainer"]["verbose"] = bool(verbose)
    if stage1_epochs is not None:
        cfg["trainer"]["stage1_epochs"] = int(stage1_epochs)
    if stage2_epochs is not None:
        cfg["trainer"]["stage2_epochs"] = int(stage2_epochs)

    ablation = build_default_ablation_block()
    ablation.update(VARIANT_PRESETS[variant_key]["switches"])
    cfg["dataset"]["ablation"] = ablation
    return cfg


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def run_single_experiment(cfg: Dict[str, Any]) -> Dict[str, Any]:
    exp_main.configure_console_output()
    exp_main.set_global_seed(int(cfg["runtime"]["seed"]))

    ablation = exp_main.get_ablation_switches(cfg)
    effective_stage_plan = exp_main.build_effective_stage_plan(cfg, ablation)
    effective_model_cfg = exp_main.build_effective_model_config(cfg, ablation)

    bundle = exp_main.build_configured_dataloaders(cfg)
    class_name_mapping = exp_main.resolve_class_name_mapping(
        cfg["dataset"], bundle.inverse_label_mapping, bundle.num_classes
    )

    model = SPARCNet(
        in_channels=bundle.num_bands,
        num_classes=bundle.num_classes,
        class_counts=bundle.split_result.train_class_counts,
        **effective_model_cfg,
    )

    classification_loss, _ = build_recommended_hsi_imbalance_loss(
        class_counts=bundle.split_result.train_class_counts,
        total_epochs=cfg["loss"]["total_epochs"],
        max_margin=cfg["loss"]["max_margin"],
        beta=cfg["loss"]["beta"],
        gamma=cfg["loss"]["gamma"],
        drw_ratio=cfg["loss"]["drw_ratio"],
        margin_mode=model.classifier_margin_mode,
    )

    contrastive_loss = None
    if ablation["use_contrastive_loss"]:
        contrastive_loss, _ = build_recommended_hsi_contrastive_loss(
            class_counts=bundle.split_result.train_class_counts,
            temperature=cfg["loss"]["contrastive_temperature"],
            beta=cfg["loss"]["contrastive_beta"],
        )

    calibration_cfg = None
    if effective_stage_plan["use_calibration"]:
        calibration_cfg = PostHocCalibrationConfig(
            epochs=cfg["calibration"]["epochs"],
            lr=cfg["calibration"]["lr"],
            weight_decay=cfg["calibration"]["weight_decay"],
            learn_class_scales=cfg["calibration"]["learn_class_scales"],
            learn_logit_mixer=cfg["calibration"]["learn_logit_mixer"],
            normalize_scales=cfg["calibration"]["normalize_scales"],
            max_log_scale=cfg["calibration"]["max_log_scale"],
            max_logit_mixer_residual=cfg["calibration"]["max_logit_mixer_residual"],
            train_loader_preference=cfg["calibration"]["train_loader_preference"],
            prior_modes=cfg["calibration"]["prior_modes"],
            prior_alpha_candidates=cfg["calibration"]["prior_alpha_candidates"],
            default_prior_mode=cfg["calibration"]["default_prior_mode"],
            default_prior_alpha=cfg["calibration"]["default_prior_alpha"],
            effective_num_beta=cfg["calibration"]["effective_num_beta"],
            monitor=cfg["calibration"]["monitor"],
            blend_candidates=cfg["calibration"]["blend_candidates"],
            min_val_gain=cfg["calibration"]["min_val_gain"],
            min_val_loss_gain=cfg["calibration"]["min_val_loss_gain"],
        )

    stage2_reset_classifier = (
        cfg["trainer"]["stage2_reset_classifier"] if effective_stage_plan["use_stage2"] else False
    )
    if ablation["use_prototype_branch"]:
        stage2_reset_classifier = False

    trainer = DecoupledLongTailTrainer(
        model,
        classification_loss,
        contrastive_loss=contrastive_loss,
        config=DecoupledTrainerConfig(
            stage1=TrainingStageConfig(
                name="stage1",
                epochs=effective_stage_plan["stage1_epochs"],
                lr=cfg["trainer"]["stage1_lr"],
                weight_decay=cfg["trainer"]["stage1_weight_decay"],
                freeze_backbone=False,
                reset_classifier=False,
                use_aux_loss=ablation["use_auxiliary_heads"],
                use_contrastive_loss=ablation["use_contrastive_loss"],
                use_balanced_loader=False,
                head_regularization_weight=(
                    cfg["trainer"]["stage1_head_regularization_weight"] if ablation["use_prototype_branch"] else 0.0
                ),
                head_output_mode="main",
                update_prototypes=False,
                guided_context=False,
            ),
            stage2=TrainingStageConfig(
                name="stage2",
                epochs=effective_stage_plan["stage2_epochs"],
                lr=cfg["trainer"]["stage2_lr"],
                weight_decay=cfg["trainer"]["stage2_weight_decay"],
                freeze_backbone=True,
                reset_classifier=stage2_reset_classifier,
                use_aux_loss=False,
                use_contrastive_loss=False,
                use_balanced_loader=(
                    cfg["trainer"]["stage2_use_balanced_loader"] if effective_stage_plan["use_stage2"] else False
                ),
                head_regularization_weight=(
                    cfg["trainer"]["stage2_head_regularization_weight"] if ablation["use_prototype_branch"] else 0.0
                ),
                head_output_mode="corrected" if ablation["use_prototype_branch"] else "main",
                update_prototypes=ablation["use_prototype_branch"],
                guided_context=ablation["use_prototype_branch"],
            ),
            aux_mid_weight=cfg["trainer"]["aux_mid_weight"],
            aux_early_weight=cfg["trainer"]["aux_early_weight"],
            contrastive_weight=cfg["trainer"]["contrastive_weight"],
            amp=cfg["trainer"]["amp"],
            stage2_tau_norm=cfg["trainer"]["stage2_tau_norm"] if effective_stage_plan["use_stage2"] else None,
            monitor=cfg["trainer"]["monitor"],
            stage2_start_from_best_stage1=cfg["trainer"]["stage2_start_from_best_stage1"],
            stage2_revert_if_no_val_gain=cfg["trainer"]["stage2_revert_if_no_val_gain"],
            stage2_min_val_gain=cfg["trainer"]["stage2_min_val_gain"],
            stage2_skip_if_stage1_val_at_least=cfg["trainer"]["stage2_skip_if_stage1_val_at_least"],
            prototype_correction_head_drop_tolerance=cfg["trainer"]["prototype_correction_head_drop_tolerance"],
            calibration=calibration_cfg,
            verbose=cfg["trainer"]["verbose"],
            print_every=cfg["trainer"]["print_every"],
        ),
        device=cfg["runtime"]["device"],
    )

    balanced_train_loader = exp_main.build_balanced_train_loader(bundle, cfg["loader"])
    effective_val_loader = None if len(bundle.val_dataset) == 0 else bundle.val_loader
    history = trainer.fit(
        bundle.train_loader,
        val_loader=effective_val_loader,
        # The benchmark reports the final selected model, so running the full
        # test split after every epoch only slows large scenes such as PaviaU.
        test_loader=None,
        balanced_train_loader=balanced_train_loader,
    )

    # Prefer the final frozen evaluation after post-hoc calibration and
    # validation-selected prototype correction. `history.best_test_eval`
    # is useful for diagnostics, but it is collected before the final
    # prototype-correction search and would otherwise hide Innovation2's
    # selected inference-time behavior.
    eval_output = trainer.test_metrics if trainer.test_metrics is not None else trainer.evaluate(bundle.test_loader)
    metrics = eval_output["classification_metrics"]
    summary = metrics.to_summary_dict()

    class_rows: List[Dict[str, Any]] = []
    for class_id in range(bundle.num_classes):
        display_id = int(bundle.inverse_label_mapping.get(class_id, class_id + 1))
        support = int(metrics.support[class_id].item())
        correct = int(metrics.confusion_matrix[class_id, class_id].item())
        class_rows.append(
            {
                "class_index": int(class_id),
                "display_id": display_id,
                "class_name": class_name_mapping.get(class_id, f"Class_{display_id}"),
                "support": support,
                "correct": correct,
                "acc": float(metrics.per_class_accuracy[class_id].item()),
                "precision": float(metrics.per_class_precision[class_id].item()),
                "recall": float(metrics.per_class_recall[class_id].item()),
                "f1": float(metrics.per_class_f1[class_id].item()),
            }
        )

    prototype_correction_scale = (
        model.get_prototype_correction_scale()
        if hasattr(model, "get_prototype_correction_scale")
        else None
    )

    result = {
        "seed": int(cfg["runtime"]["seed"]),
        "dataset": cfg["dataset"]["name"],
        "train_mode": cfg["split"]["train_mode"],
        "train_ratio": float(cfg["split"]["train_ratio"]) if cfg["split"]["train_mode"] == "ratio" else None,
        "train_samples_per_class": (
            int(cfg["split"]["train_samples_per_class"])
            if cfg["split"]["train_mode"] == "fixed_per_class"
            else None
        ),
        "train_samples_per_class_by_original_label": (
            {
                str(key): int(value)
                for key, value in cfg["split"]["train_samples_per_class_by_original_label"].items()
            }
            if cfg["split"]["train_mode"] == "custom_per_class"
            and cfg["split"]["train_samples_per_class_by_original_label"] is not None
            else None
        ),
        "val_ratio": float(cfg["split"]["val_ratio"]),
        "best_stage": history.best_stage,
        "best_epoch": None if history.best_epoch is None else int(history.best_epoch) + 1,
        "best_metric": _to_float(history.best_metric),
        "best_test_stage": history.best_test_stage,
        "best_test_epoch": None if history.best_test_epoch is None else int(history.best_test_epoch) + 1,
        "best_test_metric": _to_float(history.best_test_metric),
        "final_eval_stage": "final_calibrated_prototype_tuned",
        "prototype_correction_scale": (
            None if prototype_correction_scale is None else float(prototype_correction_scale)
        ),
        "classifier_name": type(model.classifier).__name__ if model.classifier is not None else None,
        "prototype_head_active": getattr(model, "prototype_head_active", None),
        "summary_metrics": {key: float(summary[key]) for key in summary},
        "per_class": class_rows,
        "calibration": None
        if trainer.calibration_result is None
        else {
            "prior_mode": trainer.calibration_result.prior_mode,
            "prior_alpha": float(trainer.calibration_result.prior_alpha),
            "blend_strength": float(trainer.calibration_result.blend_strength),
            "train_macro_acc": float(trainer.calibration_result.train_macro_acc),
            "baseline_val_macro_acc": None
            if trainer.calibration_result.baseline_val_macro_acc is None
            else float(trainer.calibration_result.baseline_val_macro_acc),
            "val_macro_acc": None
            if trainer.calibration_result.val_macro_acc is None
            else float(trainer.calibration_result.val_macro_acc),
        },
        "split_counts": {
            "train": int(len(bundle.train_dataset)),
            "val": int(len(bundle.val_dataset)),
            "test": int(len(bundle.test_dataset)),
        },
    }

    del trainer
    del model
    del classification_loss
    del contrastive_loss
    del balanced_train_loader
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def aggregate_values(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "std": 0.0, "var": 0.0, "max": 0.0, "min": 0.0}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "var": float(array.var(ddof=0)),
        "max": float(array.max()),
        "min": float(array.min()),
    }


def build_aggregated_results(store: Dict[str, Any]) -> Dict[str, Any]:
    aggregated: Dict[str, Any] = {}
    for variant_key, variant_info in store["variants"].items():
        completed_runs = [run for run in variant_info["runs"] if "summary_metrics" in run]
        if not completed_runs:
            continue

        metric_stats = {
            metric_key: aggregate_values(run["summary_metrics"].get(metric_key, 0.0) for run in completed_runs)
            for metric_key in SUMMARY_METRIC_KEYS
        }

        best_run = max(completed_runs, key=lambda item: item["summary_metrics"]["oa"])
        per_class_stats = []
        num_classes = len(best_run["per_class"])
        for class_index in range(num_classes):
            rows = [run["per_class"][class_index] for run in completed_runs]
            acc_stats = aggregate_values(row["acc"] for row in rows)
            per_class_stats.append(
                {
                    "class_index": rows[0]["class_index"],
                    "display_id": rows[0]["display_id"],
                    "class_name": rows[0]["class_name"],
                    "support_mean": float(np.mean([row["support"] for row in rows])),
                    "acc_mean": acc_stats["mean"],
                    "acc_std": acc_stats["std"],
                    "acc_var": acc_stats["var"],
                    "acc_max": acc_stats["max"],
                }
            )

        aggregated[variant_key] = {
            "label": variant_info["label"],
            "description": variant_info["description"],
            "completed_runs": len(completed_runs),
            "metric_stats": metric_stats,
            "best_run": best_run,
            "per_class_stats": per_class_stats,
        }
    return aggregated


def write_summary_csv(output_dir: Path, aggregated: Dict[str, Any]) -> None:
    path = output_dir / "summary_metrics.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "variant",
                "label",
                "runs",
                "metric",
                "mean",
                "std",
                "var",
                "max",
                "min",
                "best_seed_by_oa",
                "best_oa",
                "best_aa",
                "best_kappa",
                "best_macro_f1",
                "best_head_aa",
                "best_medium_aa",
                "best_tail_aa",
            ]
        )
        for variant_key, info in aggregated.items():
            best_run = info["best_run"]
            best_summary = best_run["summary_metrics"]
            for metric_key, stats in info["metric_stats"].items():
                writer.writerow(
                    [
                        variant_key,
                        info["label"],
                        info["completed_runs"],
                        metric_key,
                        stats["mean"],
                        stats["std"],
                        stats["var"],
                        stats["max"],
                        stats["min"],
                        best_run["seed"],
                        best_summary.get("oa", 0.0),
                        best_summary.get("aa", 0.0),
                        best_summary.get("kappa", 0.0),
                        best_summary.get("macro_f1", 0.0),
                        best_summary.get("head_aa", 0.0),
                        best_summary.get("medium_aa", 0.0),
                        best_summary.get("tail_aa", 0.0),
                    ]
                )


def write_per_class_csv(output_dir: Path, aggregated: Dict[str, Any]) -> None:
    path = output_dir / "per_class_metrics.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "variant",
                "label",
                "class_index",
                "display_id",
                "class_name",
                "support_mean",
                "acc_mean_percent",
                "acc_std_percent",
                "acc_var",
                "acc_max_percent",
            ]
        )
        for variant_key, info in aggregated.items():
            for row in info["per_class_stats"]:
                writer.writerow(
                    [
                        variant_key,
                        info["label"],
                        row["class_index"],
                        row["display_id"],
                        row["class_name"],
                        row["support_mean"],
                        row["acc_mean"] * 100.0,
                        row["acc_std"] * 100.0,
                        row["acc_var"],
                        row["acc_max"] * 100.0,
                    ]
                )


def write_runs_csv(output_dir: Path, store: Dict[str, Any]) -> None:
    path = output_dir / "individual_runs.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "variant",
                "label",
                "seed",
                "best_stage",
                "best_epoch",
                "best_test_stage",
                "best_test_epoch",
                "oa",
                "aa",
                "kappa",
                "macro_f1",
                "head_aa",
                "medium_aa",
                "tail_aa",
                "prototype_correction_scale",
                "prior_mode",
                "prior_alpha",
            ]
        )
        for variant_key, info in store["variants"].items():
            for run in info["runs"]:
                if "summary_metrics" not in run:
                    continue
                summary = run["summary_metrics"]
                calibration = run.get("calibration") or {}
                writer.writerow(
                    [
                        variant_key,
                        info["label"],
                        run["seed"],
                        run["best_stage"],
                        run["best_epoch"],
                        run["best_test_stage"],
                        run["best_test_epoch"],
                        summary.get("oa", 0.0),
                        summary.get("aa", 0.0),
                        summary.get("kappa", 0.0),
                        summary.get("macro_f1", 0.0),
                        summary.get("head_aa", 0.0),
                        summary.get("medium_aa", 0.0),
                        summary.get("tail_aa", 0.0),
                        run.get("prototype_correction_scale"),
                        calibration.get("prior_mode"),
                        calibration.get("prior_alpha"),
                    ]
                )


def write_markdown_report(output_dir: Path, store: Dict[str, Any], aggregated: Dict[str, Any]) -> None:
    title_dataset = store["dataset"]
    if store["train_mode"] == "ratio":
        split_desc = f"ratio={store['train_ratio']:.4f}"
    elif store["train_mode"] == "fixed_per_class":
        split_desc = f"fixed_per_class={store['train_samples_per_class']}"
    else:
        split_desc = f"custom_per_class preset={store['train_count_preset']}"

    path = output_dir / "report.md"
    lines: List[str] = []
    lines.append(f"# {title_dataset} Ablation Benchmark")
    lines.append("")
    lines.append(f"- Dataset: {title_dataset}")
    lines.append(f"- Train mode: {store['train_mode']}")
    lines.append(f"- Split setting: {split_desc}")
    lines.append(f"- Val ratio: {store['val_ratio']:.4f}")
    lines.append(f"- Repeats: {store['repeats']}")
    lines.append(f"- Seeds: {', '.join(str(seed) for seed in store['seeds'])}")
    lines.append("")
    for variant_key, info in aggregated.items():
        lines.append(f"## {info['label']}")
        lines.append("")
        lines.append(info["description"])
        lines.append("")
        lines.append("| Metric | Mean | Std | Var | Max |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for metric_key in SUMMARY_METRIC_KEYS:
            stats = info["metric_stats"][metric_key]
            lines.append(
                f"| {metric_key} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['var']:.6f} | {stats['max']:.4f} |"
            )
        best_run = info["best_run"]
        lines.append("")
        lines.append(
            f"Best OA run: seed={best_run['seed']}, OA={best_run['summary_metrics']['oa']:.4f}, "
            f"AA={best_run['summary_metrics']['aa']:.4f}, Kappa={best_run['summary_metrics']['kappa']:.4f}, "
            f"Macro-F1={best_run['summary_metrics']['macro_f1']:.4f}"
        )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_store(output_dir: Path, store: Dict[str, Any]) -> None:
    (output_dir / "results.json").write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")


def load_store(output_dir: Path) -> Optional[Dict[str, Any]]:
    path = output_dir / "results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def find_existing_run(runs: List[Dict[str, Any]], seed: int) -> Optional[Dict[str, Any]]:
    for run in runs:
        if int(run.get("seed", -1)) == int(seed) and "summary_metrics" in run:
            return run
    return None


def build_output_dir(args: argparse.Namespace, dataset_name: str) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    if args.train_mode == "ratio":
        split_tag = f"ratio_{str(args.train_ratio).replace('.', 'p')}"
    elif args.train_mode == "fixed_per_class":
        split_tag = f"spc_{args.train_samples_per_class}"
    else:
        split_tag = f"custom_{args.train_counts_preset}"
    return PROJECT_DIR / "logs" / f"{dataset_name}_ablation_{split_tag}{tag}_{timestamp}"


def main() -> None:
    args = parse_args()
    dataset_cfg = resolve_dataset_config(args)
    preset_cfg = resolve_train_count_preset(args.train_counts_preset, dataset_cfg["name"])
    gpu = None if args.gpu < 0 else int(args.gpu)
    seeds = [args.seed_start + offset for offset in range(args.repeats)]
    output_dir = build_output_dir(args, dataset_cfg["name"])
    output_dir.mkdir(parents=True, exist_ok=True)

    store = None
    if args.resume:
        store = load_store(output_dir)

    if store is None:
        store = {
            "dataset": dataset_cfg["name"],
            "train_mode": args.train_mode,
            "train_ratio": float(args.train_ratio),
            "train_samples_per_class": int(args.train_samples_per_class),
            "train_count_preset": args.train_counts_preset,
            "train_samples_per_class_by_original_label": (
                {str(key): int(value) for key, value in preset_cfg["train_counts_by_original_label"].items()}
                if preset_cfg is not None
                else None
            ),
            "val_ratio": (
                float(args.val_ratio)
                if args.val_ratio is not None
                else float(preset_cfg["val_ratio"])
                if preset_cfg is not None and "val_ratio" in preset_cfg
                else float(exp_main.CONFIG["split"]["val_ratio"])
            ),
            "repeats": int(args.repeats),
            "seeds": seeds,
            "gpu": gpu,
            "tag": args.tag,
            "variants": {
                variant_key: {
                    "label": VARIANT_PRESETS[variant_key]["label"],
                    "description": VARIANT_PRESETS[variant_key]["description"],
                    "runs": [],
                }
                for variant_key in args.variants
            },
        }
        save_store(output_dir, store)

    print(f"[Output Dir] {output_dir}")
    print(f"[Dataset] {dataset_cfg['name']}")
    print(f"[Train Mode] {args.train_mode}")
    if args.train_mode == "ratio":
        print(f"[Train Ratio] {args.train_ratio}")
    elif args.train_mode == "fixed_per_class":
        print(f"[Train Samples/Class] {args.train_samples_per_class}")
    else:
        print(f"[Train Count Preset] {args.train_counts_preset}")
    if args.val_ratio is not None:
        print(f"[Val Ratio Override] {args.val_ratio}")
    elif preset_cfg is not None and "val_ratio" in preset_cfg:
        print(f"[Val Ratio Preset] {preset_cfg['val_ratio']}")
    print(f"[GPU] {'auto' if gpu is None else gpu}")
    print(f"[Variants] {', '.join(args.variants)}")
    print(f"[Seeds] {', '.join(str(seed) for seed in seeds)}")
    print(flush=True)

    for variant_key in args.variants:
        label = VARIANT_PRESETS[variant_key]["label"]
        for seed in seeds:
            existing = find_existing_run(store["variants"][variant_key]["runs"], seed)
            if existing is not None:
                print(
                    f"[Skip] {label} | seed={seed} already completed | OA={existing['summary_metrics']['oa']:.4f}",
                    flush=True,
                )
                continue

            print(f"[Run] {label} | seed={seed} | start", flush=True)
            cfg = build_config(
                dataset_cfg=dataset_cfg,
                train_mode=args.train_mode,
                train_ratio=args.train_ratio,
                train_samples_per_class=args.train_samples_per_class,
                train_count_preset=args.train_counts_preset,
                val_ratio=args.val_ratio,
                seed=seed,
                gpu=gpu,
                variant_key=variant_key,
                stage1_epochs=args.stage1_epochs,
                stage2_epochs=args.stage2_epochs,
                amp=args.amp,
                print_every=args.print_every,
                verbose=args.verbose,
            )
            try:
                run_result = run_single_experiment(cfg)
                store["variants"][variant_key]["runs"].append(run_result)
                save_store(output_dir, store)
                summary = run_result["summary_metrics"]
                print(
                    f"[Done] {label} | seed={seed} | OA={summary['oa']:.4f} | AA={summary['aa']:.4f} | "
                    f"Kappa={summary['kappa']:.4f} | Macro-F1={summary['macro_f1']:.4f}",
                    flush=True,
                )
            except Exception as exc:  # pragma: no cover - long-running experiment safeguard
                failed_run = {"seed": seed, "error": repr(exc)}
                store["variants"][variant_key]["runs"].append(failed_run)
                save_store(output_dir, store)
                print(f"[Failed] {label} | seed={seed} | {exc!r}", flush=True)
                raise

    aggregated = build_aggregated_results(store)
    save_store(output_dir, {**store, "aggregated": aggregated})
    write_summary_csv(output_dir, aggregated)
    write_per_class_csv(output_dir, aggregated)
    write_runs_csv(output_dir, store)
    write_markdown_report(output_dir, store, aggregated)

    print("")
    print("[Aggregation Complete]")
    for variant_key in args.variants:
        if variant_key not in aggregated:
            continue
        info = aggregated[variant_key]
        stats = info["metric_stats"]["oa"]
        best_run = info["best_run"]
        print(
            f"{info['label']}: OA mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
            f"var={stats['var']:.6f}, max={stats['max']:.4f}, best_seed={best_run['seed']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
