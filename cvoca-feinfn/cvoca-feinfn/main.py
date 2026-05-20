from __future__ import annotations

from pathlib import Path
import random
import warnings
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import build_hsi_dataloaders, build_weighted_sampler
from losses import build_recommended_hsi_contrastive_loss, build_recommended_hsi_imbalance_loss
from metrics import format_metric_summary, format_per_class_accuracy_table
from models import SPARCNet
from trainers import (
    StagedLongTailTrainer,
    StagedTrainerConfig,
    RTPCConfig,
    TrainingStageConfig,
)


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# GPU selection
# -----------------------------------------------------------------------------
# Set to a CUDA device index such as 0 / 1 / 2 to force training on that GPU.
# Set to None to use the trainer's automatic device selection.
SELECTED_GPU_INDEX: Optional[int] = 0
RUNTIME_DEVICE = None if SELECTED_GPU_INDEX is None else f"cuda:{SELECTED_GPU_INDEX}"

# DATASET_DIR = PROJECT_DIR / "datasets" / "IndianPines"
DATASET_DIR = PROJECT_DIR / "datasets" / "PaviaU"
# DATASET_DIR = PROJECT_DIR / "datasets" / "Salinas"
# DATASET_DIR = PROJECT_DIR / "datasets" / "HOU2018"
# DATASET_DIR = PROJECT_DIR / "datasets" / "Houston13"
# DATASET_DIR = PROJECT_DIR / "datasets" / "Botswana"
# DATASET_DIR = PROJECT_DIR / "datasets" / "KSC"

CONFIG: Dict[str, Any] = {
    "dataset": {
        # "name": "IndianPines",
        # "cube_path": str(DATASET_DIR / "Indian_pines_corrected.mat"),
        # "cube_key": "indian_pines_corrected",
        # "gt_path": str(DATASET_DIR / "Indian_pines_gt.mat"),
        # "gt_key": "indian_pines_gt",
        # "input_layout": "HWC",
        # "class_names": None,

        "name": "PaviaU",
        "cube_path": str(DATASET_DIR / "PaviaU.mat"),
        "cube_key": "paviaU",
        "gt_path": str(DATASET_DIR / "PaviaU_gt.mat"),
        "gt_key": "paviaU_gt",
        "input_layout": "HWC",

        # "name": "Salinas",
        # "cube_path": str(DATASET_DIR / "Salinas_corrected.mat"),
        # "cube_key": "salinas_corrected",
        # "gt_path": str(DATASET_DIR / "Salinas_gt.mat"),
        # "gt_key": "salinas_gt",
        # "input_layout": "HWC",

        # "name": "Houston2018",
        # "cube_path": str(DATASET_DIR / "houston2018hsi.npy"),
        # "cube_key": None,
        # "gt_path": str(DATASET_DIR / "houston2018_gt.npy"),
        # "gt_key": None,
        # "input_layout": "HWC",

        # "name": "Houston2013",
        # "cube_path": str(DATASET_DIR / "Houston_recovered.mat"),
        # "cube_key": "Houston",
        # "gt_path": str(DATASET_DIR / "Houston_recovered_gt.mat"),
        # "gt_key": "Houston_gt",
        # "input_layout": "HWC",

        # "name": "Botswana",
        # "cube_path": str(DATASET_DIR / "Botswana.mat"),
        # "cube_key": "Botswana",
        # "gt_path": str(DATASET_DIR / "Botswana_gt.mat"),
        # "gt_key": "Botswana_gt",
        # "input_layout": "HWC",

        # "name": "KSC",
        # "cube_path": str(DATASET_DIR / "KSC.mat"),
        # "cube_key": "KSC",
        # "gt_path": str(DATASET_DIR / "KSC_gt.mat"),
        # "gt_key": "KSC_gt",
        # "input_layout": "HWC",

        # Placeholder only. The effective ablation configuration is reassigned
        # below in the clean ablation switch definition block.
        "ablation": {},
    },
    "split": {
        "train_mode": "ratio",  # 训练集划分方式：ratio 按比例划分，fixed_per_class 按每类固定样本数划分。
        "val_mode": "ratio",  # 验证集划分方式：ratio 按比例划分，fixed 按每类固定样本数划分。
        "train_ratio": 0.01,  # 训练集比例；仅在 train_mode='ratio' 时生效。
        "val_ratio": 0.1,  # 验证集比例；仅在 val_mode='ratio' 时生效。
        "train_samples_per_class": 10,  # 每类训练样本数；仅在 train_mode='fixed' 时生效。
        "train_samples_per_class_by_original_label": None,  # 按原始类别ID指定训练样本数；用于复现论文给出的逐类训练数量。
        "val_samples_per_class": 2,  # 每类验证样本数；仅在 val_mode='fixed' 时生效。
        "background_label": 0,  # 背景类别标签值，划分与评估时会跳过该标签。
        "min_train_per_class": 2,  # 每类最少保留的训练样本数，防止极小类别被分空。
        "min_val_per_class": 2,  # 每类最少保留的验证样本数，防止极小类别没有验证样本。
        "split_mode": "stratified_random",  # 划分策略：stratified_random 为分层随机；部分模式下也可按空间划分。
        "spatial_axis": "row",  # 空间划分时的切分方向：row 表示按行，col 表示按列。
        "seed": 0,  # 数据划分随机种子，保证不同实验可复现。
    },
    "preprocess": {
        "patch_size": 15,
        "clip_percentiles": None,
        "drop_bands": None,
        "keep_bands": None,
        "auto_detect_bad_bands": True,
        "auto_bad_band_mode": "interpolate",
        "padding_mode": "reflect",
    },
    "loader": {
        "batch_size": 64,
        "num_workers": 0,
        "pin_memory": True,
        "use_weighted_sampler": True,
        "sampler_weight_mode": "sqrt_inv",
        "sampler_beta": 0.999,
        "use_hsi_augmentation": True,
    },
    "model": {
        "base_channels": 64,
        "adapter_channels": 64,
        "token_dim": 96,
        "patch_size": 3,
        "patch_stride": 1,
        "split_mode": "mag_phase",
        "analytic_init": "hybrid",
        "use_transformer": True,
        "transformer_heads": 4,
        "transformer_depth": 1,
        "transformer_max_tokens": 256,
        "classifier_type": "long_tail",
        "head_embed_dim": None,
        "head_contrast_dim": 128,
        "use_auxiliary_heads": False,
        "alignment_scale_limit": 0.35,
        "alignment_bias_limit": 0.20,
        "freeze_source_classifier_in_stage2": True,
    },
    "loss": {
        "total_epochs": 180,
        "max_margin": 0.3,
        "beta": 0.999,
        "gamma": 0.0,
        "drw_ratio": 0.5,
        "contrastive_temperature": 0.07,
        "contrastive_beta": 0.999,
    },
    "trainer": {
        "stage1_epochs": 180,
        "stage2_epochs": 20,
        "single_stage_epochs": 150,
        "stage1_lr": 3e-4,
        "stage2_lr": 5e-4,
        "stage1_weight_decay": 1e-4,
        "stage2_weight_decay": 0.0,
        "stage2_reset_classifier": False,
        "stage2_use_balanced_loader": True,
        "aux_mid_weight": 0.3,
        "aux_early_weight": 0.15,
        "contrastive_weight": 0.1,
        "stage1_head_regularization_weight": 0.0,
        "stage2_head_regularization_weight": 0.08,
        "amp": False,
        "stage2_tau_norm": None,
        "monitor": "acc",
        "stage2_start_from_best_stage1": True,
        "stage2_revert_if_no_val_gain": True,
        "stage2_min_val_gain": 0.001,
        "stage2_skip_if_stage1_val_at_least": 0.995,
        "prototype_correction_head_drop_tolerance": 0.0015,
        "verbose": True,
        "print_every": 1,
    },
    "calibration": {
        "enabled": True,
        "epochs": 20,
        "lr": 5e-3,
        "weight_decay": 1e-3,
        "learn_class_scales": True,
        "learn_logit_mixer": True,
        "normalize_scales": True,
        "max_log_scale": 0.25,
        "max_logit_mixer_residual": 0.08,
        "train_loader_preference": "val",
        "prior_modes": ("none", "frequency", "effective_num"),
        "prior_alpha_candidates": (0.0, 0.1, 0.2, 0.35, 0.5),
        "default_prior_mode": "none",
        "default_prior_alpha": 0.0,
        "effective_num_beta": 0.999,
        "monitor": "acc",
        "blend_candidates": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "min_val_gain": 0.0002,
        "min_val_loss_gain": 1e-4,
    },
    "runtime": {
        "device": RUNTIME_DEVICE,  # 运行设备；默认按上方 SELECTED_GPU_INDEX 映射到 cuda:N。
        "seed": 0,
    },
}

# ---------------------------------------------------------------------------
# Clean ablation switch definition
# ---------------------------------------------------------------------------
# The original inline comments above may become hard to read in terminals with
# non-UTF8 console encoding. Re-assign the ablation block here so the runtime
# behavior and the user-facing comments stay explicit and easy to audit.
CONFIG["dataset"]["ablation"] = {
    # 一级总消融开关：
    # True  = 启用该创新点对应的完整创新模块。
    # False = 关闭该创新点，并切换到非创新的 baseline 模块/流程。
    "use_spap_backbone": True,  # 创新点1总开关：复值感知空频融合主干。关闭后改用普通残差CNN backbone。
    "use_mrpc_head": True,  # 创新点2总开关：动量双原型关系分类头。关闭后改用普通 cosine baseline head。
    "use_staged_training_engine": True,  # 创新点3总开关：两阶段解耦训练引擎。关闭后改为单阶段训练。
    "use_rtpc_calibration": False,  # 创新点4总开关：事后校准模块。关闭后不进行 stage3 calibration。

    # 二级子消融开关：
    # 这些开关仅在对应的一级总开关为 True 时生效，用于分析模块内部子机制。
    "backbone_detail": {
        "use_complex_attention": True,  # APCR: amplitude-phase channel recalibration.
        "use_apta": True,  # APTA: amplitude-phase token adapter.
        "use_spatial_branch": True,  # Spatial high-frequency branch.
        "use_frequency_branch": True,  # DAF Branch: dual-axis amplitude-phase frequency branch.
        "use_tsfi": True,  # TSFI: token-guided spatial-frequency interaction.
        "use_transformer": True,  # Global transformer recalibration.
        "use_multi_scale": True,  # Multi-scale context module.
        "use_sff_gate": True,  # Spectral fidelity fusion gate.
        "use_spectral_bypass": True,  # SAB: spectral anchor bypass.
    },
    "head_detail": {
        "use_prototype_branch": True,  # 原型关系分支：控制动量双原型/多原型匹配。
        "use_dynamic_gate": True,  # 动态关系门控：控制原型分支对主分类分支的轻量纠偏。
    },
    "training_detail": {
        "use_auxiliary_heads": False,  # 辅助监督头：控制deep supervision。
        "use_contrastive_loss": False,  # 对比学习损失：控制表征拉近/拉远约束。
    },
}


DATASET_CLASS_NAME_PRESETS: Dict[str, list[str]] = {
    "IndianPines": [
        "Alfalfa",
        "Corn_notill",
        "Corn_mintill",
        "Corn",
        "Grass_pasture",
        "Grass_trees",
        "Grass_pasture_mowed",
        "Hay_windrowed",
        "Oats",
        "Soybean_notill",
        "Soybean_mintill",
        "Soybean_clean",
        "Wheat",
        "Woods",
        "Buildings_Grass_Trees_Drives",
        "Stone_Steel_Towers",
    ],
    "PaviaU": [
        "Asphalt",
        "Meadows",
        "Gravel",
        "Trees",
        "Painted_metal_sheets",
        "Bare_Soil",
        "Bitumen",
        "Self_Blocking_Bricks",
        "Shadows",
    ],
    "Salinas": [
        "Brocoli_green_weeds_1",
        "Brocoli_green_weeds_2",
        "Fallow",
        "Fallow_rough_plow",
        "Fallow_smooth",
        "Stubble",
        "Celery",
        "Grapes_untrained",
        "Soil_vinyard_develop",
        "Corn_senesced_green_weeds",
        "Lettuce_romaine_4wk",
        "Lettuce_romaine_5wk",
        "Lettuce_romaine_6wk",
        "Lettuce_romaine_7wk",
        "Vinyard_untrained",
        "Vinyard_vertical_trellis",
    ],
    "Houston2013": [
        "Healthy_grass",
        "Stressed_grass",
        "Synthetic_grass",
        "Trees",
        "Soil",
        "Water",
        "Residential",
        "Commercial",
        "Road",
        "Highway",
        "Railway",
        "Parking_lot_1",
        "Parking_lot_2",
        "Tennis_court",
        "Running_track",
    ],
    "Houston13": [
        "Healthy_grass",
        "Stressed_grass",
        "Synthetic_grass",
        "Trees",
        "Soil",
        "Water",
        "Residential",
        "Commercial",
        "Road",
        "Highway",
        "Railway",
        "Parking_lot_1",
        "Parking_lot_2",
        "Tennis_court",
        "Running_track",
    ],
    "Houston2018": [
        "Healthy_grass",
        "Stressed_grass",
        "Synthetic_grass",
        "Evergreen_grass",
        "Deciduous_grass",
        "Soil",
        "Water",
        "Residential",
        "Commercial",
        "Road",
        "Sidewalk",
        "Crosswalk",
        "Major_Thoroughfares",
        "Highway",
        "Railway",
        "Paved_Parking_Lot",
        "Gravel_Parking_Lot",
        "Cars",
        "Trains",
        "Seats",
    ],
}


DEFAULT_INNOVATION_SWITCHES: Dict[str, bool] = {
    "use_spap_backbone": True,
    "use_mrpc_head": True,
    "use_staged_training_engine": True,
    "use_rtpc_calibration": True,
}

INNOVATION_SWITCH_ALIASES: Dict[str, str] = {
    "use_phase_amplitude_fusion_backbone": "use_spap_backbone",
    "use_dynamic_prototype_head": "use_mrpc_head",
    "use_decoupled_training_engine": "use_staged_training_engine",
    "use_adaptive_posthoc_calibration": "use_rtpc_calibration",
}

DEFAULT_BACKBONE_DETAIL_SWITCHES: Dict[str, bool] = {
    "use_complex_attention": True,
    "use_apta": True,
    "use_spatial_branch": True,
    "use_frequency_branch": True,
    "use_tsfi": True,
    "use_transformer": True,
    "use_multi_scale": True,
    "use_sff_gate": True,
    "use_spectral_bypass": True,
}

DEFAULT_HEAD_DETAIL_SWITCHES: Dict[str, bool] = {
    "use_prototype_branch": True,
    "use_dynamic_gate": True,
}

DEFAULT_TRAINING_DETAIL_SWITCHES: Dict[str, bool] = {
    "use_auxiliary_heads": False,
    "use_contrastive_loss": False,
}


def _normalize_innovation_switch_names(user_switches: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(user_switches)
    for old_key, new_key in INNOVATION_SWITCH_ALIASES.items():
        if old_key not in normalized:
            continue
        old_value = normalized.pop(old_key)
        if new_key in normalized and normalized[new_key] != old_value:
            raise ValueError(f"Conflicting ablation switches: {old_key} and {new_key}.")
        normalized[new_key] = old_value
    return normalized


def get_innovation_switches(config: Dict[str, Any]) -> Dict[str, bool]:
    dataset_cfg = config["dataset"]
    user_switches = dataset_cfg.get("ablation", {})
    if not isinstance(user_switches, dict):
        raise ValueError("CONFIG['dataset']['ablation'] must be a dict.")
    user_switches = _normalize_innovation_switch_names(user_switches)

    detail_keys = {"backbone_detail", "head_detail", "training_detail"}
    unknown_keys = sorted(set(user_switches) - set(DEFAULT_INNOVATION_SWITCHES) - detail_keys)
    if unknown_keys:
        raise ValueError(f"Unknown ablation switches: {unknown_keys}")

    switches = DEFAULT_INNOVATION_SWITCHES.copy()
    for key, value in user_switches.items():
        if key not in DEFAULT_INNOVATION_SWITCHES:
            continue
        if not isinstance(value, bool):
            raise ValueError(f"Ablation switch '{key}' must be bool, got {type(value).__name__}.")
        switches[key] = value
    return switches


def _resolve_detail_switches(
    section_name: str,
    defaults: Dict[str, bool],
    user_values: Any,
) -> Dict[str, bool]:
    if user_values is None:
        return defaults.copy()
    if not isinstance(user_values, dict):
        raise ValueError(f"CONFIG['dataset']['ablation']['{section_name}'] must be a dict.")
    if section_name == "backbone_detail" and "use_sfid" in user_values:
        user_values = dict(user_values)
        use_sfid = user_values.pop("use_sfid")
        if "use_tsfi" in user_values and user_values["use_tsfi"] != use_sfid:
            raise ValueError("Conflicting ablation switches: backbone_detail.use_sfid and use_tsfi.")
        user_values["use_tsfi"] = use_sfid

    unknown_keys = sorted(set(user_values) - set(defaults))
    if unknown_keys:
        raise ValueError(f"Unknown switches under '{section_name}': {unknown_keys}")

    resolved = defaults.copy()
    for key, value in user_values.items():
        if not isinstance(value, bool):
            raise ValueError(f"Ablation switch '{section_name}.{key}' must be bool, got {type(value).__name__}.")
        resolved[key] = value
    return resolved


def get_ablation_detail_switches(config: Dict[str, Any]) -> Dict[str, Dict[str, bool]]:
    dataset_cfg = config["dataset"]
    user_switches = dataset_cfg.get("ablation", {})
    if not isinstance(user_switches, dict):
        raise ValueError("CONFIG['dataset']['ablation'] must be a dict.")

    return {
        "backbone_detail": _resolve_detail_switches(
            "backbone_detail",
            DEFAULT_BACKBONE_DETAIL_SWITCHES,
            user_switches.get("backbone_detail"),
        ),
        "head_detail": _resolve_detail_switches(
            "head_detail",
            DEFAULT_HEAD_DETAIL_SWITCHES,
            user_switches.get("head_detail"),
        ),
        "training_detail": _resolve_detail_switches(
            "training_detail",
            DEFAULT_TRAINING_DETAIL_SWITCHES,
            user_switches.get("training_detail"),
        ),
    }


def get_ablation_switches(config: Dict[str, Any]) -> Dict[str, Any]:
    innovation = get_innovation_switches(config)
    detail = get_ablation_detail_switches(config)
    backbone_on = innovation["use_spap_backbone"]
    prototype_head_on = innovation["use_mrpc_head"]
    training_engine_on = innovation["use_staged_training_engine"]
    calibration_on = innovation["use_rtpc_calibration"]

    return {
        "backbone_mode": "innovation1" if backbone_on else "baseline",
        "head_mode": "innovation2" if prototype_head_on else "baseline",
        "use_complex_attention": backbone_on and detail["backbone_detail"]["use_complex_attention"],
        "use_apta": backbone_on and detail["backbone_detail"]["use_apta"],
        "use_spatial_branch": backbone_on and detail["backbone_detail"]["use_spatial_branch"],
        "use_frequency_branch": backbone_on and detail["backbone_detail"]["use_frequency_branch"],
        "use_tsfi": backbone_on and detail["backbone_detail"]["use_tsfi"],
        "use_transformer": backbone_on and detail["backbone_detail"]["use_transformer"],
        "use_multi_scale": backbone_on and detail["backbone_detail"]["use_multi_scale"],
        "use_sff_gate": backbone_on and detail["backbone_detail"]["use_sff_gate"],
        "use_spectral_bypass": backbone_on and detail["backbone_detail"]["use_spectral_bypass"],
        "use_prototype_branch": prototype_head_on and detail["head_detail"]["use_prototype_branch"],
        "use_dynamic_gate": prototype_head_on and detail["head_detail"]["use_dynamic_gate"],
        "use_auxiliary_heads": training_engine_on and detail["training_detail"]["use_auxiliary_heads"],
        "use_contrastive_loss": training_engine_on and detail["training_detail"]["use_contrastive_loss"],
        "use_staged_training": training_engine_on,
        "use_calibration": calibration_on,
    }


def build_effective_stage_plan(config: Dict[str, Any], ablation: Dict[str, Any]) -> Dict[str, Any]:
    trainer_cfg = config["trainer"]
    stage1_epochs = int(trainer_cfg["stage1_epochs"])
    stage2_epochs = int(trainer_cfg["stage2_epochs"])

    if not ablation["use_staged_training"]:
        stage1_epochs = int(trainer_cfg.get("single_stage_epochs", stage1_epochs))
        stage2_epochs = 0

    return {
        "stage1_epochs": stage1_epochs,
        "stage2_epochs": stage2_epochs,
        "use_stage2": stage2_epochs > 0,
        "use_calibration": bool(config["calibration"]["enabled"] and ablation["use_calibration"]),
    }


def build_effective_model_config(config: Dict[str, Any], ablation: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = dict(config["model"])
    model_cfg.update(
        {
            "backbone_mode": ablation["backbone_mode"],
            "head_mode": ablation["head_mode"],
            "use_complex_attention": ablation["use_complex_attention"],
            "use_apta": ablation["use_apta"],
            "use_spatial_branch": ablation["use_spatial_branch"],
            "use_frequency_branch": ablation["use_frequency_branch"],
            "use_tsfi": ablation["use_tsfi"],
            "use_transformer": ablation["use_transformer"],
            "use_multi_scale": ablation["use_multi_scale"],
            "use_sff_gate": ablation["use_sff_gate"],
            "use_spectral_bypass": ablation["use_spectral_bypass"],
            "use_prototype_branch": ablation["use_prototype_branch"],
            "use_dynamic_gate": ablation["use_dynamic_gate"],
            "use_auxiliary_heads": ablation["use_auxiliary_heads"],
        }
    )
    return model_cfg


def resolve_class_name_mapping(
    dataset_cfg: Dict[str, Any],
    inverse_label_mapping: Dict[int, int],
    num_classes: int,
) -> Dict[int, str]:
    custom_names = dataset_cfg.get("class_names")
    if isinstance(custom_names, (list, tuple)):
        if len(custom_names) != num_classes:
            raise ValueError(
                f"CONFIG['dataset']['class_names'] has {len(custom_names)} names, expected {num_classes}."
            )
        return {class_id: str(custom_names[class_id]) for class_id in range(num_classes)}

    if isinstance(custom_names, dict):
        mapping: Dict[int, str] = {}
        for class_id in range(num_classes):
            orig_label = int(inverse_label_mapping.get(class_id, class_id + 1))
            if class_id in custom_names:
                mapping[class_id] = str(custom_names[class_id])
            elif orig_label in custom_names:
                mapping[class_id] = str(custom_names[orig_label])
            else:
                mapping[class_id] = f"Class_{orig_label}"
        return mapping

    preset_names = DATASET_CLASS_NAME_PRESETS.get(str(dataset_cfg.get("name", "")))
    if preset_names is not None:
        mapping = {}
        for class_id in range(num_classes):
            orig_label = int(inverse_label_mapping.get(class_id, class_id + 1))
            if 1 <= orig_label <= len(preset_names):
                mapping[class_id] = preset_names[orig_label - 1]
            else:
                mapping[class_id] = f"Class_{orig_label}"
        return mapping

    return {
        class_id: f"Class_{int(inverse_label_mapping.get(class_id, class_id + 1))}"
        for class_id in range(num_classes)
    }


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_array_file(path: str | Path, key: Optional[str] = None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        npz = np.load(path)
        if key is None:
            if len(npz.files) != 1:
                raise ValueError(f"{path} contains multiple arrays. Please provide key explicitly.")
            key = npz.files[0]
        return npz[key]
    if suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            if key is None:
                if len(obj) != 1:
                    raise ValueError(f"{path} contains multiple entries. Please provide key explicitly.")
                key = next(iter(obj.keys()))
            return obj[key]
        return obj
    if suffix == ".mat":
        try:
            from scipy.io import loadmat  # type: ignore

            data = loadmat(path)
            if key is None:
                valid_keys = [name for name in data.keys() if not name.startswith("__")]
                if len(valid_keys) != 1:
                    raise ValueError(f"{path} contains multiple MAT variables. Please provide key explicitly.")
                key = valid_keys[0]
            return data[key]
        except Exception:
            try:
                import h5py  # type: ignore

                with h5py.File(path, "r") as data:
                    if key is None:
                        valid_keys = list(data.keys())
                        if len(valid_keys) != 1:
                            raise ValueError(f"{path} contains multiple HDF5 datasets. Please provide key explicitly.")
                        key = valid_keys[0]
                    array = np.array(data[key])
                    return array
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load MAT file {path}. Please install scipy or h5py, and check key={key}."
                ) from exc

    raise ValueError(f"Unsupported file type: {path.suffix}")


def load_dataset_from_config(dataset_cfg: Dict[str, Any]):
    cube = load_array_file(dataset_cfg["cube_path"], dataset_cfg.get("cube_key"))
    gt_path = dataset_cfg.get("gt_path", dataset_cfg["cube_path"])
    gt = load_array_file(gt_path, dataset_cfg.get("gt_key"))
    return cube, gt


def build_plain_eval_loader(dataset, loader_cfg: Dict[str, Any]) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=loader_cfg["batch_size"],
        shuffle=False,
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg["pin_memory"],
        persistent_workers=loader_cfg["num_workers"] > 0,
    )


def build_balanced_train_loader(bundle, loader_cfg: Dict[str, Any]) -> DataLoader:
    sampler, _ = build_weighted_sampler(
        bundle.train_dataset.labels,
        num_classes=bundle.num_classes,
        weight_mode=loader_cfg["sampler_weight_mode"],
        beta=loader_cfg["sampler_beta"],
    )
    return DataLoader(
        bundle.train_dataset,
        batch_size=loader_cfg["batch_size"],
        shuffle=False,
        sampler=sampler,
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg["pin_memory"],
        persistent_workers=loader_cfg["num_workers"] > 0,
    )


def build_configured_dataloaders(config: Dict[str, Any]):
    dataset_cfg = config["dataset"]
    split_cfg = config["split"]
    preprocess_cfg = config["preprocess"]
    loader_cfg = config["loader"]
    cube, gt = load_dataset_from_config(dataset_cfg)

    train_samples_per_class = (
        split_cfg["train_samples_per_class"] if split_cfg["train_mode"] == "fixed_per_class" else None
    )
    train_samples_per_class_by_original_label = (
        split_cfg.get("train_samples_per_class_by_original_label")
        if split_cfg["train_mode"] == "custom_per_class"
        else None
    )
    val_samples_per_class = split_cfg["val_samples_per_class"] if split_cfg["val_mode"] == "fixed_per_class" else None

    bundle = build_hsi_dataloaders(
        cube=cube,
        gt=gt,
        input_layout=dataset_cfg.get("input_layout", "HWC"),
        patch_size=preprocess_cfg["patch_size"],
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        train_samples_per_class=train_samples_per_class,
        train_samples_per_class_by_original_label=train_samples_per_class_by_original_label,
        val_samples_per_class=val_samples_per_class,
        background_label=split_cfg["background_label"],
        min_train_per_class=split_cfg["min_train_per_class"],
        min_val_per_class=split_cfg["min_val_per_class"],
        split_mode=split_cfg["split_mode"],
        spatial_axis=split_cfg["spatial_axis"],
        seed=split_cfg["seed"],
        clip_percentiles=preprocess_cfg["clip_percentiles"],
        drop_bands=preprocess_cfg["drop_bands"],
        keep_bands=preprocess_cfg["keep_bands"],
        auto_detect_bad_bands=preprocess_cfg["auto_detect_bad_bands"],
        auto_bad_band_mode=preprocess_cfg["auto_bad_band_mode"],
        padding_mode=preprocess_cfg["padding_mode"],
        batch_size=loader_cfg["batch_size"],
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg["pin_memory"],
        use_weighted_sampler=loader_cfg["use_weighted_sampler"],
        sampler_weight_mode=loader_cfg["sampler_weight_mode"],
        sampler_beta=loader_cfg["sampler_beta"],
        use_hsi_augmentation=loader_cfg["use_hsi_augmentation"],
    )
    return bundle


def configure_console_output() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer\.norm_first was True",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"networkx backend defined more than once: .*",
        category=RuntimeWarning,
    )


def print_section(title: str) -> None:
    line = "=" * 88
    print(line)
    print(title)
    print(line)


def format_key_value_block(rows: list[tuple[str, Any]]) -> str:
    label_width = max(len(label) for label, _ in rows)
    return "\n".join(f"{label:<{label_width}} : {value}" for label, value in rows)


def count_subset_labels(labels, num_classes: int) -> np.ndarray:
    array = np.asarray(labels, dtype=np.int64)
    if array.size == 0:
        return np.zeros(num_classes, dtype=np.int64)
    return np.bincount(array, minlength=num_classes)


def format_split_table(bundle) -> str:
    split_result = bundle.split_result
    num_classes = bundle.num_classes
    train_counts = count_subset_labels(split_result.train.labels, num_classes)
    val_counts = count_subset_labels(split_result.val.labels, num_classes)
    test_counts = count_subset_labels(split_result.test.labels, num_classes)

    headers = ["Class", "Orig", "Total", "Train", "Val", "Test", "Train%"]
    rows = []
    for class_id in range(num_classes):
        orig_label = split_result.inverse_label_mapping.get(class_id, class_id)
        total_count = int(split_result.total_class_counts.get(class_id, 0))
        train_count = int(train_counts[class_id])
        val_count = int(val_counts[class_id])
        test_count = int(test_counts[class_id])
        train_ratio = f"{(100.0 * train_count / total_count):6.2f}%" if total_count > 0 else "  0.00%"
        rows.append(
            [
                str(class_id),
                str(orig_label),
                str(total_count),
                str(train_count),
                str(val_count),
                str(test_count),
                train_ratio,
            ]
        )

    widths = [
        max(len(headers[col]), max((len(row[col]) for row in rows), default=0))
        for col in range(len(headers))
    ]
    header_line = "  ".join(f"{headers[col]:<{widths[col]}}" for col in range(len(headers)))
    separator_line = "  ".join("-" * widths[col] for col in range(len(headers)))
    body_lines = [
        "  ".join(f"{row[col]:<{widths[col]}}" for col in range(len(headers)))
        for row in rows
    ]
    return "\n".join([header_line, separator_line, *body_lines])


def print_split_summary(bundle, config: Dict[str, Any]) -> None:
    split_cfg = config["split"]
    preprocess_cfg = config["preprocess"]
    loader_cfg = config["loader"]
    _, scene_h, scene_w = bundle.normalized_cube_chw.shape
    total_labeled = int(sum(bundle.split_result.total_class_counts.values()))

    print_section("Dataset Summary")
    print(
        format_key_value_block(
            [
                ("Dataset", config["dataset"]["name"]),
                ("Cube shape", f"{scene_h} x {scene_w} x {bundle.num_bands}"),
                ("Patch size", preprocess_cfg["patch_size"]),
                ("Classes", bundle.num_classes),
                ("Total labeled samples", total_labeled),
                ("Train / Val / Test", f"{len(bundle.train_dataset)} / {len(bundle.val_dataset)} / {len(bundle.test_dataset)}"),
                ("Split mode", split_cfg["split_mode"]),
                ("Train mode", split_cfg["train_mode"]),
                ("Val mode", split_cfg["val_mode"]),
                ("Batch size", loader_cfg["batch_size"]),
                ("Weighted sampler", loader_cfg["use_weighted_sampler"]),
                ("HSI augmentation", loader_cfg["use_hsi_augmentation"]),
            ]
        )
    )
    print("")
    print_section("Per-Class Split Counts")
    print("Class is the remapped class id used by the model, Orig is the original label id.")
    print(format_split_table(bundle))
    print("", flush=True)


def print_ablation_plan(config: Dict[str, Any]) -> None:
    innovation = get_innovation_switches(config)
    ablation = get_ablation_switches(config)
    print_section("SPARC-Net Ablation Switches")
    print(
        format_key_value_block(
            [
                ("SPAP Backbone", innovation["use_spap_backbone"]),
                ("MRPC Head", innovation["use_mrpc_head"]),
                ("Staged Training Engine", innovation["use_staged_training_engine"]),
                ("RTPC Calibration", innovation["use_rtpc_calibration"]),
            ]
        )
    )
    print("")
    print(
        format_key_value_block(
            [
                ("Effective backbone mode", ablation["backbone_mode"]),
                ("Effective head mode", ablation["head_mode"]),
                ("Backbone detail - APCR", ablation["use_complex_attention"]),
                ("Backbone detail - APTA", ablation["use_apta"]),
                ("Backbone detail - spatial branch", ablation["use_spatial_branch"]),
                ("Backbone detail - DAF Branch", ablation["use_frequency_branch"]),
                ("Backbone detail - TSFI", ablation["use_tsfi"]),
                ("Backbone detail - transformer", ablation["use_transformer"]),
                ("Backbone detail - multi-scale", ablation["use_multi_scale"]),
                ("Backbone detail - SFF gate", ablation["use_sff_gate"]),
                ("Backbone detail - SAB", ablation["use_spectral_bypass"]),
                ("Head detail - prototype branch", ablation["use_prototype_branch"]),
                ("Head detail - dynamic gate", ablation["use_dynamic_gate"]),
                ("Training detail - auxiliary heads", ablation["use_auxiliary_heads"]),
                ("Training detail - contrastive", ablation["use_contrastive_loss"]),
            ]
        )
    )
    print("", flush=True)


def print_training_plan(config: Dict[str, Any]) -> None:
    ablation = get_ablation_switches(config)
    effective_stage_plan = build_effective_stage_plan(config, ablation)
    trainer_cfg = config["trainer"]
    loss_cfg = config["loss"]
    model_cfg = config["model"]
    runtime_cfg = config["runtime"]

    print_section("Training Plan")
    print(
        format_key_value_block(
            [
                ("Stage1 epochs", effective_stage_plan["stage1_epochs"]),
                ("Stage2 epochs", effective_stage_plan["stage2_epochs"]),
                ("Total epochs", loss_cfg["total_epochs"]),
                ("Requested device", runtime_cfg["device"] if runtime_cfg["device"] is not None else "auto"),
                ("Backbone mode", ablation["backbone_mode"]),
                ("Head mode", ablation["head_mode"]),
                ("Stage1 lr", trainer_cfg["stage1_lr"]),
                ("Stage2 lr", trainer_cfg["stage2_lr"]),
                ("LDAM max_margin", loss_cfg["max_margin"]),
                ("LDAM beta", loss_cfg["beta"]),
                ("Align scale limit", model_cfg["alignment_scale_limit"]),
                ("Align bias limit", model_cfg["alignment_bias_limit"]),
                ("Freeze source classifier", model_cfg["freeze_source_classifier_in_stage2"]),
                ("Contrastive weight", trainer_cfg["contrastive_weight"]),
                ("Stage1 head reg", trainer_cfg["stage1_head_regularization_weight"]),
                ("Stage2 head reg", trainer_cfg["stage2_head_regularization_weight"]),
                ("Prototype head-drop tol", trainer_cfg["prototype_correction_head_drop_tolerance"]),
                ("AMP", trainer_cfg["amp"]),
                ("Monitor metric", trainer_cfg["monitor"]),
                ("Calibration enabled", effective_stage_plan["use_calibration"]),
            ]
        )
    )
    print("", flush=True)


def print_eval_report(
    split_name: str,
    eval_output: Dict[str, Any],
    inverse_label_mapping: Dict[int, int],
    class_name_mapping: Dict[int, str],
) -> None:
    metrics = eval_output["classification_metrics"]
    print_section(f"{split_name} Evaluation")
    print(f"{split_name} Summary")
    print(format_metric_summary(metrics, prefix=split_name))
    print("")
    print(f"{split_name} Per-Class Accuracy")
    print(
        format_per_class_accuracy_table(
            metrics,
            class_names=class_name_mapping,
            inverse_label_mapping=inverse_label_mapping,
        )
    )
    print("")
    print(f"{split_name} Confusion Matrix")
    print(metrics.confusion_matrix.cpu().numpy())
    print("", flush=True)


def main() -> None:
    configure_console_output()
    set_global_seed(CONFIG["runtime"]["seed"])
    ablation = get_ablation_switches(CONFIG)
    effective_stage_plan = build_effective_stage_plan(CONFIG, ablation)
    effective_model_cfg = build_effective_model_config(CONFIG, ablation)

    bundle = build_configured_dataloaders(CONFIG)
    class_name_mapping = resolve_class_name_mapping(CONFIG["dataset"], bundle.inverse_label_mapping, bundle.num_classes)
    print_split_summary(bundle, CONFIG)
    print_ablation_plan(CONFIG)
    print_training_plan(CONFIG)

    model = SPARCNet(
        in_channels=bundle.num_bands,
        num_classes=bundle.num_classes,
        class_counts=bundle.split_result.train_class_counts,
        **effective_model_cfg,
    )

    classification_loss, _ = build_recommended_hsi_imbalance_loss(
        class_counts=bundle.split_result.train_class_counts,
        total_epochs=CONFIG["loss"]["total_epochs"],
        max_margin=CONFIG["loss"]["max_margin"],
        beta=CONFIG["loss"]["beta"],
        gamma=CONFIG["loss"]["gamma"],
        drw_ratio=CONFIG["loss"]["drw_ratio"],
        margin_mode=model.classifier_margin_mode,
    )

    contrastive_loss = None
    if ablation["use_contrastive_loss"]:
        contrastive_loss, _ = build_recommended_hsi_contrastive_loss(
            class_counts=bundle.split_result.train_class_counts,
            temperature=CONFIG["loss"]["contrastive_temperature"],
            beta=CONFIG["loss"]["contrastive_beta"],
        )

    calibration_cfg = None
    if effective_stage_plan["use_calibration"]:
        calibration_cfg = RTPCConfig(
            epochs=CONFIG["calibration"]["epochs"],
            lr=CONFIG["calibration"]["lr"],
            weight_decay=CONFIG["calibration"]["weight_decay"],
            learn_class_scales=CONFIG["calibration"]["learn_class_scales"],
            learn_logit_mixer=CONFIG["calibration"]["learn_logit_mixer"],
            normalize_scales=CONFIG["calibration"]["normalize_scales"],
            max_log_scale=CONFIG["calibration"]["max_log_scale"],
            max_logit_mixer_residual=CONFIG["calibration"]["max_logit_mixer_residual"],
            train_loader_preference=CONFIG["calibration"]["train_loader_preference"],
            prior_modes=CONFIG["calibration"]["prior_modes"],
            prior_alpha_candidates=CONFIG["calibration"]["prior_alpha_candidates"],
            default_prior_mode=CONFIG["calibration"]["default_prior_mode"],
            default_prior_alpha=CONFIG["calibration"]["default_prior_alpha"],
            effective_num_beta=CONFIG["calibration"]["effective_num_beta"],
            monitor=CONFIG["calibration"]["monitor"],
            blend_candidates=CONFIG["calibration"]["blend_candidates"],
            min_val_gain=CONFIG["calibration"]["min_val_gain"],
            min_val_loss_gain=CONFIG["calibration"]["min_val_loss_gain"],
        )

    stage2_reset_classifier = CONFIG["trainer"]["stage2_reset_classifier"] if effective_stage_plan["use_stage2"] else False
    # The upgraded innovation-2 head follows a DisAlign-style stage-2 setup:
    # keep the source cosine classifier learned in stage 1, then calibrate it
    # instead of reinitializing it.
    if ablation["use_prototype_branch"]:
        stage2_reset_classifier = False

    trainer = StagedLongTailTrainer(
        model,
        classification_loss,
        contrastive_loss=contrastive_loss,
        config=StagedTrainerConfig(
            stage1=TrainingStageConfig(
                name="stage1",
                epochs=effective_stage_plan["stage1_epochs"],
                lr=CONFIG["trainer"]["stage1_lr"],
                weight_decay=CONFIG["trainer"]["stage1_weight_decay"],
                freeze_backbone=False,
                reset_classifier=False,
                use_aux_loss=ablation["use_auxiliary_heads"],
                use_contrastive_loss=ablation["use_contrastive_loss"],
                use_balanced_loader=False,
                head_regularization_weight=CONFIG["trainer"]["stage1_head_regularization_weight"] if ablation["use_prototype_branch"] else 0.0,
                head_output_mode="main",
                update_prototypes=False,
                guided_context=False,
            ),
            stage2=TrainingStageConfig(
                name="stage2",
                epochs=effective_stage_plan["stage2_epochs"],
                lr=CONFIG["trainer"]["stage2_lr"],
                weight_decay=CONFIG["trainer"]["stage2_weight_decay"],
                freeze_backbone=True,
                reset_classifier=stage2_reset_classifier,
                use_aux_loss=False,
                use_contrastive_loss=False,
                use_balanced_loader=CONFIG["trainer"]["stage2_use_balanced_loader"] if effective_stage_plan["use_stage2"] else False,
                head_regularization_weight=CONFIG["trainer"]["stage2_head_regularization_weight"] if ablation["use_prototype_branch"] else 0.0,
                head_output_mode="corrected" if ablation["use_prototype_branch"] else "main",
                update_prototypes=ablation["use_prototype_branch"],
                guided_context=ablation["use_prototype_branch"],
            ),
            aux_mid_weight=CONFIG["trainer"]["aux_mid_weight"],
            aux_early_weight=CONFIG["trainer"]["aux_early_weight"],
            contrastive_weight=CONFIG["trainer"]["contrastive_weight"],
            amp=CONFIG["trainer"]["amp"],
            stage2_tau_norm=CONFIG["trainer"]["stage2_tau_norm"] if effective_stage_plan["use_stage2"] else None,
            monitor=CONFIG["trainer"]["monitor"],
            stage2_start_from_best_stage1=CONFIG["trainer"]["stage2_start_from_best_stage1"],
            stage2_revert_if_no_val_gain=CONFIG["trainer"]["stage2_revert_if_no_val_gain"],
            stage2_min_val_gain=CONFIG["trainer"]["stage2_min_val_gain"],
            stage2_skip_if_stage1_val_at_least=CONFIG["trainer"]["stage2_skip_if_stage1_val_at_least"],
            prototype_correction_head_drop_tolerance=CONFIG["trainer"]["prototype_correction_head_drop_tolerance"],
            calibration=calibration_cfg,
            verbose=CONFIG["trainer"]["verbose"],
            print_every=CONFIG["trainer"]["print_every"],
        ),
        device=CONFIG["runtime"]["device"],
    )

    balanced_train_loader = build_balanced_train_loader(bundle, CONFIG["loader"])
    effective_val_loader = None if len(bundle.val_dataset) == 0 else bundle.val_loader
    history = trainer.fit(
        bundle.train_loader,
        val_loader=effective_val_loader,
        test_loader=bundle.test_loader,
        balanced_train_loader=balanced_train_loader,
    )

    print_section("Training Finished")
    print(f"Best stage: {history.best_stage}")
    print(f"Best epoch: {history.best_epoch}")
    print(f"Best monitored metric: {history.best_metric:.4f}")
    if trainer.calibration_result is not None:
        print(
            f"Calibration: prior_mode={trainer.calibration_result.prior_mode}, "
            f"prior_alpha={trainer.calibration_result.prior_alpha:.4f}, "
            f"blend={trainer.calibration_result.blend_strength:.2f}"
        )
    if history.best_test_eval is not None:
        print(
            f"Best test stage: {history.best_test_stage}\n"
            f"Best test epoch: {history.best_test_epoch}\n"
            f"Best test monitored metric: {history.best_test_metric:.4f}"
        )
    print("", flush=True)

    plain_train_loader = build_plain_eval_loader(bundle.train_dataset, CONFIG["loader"])
    train_eval = trainer.evaluate(plain_train_loader)
    val_eval = None if len(bundle.val_dataset) == 0 else trainer.evaluate(bundle.val_loader)
    test_eval = trainer.evaluate(bundle.test_loader)

    if history.best_test_eval is not None:
        print_eval_report("Best Test", history.best_test_eval, bundle.inverse_label_mapping, class_name_mapping)
    print_eval_report("Train", train_eval, bundle.inverse_label_mapping, class_name_mapping)
    if val_eval is not None:
        print_eval_report("Val", val_eval, bundle.inverse_label_mapping, class_name_mapping)
    print_eval_report("Test", test_eval, bundle.inverse_label_mapping, class_name_mapping)


if __name__ == "__main__":
    main()
