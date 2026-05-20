from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch import nn
from PIL import Image

import main as exp_main
from datasets.patch_dataset import HSIPatchDataset
from losses import build_recommended_hsi_contrastive_loss, build_recommended_hsi_imbalance_loss
from metrics import compute_classification_metrics
from models import SPARCNet
from trainers import (
    StagedLongTailTrainer,
    StagedTrainerConfig,
    RTPCConfig,
    TrainingStageConfig,
)


PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
METHOD_NAME = "SPARC-Net"


DATASET_PRESETS: Dict[str, Dict[str, Any]] = {
    "IndianPines": {
        "name": "IndianPines",
        "cube_path": str(PROJECT_DIR / "datasets" / "IndianPines" / "Indian_pines_corrected.npy"),
        "gt_path": str(PROJECT_DIR / "datasets" / "IndianPines" / "Indian_pines_gt.npy"),
        "cube_key": None,
        "gt_key": None,
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
    "PaviaU": {
        "name": "PaviaU",
        "cube_path": str(PROJECT_DIR / "datasets" / "PaviaU" / "PaviaU.mat"),
        "gt_path": str(PROJECT_DIR / "datasets" / "PaviaU" / "PaviaU_gt.mat"),
        "cube_key": "paviaU",
        "gt_key": "paviaU_gt",
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
}


SUMMARY_METRIC_KEYS = ("oa", "aa", "kappa", "macro_f1")
RESOURCE_KEYS = (
    "train_seconds",
    "test_seconds",
    "flops_g",
    "params_m",
    "gpu_peak_allocated_mb",
    "gpu_peak_reserved_mb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the current full SPARC-Net model and save maps/resources.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["IndianPines", "Botswana", "PaviaU", "Houston2018"],
        choices=sorted(DATASET_PRESETS),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--splits", type=str, default=str(ROOT_DIR / "other-Method" / "benchmark_splits.json"))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--map-dir", type=str, default=str(ROOT_DIR / "fig"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--stage1-epochs", type=int, default=None)
    parser.add_argument("--stage2-epochs", type=int, default=None)
    parser.add_argument("--houston-stage1-epochs", type=int, default=None)
    parser.add_argument("--houston-stage2-epochs", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def aggregate_values(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def count_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def estimate_model_flops(model: nn.Module, input_shape: tuple[int, ...], device: torch.device) -> float:
    flops = 0.0
    handles = []

    def conv_hook(module, inputs, output):
        nonlocal flops
        if not torch.is_tensor(output):
            return
        kernel_ops = int(np.prod(module.kernel_size)) * (module.in_channels // module.groups)
        flops += float(output.numel() * kernel_ops * 2)

    def linear_hook(module, inputs, output):
        nonlocal flops
        if torch.is_tensor(output):
            out_numel = output.numel()
        elif isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
            out_numel = output[0].numel()
        else:
            return
        flops += float(out_numel * module.in_features * 2)

    def mha_hook(module, inputs, output):
        nonlocal flops
        if not inputs or not torch.is_tensor(inputs[0]):
            return
        q = inputs[0]
        if q.ndim != 3:
            return
        if getattr(module, "batch_first", False):
            batch, tokens, embed = q.shape
        else:
            tokens, batch, embed = q.shape
        heads = int(module.num_heads)
        head_dim = embed // max(1, heads)
        flops += float(batch * tokens * embed * embed * 3 * 2)
        flops += float(batch * heads * tokens * tokens * head_dim * 2)
        flops += float(batch * heads * tokens * tokens * head_dim * 2)
        flops += float(batch * tokens * embed * embed * 2)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.MultiheadAttention):
            handles.append(module.register_forward_hook(mha_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(input_shape, device=device)
        model(dummy)
    if was_training:
        model.train()
    for handle in handles:
        handle.remove()
    sync_device(device)
    return float(flops)


def load_splits(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_default_ablation_block() -> Dict[str, Any]:
    return {
        **exp_main.DEFAULT_INNOVATION_SWITCHES,
        "backbone_detail": exp_main.DEFAULT_BACKBONE_DETAIL_SWITCHES.copy(),
        "head_detail": exp_main.DEFAULT_HEAD_DETAIL_SWITCHES.copy(),
        "training_detail": exp_main.DEFAULT_TRAINING_DETAIL_SWITCHES.copy(),
    }


def build_config(dataset_name: str, split_cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = copy.deepcopy(exp_main.CONFIG)
    dataset_cfg = DATASET_PRESETS[dataset_name]
    cfg["dataset"]["name"] = dataset_cfg["name"]
    cfg["dataset"]["cube_path"] = dataset_cfg["cube_path"]
    cfg["dataset"]["cube_key"] = dataset_cfg["cube_key"]
    cfg["dataset"]["gt_path"] = dataset_cfg["gt_path"]
    cfg["dataset"]["gt_key"] = dataset_cfg["gt_key"]
    cfg["dataset"]["input_layout"] = dataset_cfg["input_layout"]
    cfg["dataset"]["ablation"] = build_default_ablation_block()

    cfg["split"]["train_mode"] = "custom_per_class"
    cfg["split"]["train_samples_per_class_by_original_label"] = {
        int(key): int(value) for key, value in split_cfg["train_counts_by_original_label"].items()
    }
    cfg["split"]["val_ratio"] = float(split_cfg.get("val_ratio", 0.1))
    cfg["split"]["min_train_per_class"] = 1
    cfg["split"]["min_val_per_class"] = 0 if cfg["split"]["val_ratio"] <= 0.0 else 1
    cfg["split"]["seed"] = int(args.seed)

    cfg["loader"]["batch_size"] = int(args.batch_size)
    cfg["loader"]["use_weighted_sampler"] = True
    cfg["loader"]["use_hsi_augmentation"] = True

    cfg["runtime"]["seed"] = int(args.seed)
    cfg["runtime"]["device"] = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"

    cfg["trainer"]["print_every"] = int(args.print_every)
    cfg["trainer"]["verbose"] = bool(args.print_every)
    cfg["trainer"]["amp"] = bool(args.amp)
    if args.stage1_epochs is not None:
        cfg["trainer"]["stage1_epochs"] = int(args.stage1_epochs)
    if args.stage2_epochs is not None:
        cfg["trainer"]["stage2_epochs"] = int(args.stage2_epochs)
    if dataset_name == "Houston2018":
        if args.houston_stage1_epochs is not None:
            cfg["trainer"]["stage1_epochs"] = int(args.houston_stage1_epochs)
        if args.houston_stage2_epochs is not None:
            cfg["trainer"]["stage2_epochs"] = int(args.houston_stage2_epochs)
    return cfg


def make_eval_loader(dataset, batch_size: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


def build_palette(num_classes: int) -> np.ndarray:
    base = np.array(
        [
            [0, 0, 0],
            [230, 25, 75],
            [60, 180, 75],
            [255, 225, 25],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
            [250, 190, 190],
            [0, 128, 128],
            [230, 190, 255],
            [170, 110, 40],
            [255, 250, 200],
            [128, 0, 0],
            [170, 255, 195],
            [128, 128, 0],
            [255, 215, 180],
            [0, 0, 128],
            [128, 128, 128],
        ],
        dtype=np.uint8,
    )
    if num_classes + 1 <= base.shape[0]:
        return base[: num_classes + 1]
    rng = np.random.default_rng(2026)
    extra = rng.integers(0, 255, size=(num_classes + 1 - base.shape[0], 3), dtype=np.uint8)
    return np.concatenate([base, extra], axis=0)


def apply_final_logits(trainer: StagedLongTailTrainer, logits: torch.Tensor) -> torch.Tensor:
    if trainer.logit_calibrator is not None:
        return trainer.logit_calibrator(logits)
    return logits


@torch.no_grad()
def save_labeled_classification_map(
    trainer: StagedLongTailTrainer,
    bundle,
    device: torch.device,
    save_path: Path,
    eval_batch_size: int,
) -> None:
    split = bundle.split_result
    coords = np.concatenate([split.train.coords, split.val.coords, split.test.coords], axis=0)
    labels = np.zeros(coords.shape[0], dtype=np.int64)
    dataset = HSIPatchDataset(
        cube_chw=bundle.padded_cube_chw,
        coords=coords,
        labels=labels,
        patch_size=bundle.train_dataset.patch_size,
        is_already_padded=True,
        return_coords=True,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    _, height, width = bundle.normalized_cube_chw.shape
    pred_map = np.zeros((height, width), dtype=np.uint16)
    trainer.model.eval()
    for batch in loader:
        x = batch["patch"].to(device, non_blocking=True)
        batch_coords = batch["coord"].cpu().numpy()
        outputs = trainer.model(x, return_aux=True)
        logits = apply_final_logits(trainer, outputs["logits"])
        pred = logits.argmax(dim=1).cpu().numpy()
        for idx, class_id in enumerate(pred.tolist()):
            row, col = batch_coords[idx]
            pred_map[int(row), int(col)] = int(bundle.inverse_label_mapping.get(int(class_id), int(class_id) + 1))
    palette = build_palette(int(pred_map.max()))
    rgb = palette[np.clip(pred_map, 0, palette.shape[0] - 1)]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(save_path)


def per_class_rows(metrics, class_name_mapping: Dict[int, str], inverse_label_mapping: Dict[int, int]) -> List[Dict[str, Any]]:
    rows = []
    num_classes = int(metrics.support.numel())
    for class_id in range(num_classes):
        display_id = int(inverse_label_mapping.get(class_id, class_id + 1))
        rows.append(
            {
                "class_index": int(class_id),
                "display_id": display_id,
                "class_name": class_name_mapping.get(class_id, f"Class_{display_id}"),
                "support": int(metrics.support[class_id].item()),
                "correct": int(metrics.confusion_matrix[class_id, class_id].item()),
                "acc": float(metrics.per_class_accuracy[class_id].item()),
                "precision": float(metrics.per_class_precision[class_id].item()),
                "recall": float(metrics.per_class_recall[class_id].item()),
                "f1": float(metrics.per_class_f1[class_id].item()),
            }
        )
    return rows


@torch.no_grad()
def evaluate_for_metrics(trainer: StagedLongTailTrainer, loader, num_classes: int) -> Dict[str, Any]:
    trainer.model.eval()
    preds = []
    labels = []
    total_loss = 0.0
    total = 0
    device = trainer.device
    for batch in loader:
        if isinstance(batch, dict):
            patches, y = batch["patch"], batch["label"]
        else:
            patches, y = batch[0], batch[1]
        patches = patches.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        outputs = trainer.model(patches, return_aux=True)
        logits = apply_final_logits(trainer, outputs["logits"])
        loss = torch.nn.functional.cross_entropy(logits, y)
        total_loss += float(loss.item()) * int(y.numel())
        total += int(y.numel())
        preds.append(logits.argmax(dim=1).cpu())
        labels.append(y.cpu())
    if total == 0:
        raise RuntimeError("Evaluation split is empty.")
    pred = torch.cat(preds)
    target = torch.cat(labels)
    metrics = compute_classification_metrics(labels=target, preds=pred, num_classes=num_classes)
    return {"loss": total_loss / max(total, 1), "metrics": metrics}


def build_trainer(cfg: Dict[str, Any], bundle, model: SPARCNet) -> StagedLongTailTrainer:
    ablation = exp_main.get_ablation_switches(cfg)
    effective_stage_plan = exp_main.build_effective_stage_plan(cfg, ablation)
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
        calibration_cfg = RTPCConfig(
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

    stage2_reset_classifier = cfg["trainer"]["stage2_reset_classifier"] if effective_stage_plan["use_stage2"] else False
    if ablation["use_prototype_branch"]:
        stage2_reset_classifier = False

    return StagedLongTailTrainer(
        model,
        classification_loss,
        contrastive_loss=contrastive_loss,
        config=StagedTrainerConfig(
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
                head_regularization_weight=cfg["trainer"]["stage1_head_regularization_weight"] if ablation["use_prototype_branch"] else 0.0,
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
                use_balanced_loader=cfg["trainer"]["stage2_use_balanced_loader"] if effective_stage_plan["use_stage2"] else False,
                head_regularization_weight=cfg["trainer"]["stage2_head_regularization_weight"] if ablation["use_prototype_branch"] else 0.0,
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


def run_dataset(dataset_name: str, split_cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    exp_main.configure_console_output()
    set_seed(int(args.seed))
    cfg = build_config(dataset_name, split_cfg, args)
    device = torch.device(cfg["runtime"]["device"])
    bundle = exp_main.build_configured_dataloaders(cfg)
    class_name_mapping = exp_main.resolve_class_name_mapping(
        cfg["dataset"], bundle.inverse_label_mapping, bundle.num_classes
    )
    ablation = exp_main.get_ablation_switches(cfg)
    effective_model_cfg = exp_main.build_effective_model_config(cfg, ablation)
    model = SPARCNet(
        in_channels=bundle.num_bands,
        num_classes=bundle.num_classes,
        class_counts=bundle.split_result.train_class_counts,
        **effective_model_cfg,
    )
    params = count_parameters(model)
    model.to(device)
    flops = estimate_model_flops(model, (1, bundle.num_bands, cfg["preprocess"]["patch_size"], cfg["preprocess"]["patch_size"]), device)
    trainer = build_trainer(cfg, bundle, model)
    balanced_train_loader = exp_main.build_balanced_train_loader(bundle, cfg["loader"])
    effective_val_loader = None if len(bundle.val_dataset) == 0 else bundle.val_loader
    test_loader = make_eval_loader(bundle.test_dataset, int(args.eval_batch_size))

    sync_device(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_start = time.perf_counter()
    history = trainer.fit(
        bundle.train_loader,
        val_loader=effective_val_loader,
        test_loader=None,
        balanced_train_loader=balanced_train_loader,
    )
    sync_device(device)
    train_seconds = time.perf_counter() - train_start

    test_start = time.perf_counter()
    eval_output = evaluate_for_metrics(trainer, test_loader, bundle.num_classes)
    sync_device(device)
    test_seconds = time.perf_counter() - test_start

    metrics = eval_output["metrics"]
    summary = metrics.to_summary_dict()
    map_path = Path(args.map_dir) / f"{dataset_name}-{METHOD_NAME}.png"
    save_labeled_classification_map(trainer, bundle, device, map_path, int(args.eval_batch_size))

    peak_allocated_mb = 0.0
    peak_reserved_mb = 0.0
    if device.type == "cuda":
        peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)

    run = {
        "seed": int(args.seed),
        "method": METHOD_NAME,
        "dataset": dataset_name,
        "best_stage": history.best_stage,
        "best_epoch": None if history.best_epoch is None else int(history.best_epoch) + 1,
        "best_metric": float(history.best_metric),
        "summary_metrics": {key: float(summary[key]) for key in summary},
        "resource_metrics": {
            "train_seconds": float(train_seconds),
            "test_seconds": float(test_seconds),
            "flops": float(flops),
            "flops_g": float(flops / 1e9),
            "params": int(params),
            "params_m": float(params / 1e6),
            "gpu_peak_allocated_mb": float(peak_allocated_mb),
            "gpu_peak_reserved_mb": float(peak_reserved_mb),
        },
        "per_class": per_class_rows(metrics, class_name_mapping, bundle.inverse_label_mapping),
        "split_counts": {
            "train": int(len(bundle.train_dataset)),
            "val": int(len(bundle.val_dataset)),
            "test": int(len(bundle.test_dataset)),
        },
        "map_path": str(map_path),
    }
    del trainer, model, balanced_train_loader, test_loader, bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return run


def output_dir_from_args(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_DIR / "logs" / f"full_resource_maps_{stamp}"


def load_store(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_store(path: Path, store: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")


def write_reports(output_dir: Path, store: Dict[str, Any]) -> None:
    runs = [run for run in store["runs"] if "summary_metrics" in run]
    with (output_dir / "summary_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "dataset", "seed", "oa", "aa", "kappa", "macro_f1", "map_path"])
        for run in runs:
            s = run["summary_metrics"]
            writer.writerow(
                [
                    run["method"],
                    run["dataset"],
                    run["seed"],
                    f"{s['oa']:.6f}",
                    f"{s['aa']:.6f}",
                    f"{s['kappa']:.6f}",
                    f"{s['macro_f1']:.6f}",
                    run["map_path"],
                ]
            )

    with (output_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "dataset", "display_id", "class_name", "support", "acc", "precision", "recall", "f1"])
        for run in runs:
            for row in run["per_class"]:
                writer.writerow(
                    [
                        run["method"],
                        run["dataset"],
                        row["display_id"],
                        row["class_name"],
                        row["support"],
                        f"{row['acc']:.6f}",
                        f"{row['precision']:.6f}",
                        f"{row['recall']:.6f}",
                        f"{row['f1']:.6f}",
                    ]
                )

    with (output_dir / "resource_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "dataset",
                "seed",
                "Train/s",
                "Test/s",
                "FLOPs/G",
                "Params/M",
                "GPU Peak Allocated/MB",
                "GPU Peak Reserved/MB",
                "map_path",
            ]
        )
        for run in runs:
            r = run["resource_metrics"]
            writer.writerow(
                [
                    run["method"],
                    run["dataset"],
                    run["seed"],
                    f"{r['train_seconds']:.4f}",
                    f"{r['test_seconds']:.4f}",
                    f"{r['flops_g']:.4f}",
                    f"{r['params_m']:.4f}",
                    f"{r['gpu_peak_allocated_mb']:.2f}",
                    f"{r['gpu_peak_reserved_mb']:.2f}",
                    run["map_path"],
                ]
            )


def main() -> None:
    args = parse_args()
    output_dir = output_dir_from_args(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    store_path = output_dir / "results.json"
    splits = load_splits(args.splits)
    store = load_store(store_path) if args.resume else None
    if store is None:
        store = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "method": METHOD_NAME,
            "datasets": args.datasets,
            "seed": args.seed,
            "gpu": args.gpu,
            "splits": {name: splits[name] for name in args.datasets},
            "runs": [],
        }
        save_store(store_path, store)

    print(f"[Output Dir] {output_dir}", flush=True)
    print(f"[Method] {METHOD_NAME}", flush=True)
    print(f"[Datasets] {', '.join(args.datasets)}", flush=True)
    print(f"[GPU] {args.gpu}", flush=True)
    print(flush=True)

    done = {(run.get("dataset"), int(run.get("seed", -1))) for run in store["runs"] if "summary_metrics" in run}
    for dataset_name in args.datasets:
        if (dataset_name, int(args.seed)) in done:
            print(f"[Skip] {METHOD_NAME} | {dataset_name} | seed={args.seed}", flush=True)
            continue
        print(f"[Run] {METHOD_NAME} | {dataset_name} | seed={args.seed} | start", flush=True)
        try:
            run = run_dataset(dataset_name, splits[dataset_name], args)
            store["runs"].append(run)
            save_store(store_path, store)
            write_reports(output_dir, store)
            s = run["summary_metrics"]
            r = run["resource_metrics"]
            print(
                f"[Done] {METHOD_NAME} | {dataset_name} | seed={args.seed} | "
                f"OA={s['oa']:.4f} | AA={s['aa']:.4f} | Kappa={s['kappa']:.4f} | "
                f"Train/s={r['train_seconds']:.2f} | Test/s={r['test_seconds']:.2f} | FLOPs/G={r['flops_g']:.4f}",
                flush=True,
            )
        except Exception as exc:
            store["runs"].append({"dataset": dataset_name, "seed": int(args.seed), "error": repr(exc)})
            save_store(store_path, store)
            print(f"[Failed] {METHOD_NAME} | {dataset_name} | seed={args.seed} | {exc!r}", flush=True)
            raise
    write_reports(output_dir, store)
    print("[Aggregation Complete]", flush=True)


if __name__ == "__main__":
    main()
