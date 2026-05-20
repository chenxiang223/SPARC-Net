from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
OTHER_METHOD_ROOT = THIS_DIR.parent
PROJECT_ROOT = OTHER_METHOD_ROOT.parent / "cvoca-feinfn"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as exp_main  # noqa: E402
from datasets import build_hsi_dataloaders  # noqa: E402
from datasets.patch_dataset import HSIPatchDataset  # noqa: E402
from losses import CBLDAMLoss  # noqa: E402
from metrics import compute_classification_metrics  # noqa: E402

from .model_zoo import METHOD_REGISTRY, build_method_model


LOSS_REGISTRY = ("ce", "ldam", "bs")


DATASET_PRESETS = {
    "IndianPines": {
        "name": "IndianPines",
        "cube_path": str(PROJECT_ROOT / "datasets" / "IndianPines" / "Indian_pines_corrected.npy"),
        "gt_path": str(PROJECT_ROOT / "datasets" / "IndianPines" / "Indian_pines_gt.npy"),
        "cube_key": None,
        "gt_key": None,
        "input_layout": "HWC",
    },
    "Botswana": {
        "name": "Botswana",
        "cube_path": str(PROJECT_ROOT / "datasets" / "Botswana" / "Botswana.npy"),
        "gt_path": str(PROJECT_ROOT / "datasets" / "Botswana" / "Botswana_gt.npy"),
        "cube_key": None,
        "gt_key": None,
        "input_layout": "HWC",
    },
    "PaviaU": {
        "name": "PaviaU",
        "cube_path": str(PROJECT_ROOT / "datasets" / "PaviaU" / "PaviaU.mat"),
        "gt_path": str(PROJECT_ROOT / "datasets" / "PaviaU" / "PaviaU_gt.mat"),
        "cube_key": "paviaU",
        "gt_key": "paviaU_gt",
        "input_layout": "HWC",
    },
    "Houston2018": {
        "name": "Houston2018",
        "cube_path": str(PROJECT_ROOT / "datasets" / "HOU2018" / "houston2018hsi.npy"),
        "gt_path": str(PROJECT_ROOT / "datasets" / "HOU2018" / "houston2018_gt.npy"),
        "cube_key": None,
        "gt_key": None,
        "input_layout": "HWC",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified reproduction benchmark for comparison HSI methods.")
    parser.add_argument("--methods", nargs="+", default=sorted(METHOD_REGISTRY), choices=sorted(METHOD_REGISTRY))
    parser.add_argument("--losses", nargs="+", default=["ce"], choices=LOSS_REGISTRY)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["IndianPines", "Botswana", "PaviaU", "Houston2018"],
        choices=sorted(DATASET_PRESETS),
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--houston-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ldam-max-margin", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--print-every", type=int, default=0)
    parser.add_argument("--splits", type=str, default=str(OTHER_METHOD_ROOT / "benchmark_splits.json"))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-maps", action="store_true", help="Save one labeled classification map for seed-start.")
    parser.add_argument("--map-dir", type=str, default=r"C:\Users\PC\Desktop\cvoca-feinfn\fig")
    parser.add_argument("--tag", type=str, default="")
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


def ensure_batch_tuple(batch):
    if isinstance(batch, dict):
        return batch["patch"], batch["label"]
    return batch


def aggregate_values(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "var": 0.0, "max": 0.0, "min": 0.0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "var": float(arr.var(ddof=0)),
        "max": float(arr.max()),
        "min": float(arr.min()),
    }


def format_method_label(method: str, loss_type: str) -> str:
    if loss_type == "ce":
        return method
    return f"{method}+{loss_type.upper()}"


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, class_counts: Iterable[int] | Dict[int, int], eps: float = 1e-12) -> None:
        super().__init__()
        if isinstance(class_counts, dict):
            if not class_counts:
                raise ValueError("class_counts cannot be empty.")
            max_key = max(int(key) for key in class_counts)
            counts = torch.zeros(max_key + 1, dtype=torch.float32)
            for key, value in class_counts.items():
                counts[int(key)] = float(value)
        else:
            counts = torch.as_tensor(class_counts, dtype=torch.float32)
        if counts.ndim != 1 or counts.numel() == 0:
            raise ValueError("class_counts must be a non-empty 1D tensor.")
        self.register_buffer("class_counts", counts.clamp(min=eps))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_counts = self.class_counts.to(device=logits.device, dtype=logits.dtype).log()
        return F.cross_entropy(logits + log_counts.unsqueeze(0), targets)


def build_loss_fn(loss_type: str, class_counts: Iterable[int] | Dict[int, int], args: argparse.Namespace, epochs: int) -> nn.Module:
    if loss_type == "ce":
        return nn.CrossEntropyLoss()
    if loss_type == "bs":
        return BalancedSoftmaxLoss(class_counts)
    if loss_type == "ldam":
        # This is the LDAM row from the comparison table: class-dependent margin
        # only, without the project's later class-balanced reweighting preset.
        return CBLDAMLoss(
            class_counts=class_counts,
            max_margin=args.ldam_max_margin,
            scale=1.0,
            beta=0.999,
            gamma=0.0,
            drw_start_epoch=epochs + 1,
            margin_mode="linear",
        )
    raise ValueError(f"Unknown loss_type: {loss_type}")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def get_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    return OTHER_METHOD_ROOT / "logs" / f"comparison_{stamp}{tag}"


def load_splits(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_bundle(dataset_name: str, split_cfg: Dict[str, Any], args: argparse.Namespace, seed: int):
    dataset_cfg = dict(DATASET_PRESETS[dataset_name])
    cube, gt = exp_main.load_dataset_from_config(dataset_cfg)
    val_ratio = float(split_cfg.get("val_ratio", 0.1))
    train_counts = {int(k): int(v) for k, v in split_cfg["train_counts_by_original_label"].items()}
    return build_hsi_dataloaders(
        cube=cube,
        gt=gt,
        input_layout=dataset_cfg.get("input_layout", "HWC"),
        patch_size=args.patch_size,
        train_ratio=0.01,
        val_ratio=val_ratio,
        train_samples_per_class=None,
        train_samples_per_class_by_original_label=train_counts,
        val_samples_per_class=None,
        background_label=0,
        min_train_per_class=1,
        min_val_per_class=0 if val_ratio <= 0.0 else 1,
        split_mode="stratified_random",
        seed=seed,
        clip_percentiles=None,
        auto_detect_bad_bands=True,
        auto_bad_band_mode="interpolate",
        padding_mode="reflect",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        use_weighted_sampler=False,
        use_hsi_augmentation=True,
    )


def make_eval_loader(dataset, args: argparse.Namespace):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )


@torch.no_grad()
def evaluate_model(model: nn.Module, loader, device: torch.device, num_classes: int, class_counts=None) -> Dict[str, Any]:
    model.eval()
    preds: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    total_loss = 0.0
    total = 0
    loss_fn = nn.CrossEntropyLoss()
    for batch in loader:
        x, y = ensure_batch_tuple(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += float(loss.item()) * int(y.numel())
        total += int(y.numel())
        preds.append(logits.argmax(dim=1).cpu())
        labels.append(y.cpu())
    if total == 0:
        return {"loss": math.inf, "metrics": None, "preds": torch.empty(0), "labels": torch.empty(0)}
    pred = torch.cat(preds)
    target = torch.cat(labels)
    metrics = compute_classification_metrics(labels=target, preds=pred, num_classes=num_classes, class_counts=class_counts)
    return {"loss": total_loss / total, "metrics": metrics, "preds": pred, "labels": target}


def metrics_to_summary(metrics) -> Dict[str, float]:
    summary = metrics.to_summary_dict()
    selected = {
        "oa": float(summary["oa"]),
        "aa": float(summary["aa"]),
        "kappa": float(summary["kappa"]),
        "macro_f1": float(summary["macro_f1"]),
        "macro_precision": float(summary["macro_precision"]),
        "macro_recall": float(summary["macro_recall"]),
    }
    for key in ("head_aa", "medium_aa", "tail_aa"):
        if key in summary:
            selected[key] = float(summary[key])
    return selected


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


@torch.no_grad()
def save_labeled_classification_map(
    model: nn.Module,
    bundle,
    device: torch.device,
    save_path: Path,
    args: argparse.Namespace,
) -> None:
    split = bundle.split_result
    coords = np.concatenate([split.train.coords, split.val.coords, split.test.coords], axis=0)
    labels = np.zeros(coords.shape[0], dtype=np.int64)
    dataset = HSIPatchDataset(
        cube_chw=bundle.padded_cube_chw,
        coords=coords,
        labels=labels,
        patch_size=args.patch_size,
        is_already_padded=True,
        return_coords=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    _, height, width = bundle.normalized_cube_chw.shape
    pred_map = np.zeros((height, width), dtype=np.uint16)
    model.eval()
    for batch in loader:
        x = batch["patch"].to(device, non_blocking=True)
        batch_coords = batch["coord"].cpu().numpy()
        pred = model(x).argmax(dim=1).cpu().numpy()
        for idx, class_id in enumerate(pred.tolist()):
            row, col = batch_coords[idx]
            pred_map[int(row), int(col)] = int(bundle.inverse_label_mapping.get(int(class_id), int(class_id) + 1))
    palette = build_palette(int(pred_map.max()))
    rgb = palette[np.clip(pred_map, 0, palette.shape[0] - 1)]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(save_path)


def train_one_run(method: str, loss_type: str, dataset_name: str, seed: int, split_cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    bundle = build_bundle(dataset_name, split_cfg, args, seed)
    dataset_cfg = dict(DATASET_PRESETS[dataset_name])
    class_name_mapping = exp_main.resolve_class_name_mapping(dataset_cfg, bundle.inverse_label_mapping, bundle.num_classes)
    model = build_method_model(method, bundle.num_bands, bundle.num_classes).to(device)
    params = count_parameters(model)
    flops = estimate_model_flops(model, (1, bundle.num_bands, args.patch_size, args.patch_size), device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    epochs = args.houston_epochs if dataset_name == "Houston2018" else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    loss_fn = build_loss_fn(loss_type, bundle.split_result.train_class_counts, args, epochs).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    val_loader = make_eval_loader(bundle.val_dataset, args)
    test_loader = make_eval_loader(bundle.test_dataset, args)

    best_state = None
    best_epoch = -1
    best_metric = -float("inf")
    stale = 0
    history = []
    has_val = len(bundle.val_dataset) > 0

    sync_device(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        if hasattr(loss_fn, "set_epoch"):
            loss_fn.set_epoch(epoch - 1)
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch in bundle.train_loader:
            x, y = ensure_batch_tuple(batch)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item()) * int(y.numel())
            train_correct += int((logits.argmax(dim=1) == y).sum().item())
            train_total += int(y.numel())
        scheduler.step()
        train_acc = train_correct / max(1, train_total)

        if has_val:
            val_eval = evaluate_model(model, val_loader, device, bundle.num_classes, bundle.split_result.train_class_counts)
            metric = float(val_eval["metrics"].oa) if val_eval["metrics"] is not None else train_acc
            val_loss = float(val_eval["loss"])
        else:
            metric = train_acc
            val_loss = math.inf

        history.append({"epoch": epoch, "train_loss": train_loss / max(1, train_total), "train_acc": train_acc, "val_loss": val_loss, "monitor": metric})
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if args.print_every and (epoch == 1 or epoch % args.print_every == 0 or epoch == epochs):
            print(
                f"[Epoch] {method} {dataset_name} seed={seed} epoch={epoch}/{epochs} "
                f"train_acc={train_acc:.4f} monitor={metric:.4f}",
                flush=True,
            )
        if has_val and stale >= args.patience:
            break
    sync_device(device)
    train_seconds = time.perf_counter() - train_start

    if best_state is not None:
        model.load_state_dict(best_state)
    sync_device(device)
    test_start = time.perf_counter()
    test_eval = evaluate_model(model, test_loader, device, bundle.num_classes, bundle.split_result.train_class_counts)
    sync_device(device)
    test_seconds = time.perf_counter() - test_start
    metrics = test_eval["metrics"]
    if metrics is None:
        raise RuntimeError("Test split is empty.")
    summary = metrics_to_summary(metrics)
    peak_allocated_mb = 0.0
    peak_reserved_mb = 0.0
    if device.type == "cuda":
        peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)
    if args.save_maps and int(seed) == int(args.seed_start):
        map_path = Path(args.map_dir) / f"{dataset_name}-{method}.png"
        save_labeled_classification_map(model, bundle, device, map_path, args)
    run = {
        "seed": int(seed),
        "method": method,
        "loss": loss_type,
        "method_label": format_method_label(method, loss_type),
        "dataset": dataset_name,
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "epochs_ran": int(history[-1]["epoch"] if history else 0),
        "summary_metrics": summary,
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
    }
    del model, optimizer, scheduler, bundle, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return run


def load_store(output_dir: Path) -> Optional[Dict[str, Any]]:
    path = output_dir / "results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_store(output_dir: Path, store: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")


def find_done(runs: List[Dict[str, Any]], seed: int) -> Optional[Dict[str, Any]]:
    for run in runs:
        if int(run.get("seed", -1)) == int(seed) and "summary_metrics" in run:
            return run
    return None


def build_store(args: argparse.Namespace, output_dir: Path, splits: Dict[str, Any]) -> Dict[str, Any]:
    seeds = [args.seed_start + idx for idx in range(args.repeats)]
    method_labels = [format_method_label(method, loss_type) for method in args.methods for loss_type in args.losses]
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "methods": args.methods,
        "losses": args.losses,
        "method_labels": method_labels,
        "datasets": args.datasets,
        "repeats": args.repeats,
        "seeds": seeds,
        "gpu": args.gpu,
        "epochs": args.epochs,
        "houston_epochs": args.houston_epochs,
        "splits": splits,
        "results": {
            method_label: {
                dataset: {"runs": []}
                for dataset in args.datasets
            }
            for method_label in method_labels
        },
    }


def aggregate_store(store: Dict[str, Any]) -> Dict[str, Any]:
    aggregated: Dict[str, Any] = {}
    for method, datasets in store["results"].items():
        aggregated[method] = {}
        for dataset, info in datasets.items():
            runs = [run for run in info.get("runs", []) if "summary_metrics" in run]
            if not runs:
                continue
            metrics = {}
            for key in ("oa", "aa", "kappa", "macro_f1", "tail_aa"):
                metrics[key] = aggregate_values(run["summary_metrics"][key] for run in runs)
            resource = {}
            for key in (
                "train_seconds",
                "test_seconds",
                "flops_g",
                "params_m",
                "gpu_peak_allocated_mb",
                "gpu_peak_reserved_mb",
            ):
                resource[key] = aggregate_values(run.get("resource_metrics", {}).get(key, 0.0) for run in runs)
            num_classes = len(runs[0]["per_class"])
            per_class = []
            for idx in range(num_classes):
                rows = [run["per_class"][idx] for run in runs]
                acc = aggregate_values(row["acc"] for row in rows)
                per_class.append(
                    {
                        "display_id": rows[0]["display_id"],
                        "class_name": rows[0]["class_name"],
                        "support_mean": float(np.mean([row["support"] for row in rows])),
                        "acc_mean": acc["mean"],
                        "acc_std": acc["std"],
                    }
                )
            aggregated[method][dataset] = {
                "runs": len(runs),
                "metrics": metrics,
                "resource": resource,
                "per_class": per_class,
                "best_seed": max(runs, key=lambda item: item["summary_metrics"]["oa"])["seed"],
            }
    return aggregated


def pct(stats: Dict[str, float]) -> str:
    return f"{stats['mean'] * 100.0:.2f}±{stats['std'] * 100.0:.2f}"


def write_reports(output_dir: Path, store: Dict[str, Any]) -> None:
    aggregated = aggregate_store(store)
    (output_dir / "aggregated.json").write_text(json.dumps(aggregated, indent=2, ensure_ascii=False), encoding="utf-8")

    with (output_dir / "summary_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "dataset", "runs", "oa", "aa", "kappa", "macro_f1", "tail_aa", "best_seed"])
        for method, datasets in aggregated.items():
            for dataset, item in datasets.items():
                writer.writerow(
                    [
                        method,
                        dataset,
                        item["runs"],
                        pct(item["metrics"]["oa"]),
                        pct(item["metrics"]["aa"]),
                        pct(item["metrics"]["kappa"]),
                        pct(item["metrics"]["macro_f1"]),
                        pct(item["metrics"]["tail_aa"]),
                        item["best_seed"],
                    ]
                )

    with (output_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "dataset", "display_id", "class_name", "support_mean", "acc"])
        for method, datasets in aggregated.items():
            for dataset, item in datasets.items():
                for row in item["per_class"]:
                    writer.writerow(
                        [
                            method,
                            dataset,
                            row["display_id"],
                            row["class_name"],
                            f"{row['support_mean']:.1f}",
                            f"{row['acc_mean'] * 100.0:.2f}±{row['acc_std'] * 100.0:.2f}",
                        ]
                    )

    with (output_dir / "resource_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "dataset",
                "runs",
                "Train/s",
                "Test/s",
                "FLOPs/G",
                "Params/M",
                "GPU Peak Allocated/MB",
                "GPU Peak Reserved/MB",
            ]
        )
        for method, datasets in aggregated.items():
            for dataset, item in datasets.items():
                resource = item["resource"]
                writer.writerow(
                    [
                        method,
                        dataset,
                        item["runs"],
                        f"{resource['train_seconds']['mean']:.4f}",
                        f"{resource['test_seconds']['mean']:.4f}",
                        f"{resource['flops_g']['mean']:.4f}",
                        f"{resource['params_m']['mean']:.4f}",
                        f"{resource['gpu_peak_allocated_mb']['mean']:.2f}",
                        f"{resource['gpu_peak_reserved_mb']['mean']:.2f}",
                    ]
                )


def main() -> None:
    args = parse_args()
    output_dir = get_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = load_splits(args.splits)
    store = load_store(output_dir) if args.resume else None
    if store is None:
        store = build_store(args, output_dir, splits)
        save_store(output_dir, store)

    seeds = [args.seed_start + idx for idx in range(args.repeats)]
    print(f"[Output Dir] {output_dir}", flush=True)
    print(f"[Methods] {', '.join(args.methods)}", flush=True)
    print(f"[Losses] {', '.join(args.losses)}", flush=True)
    print(f"[Datasets] {', '.join(args.datasets)}", flush=True)
    print(f"[Seeds] {', '.join(str(seed) for seed in seeds)}", flush=True)
    print(f"[GPU] {args.gpu}", flush=True)
    print(flush=True)

    for method in args.methods:
        for loss_type in args.losses:
            method_label = format_method_label(method, loss_type)
            for dataset in args.datasets:
                split_cfg = splits[dataset]
                runs = store["results"].setdefault(method_label, {}).setdefault(dataset, {"runs": []})["runs"]
                for seed in seeds:
                    existing = find_done(runs, seed)
                    if existing is not None:
                        print(f"[Skip] {method_label} {dataset} seed={seed} OA={existing['summary_metrics']['oa']:.4f}", flush=True)
                        continue
                    print(f"[Run] {method_label} | {dataset} | seed={seed} | start", flush=True)
                    try:
                        run = train_one_run(method, loss_type, dataset, seed, split_cfg, args)
                        runs.append(run)
                        save_store(output_dir, store)
                        s = run["summary_metrics"]
                        print(
                            f"[Done] {method_label} | {dataset} | seed={seed} | "
                            f"OA={s['oa']:.4f} | AA={s['aa']:.4f} | Kappa={s['kappa']:.4f} | "
                            f"Macro-F1={s['macro_f1']:.4f} | Tail-AA={s.get('tail_aa', 0.0):.4f}",
                            flush=True,
                        )
                        write_reports(output_dir, store)
                    except Exception as exc:
                        runs.append({"seed": int(seed), "loss": loss_type, "error": repr(exc)})
                        save_store(output_dir, store)
                        print(f"[Failed] {method_label} | {dataset} | seed={seed} | {exc!r}", flush=True)
                        raise

    write_reports(output_dir, store)
    print("[Aggregation Complete]", flush=True)


if __name__ == "__main__":
    main()
