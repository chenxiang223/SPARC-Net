from __future__ import annotations

import argparse
import copy
from pathlib import Path

import main as exp_main


PROJECT_DIR = Path(__file__).resolve().parent


DATASET_PRESETS = {
    "IndianPines": {
        "cube_path": str(PROJECT_DIR / "datasets" / "IndianPines" / "Indian_pines_corrected.npy"),
        "gt_path": str(PROJECT_DIR / "datasets" / "IndianPines" / "Indian_pines_gt.npy"),
        "cube_key": None,
        "gt_key": None,
        "input_layout": "HWC",
    },
    "Botswana": {
        "cube_path": str(PROJECT_DIR / "datasets" / "Botswana" / "Botswana.npy"),
        "gt_path": str(PROJECT_DIR / "datasets" / "Botswana" / "Botswana_gt.npy"),
        "cube_key": None,
        "gt_key": None,
        "input_layout": "HWC",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a customized HSI experiment with the upgraded fusion backbone.")
    parser.add_argument("--dataset", choices=sorted(DATASET_PRESETS), required=True)
    parser.add_argument("--train-mode", choices=("ratio", "fixed_per_class"), required=True)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--train-samples-per-class", type=int, default=None)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--stage1-epochs", type=int, default=None)
    parser.add_argument("--stage2-epochs", type=int, default=None)
    parser.add_argument("--disable-calibration", action="store_true")
    parser.add_argument("--disable-prototype-head", action="store_true")
    parser.add_argument("--disable-decoupled-engine", action="store_true")
    parser.add_argument("--amp", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    preset = DATASET_PRESETS[args.dataset]
    cfg = copy.deepcopy(exp_main.CONFIG)

    cfg["dataset"]["name"] = args.dataset
    cfg["dataset"]["cube_path"] = preset["cube_path"]
    cfg["dataset"]["cube_key"] = preset["cube_key"]
    cfg["dataset"]["gt_path"] = preset["gt_path"]
    cfg["dataset"]["gt_key"] = preset["gt_key"]
    cfg["dataset"]["input_layout"] = preset["input_layout"]

    cfg["split"]["train_mode"] = args.train_mode
    if args.train_mode == "ratio":
        if args.train_ratio is None:
            raise ValueError("--train-ratio is required when --train-mode=ratio")
        cfg["split"]["train_ratio"] = float(args.train_ratio)
    else:
        if args.train_samples_per_class is None:
            raise ValueError("--train-samples-per-class is required when --train-mode=fixed_per_class")
        cfg["split"]["train_samples_per_class"] = int(args.train_samples_per_class)

    cfg["trainer"]["print_every"] = int(args.print_every)
    if args.stage1_epochs is not None:
        cfg["trainer"]["stage1_epochs"] = int(args.stage1_epochs)
    if args.stage2_epochs is not None:
        cfg["trainer"]["stage2_epochs"] = int(args.stage2_epochs)
    if args.disable_calibration:
        cfg["calibration"]["enabled"] = False
    if args.disable_prototype_head:
        cfg["dataset"]["ablation"]["use_dynamic_prototype_head"] = False
    if args.disable_decoupled_engine:
        cfg["dataset"]["ablation"]["use_decoupled_training_engine"] = False
    if args.amp:
        cfg["trainer"]["amp"] = True

    exp_main.CONFIG = cfg
    if args.tag:
        print(f"[Experiment Tag] {args.tag}")
    print(f"[Dataset] {args.dataset}")
    print(f"[Train Mode] {args.train_mode}")
    if args.train_mode == "ratio":
        print(f"[Train Ratio] {cfg['split']['train_ratio']}")
    else:
        print(f"[Train Samples/Class] {cfg['split']['train_samples_per_class']}")
    print(f"[Prototype-Relation Head] {cfg['dataset']['ablation']['use_dynamic_prototype_head']}")
    print(f"[Decoupled Engine] {cfg['dataset']['ablation']['use_decoupled_training_engine']}")
    print(f"[AMP] {cfg['trainer']['amp']}")
    print(flush=True)
    exp_main.main()


if __name__ == "__main__":
    main()
