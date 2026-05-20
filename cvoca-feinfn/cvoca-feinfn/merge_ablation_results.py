from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import run_ablation_benchmark as bench


def _resolve_results_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_dir():
        path = path / "results.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_store(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _dedupe_runs(runs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_seed: Dict[int, Dict[str, Any]] = {}
    for run in runs:
        if "seed" not in run:
            continue
        seed = int(run["seed"])
        if seed not in by_seed or "summary_metrics" in run:
            by_seed[seed] = run
    return [by_seed[seed] for seed in sorted(by_seed)]


def merge_stores(stores: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not stores:
        raise ValueError("At least one results store is required.")

    merged = copy.deepcopy(stores[0])
    merged["variants"] = {}

    for store in stores:
        for variant_key, variant_info in store.get("variants", {}).items():
            if variant_key not in merged["variants"]:
                merged["variants"][variant_key] = {
                    "label": variant_info.get("label", variant_key),
                    "description": variant_info.get("description", ""),
                    "runs": [],
                }
            merged["variants"][variant_key]["runs"].extend(variant_info.get("runs", []))

    all_seeds = set()
    for variant_info in merged["variants"].values():
        variant_info["runs"] = _dedupe_runs(variant_info["runs"])
        for run in variant_info["runs"]:
            if "summary_metrics" in run and "seed" in run:
                all_seeds.add(int(run["seed"]))

    merged["seeds"] = sorted(all_seeds)
    merged["repeats"] = len(merged["seeds"])
    merged.pop("aggregated", None)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge split ablation results.json files.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Result dirs or results.json files to merge.")
    parser.add_argument("--output-dir", required=True, help="Directory for the merged report.")
    args = parser.parse_args()

    stores = [_load_store(_resolve_results_path(path_text)) for path_text in args.inputs]
    merged = merge_stores(stores)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregated = bench.build_aggregated_results(merged)
    bench.save_store(output_dir, {**merged, "aggregated": aggregated})
    bench.write_summary_csv(output_dir, aggregated)
    bench.write_per_class_csv(output_dir, aggregated)
    bench.write_runs_csv(output_dir, merged)
    bench.write_markdown_report(output_dir, merged, aggregated)
    print(f"[Merged] {output_dir}")
    for variant_key, info in aggregated.items():
        oa = info["metric_stats"]["oa"]
        print(
            f"{info['label']}: OA mean={oa['mean']:.4f}, "
            f"std={oa['std']:.4f}, runs={info['completed_runs']}"
        )


if __name__ == "__main__":
    main()
