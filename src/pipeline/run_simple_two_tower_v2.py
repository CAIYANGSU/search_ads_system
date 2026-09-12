"""Run the independent strict-temporal simple Two-Tower v2 ablation."""
from __future__ import annotations
import argparse, json, sys
from dataclasses import replace
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from search_ads_system.common.config import load_yaml_config, resolve_path
from search_ads_system.recall.simple_two_tower_v2 import SimpleTwoTowerV2Config, run_ablation

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=ROOT / "config.yaml"); parser.add_argument("--stage", choices=("smoke", "preprocess", "benchmark", "tune"), default="smoke"); args = parser.parse_args()
    raw = load_yaml_config(args.config); root = args.config.resolve().parent; opt = dict(raw.get("simple_two_tower_v2", {})); temporal = raw.get("temporal", {})
    if opt.get("cache_dir"):
        opt["cache_dir"] = resolve_path(str(opt["cache_dir"]), root)
    output = resolve_path(str(opt.get("output_dir", "outputs/temporal")), root); base = resolve_path(str(temporal.get("output_dir", "outputs/temporal")), root) / "split"
    if not (base / "past").exists() or not (base / "future_a").exists(): raise FileNotFoundError("Build the repository's strict temporal split and Future-A split first (outputs/temporal/split/past and future_a).")
    if args.stage == "smoke": opt.update({"max_users": min(int(opt.get("max_users", 100000)), 1000), "max_train_rows": int(opt.get("smoke_max_train_rows", 5000)), "epochs": 1, "top_k": 50, "batch_size": min(int(opt.get("batch_size", 16384)), 2048)})
    cfg = SimpleTwoTowerV2Config(past_path=base / "past", future_a_path=base / "future_a", output_dir=output / ("simple_two_tower_v2_smoke" if args.stage == "smoke" else "."), **{key: value for key, value in opt.items() if key in SimpleTwoTowerV2Config.__dataclass_fields__})
    print("Resolved configuration:\n" + json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in cfg.__dict__.items()}, indent=2, default=str))
    if args.stage == "preprocess":
        from search_ads_system.recall.simple_two_tower_v2 import prepare_data
        data = prepare_data(cfg)
        print(json.dumps({"preprocessing": data.metadata, "cache_dir": str(cfg.cache_dir or cfg.output_dir / "simple_two_tower_v2_cache")}, indent=2, default=str))
    elif args.stage == "tune":
        reports = []
        for batch_size in (8192, 16384, 32768):
            tuned = replace(cfg, output_dir=output / "simple_two_tower_v2_tuning" / f"batch_{batch_size}", variants=("v0_id_only",), batch_size=batch_size, epochs=1, max_train_rows=min(int(opt.get("tuning_max_train_rows", 65536)), 65536), max_users=min(cfg.max_users, int(opt.get("tuning_max_users", 5000))))
            reports.append(run_ablation(tuned))
        print(__import__("pandas").concat(reports, ignore_index=True)[["model", "training_rows_per_second", "training_batches_per_second", "peak_gpu_mb", "peak_host_ram_mb"]].to_string(index=False))
    else: print(run_ablation(cfg).to_string(index=False))
if __name__ == "__main__": main()
