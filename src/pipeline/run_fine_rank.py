"""Command-line entry point for DCNv2 multi-task fine ranking."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from search_ads_system.common.config import load_yaml_config  # noqa: E402
from search_ads_system.evaluation.final_holdout import future_b_opened_warning  # noqa: E402
from search_ads_system.evaluation.temporal import build_future_ab_split, build_temporal_split, parse_temporal_config  # noqa: E402
from search_ads_system.ranking.fine_rank import build_dataset, evaluate_fine_ranker, load_fine_ranker, parse_fine_rank_config, run_fine_rank, train_fine_ranker  # noqa: E402
from search_ads_system.ranking.fine_rank_audit import parse_fine_rank_audit_config, run_fine_rank_audit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build, train, evaluate, infer, or audit DCNv2 fine ranking.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--stage", choices=("build_dataset", "train", "evaluate", "infer", "audit", "temporal_sanity", "all"), default="all")
    parser.add_argument("--temporal", action="store_true", help="Use isolated outputs/temporal Fine Rank artifacts for the selected stage.")
    parser.add_argument("--with-id-ablation", action="store_true", help="Run temporary ID ablation; in temporal mode also runs the compact 1/3/5-epoch sweep.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config_path = args.config.resolve()
    if warning := future_b_opened_warning(config_path): logging.warning(warning)
    raw = load_yaml_config(config_path)
    if args.temporal:
        fine = dict(raw.get("fine_rank", {})); fine["mode"] = "temporal"; raw["fine_rank"] = fine
    if args.stage == "temporal_sanity":
        fine = dict(raw.get("fine_rank", {})); temporal = dict(raw.get("temporal", {})); temporal_fine = dict(temporal.get("fine_rank", {})); sanity = temporal_fine.pop("sanity", {})
        if not isinstance(sanity, dict):
            raise ValueError("temporal.fine_rank.sanity must be a mapping")
        temporal_fine.update(sanity); temporal["fine_rank"] = temporal_fine; fine["mode"] = "temporal"; raw["fine_rank"] = fine; raw["temporal"] = temporal
    config = parse_fine_rank_config(raw, config_path)
    if args.stage == "temporal_sanity":
        temporal_root = config.metrics_path.parent.parent
        config = replace(
            config,
            output_path=temporal_root / "ranking" / "fine_rank_sanity_topk.csv",
            model_path=temporal_root / "models" / "fine_rank_dcnv2_sanity.pt",
            cache_dir=temporal_root / "ranking" / "fine_rank_sanity" / "train",
            metrics_path=temporal_root / "metrics" / "fine_rank_sanity_metrics.json",
        )
    if config.mode == "temporal":
        temporal_config = parse_temporal_config(raw, config_path)
        build_temporal_split(temporal_config)
        build_future_ab_split(temporal_config)
    if args.stage == "temporal_sanity":
        dataset = build_dataset(config)
        training = train_fine_ranker(config, dataset) if config.train else {"skipped": "fine_rank.train=false"}
        model, _ = load_fine_ranker(config)
        evaluation = evaluate_fine_ranker(model, config, metadata=dataset)
        config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        config.metrics_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True), encoding="utf-8")
        audit_config = replace(parse_fine_rank_audit_config(raw, config_path, config), output_path=config.metrics_path.parent / "fine_rank_sanity_audit.json")
        result = {"temporal_split": str(temporal_config.output_dir / "split" / "metadata.json"), "dataset": dataset, "training": training, "evaluation": evaluation, "audit": run_fine_rank_audit(config, audit_config, include_ablation=args.with_id_ablation)}
    else:
        result = (
            {"audit": run_fine_rank_audit(config, parse_fine_rank_audit_config(raw, config_path, config), include_ablation=args.with_id_ablation)}
            if args.stage == "audit"
            else run_fine_rank(config, stage=args.stage)
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
