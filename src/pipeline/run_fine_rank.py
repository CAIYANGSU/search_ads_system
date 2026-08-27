"""Command-line entry point for DCNv2 multi-task fine ranking."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from search_ads_system.common.config import load_yaml_config  # noqa: E402
from search_ads_system.ranking.fine_rank import parse_fine_rank_config, run_fine_rank  # noqa: E402
from search_ads_system.ranking.fine_rank_audit import parse_fine_rank_audit_config, run_fine_rank_audit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build, train, evaluate, infer, or audit DCNv2 fine ranking.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--stage", choices=("build_dataset", "train", "evaluate", "infer", "audit", "all"), default="all")
    parser.add_argument("--with-id-ablation", action="store_true", help="Run the small temporary A/B/C/D ID memorization experiment during --stage audit.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config_path = args.config.resolve()
    raw = load_yaml_config(config_path)
    config = parse_fine_rank_config(raw, config_path)
    result = (
        {"audit": run_fine_rank_audit(config, parse_fine_rank_audit_config(raw, config_path, config), include_ablation=args.with_id_ablation)}
        if args.stage == "audit"
        else run_fine_rank(config, stage=args.stage)
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
