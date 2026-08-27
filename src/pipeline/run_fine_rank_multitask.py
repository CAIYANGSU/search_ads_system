"""Run strict-temporal Search Conversion multi-model fine ranking."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from search_ads_system.common.config import load_yaml_config
from search_ads_system.ranking.fine_rank_multitask import parse_fine_rank_multitask_config, run_fine_rank_multitask


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Conversion clicked-interaction fine rank")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--stage", choices=("sanity", "all"), required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config_path = args.config.resolve()
    config = parse_fine_rank_multitask_config(load_yaml_config(config_path), config_path, stage=args.stage)
    print(json.dumps(run_fine_rank_multitask(config, stage=args.stage), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
